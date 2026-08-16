from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_toxin_predict.datasets import (
    BASE_FEATURES,
    TOXIN_FEATURES,
    extract_features,
    read_workbook_sheet,
)
from ml_toxin_predict.modeling import knn_param_grid
from ml_toxin_predict.preprocessing import clean_numeric_frame
from ml_toxin_predict.workflow import TOXIN_MODEL_FEATURES


REGRESSION_METRICS = ("mae", "rmse", "r2")
EVENT_METRICS = ("pod", "far", "csi")
SUMMARY_METRICS = REGRESSION_METRICS + EVENT_METRICS


@dataclass(frozen=True)
class BlockedSplit:
    split_type: str
    fold: str
    train_index: pd.Index
    test_index: pd.Index


def prepare_observed_toxin_data(workbook: Path, sheet_name: str) -> pd.DataFrame:
    """Read observed station data without temporal interpolation."""
    raw = read_workbook_sheet(workbook, sheet_name)
    extracted = extract_features(
        raw,
        TOXIN_FEATURES + BASE_FEATURES,
        include_date=True,
        include_site=True,
    )
    numeric = clean_numeric_frame(extracted.drop(columns=["Time", "Site"]))
    data = extracted[["Time", "Site"]].join(numeric)
    data["Time"] = pd.to_datetime(data["Time"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["Time", "Site"])

    numeric_columns = list(numeric.columns)
    data = (
        data.groupby(["Site", "Time"], as_index=False, sort=True)[numeric_columns]
        .mean()
        .sort_values(["Site", "Time"])
        .reset_index(drop=True)
    )
    data["Total microcystin"] = (
        data["Dissolved Microcystin"] + data["Particulate Microcystin"]
    )
    return data


def make_weekly_observed_pairs(
    observations: pd.DataFrame,
    *,
    nominal_horizon_days: int = 7,
    tolerance_days: int = 2,
) -> pd.DataFrame:
    """Pair consecutive same-site observations near a nominal weekly horizon."""
    if nominal_horizon_days < 1:
        raise ValueError("nominal_horizon_days must be at least 1")
    if tolerance_days < 0 or tolerance_days >= nominal_horizon_days:
        raise ValueError("tolerance_days must be between 0 and horizon_days - 1")

    minimum_days = nominal_horizon_days - tolerance_days
    maximum_days = nominal_horizon_days + tolerance_days
    rows: list[dict] = []

    for site, group in observations.groupby("Site", sort=False):
        group = group.sort_values("Time").reset_index(drop=True)
        for index in range(len(group) - 1):
            current = group.iloc[index]
            future = group.iloc[index + 1]
            actual_horizon_days = int((future["Time"] - current["Time"]).days)
            if not minimum_days <= actual_horizon_days <= maximum_days:
                continue
            if pd.isna(current["Total microcystin"]) or pd.isna(
                future["Total microcystin"]
            ):
                continue

            rows.append(
                {
                    **{feature: current[feature] for feature in TOXIN_MODEL_FEATURES},
                    "Site": site,
                    "current_date": current["Time"],
                    "target_date": future["Time"],
                    "target_year": int(future["Time"].year),
                    "station_year": f"{site}-{future['Time'].year}",
                    "nominal_horizon_days": nominal_horizon_days,
                    "actual_horizon_days": actual_horizon_days,
                    "Total microcystin_target": future["Total microcystin"],
                }
            )

    if not rows:
        raise ValueError(
            "No same-site consecutive observations matched the requested horizon window"
        )
    return pd.DataFrame(rows).sort_values(["target_date", "Site"]).reset_index(drop=True)


def build_weekly_dataset(
    workbook: Path,
    sheet_name: str,
    *,
    nominal_horizon_days: int = 7,
    tolerance_days: int = 2,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    observations = prepare_observed_toxin_data(workbook, sheet_name)
    pairs = make_weekly_observed_pairs(
        observations,
        nominal_horizon_days=nominal_horizon_days,
        tolerance_days=tolerance_days,
    )
    X = pairs[TOXIN_MODEL_FEATURES].copy()
    y = pairs["Total microcystin_target"].copy()
    metadata = pairs.drop(
        columns=[*TOXIN_MODEL_FEATURES, "Total microcystin_target"]
    ).copy()
    return X, y, metadata, pairs


def iter_blocked_splits(
    metadata: pd.DataFrame,
    split_type: str,
) -> list[BlockedSplit]:
    if split_type == "year":
        group_column = "target_year"
    elif split_type == "station_year":
        group_column = "station_year"
    else:
        raise ValueError("split_type must be 'year' or 'station_year'")

    groups = metadata[group_column]
    splits = []
    for fold in sorted(groups.unique(), key=str):
        test_mask = groups == fold
        splits.append(
            BlockedSplit(
                split_type=split_type,
                fold=str(fold),
                train_index=metadata.index[~test_mask],
                test_index=metadata.index[test_mask],
            )
        )
    return splits


def regression_and_event_metrics(
    y_true,
    y_pred,
    *,
    threshold: float = 1.0,
) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(truth, prediction)

    observed_event = truth >= threshold
    predicted_event = prediction >= threshold
    hits = int(np.sum(observed_event & predicted_event))
    misses = int(np.sum(observed_event & ~predicted_event))
    false_alarms = int(np.sum(~observed_event & predicted_event))
    correct_negatives = int(np.sum(~observed_event & ~predicted_event))

    pod_denominator = hits + misses
    far_denominator = hits + false_alarms
    csi_denominator = hits + misses + false_alarms
    r2 = (
        float(r2_score(truth, prediction))
        if len(truth) >= 2 and not np.isclose(np.var(truth), 0.0)
        else np.nan
    )
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mse**0.5),
        "r2": r2,
        "pod": hits / pod_denominator if pod_denominator else np.nan,
        "far": false_alarms / far_denominator if far_denominator else np.nan,
        "csi": hits / csi_denominator if csi_denominator else np.nan,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
    }


def seasonal_climatology_predictions(
    y_train: pd.Series,
    metadata_train: pd.DataFrame,
    metadata_test: pd.DataFrame,
) -> np.ndarray:
    """Predict training-only site-month climatology with conservative fallbacks."""
    climatology = metadata_train[["Site", "target_date"]].copy()
    climatology["target"] = y_train.to_numpy()
    climatology["month"] = pd.to_datetime(climatology["target_date"]).dt.month

    site_month = climatology.groupby(["Site", "month"])["target"].mean()
    month = climatology.groupby("month")["target"].mean()
    global_mean = float(climatology["target"].mean())

    predictions = []
    for row in metadata_test.itertuples():
        target_month = pd.Timestamp(row.target_date).month
        site_month_key = (row.Site, target_month)
        if site_month_key in site_month.index:
            predictions.append(float(site_month.loc[site_month_key]))
        elif target_month in month.index:
            predictions.append(float(month.loc[target_month]))
        else:
            predictions.append(global_mean)
    return np.asarray(predictions)


def make_knn_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor()),
        ]
    )


def fit_blocked_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    inner_groups: pd.Series,
    *,
    inner_cv: int = 5,
    n_jobs: int = 1,
    verbose: int = 0,
) -> tuple[Pipeline, dict[str, object]]:
    unique_groups = inner_groups.nunique()
    if unique_groups < 2:
        raise ValueError("At least two training groups are required for blocked tuning")
    n_splits = min(inner_cv, unique_groups)
    grid = GridSearchCV(
        estimator=make_knn_pipeline(),
        param_grid={
            f"model__{parameter}": values
            for parameter, values in knn_param_grid().items()
        },
        cv=GroupKFold(n_splits=n_splits),
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        verbose=verbose,
    )
    grid.fit(X_train, y_train, groups=inner_groups)
    details = {
        "best_params": {
            key.removeprefix("model__"): value
            for key, value in grid.best_params_.items()
        },
        "best_inner_cv_rmse": float((-grid.best_score_) ** 0.5),
        "inner_cv_splits": int(n_splits),
    }
    return grid.best_estimator_, details


def evaluate_blocked_validation(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    *,
    split_types: tuple[str, ...] = ("year", "station_year"),
    threshold: float = 1.0,
    inner_cv: int = 5,
    n_jobs: int = 1,
    verbose: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    prediction_frames = []
    horizon_days = int(metadata["nominal_horizon_days"].iloc[0])

    for split_type in split_types:
        for split in iter_blocked_splits(metadata, split_type):
            X_train = X.loc[split.train_index]
            X_test = X.loc[split.test_index]
            y_train = y.loc[split.train_index]
            y_test = y.loc[split.test_index]
            metadata_train = metadata.loc[split.train_index]
            metadata_test = metadata.loc[split.test_index]
            inner_group_column = (
                "target_year" if split_type == "year" else "station_year"
            )

            model, model_details = fit_blocked_knn(
                X_train,
                y_train,
                metadata_train[inner_group_column],
                inner_cv=inner_cv,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            methods = {
                "knn": model.predict(X_test),
                "persistence": X_test["Total microcystin"].to_numpy(),
                "seasonal_climatology": seasonal_climatology_predictions(
                    y_train,
                    metadata_train,
                    metadata_test,
                ),
            }

            for method, prediction in methods.items():
                metrics = regression_and_event_metrics(
                    y_test,
                    prediction,
                    threshold=threshold,
                )
                is_model = method == "knn"
                fold_rows.append(
                    {
                        "horizon_days": horizon_days,
                        "split_type": split_type,
                        "fold": split.fold,
                        "method": method,
                        "model": "KNN" if is_model else "none",
                        "baseline": "none" if is_model else method,
                        "train_rows": len(X_train),
                        "test_rows": len(X_test),
                        "threshold": threshold,
                        "best_params": (
                            json.dumps(model_details["best_params"], sort_keys=True)
                            if is_model
                            else ""
                        ),
                        "best_inner_cv_rmse": (
                            model_details["best_inner_cv_rmse"] if is_model else np.nan
                        ),
                        "inner_cv_splits": (
                            model_details["inner_cv_splits"] if is_model else np.nan
                        ),
                        **metrics,
                    }
                )

                predictions = metadata_test.reset_index(names="sample_index").copy()
                predictions.insert(0, "split_type", split_type)
                predictions.insert(1, "fold", split.fold)
                predictions.insert(2, "method", method)
                predictions["y_true"] = y_test.to_numpy()
                predictions["y_pred"] = prediction
                prediction_frames.append(predictions)

    return pd.DataFrame(fold_rows), pd.concat(prediction_frames, ignore_index=True)


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "horizon_days",
        "split_type",
        "method",
        "model",
        "baseline",
        "threshold",
    ]
    rows = []
    for keys, group in fold_metrics.groupby(group_columns, sort=False, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        row["folds"] = group["fold"].nunique()
        row["test_rows"] = int(group["test_rows"].sum())
        for metric in SUMMARY_METRICS:
            values = group[metric].dropna()
            row[f"{metric}_valid_folds"] = int(len(values))
            mean = float(values.mean()) if len(values) else np.nan
            sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_mean_sd"] = _format_mean_sd(mean, sd)
        rows.append(row)
    return pd.DataFrame(rows)


def _format_mean_sd(mean: float, sd: float) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(sd):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {sd:.4f}"
