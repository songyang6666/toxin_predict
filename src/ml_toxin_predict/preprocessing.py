from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer


@dataclass(frozen=True)
class OutlierReport:
    column: str
    original_non_nan: int
    filtered_non_nan: int
    outliers: int


def coerce_detection_limit(value):
    """Convert strings like '<0.05' to half the reported detection limit."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return np.nan
        if value.lower() in {"bdl", "nd"}:
            return np.nan
        if "<" in value:
            return float(value.split("<")[-1]) / 2
    return float(value)


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda column: column.map(coerce_detection_limit))


def set_outliers_to_nan(
    column: pd.Series,
    contamination: float = 0.005,
    random_state: int = 42,
) -> tuple[np.ndarray, OutlierReport]:
    non_nan_mask = ~pd.isna(column)
    clean_column = column[non_nan_mask].astype(float).to_numpy(copy=True).reshape(-1, 1)
    original_non_nan = int(non_nan_mask.sum())

    full_column = np.full(column.shape, np.nan)
    if len(clean_column) == 0:
        return full_column, OutlierReport(column.name, original_non_nan, 0, 0)

    model = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_mask = model.fit_predict(clean_column) == -1
    clean_column[outlier_mask] = np.nan
    full_column[non_nan_mask] = clean_column.flatten()

    filtered_non_nan = int(np.sum(~np.isnan(full_column)))
    report = OutlierReport(
        column=str(column.name),
        original_non_nan=original_non_nan,
        filtered_non_nan=filtered_non_nan,
        outliers=int(np.sum(outlier_mask)),
    )
    return full_column, report


def remove_column_outliers(
    df: pd.DataFrame,
    contamination: float = 0.005,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[OutlierReport]]:
    filtered = pd.DataFrame(index=df.index)
    reports: list[OutlierReport] = []
    for column_name in df.columns:
        values, report = set_outliers_to_nan(
            df[column_name],
            contamination=contamination,
            random_state=random_state,
        )
        filtered[column_name] = values
        reports.append(report)
    return filtered, reports


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)


def prepare_numeric_features(
    df: pd.DataFrame,
    *,
    outlier_contamination: float = 0.005,
    impute_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[OutlierReport]]:
    cleaned = clean_numeric_frame(df)
    filtered, reports = remove_column_outliers(
        cleaned,
        contamination=outlier_contamination,
        random_state=random_state,
    )
    imputed = impute_knn(filtered, n_neighbors=impute_neighbors)
    return imputed, reports
