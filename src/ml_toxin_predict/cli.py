from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_toxin_predict.config import DEFAULT_WORKBOOK, OUTPUT_DIR, PROCESSED_DIR
from ml_toxin_predict.modeling import fit_knn_grid_search, knn_param_grid, write_json
from ml_toxin_predict.workflow import build_horizon_dataset, prepare_daily_toxin_data


def build_dataset(args: argparse.Namespace):
    """Prepare the total microcystin dataset for the requested horizon."""
    daily, reports = prepare_daily_toxin_data(
        Path(args.workbook),
        args.sheet,
        outlier_contamination=args.outlier_contamination,
        impute_neighbors=args.impute_neighbors,
        random_state=args.random_state,
    )
    X, y, predict = build_horizon_dataset(daily, args.horizon_days)
    return X, y, reports, daily, predict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible microcystin prediction experiment.")
    parser.add_argument("--workbook", type=str, default=str(DEFAULT_WORKBOOK))
    parser.add_argument("--sheet", default="2012-2022_surface_predict_toxi")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outlier-contamination", type=float, default=0.005)
    parser.add_argument("--impute-neighbors", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--write-processed", action="store_true")
    parser.add_argument("--make-explanations", action="store_true")
    parser.add_argument("--shap-samples", type=int, default=200)
    parser.add_argument("--dependence-feature", default="DO")
    parser.add_argument("--interaction-feature", default="Cond")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR / "toxin_predict"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    X, y, reports, daily, predict = build_dataset(args)
    param_grid = knn_param_grid()
    best_model, test_metrics, predictions, metadata, cv_results = fit_knn_grid_search(
        X,
        y,
        param_grid=param_grid,
        test_size=args.test_size,
        random_state=args.random_state,
        cv=args.cv,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dates = pd.to_datetime(
        predict.loc[predictions["sample_index"], "Date"]
    ).dt.strftime("%Y-%m-%d")
    predictions.insert(1, "date", prediction_dates.to_numpy())
    if "Site" in predict.columns:
        prediction_sites = predict.loc[predictions["sample_index"], "Site"]
        predictions.insert(2, "site", prediction_sites.to_numpy())
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)

    if args.write_processed:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_csv(PROCESSED_DIR / "toxin_daily_interpolated.csv", index=False)
        predict.to_csv(PROCESSED_DIR / "toxin_predict_dataset.csv", index=False)

    if args.make_explanations:
        from ml_toxin_predict.explainability import write_permutation_importance, write_shap_analysis

        X_train, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        write_permutation_importance(
            best_model,
            X_test,
            y_test,
            list(X.columns),
            output_dir / "permutation_importance.csv",
            random_state=args.random_state,
        )
        write_shap_analysis(
            best_model,
            X_test,
            X_test,
            output_dir / "shap",
            max_samples=args.shap_samples,
            random_state=args.random_state,
            dependence_feature=args.dependence_feature,
            interaction_feature=args.interaction_feature,
        )

    write_json(
        output_dir / "metrics.json",
        {
            "task": "toxin_predict",
            "model": "KNeighborsRegressor",
            "sheet": args.sheet,
            "horizon_days": args.horizon_days,
            "prediction_grouping": ["Site", "calendar_year"],
            "rows": len(X),
            "features": list(X.columns),
            "param_grid": param_grid,
            **metadata,
            "outliers": [report.__dict__ for report in reports],
        },
    )
    print(test_metrics)
