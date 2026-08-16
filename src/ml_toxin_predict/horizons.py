from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from ml_toxin_predict.config import DEFAULT_WORKBOOK, OUTPUT_DIR, PROJECT_ROOT
from ml_toxin_predict.modeling import fit_knn_grid_search, knn_param_grid, write_json
from ml_toxin_predict.workflow import build_horizon_dataset, prepare_daily_toxin_data


os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("prediction horizons must be positive integers")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare total microcystin prediction performance across time horizons."
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--sheet", default="2012-2022_surface_predict_toxi")
    parser.add_argument(
        "--horizons",
        type=positive_integer,
        nargs="+",
        default=list(range(1, 8)),
        help="Prediction horizons in days (default: 1 2 3 4 5 6 7).",
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outlier-contamination", type=float, default=0.005)
    parser.add_argument("--impute-neighbors", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "horizon_1_7_comparison",
    )
    return parser.parse_args(argv)


def write_comparison_plot(summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, r2_axis = plt.subplots(figsize=(7.2, 4.8))
    rmse_axis = r2_axis.twinx()

    r2_line = r2_axis.plot(
        summary["horizon_days"],
        summary["r2"],
        color="#2166ac",
        marker="o",
        linewidth=2,
        label="R2",
    )
    rmse_line = rmse_axis.plot(
        summary["horizon_days"],
        summary["rmse"],
        color="#b2182b",
        marker="s",
        linewidth=2,
        label="RMSE",
    )
    r2_axis.set_xlabel("Prediction horizon (days)")
    r2_axis.set_ylabel("R2", color="#2166ac")
    rmse_axis.set_ylabel("RMSE", color="#b2182b")
    r2_axis.set_xticks(summary["horizon_days"])
    r2_axis.grid(axis="y", alpha=0.25)
    r2_axis.legend(r2_line + rmse_line, ["R2", "RMSE"], loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_comparison(args: argparse.Namespace) -> pd.DataFrame:
    horizons = sorted(set(args.horizons))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily, reports = prepare_daily_toxin_data(
        Path(args.workbook),
        args.sheet,
        outlier_contamination=args.outlier_contamination,
        impute_neighbors=args.impute_neighbors,
        random_state=args.random_state,
    )
    param_grid = knn_param_grid()
    summary_rows = []

    for horizon_days in horizons:
        X, y, prediction_data = build_horizon_dataset(daily, horizon_days)
        _, test_metrics, predictions, metadata, cv_results = fit_knn_grid_search(
            X,
            y,
            param_grid=param_grid,
            test_size=args.test_size,
            random_state=args.random_state,
            cv=args.cv,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )

        horizon_dir = output_dir / f"{horizon_days}d"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        prediction_dates = pd.to_datetime(
            prediction_data.loc[predictions["sample_index"], "Date"]
        ).dt.strftime("%Y-%m-%d")
        predictions.insert(1, "date", prediction_dates.to_numpy())
        if "Site" in prediction_data.columns:
            prediction_sites = prediction_data.loc[predictions["sample_index"], "Site"]
            predictions.insert(2, "site", prediction_sites.to_numpy())
        predictions.to_csv(horizon_dir / "predictions.csv", index=False)
        cv_results.to_csv(horizon_dir / "cv_results.csv", index=False)

        metrics_payload = {
            "task": "toxin_predict",
            "model": "KNeighborsRegressor",
            "sheet": args.sheet,
            "horizon_days": horizon_days,
            "prediction_grouping": ["Site", "calendar_year"],
            "rows": len(X),
            "features": list(X.columns),
            "param_grid": param_grid,
            **metadata,
        }
        write_json(horizon_dir / "metrics.json", metrics_payload)

        best_params = metadata["best_params"]
        summary_rows.append(
            {
                "horizon_days": horizon_days,
                "rows": len(X),
                **test_metrics,
                "best_n_neighbors": best_params["n_neighbors"],
                "best_weights": best_params["weights"],
                "best_p": best_params["p"],
                "best_cv_rmse": metadata["best_cv_mse"] ** 0.5,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    write_json(
        output_dir / "summary.json",
        {
            "task": "toxin_horizon_comparison",
            "model": "KNeighborsRegressor",
            "sheet": args.sheet,
            "horizons": horizons,
            "prediction_grouping": ["Site", "calendar_year"],
            "split_strategy": "random",
            "test_size": args.test_size,
            "random_state": args.random_state,
            "cv": args.cv,
            "outliers": [report.__dict__ for report in reports],
            "results": summary.to_dict(orient="records"),
        },
    )
    write_comparison_plot(summary, output_dir / "horizon_r2_rmse.png")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_comparison(args)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
