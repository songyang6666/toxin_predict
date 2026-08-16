from __future__ import annotations

import argparse
from pathlib import Path

from ml_toxin_predict.config import DEFAULT_WORKBOOK, OUTPUT_DIR
from ml_toxin_predict.modeling import write_json
from ml_toxin_predict.weekly import (
    EVENT_METRICS,
    REGRESSION_METRICS,
    SUPPORTED_MODELS,
    build_weekly_dataset,
    evaluate_blocked_validation,
    summarize_fold_metrics,
    summarize_pooled_predictions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate non-interpolated weekly toxin predictions with blocked "
            "validation and reviewer-requested baselines."
        )
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--sheet", default="2012-2022_surface_predict_toxi")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--tolerance-days", type=int, default=2)
    parser.add_argument(
        "--split-types",
        choices=["year", "station_year"],
        nargs="+",
        default=["year", "station_year"],
    )
    parser.add_argument(
        "--models",
        choices=SUPPORTED_MODELS,
        nargs="+",
        default=list(SUPPORTED_MODELS),
    )
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "weekly_blocked_validation",
    )
    return parser.parse_args(argv)


def render_summary_text(summary) -> str:
    display = summary[
        [
            "horizon_days",
            "split_type",
            "model",
            "baseline",
            "folds",
            "test_rows",
            *[f"{metric}_mean_sd" for metric in REGRESSION_METRICS],
            *[f"{metric}_mean_sd" for metric in EVENT_METRICS],
        ]
    ].rename(
        columns={
            "horizon_days": "horizon",
            "test_rows": "N",
            "mae_mean_sd": "MAE (mean +/- sd)",
            "rmse_mean_sd": "RMSE (mean +/- sd)",
            "r2_mean_sd": "R2 (mean +/- sd)",
            "pod_mean_sd": "POD (mean +/- sd)",
            "far_mean_sd": "FAR (mean +/- sd)",
            "csi_mean_sd": "CSI (mean +/- sd)",
        }
    )
    return display.to_string(index=False)


def run_validation(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata, pairs = build_weekly_dataset(
        Path(args.workbook),
        args.sheet,
        nominal_horizon_days=args.horizon_days,
        tolerance_days=args.tolerance_days,
    )
    fold_metrics, predictions = evaluate_blocked_validation(
        X,
        y,
        metadata,
        split_types=tuple(dict.fromkeys(args.split_types)),
        model_names=tuple(dict.fromkeys(args.models)),
        threshold=args.threshold,
        inner_cv=args.inner_cv,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
    summary = summarize_fold_metrics(fold_metrics)
    pooled_metrics = summarize_pooled_predictions(
        predictions,
        threshold=args.threshold,
    )
    summary_text = render_summary_text(summary)

    pairs.to_csv(output_dir / "weekly_observed_pairs.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pooled_metrics.to_csv(output_dir / "pooled_metrics.csv", index=False)
    (output_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    write_json(
        output_dir / "summary.json",
        {
            "task": "toxin_weekly_blocked_validation",
            "pairing": (
                "consecutive same-site observed samples; no temporal interpolation"
            ),
            "nominal_horizon_days": args.horizon_days,
            "accepted_actual_horizon_days": [
                args.horizon_days - args.tolerance_days,
                args.horizon_days + args.tolerance_days,
            ],
            "rows": len(pairs),
            "split_types": list(dict.fromkeys(args.split_types)),
            "split_definitions": {
                "year": "leave one target year out",
                "station_year": "leave one monitoring-site target-year block out",
            },
            "models": list(dict.fromkeys(args.models)),
            "model_selection": (
                "nested group-blocked CV with fold-local preprocessing and "
                "nonnegative predictions"
            ),
            "baselines": {
                "persistence": "current observed total microcystin",
                "seasonal_climatology": (
                    "training-only site-month mean with month and global fallbacks"
                ),
            },
            "event_threshold_micrograms_per_liter": args.threshold,
            "event_metrics": {
                "POD": "hits / (hits + misses)",
                "FAR": "false alarms / (hits + false alarms)",
                "CSI": "hits / (hits + misses + false alarms)",
            },
            "aggregation": "unweighted mean and sample SD across held-out blocks",
            "results": summary.astype(object)
            .where(summary.notna(), None)
            .to_dict(orient="records"),
            "supplementary_pooled_results": pooled_metrics.astype(object)
            .where(pooled_metrics.notna(), None)
            .to_dict(orient="records"),
        },
    )
    return summary, summary_text


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _, summary_text = run_validation(args)
    print(summary_text)


if __name__ == "__main__":
    main()
