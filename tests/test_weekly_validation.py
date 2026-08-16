from __future__ import annotations

import numpy as np
import pandas as pd

from ml_toxin_predict.weekly import (
    SUPPORTED_MODELS,
    iter_blocked_splits,
    make_model_spec,
    make_weekly_observed_pairs,
    regression_and_event_metrics,
    seasonal_climatology_predictions,
    summarize_pooled_predictions,
)
from ml_toxin_predict.workflow import TOXIN_MODEL_FEATURES


def make_observations() -> pd.DataFrame:
    observations = pd.DataFrame(
        {
            "Site": ["A", "A", "A", "B", "B"],
            "Time": pd.to_datetime(
                ["2020-06-01", "2020-06-08", "2020-06-22", "2020-06-01", "2020-06-10"]
            ),
        }
    )
    for feature in TOXIN_MODEL_FEATURES:
        observations[feature] = np.arange(len(observations), dtype=float)
    observations["Total microcystin"] = [0.5, 1.5, 4.0, 2.0, 3.0]
    return observations


def test_weekly_pairs_use_only_consecutive_observed_dates():
    pairs = make_weekly_observed_pairs(
        make_observations(),
        nominal_horizon_days=7,
        tolerance_days=2,
    )

    assert pairs[["Site", "actual_horizon_days"]].to_records(index=False).tolist() == [
        ("A", 7),
        ("B", 9),
    ]
    assert pairs["target_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-06-08",
        "2020-06-10",
    ]
    assert pairs["Total microcystin_target"].tolist() == [1.5, 3.0]


def test_blocked_splits_hold_out_complete_groups():
    metadata = pd.DataFrame(
        {
            "target_year": [2020, 2020, 2021, 2021],
            "station_year": ["A-2020", "B-2020", "A-2021", "B-2021"],
        }
    )

    for split_type, group_column in [
        ("year", "target_year"),
        ("station_year", "station_year"),
    ]:
        for split in iter_blocked_splits(metadata, split_type):
            train_groups = set(metadata.loc[split.train_index, group_column])
            test_groups = set(metadata.loc[split.test_index, group_column])
            assert train_groups.isdisjoint(test_groups)
            assert len(test_groups) == 1


def test_event_metrics_at_one_microgram_per_liter():
    metrics = regression_and_event_metrics(
        [2.0, 2.0, 0.0, 0.0],
        [2.0, 0.0, 2.0, 0.0],
        threshold=1.0,
    )

    assert metrics["hits"] == 1
    assert metrics["misses"] == 1
    assert metrics["false_alarms"] == 1
    assert metrics["pod"] == 0.5
    assert metrics["far"] == 0.5
    assert metrics["csi"] == 1 / 3


def test_seasonal_climatology_uses_training_data_only():
    y_train = pd.Series([1.0, 3.0, 10.0])
    metadata_train = pd.DataFrame(
        {
            "Site": ["A", "A", "B"],
            "target_date": pd.to_datetime(
                ["2019-06-01", "2020-06-15", "2020-07-01"]
            ),
        }
    )
    metadata_test = pd.DataFrame(
        {
            "Site": ["A", "C", "C"],
            "target_date": pd.to_datetime(
                ["2021-06-01", "2021-07-01", "2021-08-01"]
            ),
        }
    )

    predictions = seasonal_climatology_predictions(
        y_train,
        metadata_train,
        metadata_test,
    )

    assert predictions.tolist() == [2.0, 10.0, 14.0 / 3.0]


def test_all_comparison_models_have_unique_specs():
    specs = [make_model_spec(name) for name in SUPPORTED_MODELS]

    assert [spec.name for spec in specs] == list(SUPPORTED_MODELS)
    assert len({spec.label for spec in specs}) == len(SUPPORTED_MODELS)
    assert all(spec.pipeline.steps for spec in specs)


def test_pooled_metrics_keep_methods_separate():
    predictions = pd.DataFrame(
        {
            "split_type": ["year"] * 4,
            "method": ["linear", "linear", "persistence", "persistence"],
            "model": ["Linear Regression", "Linear Regression", "none", "none"],
            "baseline": ["none", "none", "persistence", "persistence"],
            "nominal_horizon_days": [7] * 4,
            "y_true": [0.0, 2.0, 0.0, 2.0],
            "y_pred": [0.0, 2.0, 2.0, 0.0],
        }
    )

    pooled = summarize_pooled_predictions(predictions)

    assert pooled["method"].tolist() == ["linear", "persistence"]
    assert pooled["mae"].tolist() == [0.0, 2.0]
