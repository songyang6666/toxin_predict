from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from ml_toxin_predict.cli import build_dataset, parse_args
from ml_toxin_predict.modeling import fit_knn_grid_search
from ml_toxin_predict.workflow import TOXIN_MODEL_FEATURES, build_horizon_dataset


def test_build_dataset_has_expected_shape():
    args = parse_args(["--verbose", "0"])
    X, y, reports, _, _ = build_dataset(args)

    assert len(X) == len(y)
    assert len(X) > 0
    assert "Total microcystin" in X.columns
    assert "Chla" in X.columns
    assert reports


def test_horizon_target_does_not_cross_calendar_years():
    dates = pd.to_datetime(
        ["2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02", "2021-01-03"]
    )
    daily = pd.DataFrame({"Date": dates})
    for feature in TOXIN_MODEL_FEATURES:
        daily[feature] = range(len(dates))

    _, y, prediction_data = build_horizon_dataset(daily, horizon_days=2)

    assert prediction_data["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2021-01-01"]
    assert y.tolist() == [4]


def test_horizon_target_does_not_cross_sites():
    dates = pd.to_datetime(["2021-06-01", "2021-06-02"] * 2)
    daily = pd.DataFrame({"Date": dates, "Site": ["A", "A", "B", "B"]})
    for feature in TOXIN_MODEL_FEATURES:
        daily[feature] = [0, 1, 10, 11]

    _, y, prediction_data = build_horizon_dataset(daily, horizon_days=1)

    assert prediction_data["Site"].tolist() == ["A", "B"]
    assert y.tolist() == [1, 11]


def test_grid_search_uses_a_scaling_pipeline():
    X = pd.DataFrame(
        {
            "feature_a": range(20),
            "feature_b": [value % 3 for value in range(20)],
        }
    )
    y = pd.Series([value * 0.5 for value in range(20)])

    model, _, predictions, metadata, _ = fit_knn_grid_search(
        X,
        y,
        param_grid={"n_neighbors": [3], "weights": ["distance"], "p": [1]},
        cv=2,
        verbose=0,
    )

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["scaler", "model"]
    assert metadata["best_params"] == {"n_neighbors": 3, "p": 1, "weights": "distance"}
    assert list(predictions.columns) == ["sample_index", "y_true", "y_pred"]
