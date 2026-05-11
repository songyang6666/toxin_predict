from __future__ import annotations

from ml_toxin_predict.cli import build_dataset, parse_args


def test_build_dataset_has_expected_shape():
    args = parse_args(["--verbose", "0"])
    X, y, reports, _, _ = build_dataset(args)

    assert len(X) == len(y)
    assert len(X) > 0
    assert "Total microcystin" in X.columns
    assert "Chla" in X.columns
    assert reports
