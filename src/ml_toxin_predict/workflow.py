from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_toxin_predict.datasets import (
    BASE_FEATURES,
    TOXIN_FEATURES,
    extract_features,
    read_workbook_sheet,
)
from ml_toxin_predict.predicting import add_shifted_target, interpolate_daily_within_year
from ml_toxin_predict.preprocessing import OutlierReport, prepare_numeric_features


TOXIN_MODEL_FEATURES = [
    "Total microcystin",
    "Temp",
    "Chla",
    "Depth",
    "Phycocyanin",
    "Wind",
    "DO",
    "PAR",
    "AMM",
    "Cond",
    "SD",
    "NN",
    "TP",
    "SRP",
    "TUR",
]


def prepare_daily_toxin_data(
    workbook: Path,
    sheet_name: str,
    *,
    outlier_contamination: float = 0.005,
    impute_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[OutlierReport]]:
    """Read, clean, impute, and interpolate the source observations."""
    raw = read_workbook_sheet(workbook, sheet_name)
    extracted = extract_features(
        raw,
        TOXIN_FEATURES + BASE_FEATURES,
        include_date=True,
        include_site=True,
    )
    numeric, reports = prepare_numeric_features(
        extracted.drop(columns=["Time", "Site"]),
        outlier_contamination=outlier_contamination,
        impute_neighbors=impute_neighbors,
        random_state=random_state,
    )
    prepared = extracted[["Time", "Site"]].join(numeric)
    daily = interpolate_daily_within_year(
        prepared,
        date_column="Time",
        sort_dates=True,
        group_column="Site",
    )
    daily["Total microcystin"] = (
        daily["Dissolved Microcystin"] + daily["Particulate Microcystin"]
    )
    return daily, reports


def build_horizon_dataset(
    daily: pd.DataFrame,
    horizon_days: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create model inputs and a future target for one prediction horizon."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")

    target_column = f"Total microcystin_next_{horizon_days}_days"
    prediction_data = add_shifted_target(
        daily,
        "Total microcystin",
        horizon_days=horizon_days,
        output_column=target_column,
        group_column="Site" if "Site" in daily.columns else None,
    )
    X = prediction_data[TOXIN_MODEL_FEATURES]
    y = prediction_data[target_column]
    return X, y, prediction_data
