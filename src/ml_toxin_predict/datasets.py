from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd

from .config import DEFAULT_WORKBOOK


@dataclass(frozen=True)
class FeatureSpec:
    output_name: str
    source_contains: str


BASE_FEATURES = [
    FeatureSpec("Chla", "Extracted Chlorophyll a"),
    FeatureSpec("TP", "Total Phosphorus"),
    FeatureSpec("AMM", "Ammonia"),
    FeatureSpec("SRP", "Soluble Reactive Phosphorus"),
    FeatureSpec("Temp", "CTD Temperature"),
    FeatureSpec("DO", "CTD Dissolved Oxygen"),
    FeatureSpec("TDP", "Total Dissolved Phosphorus"),
    FeatureSpec("NN", "Nitrate + Nitrite"),
    FeatureSpec("POC", "Particulate Organic Carbon"),
    FeatureSpec("Depth", "Station Depth"),
    FeatureSpec("PON", "Particulate Organic Nitrogen"),
    FeatureSpec("Cond", "CTD Specific Conductivity"),
    FeatureSpec("SD", "Secchi Depth"),
    FeatureSpec("TSS", "Total Suspended Solids"),
    FeatureSpec("Wind", "Wind speed"),
    FeatureSpec("PAR", "Photosynthetically Active Radiation"),
]

TOXIN_FEATURES = [
    FeatureSpec("Phycocyanin", "Extracted Phycocyanin"),
    FeatureSpec("Dissolved Microcystin", "Dissolved Microcystin"),
    FeatureSpec("Particulate Microcystin", "Particulate Microcystin"),
    FeatureSpec("TUR", "2_Turbidity"),
]


def read_workbook_sheet(
    workbook: Path = DEFAULT_WORKBOOK,
    sheet_name: str = "2012-2022_surface",
) -> pd.DataFrame:
    try:
        return pd.read_excel(workbook, sheet_name=sheet_name)
    except BadZipFile as exc:
        raise ValueError(
            f"Cannot read {workbook} as a valid .xlsx file. "
            "The file appears to be damaged or not saved as a standard Excel workbook. "
            "Open it in Excel/WPS/Numbers and use 'Save As' to create a new .xlsx file, "
            "then rerun the experiment with --workbook /path/to/the/new_file.xlsx."
        ) from exc


def extract_first_matching_column(df: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    matches = df.filter(like=spec.source_contains)
    if matches.empty:
        raise KeyError(f"No source column contains {spec.source_contains!r}")
    return matches.iloc[:, 0].rename(spec.output_name)


def extract_features(
    df: pd.DataFrame,
    specs: list[FeatureSpec],
    *,
    include_date: bool = False,
    include_site: bool = False,
) -> pd.DataFrame:
    columns = []
    if include_date:
        date_columns = df.filter(like="Date")
        if date_columns.empty:
            raise KeyError("No Date column found")
        columns.append(date_columns.iloc[:, 0].rename("Time"))

    if include_site:
        site_columns = df.filter(like="Site")
        if site_columns.empty:
            raise KeyError("No Site column found")
        columns.append(site_columns.iloc[:, 0].rename("Site"))

    for spec in specs:
        columns.append(extract_first_matching_column(df, spec))
    return pd.concat(columns, axis=1)
