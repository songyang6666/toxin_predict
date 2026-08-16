from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def interpolate_daily_within_year(
    df: pd.DataFrame,
    date_column: str = "Time",
    sort_dates: bool = False,
    include_source_date_column: bool = False,
    group_column: str | None = None,
) -> pd.DataFrame:
    if group_column is not None:
        if group_column not in df.columns:
            raise KeyError(f"No grouping column named {group_column!r}")

        interpolated_groups = []
        for group_value, group in df.groupby(group_column, sort=False, dropna=False):
            interpolated = interpolate_daily_within_year(
                group.drop(columns=[group_column]),
                date_column=date_column,
                sort_dates=sort_dates,
                include_source_date_column=include_source_date_column,
            )
            interpolated.insert(1, group_column, group_value)
            interpolated_groups.append(interpolated)
        if not interpolated_groups:
            return pd.DataFrame(columns=["Date", group_column])
        return pd.concat(interpolated_groups, ignore_index=True)

    data = df.copy()
    data[date_column] = pd.to_datetime(data[date_column], format="%m/%d/%Y", errors="coerce")
    data = data.dropna(subset=[date_column])
    if sort_dates:
        data = data.sort_values(date_column)
    data = data.reset_index(drop=True)

    rows = []
    if include_source_date_column:
        value_columns = list(data.columns)
    else:
        value_columns = [column for column in data.columns if column != date_column]

    for index in range(len(data) - 1):
        current = data.iloc[index]
        next_row = data.iloc[index + 1]
        days = int((next_row[date_column] - current[date_column]).days)

        if current[date_column].year != next_row[date_column].year:
            rows.append([current[date_column], *[current[column] for column in value_columns]])
            continue

        if days <= 0:
            continue

        for offset in range(days):
            ratio = offset / days
            date = current[date_column] + timedelta(days=offset)
            values = [
                current[column] + (next_row[column] - current[column]) * ratio
                for column in value_columns
            ]
            rows.append([date, *values])

    if len(data):
        last = data.iloc[-1]
        rows.append([last[date_column], *[last[column] for column in value_columns]])

    return pd.DataFrame(rows, columns=["Date", *value_columns])


def add_shifted_target(
    df: pd.DataFrame,
    target_column: str,
    *,
    horizon_days: int = 7,
    date_column: str = "Date",
    output_column: str | None = None,
    drop_all_na: bool = False,
    group_column: str | None = None,
) -> pd.DataFrame:
    output_column = output_column or f"{target_column}_next_{horizon_days}_days"
    data = df.copy()
    data[output_column] = data[target_column].shift(-horizon_days)
    year = pd.to_datetime(data[date_column]).dt.year
    valid_target = year == year.shift(-horizon_days)
    if group_column is not None:
        if group_column not in data.columns:
            raise KeyError(f"No grouping column named {group_column!r}")
        valid_target &= data[group_column] == data[group_column].shift(-horizon_days)
    data.loc[~valid_target, output_column] = np.nan
    if drop_all_na:
        return data.dropna().reset_index(drop=True)
    return data.dropna(subset=[output_column]).reset_index(drop=True)
