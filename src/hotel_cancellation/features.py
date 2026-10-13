"""Point-in-time feature engineering shared by training and inference."""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_REQUIRED_COLUMNS

MONTH_TO_NUMBER = {name: number for number, name in enumerate(calendar.month_name) if name}


class FeatureContractError(ValueError):
    """Raised when booking data cannot satisfy the scoring contract."""


def arrival_dates(raw: pd.DataFrame) -> pd.Series:
    month = raw["arrival_date_month"].map(MONTH_TO_NUMBER)
    if month.isna().any():
        invalid = ", ".join(map(str, raw.loc[month.isna(), "arrival_date_month"].unique()[:3]))
        raise FeatureContractError(f"Unsupported arrival month values: {invalid}.")
    return pd.to_datetime(
        {
            "year": raw["arrival_date_year"],
            "month": month,
            "day": raw["arrival_date_day_of_month"],
        },
        errors="raise",
    )


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise FeatureContractError(f"Missing required columns: {', '.join(missing[:8])}.")

    frame = raw.copy()
    month_number = frame["arrival_date_month"].map(MONTH_TO_NUMBER)
    if month_number.isna().any():
        raise FeatureContractError("arrival_date_month contains unsupported values.")

    frame["total_nights"] = (
        frame["stays_in_weekend_nights"] + frame["stays_in_week_nights"]
    )
    frame["total_guests"] = frame[["adults", "children", "babies"]].fillna(0).sum(axis=1)
    frame["is_family"] = ((frame["children"].fillna(0) + frame["babies"].fillna(0)) > 0).astype(
        int
    )
    frame["has_agent"] = frame["agent"].notna().astype(int)
    frame["has_company"] = frame["company"].notna().astype(int)
    frame["agent_code"] = frame["agent"].fillna("Direct").astype(str)
    frame["arrival_month_sin"] = np.sin(2 * np.pi * month_number / 12)
    frame["arrival_month_cos"] = np.cos(2 * np.pi * month_number / 12)
    frame["arrival_week_sin"] = np.sin(2 * np.pi * frame["arrival_date_week_number"] / 53)
    frame["arrival_week_cos"] = np.cos(2 * np.pi * frame["arrival_date_week_number"] / 53)

    return frame[[*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]].copy()
