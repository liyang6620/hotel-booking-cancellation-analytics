from __future__ import annotations

import pandas as pd
import pytest

from hotel_cancellation.config import (
    CATEGORICAL_FEATURES,
    DIRECT_LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    POST_BOOKING_COLUMNS,
)
from hotel_cancellation.features import FeatureContractError, arrival_dates, build_features


def test_feature_contract_rejects_missing_required_fields(raw_data: pd.DataFrame) -> None:
    incomplete = raw_data.head(1).drop(columns="lead_time")

    with pytest.raises(FeatureContractError, match="lead_time"):
        build_features(incomplete)


def test_booking_creation_features_exclude_future_information(raw_data: pd.DataFrame) -> None:
    features = build_features(raw_data.head(10))

    assert features.columns.tolist() == [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
    prohibited = set(DIRECT_LEAKAGE_COLUMNS) | set(POST_BOOKING_COLUMNS)
    assert prohibited.isdisjoint(features.columns)


def test_invalid_arrival_month_is_rejected(raw_data: pd.DataFrame) -> None:
    invalid = raw_data.head(1).copy()
    invalid["arrival_date_month"] = "NotAMonth"

    with pytest.raises(FeatureContractError, match="Unsupported arrival month"):
        arrival_dates(invalid)
