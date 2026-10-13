from __future__ import annotations

import pandas as pd
import pytest

from hotel_cancellation.config import ARTIFACTS_DIR, TARGET
from hotel_cancellation.scoring import ModelArtifacts, score_bookings


def test_duplicate_policy_matches_source_data(
    raw_data: pd.DataFrame, metadata: dict[str, object]
) -> None:
    duplicate_count = int(raw_data.duplicated(keep="first").sum())

    assert metadata["source_rows"] == len(raw_data)
    assert metadata["exact_duplicate_rows_removed"] == duplicate_count
    assert metadata["model_rows"] == len(raw_data) - duplicate_count


def test_persisted_split_is_complete_and_chronological(
    metadata: dict[str, object],
) -> None:
    split = pd.read_csv(ARTIFACTS_DIR / "split_assignments.csv", parse_dates=["arrival_date"])

    assert set(split["split"]) == {"train", "validation", "test"}
    assert len(split) == metadata["model_rows"]
    assert split.loc[split["split"] == "train", "arrival_date"].max() < pd.Timestamp(
        "2016-09-01"
    )
    validation_dates = split.loc[split["split"] == "validation", "arrival_date"]
    assert validation_dates.min() >= pd.Timestamp("2016-09-01")
    assert validation_dates.max() < pd.Timestamp("2017-01-01")
    assert split.loc[split["split"] == "test", "arrival_date"].min() >= pd.Timestamp(
        "2017-01-01"
    )
    assert split["source_row"].is_unique
    assert split[TARGET].isin([0, 1]).all()


def test_saved_artifacts_match_score_regression_fixture(
    raw_data: pd.DataFrame, model_artifacts: ModelArtifacts
) -> None:
    expected = pd.read_csv(ARTIFACTS_DIR / "score_regression.csv")
    bookings = raw_data.iloc[expected["source_row"].astype(int)].copy()

    actual = score_bookings(bookings, model_artifacts)

    assert actual.between(0, 1).all()
    assert actual.to_numpy() == pytest.approx(
        expected["cancellation_probability"].to_numpy(), abs=1e-7
    )


def test_single_and_batch_scoring_are_consistent(
    raw_data: pd.DataFrame, model_artifacts: ModelArtifacts
) -> None:
    bookings = raw_data.iloc[[0, 500, 10_000]].copy()
    batch = score_bookings(bookings, model_artifacts)
    single = pd.concat(
        [score_bookings(bookings.loc[[index]], model_artifacts) for index in bookings.index]
    )

    pd.testing.assert_series_equal(batch, single)


def test_empty_scoring_request_is_rejected(model_artifacts: ModelArtifacts) -> None:
    with pytest.raises(ValueError, match="No bookings"):
        score_bookings(pd.DataFrame(), model_artifacts)
