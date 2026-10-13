from __future__ import annotations

import json

import pandas as pd
import pytest

from hotel_cancellation.config import DATA_PATH, METADATA_PATH
from hotel_cancellation.scoring import ModelArtifacts, load_artifacts


@pytest.fixture(scope="session")
def raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="session")
def metadata() -> dict[str, object]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def model_artifacts() -> ModelArtifacts:
    return load_artifacts()
