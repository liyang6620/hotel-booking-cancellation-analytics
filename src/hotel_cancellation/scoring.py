"""Load persisted artifacts and score bookings at creation time."""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .calibration import PlattCalibrator
from .config import CALIBRATOR_PATH, MODEL_PATH, PREPROCESSOR_PATH
from .features import build_features


@dataclass(frozen=True)
class ModelArtifacts:
    preprocessor: object
    model: XGBClassifier
    calibrator: PlattCalibrator


def load_artifacts() -> ModelArtifacts:
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    calibrator = PlattCalibrator.from_dict(
        json.loads(CALIBRATOR_PATH.read_text(encoding="utf-8"))
    )
    return ModelArtifacts(
        preprocessor=joblib.load(PREPROCESSOR_PATH), model=model, calibrator=calibrator
    )


def score_bookings(raw: pd.DataFrame, artifacts: ModelArtifacts) -> pd.Series:
    if raw.empty:
        raise ValueError("No bookings were provided for scoring.")
    features = build_features(raw)
    transformed = artifacts.preprocessor.transform(features)
    raw_probabilities = artifacts.model.predict_proba(transformed)[:, 1]
    probabilities = artifacts.calibrator.predict(raw_probabilities)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("The model returned invalid probabilities.")
    return pd.Series(probabilities, index=raw.index, name="cancellation_probability")
