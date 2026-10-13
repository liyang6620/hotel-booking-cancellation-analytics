"""Portable Platt probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibrator:
    slope: float | None = None
    intercept: float | None = None

    @staticmethod
    def _logits(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        estimator = LogisticRegression(random_state=42)
        estimator.fit(self._logits(probabilities), labels)
        self.slope = float(estimator.coef_[0, 0])
        self.intercept = float(estimator.intercept_[0])
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if self.slope is None or self.intercept is None:
            raise RuntimeError("The calibrator has not been fitted.")
        logits = self.slope * self._logits(probabilities).ravel() + self.intercept
        return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))

    def to_dict(self) -> dict[str, float | str]:
        if self.slope is None or self.intercept is None:
            raise RuntimeError("The calibrator has not been fitted.")
        return {"method": "platt", "slope": self.slope, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, payload: dict[str, float | str]) -> PlattCalibrator:
        if payload.get("method") != "platt":
            raise ValueError("Unsupported calibration method.")
        return cls(slope=float(payload["slope"]), intercept=float(payload["intercept"]))
