from __future__ import annotations

import numpy as np

from hotel_cancellation.calibration import PlattCalibrator


def test_calibrator_round_trip_is_portable() -> None:
    probabilities = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    calibrator = PlattCalibrator(slope=1.2, intercept=-0.3)

    restored = PlattCalibrator.from_dict(calibrator.to_dict())

    np.testing.assert_allclose(restored.predict(probabilities), calibrator.predict(probabilities))


def test_positive_platt_slope_preserves_risk_order() -> None:
    probabilities = np.linspace(0.01, 0.99, 100)
    calibrator = PlattCalibrator(slope=1.2, intercept=-0.3)

    calibrated = calibrator.predict(probabilities)

    assert np.all(np.diff(calibrated) > 0)
    assert np.all((calibrated > 0) & (calibrated < 1))


def test_unfitted_calibrator_fails_explicitly() -> None:
    calibrator = PlattCalibrator()

    try:
        calibrator.predict(np.array([0.5]))
    except RuntimeError as error:
        assert "not been fitted" in str(error)
    else:
        raise AssertionError("An unfitted calibrator must not score probabilities.")
