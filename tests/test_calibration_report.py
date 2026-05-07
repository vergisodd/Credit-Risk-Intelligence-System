import numpy as np

from src.calibration_report import (
    calculate_calibration_by_decile,
    calculate_calibration_by_risk_tier,
)


def test_calibration_by_decile_has_10_groups():
    y_true = np.array([1] * 20 + [0] * 80)
    y_score = np.linspace(1, 0, 100)

    result = calculate_calibration_by_decile(y_true, y_score)

    assert result["decile"].nunique() == 10
    assert {"observed_default_rate", "mean_predicted_risk", "calibration_error"} <= set(
        result.columns
    )


def test_calibration_error_is_mean_score_minus_observed_rate():
    y_true = np.array([1] * 10 + [0] * 90)
    y_score = np.linspace(1, 0, 100)

    result = calculate_calibration_by_decile(y_true, y_score)
    top_decile = result.loc[result["decile"] == 1].iloc[0]

    assert np.isclose(
        top_decile["calibration_error"],
        top_decile["mean_predicted_risk"] - top_decile["observed_default_rate"],
    )


def test_calibration_by_risk_tier_uses_configured_thresholds():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_score = np.array([0.10, 0.20, 0.35, 0.55, 0.70, 0.90])
    risk_tiers = {"low": 0.30, "medium": 0.60, "high": 1.01}

    result = calculate_calibration_by_risk_tier(y_true, y_score, risk_tiers)

    assert set(result["risk_tier"].astype(str)) == {"Low", "Medium", "High"}
    assert result["applicant_count"].sum() == 6
