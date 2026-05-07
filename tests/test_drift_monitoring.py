import numpy as np
import pandas as pd

from src.drift_monitoring import build_drift_report, calculate_psi, interpret_psi


def test_psi_is_low_for_identical_distributions():
    values = np.arange(100)

    psi = calculate_psi(values, values)

    assert psi < 0.001
    assert interpret_psi(psi) == "low shift"


def test_psi_increases_for_shifted_distribution():
    expected = np.arange(100)
    actual = np.arange(100) + 50

    psi = calculate_psi(expected, actual)

    assert psi > 0


def test_build_drift_report_contains_expected_columns():
    train = pd.DataFrame(
        {
            "EXT_SOURCE_2": [0.1, 0.2, 0.3, np.nan],
            "AMT_CREDIT": [100.0, 200.0, 300.0, 400.0],
        }
    )
    holdout = pd.DataFrame(
        {
            "EXT_SOURCE_2": [0.2, 0.3, np.nan, np.nan],
            "AMT_CREDIT": [150.0, 250.0, 350.0, 450.0],
        }
    )

    result = build_drift_report(train, holdout, features=["EXT_SOURCE_2", "AMT_CREDIT"])

    assert len(result) == 2
    assert {
        "feature",
        "psi",
        "train_missing_rate",
        "holdout_missing_rate",
        "missing_rate_delta",
        "mean_delta",
        "median_delta",
    } <= set(result.columns)
    ext_source = result.loc[result["feature"] == "EXT_SOURCE_2"].iloc[0]
    assert ext_source["missing_rate_delta"] == 0.25
