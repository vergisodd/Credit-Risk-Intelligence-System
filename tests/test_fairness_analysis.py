import numpy as np
import pandas as pd

from src.fairness_analysis import disaggregated_metrics, rate_metrics


def test_rate_metrics_calculates_fpr_and_fnr():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.10, 0.80, 0.40, 0.90])
    metrics = rate_metrics(y_true, y_proba, threshold=0.50)
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1


def test_disaggregated_metrics_filters_allowed_groups():
    X_test = pd.DataFrame({"CODE_GENDER": ["F", "F", "M", "XNA"]})
    y_test = pd.Series([0, 1, 1, 0])
    y_proba = np.array([0.20, 0.70, 0.80, 0.90])
    result = disaggregated_metrics(
        X_test,
        y_test,
        y_proba,
        column="CODE_GENDER",
        threshold=0.50,
        allowed_groups=["F", "M"],
    )
    assert set(result["group"]) == {"F", "M"}
    assert "equalized_odds_gap" in result.columns
