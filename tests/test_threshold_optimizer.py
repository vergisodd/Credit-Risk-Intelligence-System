import numpy as np

from src.threshold_optimizer import find_optimal_threshold


def test_find_optimal_threshold_returns_valid_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.20, 0.90, 0.35, 0.60, 0.80])
    threshold, cost, results, f1_threshold = find_optimal_threshold(y_true, y_proba, 10, 1)
    assert 0.05 <= threshold <= 0.95
    assert cost >= 0
    assert 0.05 <= f1_threshold <= 0.95
    assert not results.empty


def test_high_fn_cost_leads_to_lower_threshold_than_high_fp_cost():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.20, 0.90, 0.35, 0.60, 0.80])
    high_fn_threshold, _, _, _ = find_optimal_threshold(y_true, y_proba, 100, 1)
    high_fp_threshold, _, _, _ = find_optimal_threshold(y_true, y_proba, 1, 100)
    assert high_fn_threshold < high_fp_threshold


def test_results_dataframe_has_expected_columns():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.10, 0.40, 0.60, 0.90])
    _, _, results, _ = find_optimal_threshold(y_true, y_proba, 3, 1)
    assert {"threshold", "TN", "FP", "FN", "TP", "total_cost", "f1_default"} <= set(results.columns)
