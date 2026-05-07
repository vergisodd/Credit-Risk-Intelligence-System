import numpy as np

from src.threshold_optimizer import find_optimal_threshold


def test_find_optimal_threshold_returns_valid_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.20, 0.90, 0.35, 0.60, 0.80])
    result = find_optimal_threshold(y_true, y_proba, 10, 1)
    assert 0.05 <= result.cost_minimizing_threshold <= 0.95
    assert result.min_cost >= 0
    assert 0.05 <= result.f1_optimal_threshold <= 0.95
    assert 0 <= result.f1_at_optimal <= 1
    assert not result.threshold_table.empty


def test_high_fn_cost_leads_to_lower_threshold_than_high_fp_cost():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.20, 0.90, 0.35, 0.60, 0.80])
    high_fn_threshold = find_optimal_threshold(y_true, y_proba, 100, 1).cost_minimizing_threshold
    high_fp_threshold = find_optimal_threshold(y_true, y_proba, 1, 100).cost_minimizing_threshold
    assert high_fn_threshold < high_fp_threshold


def test_results_dataframe_has_expected_columns():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.10, 0.40, 0.60, 0.90])
    results = find_optimal_threshold(y_true, y_proba, 3, 1).threshold_table
    assert {"threshold", "TN", "FP", "FN", "TP", "total_cost", "f1_default"} <= set(results.columns)


def test_threshold_result_summary_uses_explicit_names():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.10, 0.40, 0.60, 0.90])
    summary = find_optimal_threshold(y_true, y_proba, 3, 1).to_summary_dict()
    assert {
        "cost_minimizing_threshold",
        "min_cost",
        "f1_optimal_threshold",
        "f1_at_optimal",
        "fn_cost",
        "fp_cost",
    } <= set(summary)
