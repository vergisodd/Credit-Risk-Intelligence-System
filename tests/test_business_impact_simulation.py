import numpy as np

from src.business_impact_simulation import build_business_impact_table, summarize_policy


def test_summarize_policy_calculates_review_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.90, 0.20, 0.80, 0.10])
    y_pred = np.array([1, 0, 1, 0])

    result = summarize_policy(
        y_true,
        y_score,
        y_pred,
        "test policy",
        0.50,
        fn_cost=10,
        fp_cost=1,
    )

    assert result["applicants_reviewed"] == 2
    assert result["true_defaults_captured"] == 1
    assert result["false_reviews"] == 1
    assert result["missed_defaults"] == 1
    assert result["total_cost_units"] == 11
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_business_impact_table_contains_expected_policies():
    y_true = np.array([1] * 10 + [0] * 90)
    y_score = np.linspace(1, 0, 100)
    thresholds = {
        "default_threshold": 0.50,
        "f1_optimal_threshold": 0.70,
        "cost_minimizing_threshold": 0.30,
    }

    result = build_business_impact_table(
        y_true,
        y_score,
        fn_cost=10,
        fp_cost=1,
        thresholds=thresholds,
    )

    assert len(result) == 6
    assert {
        "Default threshold 0.50",
        "F1-optimal threshold",
        "Cost-minimizing threshold",
        "Top 10% review capacity",
        "Top 15% review capacity",
        "Top 20% review capacity",
    } == set(result["policy_name"])


def test_capacity_policy_reviews_expected_share():
    y_true = np.array([1] * 10 + [0] * 90)
    y_score = np.linspace(1, 0, 100)
    thresholds = {
        "default_threshold": 0.50,
        "f1_optimal_threshold": 0.70,
        "cost_minimizing_threshold": 0.30,
    }

    result = build_business_impact_table(
        y_true,
        y_score,
        fn_cost=10,
        fp_cost=1,
        thresholds=thresholds,
    )
    top_10 = result.loc[result["policy_name"] == "Top 10% review capacity"].iloc[0]

    assert top_10["applicants_reviewed"] == 10
    assert top_10["review_rate"] == 0.10
    assert 0 <= top_10["default_capture_rate"] <= 1
