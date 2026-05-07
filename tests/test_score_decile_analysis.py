import numpy as np

from src.score_decile_analysis import assign_score_deciles, calculate_decile_lift_table


def test_decile_output_has_10_groups_when_enough_rows_exist():
    y_true = np.array([1, 0] * 50)
    y_score = np.linspace(1, 0, 100)

    result = calculate_decile_lift_table(y_true, y_score)

    assert result["decile"].nunique() == 10
    assert set(result["decile"]) == set(range(1, 11))


def test_highest_risk_decile_has_label_1():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_score = np.array([0.10, 0.99, 0.20, 0.95, 0.30, 0.90, 0.40, 0.80, 0.50, 0.70])

    assigned = assign_score_deciles(y_true, y_score)

    assert assigned.iloc[0]["decile"] == 1
    assert assigned.iloc[0]["predicted_score"] == 0.99


def test_lift_calculation_equals_default_rate_over_base_rate():
    y_true = np.array([1] * 10 + [0] * 90)
    y_score = np.linspace(1, 0, 100)

    result = calculate_decile_lift_table(y_true, y_score)
    top_decile = result.loc[result["decile"] == 1].iloc[0]

    assert np.isclose(top_decile["lift_vs_base_rate"], top_decile["actual_default_rate"] / 0.10)


def test_cumulative_capture_rate_is_between_zero_and_one():
    y_true = np.array([1] * 25 + [0] * 75)
    y_score = np.linspace(1, 0, 100)

    result = calculate_decile_lift_table(y_true, y_score)

    assert result["cumulative_default_capture_rate"].between(0, 1).all()
    assert result["cumulative_default_capture_rate"].iloc[-1] == 1
