"""Business impact simulation for champion review-routing policies."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.champion_model import (
    get_champion_holdout,
    load_champion_model,
    load_model_manifest,
)
from src.config_loader import ensure_parent_dir, load_config
from src.model_utils import save_dataframe
from src.threshold_optimizer import find_optimal_threshold


def _threshold_predictions(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Flag applicants whose score is at or above a threshold."""
    return (y_score >= threshold).astype(int)


def _capacity_predictions(y_score: np.ndarray, capacity: float) -> np.ndarray:
    """Flag the highest-scored applicants up to a capacity share."""
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in the interval (0, 1].")
    review_count = int(np.ceil(len(y_score) * capacity))
    order = np.argsort(-y_score, kind="mergesort")
    y_pred = np.zeros(len(y_score), dtype=int)
    y_pred[order[:review_count]] = 1
    return y_pred


def summarize_policy(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    policy_name: str,
    threshold_or_capacity: str | float,
    fn_cost: float,
    fp_cost: float,
) -> dict:
    """Calculate manual review policy metrics."""
    y_true_array = np.asarray(y_true).astype(int)
    y_pred_array = np.asarray(y_pred).astype(int)
    total_defaults = int(y_true_array.sum())
    applicants_reviewed = int(y_pred_array.sum())
    true_defaults_captured = int(((y_true_array == 1) & (y_pred_array == 1)).sum())
    false_reviews = int(((y_true_array == 0) & (y_pred_array == 1)).sum())
    missed_defaults = int(((y_true_array == 1) & (y_pred_array == 0)).sum())
    applicant_count = len(y_true_array)

    return {
        "policy_name": policy_name,
        "threshold_or_capacity": threshold_or_capacity,
        "applicants_reviewed": applicants_reviewed,
        "review_rate": applicants_reviewed / applicant_count if applicant_count else np.nan,
        "true_defaults_captured": true_defaults_captured,
        "default_capture_rate": (
            true_defaults_captured / total_defaults if total_defaults else np.nan
        ),
        "false_reviews": false_reviews,
        "false_review_rate": (
            false_reviews / applicants_reviewed if applicants_reviewed else np.nan
        ),
        "missed_defaults": missed_defaults,
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "total_cost_units": float((missed_defaults * fn_cost) + (false_reviews * fp_cost)),
    }


def _manifest_thresholds(config: dict, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Resolve threshold policies from manifest, falling back to holdout optimization."""
    manifest_policy = load_model_manifest(config).get("threshold_policy", {})
    lender = config["thresholds"]["business_scenarios"]["lender"]
    optimized = find_optimal_threshold(
        y_true,
        y_score,
        fn_cost=lender["fn_cost"],
        fp_cost=lender["fp_cost"],
    )
    return {
        "default_threshold": float(config["thresholds"]["default"]),
        "f1_optimal_threshold": float(
            manifest_policy.get("f1_optimal_threshold", optimized.f1_optimal_threshold)
        ),
        "cost_minimizing_threshold": float(
            manifest_policy.get("cost_minimizing_threshold", optimized.cost_minimizing_threshold)
        ),
    }


def build_business_impact_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    fn_cost: float,
    fp_cost: float,
    thresholds: dict[str, float],
    capacities: tuple[float, ...] = (0.10, 0.15, 0.20),
) -> pd.DataFrame:
    """Compare threshold and top-capacity manual review policies."""
    y_true_array = np.asarray(y_true).astype(int)
    y_score_array = np.asarray(y_score, dtype=float)
    policies = [
        (
            "Default threshold 0.50",
            thresholds["default_threshold"],
            _threshold_predictions(y_score_array, thresholds["default_threshold"]),
        ),
        (
            "F1-optimal threshold",
            thresholds["f1_optimal_threshold"],
            _threshold_predictions(y_score_array, thresholds["f1_optimal_threshold"]),
        ),
        (
            "Cost-minimizing threshold",
            thresholds["cost_minimizing_threshold"],
            _threshold_predictions(
                y_score_array,
                thresholds["cost_minimizing_threshold"],
            ),
        ),
    ]
    for capacity in capacities:
        policies.append(
            (
                f"Top {int(capacity * 100)}% review capacity",
                f"{capacity:.0%}",
                _capacity_predictions(y_score_array, capacity),
            )
        )

    rows = [
        summarize_policy(
            y_true_array,
            y_score_array,
            y_pred,
            policy_name,
            threshold_or_capacity,
            fn_cost,
            fp_cost,
        )
        for policy_name, threshold_or_capacity, y_pred in policies
    ]
    return pd.DataFrame(rows)


def save_business_impact_visual(policy_df: pd.DataFrame, config: dict) -> None:
    """Save a compact policy comparison chart."""
    output_path = ensure_parent_dir(config["visuals"]["business_impact_policy_comparison"])
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(policy_df))
    width = 0.35
    ax1.bar(
        x - width / 2,
        policy_df["default_capture_rate"],
        width,
        label="Default capture rate",
        color="#1B4F8A",
    )
    ax1.bar(
        x + width / 2,
        policy_df["review_rate"],
        width,
        label="Review rate",
        color="#B45309",
    )
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(policy_df["policy_name"], rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, policy_df["total_cost_units"], color="#111827", marker="o", label="Cost units")
    ax2.set_ylabel("Cost Units")
    ax2.legend(loc="upper right")

    ax1.set_title("Business Impact Policy Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_business_impact_markdown(policy_df: pd.DataFrame, config: dict) -> None:
    """Write the business impact simulation report."""
    output_path = ensure_parent_dir(config["reports"]["business_impact_simulation_md"])
    lender = config["thresholds"]["business_scenarios"]["lender"]
    markdown_table = policy_df.to_markdown(index=False, floatfmt=".4f")
    report = f"""# Business Impact Simulation

This report evaluates the champion model as a **manual review-routing policy**, not an automated approval or rejection engine.

Cost units use the configured lender scenario: false negatives = {lender["fn_cost"]} units and false positives = {lender["fp_cost"]} unit. These are relative portfolio demonstration units, not dollars and not claimed financial savings.

{markdown_table}

## Interpretation

Thresholds and review-capacity rules change the tradeoff among captured defaults, false reviews, missed defaults, precision, recall, F1, and cost units. This turns the model from a leaderboard score into a decision-support workflow that can be reviewed by analysts and stakeholders.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Generate business impact simulation outputs."""
    config = load_config()
    model = load_champion_model(config)
    _, X_test, _, y_test = get_champion_holdout(config)
    y_score = model.predict_proba(X_test)[:, 1]
    lender = config["thresholds"]["business_scenarios"]["lender"]
    thresholds = _manifest_thresholds(config, y_test.to_numpy(), y_score)

    policy_df = build_business_impact_table(
        y_true=y_test,
        y_score=y_score,
        fn_cost=lender["fn_cost"],
        fp_cost=lender["fp_cost"],
        thresholds=thresholds,
    )
    save_dataframe(policy_df, config["reports"]["business_impact_simulation_csv"])
    save_business_impact_visual(policy_df, config)
    write_business_impact_markdown(policy_df, config)
    print(
        f"Business impact simulation saved to {config['reports']['business_impact_simulation_csv']}"
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        raise SystemExit(
            f"Data/model unavailable for business impact simulation: {error}"
        ) from error
