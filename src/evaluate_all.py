"""Evaluate and compare all trained model pipelines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.config_loader import ensure_parent_dir, load_config, resolve_path
from src.champion_model import get_champion_spec, save_model_manifest
from src.model_utils import (
    evaluate_predictions,
    load_engineered_data,
    load_engineered_data_with_bureau,
    load_model_artifact,
    make_train_test_split,
    save_dataframe,
)
from src.threshold_optimizer import ThresholdOptimizationResult, find_optimal_threshold


MODEL_SPECS = [
    ("Logistic Regression", "logistic_model", "baseline_metrics", "application", False),
    ("XGBoost", "xgboost_model", "xgboost_metrics", "application", False),
    ("LightGBM", "lightgbm_model", "lightgbm_metrics", "application", False),
    ("LightGBM+Bureau", "lightgbm_bureau_model", "lightgbm_bureau_metrics", "bureau", True),
]


def get_scores(config: dict, test_sets: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Load model artifacts and score the shared holdout set."""
    scores = {}
    for model_name, artifact_key, _, test_set_key, optional in MODEL_SPECS:
        artifact_path = resolve_path(config["artifacts"][artifact_key])
        if optional and not artifact_path.exists():
            continue
        model = load_model_artifact(config["artifacts"][artifact_key])
        scores[model_name] = model.predict_proba(test_sets[test_set_key])[:, 1]
    return scores


def get_cv_auc(config: dict, model_name: str, report_key: str) -> float:
    """Read the model's reported cross-validation AUC for comparison output."""
    report_path = resolve_path(config["reports"][report_key])
    if not report_path.exists():
        return float("nan")
    with report_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    if model_name in {"LightGBM", "LightGBM+Bureau"}:
        return float(metrics.get("tuned_cv_auc_mean", metrics.get("cv_auc_mean", float("nan"))))
    return float(metrics.get("cv_auc_mean", float("nan")))


def build_comparison_table(
    y_test: pd.Series | np.ndarray,
    scores: dict[str, np.ndarray],
    default_threshold: float,
    config: dict,
) -> pd.DataFrame:
    """Create the all-model comparison table."""
    rows = []
    lender = config["thresholds"]["business_scenarios"]["lender"]
    for model_name, y_proba in scores.items():
        report_key = next(
            report_key for spec_name, _, report_key, _, _ in MODEL_SPECS if spec_name == model_name
        )
        threshold_result = find_optimal_threshold(
            y_test,
            y_proba,
            fn_cost=lender["fn_cost"],
            fp_cost=lender["fp_cost"],
        )
        f1_metrics = evaluate_predictions(y_test, y_proba, threshold_result.f1_optimal_threshold)
        cost_metrics = evaluate_predictions(y_test, y_proba, threshold_result.cost_minimizing_threshold)
        default_metrics = evaluate_predictions(y_test, y_proba, default_threshold)
        rows.append(
            {
                "Model": model_name,
                "AUC-ROC": roc_auc_score(y_test, y_proba),
                "Average Precision": average_precision_score(y_test, y_proba),
                "Tuned CV AUC": get_cv_auc(config, model_name, report_key),
                "F1-Default": default_metrics["f1_default_class"],
                "Precision-Default": default_metrics["precision_default_class"],
                "Recall-Default": default_metrics["recall_default_class"],
                "F1-Optimal Threshold": threshold_result.f1_optimal_threshold,
                "F1 at F1-Optimal": threshold_result.f1_at_optimal,
                "Cost-Min Threshold": threshold_result.cost_minimizing_threshold,
                "Min Relative Cost": threshold_result.min_cost,
                "Precision-Selected": f1_metrics["precision_default_class"],
                "Recall-Selected": f1_metrics["recall_default_class"],
                "Review Volume-Selected": int(f1_metrics["predicted_default_count"]),
                "Missed Defaults-Selected": int(f1_metrics["false_negatives"]),
                "Review Volume-Cost-Min": int(cost_metrics["predicted_default_count"]),
                "Missed Defaults-Cost-Min": int(cost_metrics["false_negatives"]),
            }
        )
    return pd.DataFrame(rows)


def bureau_improvement_note(comparison_df: pd.DataFrame) -> str | None:
    """Describe the holdout AUC lift from bureau features when available."""
    model_names = set(comparison_df["Model"])
    if {"LightGBM", "LightGBM+Bureau"} - model_names:
        return None
    lightgbm_auc = float(comparison_df.loc[comparison_df["Model"] == "LightGBM", "AUC-ROC"].iloc[0])
    bureau_auc = float(comparison_df.loc[comparison_df["Model"] == "LightGBM+Bureau", "AUC-ROC"].iloc[0])
    improvement = bureau_auc - lightgbm_auc
    if improvement <= 0:
        return None
    return f"LightGBM+Bureau achieved +{improvement:.4f} AUC improvement from bureau feature integration"


def save_combined_roc(
    y_test: pd.Series | np.ndarray,
    scores: dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Save a combined ROC curve for all models."""
    output_file = ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, y_proba in scores.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_value = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{model_name} AUC={auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", label="Random")
    ax.set_title("ROC Comparison - All Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def save_combined_pr(
    y_test: pd.Series | np.ndarray,
    scores: dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Save a combined Precision-Recall curve for all models."""
    output_file = ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, y_proba in scores.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap_value = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f"{model_name} AP={ap_value:.3f}")
    ax.set_title("Precision-Recall Comparison - All Models")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def save_calibration_plot(
    y_test: pd.Series | np.ndarray,
    scores: dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Save a reliability diagram for XGBoost and LightGBM."""
    output_file = ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name in ["XGBoost", "LightGBM", "LightGBM+Bureau"]:
        if model_name not in scores:
            continue
        prob_true, prob_pred = calibration_curve(y_test, scores[model_name], n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", label="Perfect calibration")
    ax.set_title("Calibration Plot")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Default Rate")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def save_cost_threshold_outputs(
    y_test: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    config: dict,
) -> ThresholdOptimizationResult:
    """Save cost-threshold CSV and visual for the champion model."""
    scenario_a = config["thresholds"]["business_scenarios"]["lender"]
    scenario_b = config["thresholds"]["business_scenarios"]["balanced"]
    optimization_a = find_optimal_threshold(
        y_test,
        y_proba,
        fn_cost=scenario_a["fn_cost"],
        fp_cost=scenario_a["fp_cost"],
    )
    optimization_b = find_optimal_threshold(
        y_test,
        y_proba,
        fn_cost=scenario_b["fn_cost"],
        fp_cost=scenario_b["fp_cost"],
    )
    results_a = optimization_a.threshold_table
    results_b = optimization_b.threshold_table
    output_df = pd.DataFrame(
        {
            "threshold": results_a["threshold"],
            "FN": results_a["FN"],
            "FP": results_a["FP"],
            "total_cost_A": results_a["total_cost"],
            "total_cost_B": results_b["total_cost"],
            "f1_default": results_a["f1_default"],
        }
    )
    save_dataframe(output_df, config["reports"]["cost_threshold_analysis"])

    output_file = ensure_parent_dir(config["visuals"]["cost_threshold_analysis"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(output_df["threshold"], output_df["total_cost_A"], label="Scenario A cost")
    ax.plot(output_df["threshold"], output_df["total_cost_B"], label="Scenario B cost")
    ax.axvline(optimization_a.f1_optimal_threshold, color="#111827", linestyle="--", label="F1-optimal")
    ax.axvline(optimization_a.cost_minimizing_threshold, color="#1B4F8A", linestyle=":", label="Lender cost-min")
    ax.axvline(optimization_b.cost_minimizing_threshold, color="#B45309", linestyle=":", label="Balanced cost-min")
    ax.set_title("Cost Threshold Analysis")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total Relative Cost")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return optimization_a


def update_model_comparison_report(
    comparison_df: pd.DataFrame,
    config: dict,
    note: str | None = None,
) -> None:
    """Write a compact model comparison Markdown report."""
    best_row = comparison_df.sort_values("AUC-ROC", ascending=False).iloc[0]
    output_file = ensure_parent_dir(config["reports"]["model_comparison_md"])
    markdown_table = comparison_df.to_csv(sep="|", index=False, float_format="%.4f")
    markdown_table = markdown_table.replace("|", " | ")
    header, *rows = markdown_table.strip().splitlines()
    separator = " | ".join(["---"] * len(comparison_df.columns))
    markdown_table = "\n".join([f"| {header} |", f"| {separator} |", *[f"| {row} |" for row in rows]])
    low_tier = config["thresholds"]["risk_tiers"]["low"]
    high_tier = config["thresholds"]["risk_tiers"]["medium"]
    champion = get_champion_spec(config)
    note_block = f"\n{note}\n" if note else ""
    report = f"""# Model Comparison Report

## Summary

The champion model is **{champion.model_name}** (`{champion.feature_set}` feature set). The strongest holdout model in the comparison is **{best_row["Model"]}** with ROC-AUC {best_row["AUC-ROC"]:.4f} and Average Precision {best_row["Average Precision"]:.4f}.

{markdown_table}
{note_block}

## Threshold Definitions

`default_threshold` is the conventional 0.50 classifier cutoff.

`cost_minimizing_threshold` minimizes a stated false-negative/false-positive cost scenario. In the bundled lender scenario, false negatives are weighted 10x false positives.

`f1_optimal_threshold` maximizes default-class F1. This is the configured operating threshold for the portfolio review queue because it balances precision and recall for manual review prioritization.

`risk_tiers` are score bands used for analyst triage and are not the same thing as a binary classifier threshold.

## Calibration Interpretation

The calibration plot compares predicted default probabilities with observed default rates across probability bins. For risk-tiering, calibration matters because a score near {high_tier:.2f} should behave like a materially higher-risk applicant group than a score near {low_tier:.2f}. If the curve sits far from the diagonal, predicted probabilities are still useful for ranking, but the exact percentages should be treated as review scores rather than literal default-rate estimates.
"""
    output_file.write_text(report, encoding="utf-8")


def main() -> None:
    """Run all-model evaluation and comparison reporting."""
    config = load_config()
    print("Loading holdout split...")
    X, y = load_engineered_data()
    _, X_test, _, y_test = make_train_test_split(X, y, config)
    test_sets = {"application": X_test}
    bureau_artifact_path = resolve_path(config["artifacts"]["lightgbm_bureau_model"])
    if bureau_artifact_path.exists():
        X_bureau, y_bureau = load_engineered_data_with_bureau()
        _, X_test_bureau, _, y_test_bureau = make_train_test_split(X_bureau, y_bureau, config)
        if not y_test.reset_index(drop=True).equals(y_test_bureau.reset_index(drop=True)):
            raise ValueError("Bureau and application holdout splits do not align.")
        test_sets["bureau"] = X_test_bureau

    print("Scoring all trained model pipelines...")
    scores = get_scores(config, test_sets)
    comparison_df = build_comparison_table(y_test, scores, config["thresholds"]["default"], config)
    improvement_note = bureau_improvement_note(comparison_df)
    if improvement_note:
        print(improvement_note)

    print("Saving comparison outputs...")
    save_dataframe(comparison_df, config["reports"]["model_comparison_full"])
    save_combined_roc(y_test, scores, config["visuals"]["roc_comparison_all_models"])
    save_combined_pr(y_test, scores, config["visuals"]["pr_comparison_all_models"])
    save_calibration_plot(y_test, scores, config["visuals"]["calibration_plot"])
    champion_name = get_champion_spec(config).model_name
    if champion_name not in scores:
        raise FileNotFoundError(
            f"Champion model scores for {champion_name} are unavailable. "
            "Train the champion model and ensure bureau.csv is available."
        )
    threshold_optimization = save_cost_threshold_outputs(y_test, scores[champion_name], config)
    save_model_manifest(config, threshold_optimization)
    update_model_comparison_report(comparison_df, config, improvement_note)

    print()
    print(comparison_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"Comparison saved to: {resolve_path(config['reports']['model_comparison_full'])}")


if __name__ == "__main__":
    main()
