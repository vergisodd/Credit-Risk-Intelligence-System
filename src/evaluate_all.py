"""Evaluate and compare all trained model pipelines."""

from __future__ import annotations

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
from src.model_utils import evaluate_predictions, load_engineered_data, load_model_artifact, make_train_test_split, save_dataframe
from src.threshold_optimizer import find_optimal_threshold


MODEL_SPECS = [
    ("Logistic Regression", "logistic_model"),
    ("XGBoost", "xgboost_model"),
    ("LightGBM", "lightgbm_model"),
]


def get_scores(config: dict, X_test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Load model artifacts and score the shared holdout set."""
    scores = {}
    for model_name, artifact_key in MODEL_SPECS:
        model = load_model_artifact(config["artifacts"][artifact_key])
        scores[model_name] = model.predict_proba(X_test)[:, 1]
    return scores


def find_f1_optimal_threshold(y_true, y_proba) -> tuple[float, dict]:
    """Find the F1-maximizing threshold for a model."""
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        metrics = evaluate_predictions(y_true, y_proba, float(threshold))
        rows.append(metrics)
    results = pd.DataFrame(rows)
    best_row = results.loc[results["f1_default_class"].idxmax()]
    return float(best_row["threshold"]), best_row.to_dict()


def build_comparison_table(y_test, scores: dict[str, np.ndarray], default_threshold: float) -> pd.DataFrame:
    """Create the all-model comparison table."""
    rows = []
    for model_name, y_proba in scores.items():
        optimal_threshold, optimal_metrics = find_f1_optimal_threshold(y_test, y_proba)
        default_metrics = evaluate_predictions(y_test, y_proba, default_threshold)
        rows.append(
            {
                "Model": model_name,
                "AUC-ROC": roc_auc_score(y_test, y_proba),
                "Average Precision": average_precision_score(y_test, y_proba),
                "F1-Default": default_metrics["f1_default_class"],
                "Precision-Default": default_metrics["precision_default_class"],
                "Recall-Default": default_metrics["recall_default_class"],
                "Optimal Threshold": optimal_threshold,
                "FP at Optimal": int(optimal_metrics["false_positives"]),
                "FN at Optimal": int(optimal_metrics["false_negatives"]),
            }
        )
    return pd.DataFrame(rows)


def save_combined_roc(y_test, scores: dict[str, np.ndarray], output_path: str) -> None:
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


def save_combined_pr(y_test, scores: dict[str, np.ndarray], output_path: str) -> None:
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


def save_calibration_plot(y_test, scores: dict[str, np.ndarray], output_path: str) -> None:
    """Save a reliability diagram for XGBoost and LightGBM."""
    output_file = ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name in ["XGBoost", "LightGBM"]:
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


def save_cost_threshold_outputs(y_test, y_proba: np.ndarray, config: dict) -> None:
    """Save cost-threshold CSV and visual for the LightGBM model."""
    scenario_a = config["thresholds"]["business_scenarios"]["lender"]
    scenario_b = config["thresholds"]["business_scenarios"]["balanced"]
    threshold_a, _, results_a, f1_threshold = find_optimal_threshold(
        y_test,
        y_proba,
        fn_cost=scenario_a["fn_cost"],
        fp_cost=scenario_a["fp_cost"],
    )
    threshold_b, _, results_b, _ = find_optimal_threshold(
        y_test,
        y_proba,
        fn_cost=scenario_b["fn_cost"],
        fp_cost=scenario_b["fp_cost"],
    )
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
    ax.axvline(f1_threshold, color="#111827", linestyle="--", label="F1-optimal")
    ax.axvline(threshold_a, color="#1B4F8A", linestyle=":", label="Scenario A optimum")
    ax.axvline(threshold_b, color="#B45309", linestyle=":", label="Scenario B optimum")
    ax.set_title("Cost Threshold Analysis")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total Relative Cost")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def update_model_comparison_report(comparison_df: pd.DataFrame, config: dict) -> None:
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
    report = f"""# Model Comparison Report

## Summary

The strongest holdout model is **{best_row["Model"]}** with ROC-AUC {best_row["AUC-ROC"]:.4f} and Average Precision {best_row["Average Precision"]:.4f}.

{markdown_table}

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

    print("Scoring all trained model pipelines...")
    scores = get_scores(config, X_test)
    comparison_df = build_comparison_table(y_test, scores, config["thresholds"]["default"])

    print("Saving comparison outputs...")
    save_dataframe(comparison_df, config["reports"]["model_comparison_full"])
    save_combined_roc(y_test, scores, config["visuals"]["roc_comparison_all_models"])
    save_combined_pr(y_test, scores, config["visuals"]["pr_comparison_all_models"])
    save_calibration_plot(y_test, scores, config["visuals"]["calibration_plot"])
    save_cost_threshold_outputs(y_test, scores["LightGBM"], config)
    update_model_comparison_report(comparison_df, config)

    print()
    print(comparison_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"Comparison saved to: {resolve_path(config['reports']['model_comparison_full'])}")


if __name__ == "__main__":
    main()
