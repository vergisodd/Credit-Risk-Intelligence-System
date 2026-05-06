"""Evaluate the trained Logistic Regression pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import classification_report

from src.config_loader import load_config
from src.model_utils import (
    evaluate_predictions,
    load_engineered_data,
    load_model_artifact,
    make_train_test_split,
    run_threshold_analysis,
    save_confusion_matrix_visual,
    save_dataframe,
    save_json,
    save_precision_recall_curve_from_scores,
    save_roc_curve_from_scores,
)


def main() -> None:
    """Recreate the holdout split and evaluate the logistic baseline."""
    config = load_config()
    threshold = config["thresholds"]["default"]

    print("Loading trained Logistic Regression pipeline...")
    model = load_model_artifact(config["artifacts"]["logistic_model"])

    print("Loading holdout split...")
    X, y = load_engineered_data()
    _, X_test, _, y_test = make_train_test_split(X, y, config)

    print("Scoring holdout data...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    default_metrics = evaluate_predictions(y_test, y_proba, threshold)
    results = {
        "model_name": "logistic_regression_baseline",
        "roc_auc": default_metrics["roc_auc"],
        "average_precision": default_metrics["average_precision"],
        "default_threshold_metrics": default_metrics,
        "classification_report_threshold_0_50": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    threshold_df = run_threshold_analysis(
        y_true=y_test,
        y_proba=y_proba,
        thresholds=config["thresholds"]["analysis_grid"],
        risk_tiers=config["thresholds"]["risk_tiers"],
    )

    print("Saving evaluation outputs...")
    save_json(results, config["reports"]["logistic_evaluation"])
    save_dataframe(threshold_df, config["reports"]["logistic_threshold_analysis"])
    save_roc_curve_from_scores(y_test, y_proba, config["visuals"]["roc_lr"], "Logistic Regression")
    save_precision_recall_curve_from_scores(
        y_test,
        y_proba,
        config["visuals"]["pr_lr"],
        "Logistic Regression",
        config["thresholds"]["operating_points"],
    )
    save_confusion_matrix_visual(
        y_test,
        y_proba,
        threshold,
        config["visuals"]["confusion_matrix_lr"],
        f"Confusion Matrix - Logistic Regression Threshold {threshold:.2f}",
    )

    print()
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"Average Precision: {results['average_precision']:.4f}")


if __name__ == "__main__":
    main()
