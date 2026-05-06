"""Train the Logistic Regression baseline pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.config_loader import load_config
from src.model_utils import (
    build_preprocessor,
    evaluate_predictions,
    get_feature_type_lists,
    load_engineered_data,
    make_train_test_split,
    run_threshold_analysis,
    save_confusion_matrix_visual,
    save_dataframe,
    save_json,
    save_model_artifact,
    save_precision_recall_curve_from_scores,
    save_roc_curve_from_scores,
)


def build_logistic_regression_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict,
) -> Pipeline:
    """Build the full preprocessing and Logistic Regression pipeline."""
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=True,
        sparse_output=True,
    )
    model_config = config["logistic_regression"]
    classifier = LogisticRegression(
        class_weight=model_config["class_weight"],
        max_iter=model_config["max_iter"],
        solver=model_config["solver"],
        random_state=config["model"]["random_state"],
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def summarize_results(metrics: dict) -> None:
    """Print a compact model summary."""
    summary = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "AUC-ROC": metrics["roc_auc"],
                "Average Precision": metrics["average_precision"],
                "Precision": metrics["default_threshold_metrics"][
                    "precision_default_class"
                ],
                "Recall": metrics["default_threshold_metrics"]["recall_default_class"],
                "F1": metrics["default_threshold_metrics"]["f1_default_class"],
            }
        ]
    )
    print()
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    """Run the Logistic Regression training workflow."""
    config = load_config()
    threshold = config["thresholds"]["default"]

    print("Loading and engineering data...")
    X, y = load_engineered_data()
    numeric_features, categorical_features = get_feature_type_lists(X)

    print("Creating stratified train/test split...")
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)

    print("Building full sklearn Pipeline...")
    model = build_logistic_regression_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        config=config,
    )

    print("Training Logistic Regression baseline...")
    model.fit(X_train, y_train)

    print("Evaluating holdout performance...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    default_metrics = evaluate_predictions(y_test, y_proba, threshold)
    metrics = {
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

    print("Saving model, metrics, and visuals...")
    save_model_artifact(model, config["artifacts"]["logistic_model"])
    save_json(metrics, config["reports"]["baseline_metrics"])
    save_json(metrics, config["reports"]["logistic_evaluation"])
    save_dataframe(threshold_df, config["reports"]["logistic_threshold_analysis"])
    save_roc_curve_from_scores(
        y_test,
        y_proba,
        config["visuals"]["roc_lr"],
        label="Logistic Regression",
    )
    save_precision_recall_curve_from_scores(
        y_test,
        y_proba,
        config["visuals"]["pr_lr"],
        label="Logistic Regression",
        operating_points=config["thresholds"]["operating_points"],
    )
    save_confusion_matrix_visual(
        y_test,
        y_proba,
        threshold,
        config["visuals"]["confusion_matrix_lr"],
        title=f"Confusion Matrix - Logistic Regression Threshold {threshold:.2f}",
    )

    summarize_results(metrics)
    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
