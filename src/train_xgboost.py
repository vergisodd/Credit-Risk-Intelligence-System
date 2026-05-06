"""Train the XGBoost credit risk pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config_loader import load_config
from src.model_utils import (
    build_preprocessor,
    calculate_scale_pos_weight,
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


def build_xgboost_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_pos_weight: float,
    config: dict,
) -> Pipeline:
    """Build the full preprocessing and XGBoost pipeline."""
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=False,
        sparse_output=True,
    )
    model_config = config["xgboost"]
    classifier = XGBClassifier(
        n_estimators=model_config["n_estimators"],
        learning_rate=model_config["learning_rate"],
        max_depth=model_config["max_depth"],
        min_child_weight=model_config["min_child_weight"],
        subsample=model_config["subsample"],
        colsample_bytree=model_config["colsample_bytree"],
        objective=model_config["objective"],
        eval_metric=model_config["eval_metric"],
        tree_method=model_config["tree_method"],
        scale_pos_weight=scale_pos_weight,
        random_state=config["model"]["random_state"],
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def run_cross_validation(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    """Run stratified 5-fold CV on the training set."""
    cv = StratifiedKFold(
        n_splits=config["model"]["cv_folds"],
        shuffle=True,
        random_state=config["model"]["random_state"],
    )
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "auc": "roc_auc",
            "ap": "average_precision",
        },
        n_jobs=1,
        return_train_score=False,
    )
    return {
        "cv_auc_mean": float(scores["test_auc"].mean()),
        "cv_auc_std": float(scores["test_auc"].std()),
        "cv_ap_mean": float(scores["test_ap"].mean()),
        "cv_ap_std": float(scores["test_ap"].std()),
    }


def summarize_results(metrics: dict) -> None:
    """Print a compact model summary."""
    summary = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "AUC-ROC": metrics["roc_auc"],
                "Average Precision": metrics["average_precision"],
                "CV AUC": metrics["cv_auc_mean"],
                "CV AP": metrics["cv_ap_mean"],
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
    """Run the XGBoost training workflow."""
    config = load_config()
    threshold = config["thresholds"]["default"]

    print("Loading and engineering data...")
    X, y = load_engineered_data()
    numeric_features, categorical_features = get_feature_type_lists(X)

    print("Creating stratified train/test split...")
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    print("Building full sklearn Pipeline...")
    model = build_xgboost_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_pos_weight=scale_pos_weight,
        config=config,
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    print("Running 5-fold stratified CV on the training set...")
    cv_metrics = run_cross_validation(model, X_train, y_train, config)

    print("Evaluating holdout performance...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    default_metrics = evaluate_predictions(y_test, y_proba, threshold)
    metrics = {
        "model_name": "xgboost_credit_risk_model",
        "roc_auc": default_metrics["roc_auc"],
        "average_precision": default_metrics["average_precision"],
        **cv_metrics,
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
    save_model_artifact(model, config["artifacts"]["xgboost_model"])
    save_json(metrics, config["reports"]["xgboost_metrics"])
    save_dataframe(threshold_df, config["reports"]["xgboost_threshold_analysis"])
    save_roc_curve_from_scores(
        y_test,
        y_proba,
        config["visuals"]["roc_xgboost"],
        label="XGBoost",
    )
    save_precision_recall_curve_from_scores(
        y_test,
        y_proba,
        config["visuals"]["pr_xgboost"],
        label="XGBoost",
        operating_points=config["thresholds"]["operating_points"],
    )
    save_confusion_matrix_visual(
        y_test,
        y_proba,
        threshold,
        config["visuals"]["confusion_matrix_xgboost"],
        title=f"Confusion Matrix - XGBoost Threshold {threshold:.2f}",
    )

    summarize_results(metrics)
    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
