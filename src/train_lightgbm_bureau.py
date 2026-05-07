"""Train a LightGBM credit risk pipeline with bureau aggregations."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import classification_report

from src.config_loader import load_config
from src.model_utils import (
    calculate_scale_pos_weight,
    evaluate_predictions,
    get_feature_type_lists,
    load_engineered_data_with_bureau,
    make_train_test_split,
    run_threshold_analysis,
    save_confusion_matrix_visual,
    save_dataframe,
    save_json,
    save_model_artifact,
    save_precision_recall_curve_from_scores,
    save_roc_curve_from_scores,
)
from src.champion_model import save_model_manifest
from src.threshold_optimizer import find_optimal_threshold
from src.train_lightgbm_common import (
    build_lightgbm_params,
    build_lightgbm_pipeline,
    optimize_hyperparameters,
    run_base_cross_validation,
    run_tuned_cross_validation,
)


def summarize_results(metrics: dict) -> None:
    """Print a clean summary table."""
    row = {
        "Model": "LightGBM+Bureau",
        "AUC-ROC": metrics["roc_auc"],
        "Average Precision": metrics["average_precision"],
        "Base CV AUC": metrics["base_cv_auc_mean"],
        "Tuned CV AUC": metrics["tuned_cv_auc_mean"],
        "Tuned CV AUC Std": metrics["tuned_cv_auc_std"],
        "Tuned CV AP": metrics["tuned_cv_ap_mean"],
        "Tuned CV AP Std": metrics["tuned_cv_ap_std"],
        "Precision": metrics["default_threshold_metrics"]["precision_default_class"],
        "Recall": metrics["default_threshold_metrics"]["recall_default_class"],
        "F1": metrics["default_threshold_metrics"]["f1_default_class"],
    }
    print()
    print(pd.DataFrame([row]).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    """Run LightGBM training, tuning, evaluation, and artifact export."""
    config = load_config()
    threshold = config["thresholds"]["default"]

    print("Training LightGBM with bureau features...")
    print("Loading and engineering data...")
    X, y = load_engineered_data_with_bureau()
    numeric_features, categorical_features = get_feature_type_lists(X)

    print("Creating stratified train/test split...")
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    print("Running base 5-fold CV...")
    base_params = build_lightgbm_params(config, scale_pos_weight)
    base_model = build_lightgbm_pipeline(numeric_features, categorical_features, base_params)
    cv_metrics = run_base_cross_validation(base_model, X_train, y_train, config)

    print("Running Optuna hyperparameter optimization...")
    best_overrides = optimize_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_pos_weight=scale_pos_weight,
        config=config,
    )
    best_params = build_lightgbm_params(config, scale_pos_weight, best_overrides)

    print("Retraining LightGBM on the full training set...")
    model = build_lightgbm_pipeline(numeric_features, categorical_features, best_params)
    model.fit(X_train, y_train)

    print("Running tuned 5-fold CV...")
    tuned_cv_metrics = run_tuned_cross_validation(
        best_params=best_params,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        X_train=X_train,
        y_train=y_train,
        config=config,
    )

    print("Evaluating holdout performance...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    default_metrics = evaluate_predictions(y_test, y_proba, threshold)
    lender_scenario = config["thresholds"]["business_scenarios"]["lender"]
    threshold_optimization = find_optimal_threshold(
        y_test,
        y_proba,
        fn_cost=lender_scenario["fn_cost"],
        fp_cost=lender_scenario["fp_cost"],
    )
    metrics = {
        "model_name": "lightgbm_bureau_credit_risk_model",
        "roc_auc": default_metrics["roc_auc"],
        "average_precision": default_metrics["average_precision"],
        **cv_metrics,
        **tuned_cv_metrics,
        "best_params": best_params,
        "threshold_policy": threshold_optimization.to_summary_dict(),
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
    save_model_artifact(model, config["artifacts"]["lightgbm_bureau_model"])
    save_json(metrics, config["reports"]["lightgbm_bureau_metrics"])
    save_model_manifest(config, threshold_optimization, metrics)
    save_dataframe(threshold_df, config["reports"]["lgbm_bureau_threshold_analysis"])
    save_roc_curve_from_scores(
        y_test,
        y_proba,
        config["visuals"]["roc_lightgbm_bureau"],
        label="LightGBM+Bureau",
    )
    save_precision_recall_curve_from_scores(
        y_test,
        y_proba,
        config["visuals"]["pr_lightgbm_bureau"],
        label="LightGBM+Bureau",
        operating_points=config["thresholds"]["operating_points"],
    )
    save_confusion_matrix_visual(
        y_test,
        y_proba,
        threshold,
        config["visuals"]["confusion_matrix_lightgbm_bureau"],
        title=f"Confusion Matrix - LightGBM+Bureau Threshold {threshold:.2f}",
    )

    summarize_results(metrics)
    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
