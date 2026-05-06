"""Train a LightGBM credit risk pipeline with Optuna tuning."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

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


def build_lightgbm_params(config: dict, scale_pos_weight: float, overrides: dict | None = None) -> dict:
    """Create LightGBM model parameters from config and optional Optuna overrides."""
    params = {
        **config["lightgbm"],
        "objective": "binary",
        "random_state": config["model"]["random_state"],
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
    }
    if overrides:
        params.update(overrides)
    return params


def build_lightgbm_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    params: dict,
) -> Pipeline:
    """Build the full preprocessing and LightGBM pipeline."""
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=False,
        sparse_output=False,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**params)),
        ]
    )


def run_base_cross_validation(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> dict:
    """Run 5-fold CV for the base LightGBM configuration."""
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
        "base_cv_auc_mean": float(scores["test_auc"].mean()),
        "base_cv_auc_std": float(scores["test_auc"].std()),
        "base_cv_ap_mean": float(scores["test_ap"].mean()),
        "base_cv_ap_std": float(scores["test_ap"].std()),
    }


def run_tuned_cross_validation(
    best_params: dict,
    numeric_features: list[str],
    categorical_features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> dict:
    """Run 5-fold CV on the Optuna-tuned params for honest performance reporting."""
    tuned_model = build_lightgbm_pipeline(numeric_features, categorical_features, best_params)
    cv = StratifiedKFold(
        n_splits=config["model"]["cv_folds"],
        shuffle=True,
        random_state=config["model"]["random_state"],
    )
    scores = cross_validate(
        tuned_model,
        X_train,
        y_train,
        cv=cv,
        scoring={"auc": "roc_auc", "ap": "average_precision"},
        n_jobs=1,
        return_train_score=False,
    )
    return {
        "tuned_cv_auc_mean": float(scores["test_auc"].mean()),
        "tuned_cv_auc_std": float(scores["test_auc"].std()),
        "tuned_cv_ap_mean": float(scores["test_ap"].mean()),
        "tuned_cv_ap_std": float(scores["test_ap"].std()),
    }


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    scale_pos_weight: float,
    config: dict,
) -> dict:
    """Run Optuna search using 3-fold CV on the training set only."""

    def objective(trial: optuna.Trial) -> float:
        overrides = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        params = build_lightgbm_params(config, scale_pos_weight, overrides)
        model = build_lightgbm_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            params=params,
        )
        cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=config["model"]["random_state"],
        )
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
            return_train_score=False,
        )
        return -float(np.mean(scores["test_score"]))

    sampler = optuna.samplers.TPESampler(seed=config["model"]["random_state"])
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=config["model"]["lgbm_n_trials"], show_progress_bar=False)
    return study.best_params


def summarize_results(metrics: dict) -> None:
    """Print a clean summary table."""
    row = {
        "Model": "LightGBM",
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

    print("Loading and engineering data...")
    X, y = load_engineered_data()
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
    metrics = {
        "model_name": "lightgbm_credit_risk_model",
        "roc_auc": default_metrics["roc_auc"],
        "average_precision": default_metrics["average_precision"],
        **cv_metrics,
        **tuned_cv_metrics,
        "best_params": best_params,
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
    save_model_artifact(model, config["artifacts"]["lightgbm_model"])
    save_json(metrics, config["reports"]["lightgbm_metrics"])
    save_dataframe(threshold_df, config["reports"]["lightgbm_threshold_analysis"])
    save_roc_curve_from_scores(
        y_test,
        y_proba,
        config["visuals"]["roc_lightgbm"],
        label="LightGBM",
    )
    save_precision_recall_curve_from_scores(
        y_test,
        y_proba,
        config["visuals"]["pr_lightgbm"],
        label="LightGBM",
        operating_points=config["thresholds"]["operating_points"],
    )
    save_confusion_matrix_visual(
        y_test,
        y_proba,
        threshold,
        config["visuals"]["confusion_matrix_lightgbm"],
        title=f"Confusion Matrix - LightGBM Threshold {threshold:.2f}",
    )

    summarize_results(metrics)
    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
