"""Shared LightGBM training helpers for application-only and bureau models."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.model_utils import build_preprocessor


def build_lightgbm_params(
    config: dict, scale_pos_weight: float, overrides: dict | None = None
) -> dict:
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
    """Run configured CV for the base LightGBM configuration."""
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
    """Run configured CV on the Optuna-tuned params."""
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
