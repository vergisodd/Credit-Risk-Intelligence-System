"""
Shared modelling, evaluation, and plotting helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config_loader import resolve_path
from src.data_cleaning import get_feature_types, load_and_clean
from src.feature_engineering import add_all_features


def create_one_hot_encoder(sparse_output: bool = False) -> OneHotEncoder:
    """
    Create a OneHotEncoder compatible with multiple scikit-learn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse_output)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = False,
    sparse_output: bool = False,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numeric and categorical preprocessing.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", create_one_hot_encoder(sparse_output=sparse_output)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        verbose_feature_names_out=True,
    )


def load_engineered_data(validate_schema: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load, clean, and engineer features without fitting preprocessing.
    """
    X, y, _ = load_and_clean(validate_schema=validate_schema)
    X = add_all_features(X)
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create the standard stratified project split.
    """
    return train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )


def get_feature_type_lists(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return numeric and categorical feature lists for the current feature matrix.
    """
    return get_feature_types(X)


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate class imbalance weight for gradient boosting classifiers.
    """
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    if positive_count == 0:
        raise ValueError("Positive class count is zero; cannot calculate class weight.")
    return negative_count / positive_count


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """
    Evaluate binary classifier probabilities at a selected threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "precision_default_class": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_default_class": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_default_class": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_default_count": int(y_pred.sum()),
    }


def run_threshold_analysis(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    thresholds: Iterable[float],
    risk_tiers: dict[str, float],
) -> pd.DataFrame:
    """
    Evaluate model performance across classification thresholds.
    """
    rows = []
    for threshold in thresholds:
        metrics = evaluate_predictions(y_true, y_proba, threshold)
        recall = metrics["recall_default_class"]
        precision = metrics["precision_default_class"]
        if threshold < risk_tiers["low"]:
            label = "Very aggressive screening"
        elif threshold < risk_tiers["medium"]:
            label = "Broad manual review queue"
        elif threshold < risk_tiers["high"]:
            label = "Balanced risk screening"
        else:
            label = "Conservative high-risk flag"

        rows.append(
            {
                "threshold": metrics["threshold"],
                "recall": recall,
                "precision": precision,
                "f1": metrics["f1_default_class"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "business_label": label,
            }
        )
    return pd.DataFrame(rows)


def save_json(data: dict, output_path: str | Path) -> None:
    """
    Save a dictionary as pretty JSON.
    """
    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save a dataframe to CSV with parent directories created.
    """
    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def save_model_artifact(model, output_path: str | Path) -> None:
    """
    Persist a fitted model pipeline.
    """
    import joblib

    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_file)


def load_model_artifact(model_path: str | Path):
    """
    Load a fitted model pipeline with a clear missing-file error.
    """
    import joblib

    path = resolve_path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train the corresponding model first."
        )
    return joblib.load(path)


def save_roc_curve_from_scores(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    output_path: str | Path,
    label: str,
) -> None:
    """
    Save a single-model ROC curve.
    """
    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_value = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"{label} AUC={auc_value:.3f}", color="#1B4F8A")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", label="Random")
    ax.set_title(f"ROC Curve - {label}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def save_precision_recall_curve_from_scores(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    output_path: str | Path,
    label: str,
    operating_points: Iterable[float],
) -> None:
    """
    Save a precision-recall curve with selected threshold operating points.
    """
    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap_value = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#1B4F8A", label=f"{label} AP={ap_value:.3f}")
    for threshold in operating_points:
        metrics = evaluate_predictions(y_true, y_proba, threshold)
        ax.scatter(
            metrics["recall_default_class"],
            metrics["precision_default_class"],
            s=45,
            label=f"t={threshold:.2f}",
        )
    ax.set_title(f"Precision-Recall Curve - {label}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def save_confusion_matrix_visual(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    output_path: str | Path,
    title: str,
) -> None:
    """
    Save a readable confusion matrix visual.
    """
    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Default", "Default"])
    ax.set_yticklabels(["Non-Default", "Default"])

    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            ax.text(column, row, f"{cm[row, column]:,}", ha="center", va="center")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
