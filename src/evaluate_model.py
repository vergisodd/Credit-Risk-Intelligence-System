"""
Evaluate trained models for the Credit Risk Intelligence System.

This script:
- Loads the trained model artifact
- Recreates the same test split
- Evaluates default threshold performance
- Performs threshold analysis
- Saves evaluation metrics and visuals
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.model_selection import train_test_split

from data_cleaning import load_raw_data, prepare_features_and_target
from feature_engineering import add_domain_features


RAW_DATA_PATH = "data/raw/application_train.csv"
MODEL_PATH = "models/logistic_regression_baseline.joblib"

EVALUATION_OUTPUT_PATH = "reports/logistic_regression_evaluation.json"
THRESHOLD_OUTPUT_PATH = "reports/threshold_analysis.csv"
ROC_CURVE_OUTPUT_PATH = "visuals/roc_curve_logistic_regression.png"
CONFUSION_MATRIX_OUTPUT_PATH = "visuals/confusion_matrix_threshold_0_50.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_THRESHOLD = 0.50


def load_model(model_path: str):
    """
    Load trained model artifact.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run src/train_model.py first."
        )

    return joblib.load(path)


def recreate_test_data():
    """
    Recreate the same train/test split used during training.
    """
    df = load_raw_data(RAW_DATA_PATH)

    X, y, dropped_columns = prepare_features_and_target(df)
    X = add_domain_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_test, y_test


def evaluate_at_threshold(
    y_true: pd.Series,
    y_pred_proba,
    threshold: float
) -> dict:
    """
    Evaluate model at a selected classification threshold.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_default_class": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_default_class": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_default_class": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_default_count": int(y_pred.sum())
    }

    return metrics


def run_threshold_analysis(y_true: pd.Series, y_pred_proba) -> pd.DataFrame:
    """
    Evaluate model across multiple classification thresholds.
    """
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    threshold_results = [
        evaluate_at_threshold(y_true, y_pred_proba, threshold)
        for threshold in thresholds
    ]

    return pd.DataFrame(threshold_results)


def save_json(data: dict, output_path: str) -> None:
    """
    Save dictionary as JSON.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)


def save_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Save ROC curve visual.
    """
    output_file = Path(ROC_CURVE_OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve — Logistic Regression Baseline")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def save_confusion_matrix_visual(y_true: pd.Series, y_pred) -> None:
    """
    Save confusion matrix visual.
    """
    output_file = Path(CONFUSION_MATRIX_OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)

    ax.set_title("Confusion Matrix — Threshold 0.50")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Default", "Default"])
    ax.set_yticklabels(["Non-Default", "Default"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def main() -> None:
    """
    Main evaluation workflow.
    """
    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Recreating test data...")
    X_test, y_test = recreate_test_data()

    print("Generating predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_pred_proba >= DEFAULT_THRESHOLD).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_pred_proba))

    default_threshold_metrics = evaluate_at_threshold(
        y_true=y_test,
        y_pred_proba=y_pred_proba,
        threshold=DEFAULT_THRESHOLD
    )

    evaluation_results = {
        "model_name": "logistic_regression_baseline",
        "roc_auc": roc_auc,
        "default_threshold_metrics": default_threshold_metrics,
        "classification_report_threshold_0_50": classification_report(
            y_test,
            y_pred_default,
            output_dict=True,
            zero_division=0
        )
    }

    print("Running threshold analysis...")
    threshold_df = run_threshold_analysis(y_test, y_pred_proba)

    print("Saving reports...")
    save_json(evaluation_results, EVALUATION_OUTPUT_PATH)

    Path(THRESHOLD_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(THRESHOLD_OUTPUT_PATH, index=False)

    print("Saving visuals...")
    save_roc_curve(model, X_test, y_test)
    save_confusion_matrix_visual(y_test, y_pred_default)

    print()
    print("Evaluation complete.")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print()
    print("Default threshold metrics:")
    for key, value in default_threshold_metrics.items():
        print(f"{key}: {value}")

    print()
    print("Threshold analysis:")
    print(threshold_df)

    print()
    print(f"Evaluation saved to: {EVALUATION_OUTPUT_PATH}")
    print(f"Threshold analysis saved to: {THRESHOLD_OUTPUT_PATH}")
    print(f"ROC curve saved to: {ROC_CURVE_OUTPUT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_OUTPUT_PATH}")


if __name__ == "__main__":
    main()