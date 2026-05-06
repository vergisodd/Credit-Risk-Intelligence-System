"""
Train an XGBoost model for the Credit Risk Intelligence System.

This script:
- Loads raw Home Credit data
- Applies the same cleaning and feature engineering workflow
- Builds preprocessing pipeline
- Trains an XGBoost classifier
- Evaluates performance
- Saves model metrics and visuals
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_cleaning import (
    get_feature_types,
    load_raw_data,
    prepare_features_and_target
)
from feature_engineering import add_domain_features


RAW_DATA_PATH = "data/raw/application_train.csv"
MODEL_OUTPUT_PATH = "models/xgboost_credit_risk_model.joblib"
METRICS_OUTPUT_PATH = "reports/xgboost_model_metrics.json"
THRESHOLD_OUTPUT_PATH = "reports/xgboost_threshold_analysis.csv"
ROC_CURVE_OUTPUT_PATH = "visuals/roc_curve_xgboost.png"
CONFUSION_MATRIX_OUTPUT_PATH = "visuals/confusion_matrix_xgboost_threshold_0_50.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_THRESHOLD = 0.50


def create_one_hot_encoder() -> OneHotEncoder:
    """
    Create a OneHotEncoder compatible with different scikit-learn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str]
) -> ColumnTransformer:
    """
    Build preprocessing pipeline for XGBoost.

    For tree-based models, scaling is not required.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", create_one_hot_encoder())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate class imbalance weight for XGBoost.

    scale_pos_weight = negative class count / positive class count
    """
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    return negative_count / positive_count


def build_xgboost_model(
    preprocessor: ColumnTransformer,
    scale_pos_weight: float
) -> Pipeline:
    """
    Build XGBoost model pipeline.
    """
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb_model)
        ]
    )

    return model


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
    plt.title("ROC Curve — XGBoost Model")
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

    ax.set_title("Confusion Matrix — XGBoost Threshold 0.50")
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
    Main XGBoost training workflow.
    """
    print("Loading raw data...")
    df = load_raw_data(RAW_DATA_PATH)

    print("Preparing features and target...")
    X, y, dropped_columns = prepare_features_and_target(df)

    print("Adding engineered features...")
    X = add_domain_features(X)

    numeric_features, categorical_features = get_feature_types(X)

    print("Dataset summary")
    print("---------------")
    print("Raw data shape:", df.shape)
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Dropped columns:", len(dropped_columns))
    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))
    print()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    print("Class imbalance weight:")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    print()

    print("Building XGBoost pipeline...")
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    model = build_xgboost_model(
        preprocessor=preprocessor,
        scale_pos_weight=scale_pos_weight
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    print("Generating predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= DEFAULT_THRESHOLD).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_pred_proba))

    default_threshold_metrics = evaluate_at_threshold(
        y_true=y_test,
        y_pred_proba=y_pred_proba,
        threshold=DEFAULT_THRESHOLD
    )

    metrics = {
        "model_name": "xgboost_credit_risk_model",
        "roc_auc": roc_auc,
        "default_threshold_metrics": default_threshold_metrics,
        "classification_report_threshold_0_50": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )
    }

    threshold_df = run_threshold_analysis(y_test, y_pred_proba)

    print()
    print("XGBoost Evaluation Results")
    print("--------------------------")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Accuracy: {default_threshold_metrics['accuracy']:.4f}")
    print(f"Precision default class: {default_threshold_metrics['precision_default_class']:.4f}")
    print(f"Recall default class: {default_threshold_metrics['recall_default_class']:.4f}")
    print(f"F1 default class: {default_threshold_metrics['f1_default_class']:.4f}")
    print()
    print("Confusion Matrix:")
    print(
        [
            [
                default_threshold_metrics["true_negatives"],
                default_threshold_metrics["false_positives"]
            ],
            [
                default_threshold_metrics["false_negatives"],
                default_threshold_metrics["true_positives"]
            ]
        ]
    )
    print()
    print("Threshold analysis:")
    print(threshold_df)

    print()
    print("Saving model artifact...")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)

    print("Saving metrics and visuals...")
    save_json(metrics, METRICS_OUTPUT_PATH)
    threshold_df.to_csv(THRESHOLD_OUTPUT_PATH, index=False)
    save_roc_curve(model, X_test, y_test)
    save_confusion_matrix_visual(y_test, y_pred)

    print()
    print("Training complete.")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"Metrics saved to: {METRICS_OUTPUT_PATH}")
    print(f"Threshold analysis saved to: {THRESHOLD_OUTPUT_PATH}")
    print(f"ROC curve saved to: {ROC_CURVE_OUTPUT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_OUTPUT_PATH}")


if __name__ == "__main__":
    main()