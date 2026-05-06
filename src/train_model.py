"""
Train baseline machine learning models for the Credit Risk Intelligence System.

This script:
- Loads raw Home Credit data
- Cleans and prepares features
- Adds domain-specific engineered features
- Splits data into training and testing sets
- Builds a preprocessing pipeline
- Trains a Logistic Regression baseline model
- Evaluates the model
- Saves the trained model artifact
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_cleaning import (
    get_feature_types,
    load_raw_data,
    prepare_features_and_target
)
from feature_engineering import add_domain_features


RAW_DATA_PATH = "data/raw/application_train.csv"
MODEL_OUTPUT_PATH = "models/logistic_regression_baseline.joblib"
METRICS_OUTPUT_PATH = "reports/baseline_model_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def create_one_hot_encoder() -> OneHotEncoder:
    """
    Create a OneHotEncoder that works across different scikit-learn versions.

    Newer versions use sparse_output.
    Older versions use sparse.
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
    Build preprocessing pipeline for numeric and categorical features.

    Numeric features:
    - median imputation
    - standard scaling

    Categorical features:
    - most frequent imputation
    - one-hot encoding
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
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


def build_logistic_regression_model(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Build the baseline Logistic Regression pipeline.
    """
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]
    )

    return model


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate trained model on test data.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "precision_default_class": precision_score(y_test, y_pred),
        "recall_default_class": recall_score(y_test, y_pred),
        "f1_default_class": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True
        )
    }

    return metrics


def save_metrics(metrics: dict, output_path: str) -> None:
    """
    Save model metrics as a JSON file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as file:
        json.dump(metrics, file, indent=4)


def print_metrics(metrics: dict) -> None:
    """
    Print important model metrics.
    """
    print("Model Evaluation Results")
    print("------------------------")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Precision default class: {metrics['precision_default_class']:.4f}")
    print(f"Recall default class: {metrics['recall_default_class']:.4f}")
    print(f"F1 default class: {metrics['f1_default_class']:.4f}")
    print()
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


def main() -> None:
    """
    Main training workflow.
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

    print("Building model pipeline...")
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    model = build_logistic_regression_model(preprocessor)

    print("Training Logistic Regression baseline...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print_metrics(metrics)

    print()
    print("Saving model artifact...")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)

    print("Saving metrics...")
    save_metrics(metrics, METRICS_OUTPUT_PATH)

    print()
    print("Training complete.")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"Metrics saved to: {METRICS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()