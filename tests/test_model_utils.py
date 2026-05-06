import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_utils import build_preprocessor, evaluate_predictions, make_train_test_split


def test_build_preprocessor_has_numeric_and_categorical_transformers():
    preprocessor = build_preprocessor(["income"], ["gender"])
    assert isinstance(preprocessor, ColumnTransformer)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_build_preprocessor_with_scaling_includes_standard_scaler():
    preprocessor = build_preprocessor(["income"], ["gender"], scale_numeric=True)
    numeric_transformer = next(
        transformer for name, transformer, _ in preprocessor.transformers if name == "num"
    )
    assert isinstance(numeric_transformer, Pipeline)
    assert any(isinstance(step, StandardScaler) for _, step in numeric_transformer.steps)


def test_make_train_test_split_returns_stratified_split(config, synthetic_y_true):
    X = pd.DataFrame(
        {
            "income": np.linspace(50_000, 250_000, len(synthetic_y_true)),
            "gender": np.where(np.arange(len(synthetic_y_true)) % 2 == 0, "M", "F"),
        }
    )
    y = pd.Series(synthetic_y_true)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)
    assert len([X_train, X_test, y_train, y_test]) == 4
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_evaluate_predictions_returns_expected_keys(synthetic_y_true, synthetic_y_proba):
    metrics = evaluate_predictions(synthetic_y_true, synthetic_y_proba, threshold=0.5)
    expected_keys = {
        "threshold",
        "accuracy",
        "roc_auc",
        "average_precision",
        "precision_default_class",
        "recall_default_class",
        "f1_default_class",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    }
    assert expected_keys <= set(metrics)
