"""
Prediction utilities for the Credit Risk Intelligence System.

This script:
- Loads a trained model
- Loads applicant data
- Applies the same cleaning and feature engineering workflow
- Generates default-risk probabilities
- Assigns applicants into business risk tiers
"""

from pathlib import Path

import joblib
import pandas as pd

from data_cleaning import load_raw_data, prepare_features_and_target
from feature_engineering import add_domain_features


MODEL_PATH = "models/logistic_regression_baseline.joblib"
RAW_DATA_PATH = "data/raw/application_train.csv"
PREDICTION_OUTPUT_PATH = "reports/sample_predictions.csv"


def load_trained_model(model_path: str = MODEL_PATH):
    """
    Load trained model artifact.

    Parameters
    ----------
    model_path : str
        Path to saved model.

    Returns
    -------
    object
        Trained model pipeline.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run src/train_model.py first."
        )

    return joblib.load(path)


def assign_risk_tier(default_probability: float) -> str:
    """
    Convert predicted default probability into a business risk tier.

    Parameters
    ----------
    default_probability : float
        Predicted probability of payment difficulty.

    Returns
    -------
    str
        Risk tier label.
    """
    if default_probability < 0.30:
        return "Low Risk"

    if default_probability < 0.60:
        return "Medium Risk"

    return "High Risk"


def predict_risk(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate default-risk predictions for applicant data.

    Parameters
    ----------
    model : object
        Trained sklearn pipeline.
    X : pd.DataFrame
        Applicant feature dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe containing prediction results.
    """
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    prediction_results = pd.DataFrame(
        {
            "default_probability": probabilities,
            "predicted_class": predictions
        }
    )

    prediction_results["risk_tier"] = prediction_results[
        "default_probability"
    ].apply(assign_risk_tier)

    return prediction_results


def create_sample_predictions(
    raw_data_path: str = RAW_DATA_PATH,
    model_path: str = MODEL_PATH,
    sample_size: int = 20
) -> pd.DataFrame:
    """
    Create sample predictions from the raw training data.

    This is mainly for testing and demonstration.
    """
    model = load_trained_model(model_path)

    df = load_raw_data(raw_data_path)

    applicant_ids = df["SK_ID_CURR"].copy() if "SK_ID_CURR" in df.columns else None

    X, y, dropped_columns = prepare_features_and_target(df)
    X = add_domain_features(X)

    sample_X = X.head(sample_size)

    prediction_results = predict_risk(model, sample_X)

    if applicant_ids is not None:
        prediction_results.insert(
            0,
            "applicant_id",
            applicant_ids.head(sample_size).values
        )

    prediction_results["actual_target"] = y.head(sample_size).values

    return prediction_results


def main() -> None:
    """
    Main prediction workflow.
    """
    print("Creating sample predictions...")

    predictions = create_sample_predictions()

    output_file = Path(PREDICTION_OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output_file, index=False)

    print()
    print("Sample predictions:")
    print(predictions)

    print()
    print(f"Predictions saved to: {PREDICTION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()