"""Generate sample applicant predictions from a trained pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config_loader import load_config
from src.data_cleaning import DEFAULT_ID_COLUMN, load_raw_data, prepare_features_and_target
from src.feature_engineering import add_all_features
from src.model_utils import load_model_artifact, save_dataframe


def assign_risk_tier(default_probability: float, config: dict) -> str:
    """Convert predicted default probability into a configured risk tier."""
    tiers = config["thresholds"]["risk_tiers"]
    if default_probability < tiers["low"]:
        return "Low Risk"
    if default_probability < tiers["medium"]:
        return "Medium Risk"
    return "High Risk"


def predict_risk(model: object, X: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate default-risk predictions for applicant data."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= config["thresholds"]["default"]).astype(int)
    prediction_results = pd.DataFrame(
        {
            "default_probability": probabilities,
            "predicted_class": predictions,
        }
    )
    prediction_results["risk_tier"] = prediction_results["default_probability"].apply(
        lambda probability: assign_risk_tier(probability, config)
    )
    return prediction_results


def create_sample_predictions(sample_size: int = 20) -> pd.DataFrame:
    """Create sample predictions from the raw training data."""
    config = load_config()
    model = load_model_artifact(config["artifacts"]["logistic_model"])
    df = load_raw_data()
    applicant_ids = df[DEFAULT_ID_COLUMN].copy() if DEFAULT_ID_COLUMN in df.columns else None
    X, y, _ = prepare_features_and_target(df)
    X = add_all_features(X)
    prediction_results = predict_risk(model, X.head(sample_size), config)

    if applicant_ids is not None:
        prediction_results.insert(0, "applicant_id", applicant_ids.head(sample_size).values)
    prediction_results["actual_target"] = y.head(sample_size).values
    return prediction_results


def main() -> None:
    """Save sample predictions for portfolio reporting."""
    config = load_config()
    predictions = create_sample_predictions()
    save_dataframe(predictions, config["reports"]["sample_predictions"])
    print(predictions)


if __name__ == "__main__":
    main()
