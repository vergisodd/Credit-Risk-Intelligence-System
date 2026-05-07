"""Generate sample applicant predictions from a trained pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.champion_model import (
    get_persisted_operating_threshold,
    load_champion_feature_data,
    load_champion_model,
    review_recommendation,
)
from src.config_loader import load_config
from src.data_cleaning import DEFAULT_ID_COLUMN, load_raw_data
from src.model_utils import save_dataframe


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
    operating_threshold = get_operating_threshold(config)
    predictions = (probabilities >= operating_threshold).astype(int)
    prediction_results = pd.DataFrame(
        {
            "default_probability": probabilities,
            "predicted_class_at_operating_threshold": predictions,
        }
    )
    prediction_results["risk_tier"] = prediction_results["default_probability"].apply(
        lambda probability: assign_risk_tier(probability, config)
    )
    prediction_results["operating_threshold"] = operating_threshold
    prediction_results["review_recommendation"] = prediction_results["default_probability"].apply(
        lambda probability: review_recommendation(probability, operating_threshold, config)
    )
    return prediction_results


def get_operating_threshold(config: dict) -> float:
    """Read the champion operating threshold from metrics, falling back to default."""
    return get_persisted_operating_threshold(config)


def create_sample_predictions(sample_size: int = 20) -> pd.DataFrame:
    """Create sample predictions from the raw training data."""
    config = load_config()
    model = load_champion_model(config)
    df = load_raw_data()
    applicant_ids = df[DEFAULT_ID_COLUMN].copy() if DEFAULT_ID_COLUMN in df.columns else None
    X, y = load_champion_feature_data(config)
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
