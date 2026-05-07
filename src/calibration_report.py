"""Calibration reporting for champion risk scores."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.champion_model import get_champion_holdout, get_champion_spec, load_champion_model
from src.config_loader import ensure_parent_dir, load_config
from src.model_utils import save_dataframe
from src.score_decile_analysis import assign_score_deciles


def calculate_calibration_by_decile(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Calculate calibration by descending score decile."""
    score_df = assign_score_deciles(y_true, y_score, n_deciles=10)
    grouped = score_df.groupby("decile", sort=True)
    calibration_df = grouped.agg(
        applicant_count=("actual_default", "size"),
        observed_default_rate=("actual_default", "mean"),
        mean_predicted_risk=("predicted_score", "mean"),
        min_score=("predicted_score", "min"),
        max_score=("predicted_score", "max"),
    ).reset_index()
    calibration_df["calibration_error"] = (
        calibration_df["mean_predicted_risk"] - calibration_df["observed_default_rate"]
    )
    return calibration_df


def calculate_calibration_by_risk_tier(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    risk_tiers: dict[str, float],
) -> pd.DataFrame:
    """Calculate calibration summary across configured risk tiers."""
    score_df = pd.DataFrame(
        {
            "actual_default": np.asarray(y_true).astype(int),
            "predicted_score": np.asarray(y_score, dtype=float),
        }
    )
    low = float(risk_tiers["low"])
    medium = float(risk_tiers["medium"])
    score_df["risk_tier"] = pd.cut(
        score_df["predicted_score"],
        bins=[-np.inf, low, medium, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    )
    tier_df = (
        score_df.groupby("risk_tier", observed=False)
        .agg(
            applicant_count=("actual_default", "size"),
            observed_default_rate=("actual_default", "mean"),
            mean_predicted_risk=("predicted_score", "mean"),
            min_score=("predicted_score", "min"),
            max_score=("predicted_score", "max"),
        )
        .reset_index()
    )
    tier_df["calibration_error"] = tier_df["mean_predicted_risk"] - tier_df["observed_default_rate"]
    return tier_df


def save_calibration_visual(calibration_df: pd.DataFrame, config: dict) -> None:
    """Save observed-vs-predicted calibration by decile."""
    output_path = ensure_parent_dir(config["visuals"]["calibration_by_decile"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        calibration_df["decile"],
        calibration_df["observed_default_rate"],
        marker="o",
        label="Observed default rate",
        color="#1B4F8A",
    )
    ax.plot(
        calibration_df["decile"],
        calibration_df["mean_predicted_risk"],
        marker="o",
        label="Mean predicted risk",
        color="#B45309",
    )
    ax.set_title("Calibration by Score Decile")
    ax.set_xlabel("Score Decile (1 = Highest Predicted Risk)")
    ax.set_ylabel("Rate")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_calibration_markdown(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    decile_df: pd.DataFrame,
    tier_df: pd.DataFrame,
    config: dict,
) -> None:
    """Write the calibration Markdown report."""
    output_path = ensure_parent_dir(config["reports"]["calibration_report_md"])
    champion = get_champion_spec(config)
    brier = brier_score_loss(y_true, y_score)
    base_rate = float(np.mean(y_true))
    mean_predicted = float(np.mean(y_score))
    report = f"""# Calibration Report

Champion model: **{champion.model_name}**

## Summary

- Brier score: **{brier:.4f}**
- Holdout base default rate: **{base_rate:.2%}**
- Mean predicted risk score: **{mean_predicted:.2%}**

Scores are useful for prioritizing manual review. They should not be treated as perfectly calibrated probability-of-default estimates without additional validation and possible calibration.

## Calibration by Decile

{decile_df.to_markdown(index=False, floatfmt=".4f")}

## Calibration by Risk Tier

{tier_df.to_markdown(index=False, floatfmt=".4f")}

## Governance Note

Calibration needs further production validation, ideally with out-of-time data and monitored cohorts. In this portfolio project, calibration is reported as a diagnostic for score interpretation rather than as a claim that scores are legally or operationally calibrated PD estimates.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Generate calibration CSVs, Markdown, and visual."""
    config = load_config()
    model = load_champion_model(config)
    _, X_test, _, y_test = get_champion_holdout(config)
    y_score = model.predict_proba(X_test)[:, 1]

    decile_df = calculate_calibration_by_decile(y_test, y_score)
    tier_df = calculate_calibration_by_risk_tier(
        y_test,
        y_score,
        config["thresholds"]["risk_tiers"],
    )
    save_dataframe(decile_df, config["reports"]["calibration_by_decile_csv"])
    save_dataframe(tier_df, config["reports"]["calibration_by_risk_tier_csv"])
    save_calibration_visual(decile_df, config)
    write_calibration_markdown(y_test, y_score, decile_df, tier_df, config)
    print(f"Calibration report saved to {config['reports']['calibration_report_md']}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        raise SystemExit(f"Data/model unavailable for calibration report: {error}") from error
