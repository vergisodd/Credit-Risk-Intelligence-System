"""Score decile and lift analysis for the champion model."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.champion_model import get_champion_holdout, get_champion_spec, load_champion_model
from src.config_loader import ensure_parent_dir, load_config
from src.model_utils import save_dataframe


def assign_score_deciles(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Return applicant-level scores sorted into rank deciles."""
    if n_deciles < 1:
        raise ValueError("n_deciles must be at least 1.")
    score_df = pd.DataFrame(
        {
            "actual_default": np.asarray(y_true).astype(int),
            "predicted_score": np.asarray(y_score, dtype=float),
        }
    ).sort_values("predicted_score", ascending=False, kind="mergesort")
    if score_df.empty:
        raise ValueError("Score decile analysis requires at least one row.")

    ranks = np.arange(len(score_df))
    score_df["decile"] = ((ranks * n_deciles) // len(score_df)) + 1
    score_df["decile"] = score_df["decile"].clip(upper=n_deciles).astype(int)
    return score_df.reset_index(drop=True)


def calculate_decile_lift_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Calculate decile default rates, lift, and cumulative capture."""
    score_df = assign_score_deciles(y_true, y_score, n_deciles=n_deciles)
    base_rate = float(score_df["actual_default"].mean())
    total_defaults = int(score_df["actual_default"].sum())

    grouped = score_df.groupby("decile", sort=True)
    decile_df = grouped.agg(
        applicant_count=("actual_default", "size"),
        actual_default_count=("actual_default", "sum"),
        actual_default_rate=("actual_default", "mean"),
        mean_predicted_score=("predicted_score", "mean"),
        min_score=("predicted_score", "min"),
        max_score=("predicted_score", "max"),
    ).reset_index()
    decile_df["lift_vs_base_rate"] = (
        decile_df["actual_default_rate"] / base_rate if base_rate > 0 else np.nan
    )
    decile_df["cumulative_applicant_percentage"] = (
        decile_df["applicant_count"].cumsum() / decile_df["applicant_count"].sum()
    )
    decile_df["cumulative_default_capture_rate"] = (
        decile_df["actual_default_count"].cumsum() / total_defaults
        if total_defaults > 0
        else np.nan
    )
    return decile_df


def save_decile_visuals(decile_df: pd.DataFrame, config: dict) -> None:
    """Save lift and cumulative default capture visuals."""
    lift_path = ensure_parent_dir(config["visuals"]["score_decile_lift"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(decile_df["decile"].astype(str), decile_df["lift_vs_base_rate"], color="#1B4F8A")
    ax.axhline(1.0, color="#6B7280", linestyle="--", linewidth=1)
    ax.set_title("Score Decile Lift vs Base Default Rate")
    ax.set_xlabel("Score Decile (1 = Highest Predicted Risk)")
    ax.set_ylabel("Lift vs Base Rate")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(lift_path, dpi=300)
    plt.close(fig)

    capture_path = ensure_parent_dir(config["visuals"]["cumulative_default_capture"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        decile_df["cumulative_applicant_percentage"],
        decile_df["cumulative_default_capture_rate"],
        marker="o",
        color="#B45309",
    )
    ax.plot([0, 1], [0, 1], color="#6B7280", linestyle="--", label="Random ranking")
    ax.set_title("Cumulative Default Capture by Review Depth")
    ax.set_xlabel("Cumulative Applicant Share Reviewed")
    ax.set_ylabel("Cumulative Default Capture Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(capture_path, dpi=300)
    plt.close(fig)


def write_decile_markdown(decile_df: pd.DataFrame, config: dict) -> None:
    """Write a concise Markdown report for portfolio/demo use."""
    output_path = ensure_parent_dir(config["reports"]["score_decile_analysis_md"])
    base_rate = (
        decile_df["actual_default_count"].sum() / decile_df["applicant_count"].sum()
        if decile_df["applicant_count"].sum()
        else np.nan
    )
    top_decile = decile_df.loc[decile_df["decile"] == 1].iloc[0]
    markdown_table = decile_df.to_markdown(index=False, floatfmt=".4f")
    report = f"""# Score Decile and Lift Analysis

This report evaluates the configured champion model as a ranking tool for manual credit review. AUC is useful, but review teams also need to know whether the highest-risk score bands actually concentrate defaults.

Base default rate in this holdout split: **{base_rate:.2%}**.

Top score decile observed default rate: **{top_decile["actual_default_rate"]:.2%}**.

Scores are sorted descending, so decile 1 is the highest predicted-risk group.

{markdown_table}

## Interpretation

Lift above 1.0 means a score band has a higher observed default rate than the holdout base rate. This analysis supports review prioritization only; it does not approve or reject applicants.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Generate score decile CSV, Markdown, and visuals for the champion model."""
    config = load_config()
    champion = get_champion_spec(config)
    model = load_champion_model(config)
    _, X_test, _, y_test = get_champion_holdout(config)
    y_score = model.predict_proba(X_test)[:, 1]

    decile_df = calculate_decile_lift_table(y_test, y_score)
    save_dataframe(decile_df, config["reports"]["score_decile_analysis_csv"])
    save_decile_visuals(decile_df, config)
    write_decile_markdown(decile_df, config)
    print(
        f"Score decile analysis generated for {champion.model_name}: "
        f"{config['reports']['score_decile_analysis_csv']}"
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        raise SystemExit(f"Data/model unavailable for score decile analysis: {error}") from error
