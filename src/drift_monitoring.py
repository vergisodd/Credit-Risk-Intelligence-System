"""Train-vs-holdout drift monitoring simulation."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.champion_model import get_champion_holdout
from src.config_loader import ensure_parent_dir, load_config
from src.model_utils import save_dataframe


PREFERRED_DRIFT_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_MEAN",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
]


def calculate_psi(
    expected: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate Population Stability Index using expected-sample quantile bins."""
    expected_series = pd.Series(expected).dropna().astype(float)
    actual_series = pd.Series(actual).dropna().astype(float)
    if expected_series.empty or actual_series.empty:
        return float("nan")
    if expected_series.nunique() <= 1:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(expected_series.quantile(quantiles).to_numpy())
    if len(breakpoints) < 2:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = pd.cut(expected_series, bins=breakpoints, include_lowest=True).value_counts(
        sort=False
    )
    actual_counts = pd.cut(actual_series, bins=breakpoints, include_lowest=True).value_counts(
        sort=False
    )
    expected_pct = (expected_counts / expected_counts.sum()).clip(lower=epsilon)
    actual_pct = (actual_counts / actual_counts.sum()).clip(lower=epsilon)
    psi = ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
    return float(psi)


def interpret_psi(psi: float) -> str:
    """Return a simple PSI severity label."""
    if np.isnan(psi):
        return "unavailable"
    if psi < 0.10:
        return "low shift"
    if psi < 0.25:
        return "moderate shift"
    return "high shift"


def select_drift_features(
    X_train: pd.DataFrame, X_holdout: pd.DataFrame, max_features: int = 15
) -> list[str]:
    """Select numeric features for a lightweight drift simulation report."""
    common_columns = [column for column in X_train.columns if column in X_holdout.columns]
    numeric_columns = [
        column
        for column in common_columns
        if pd.api.types.is_numeric_dtype(X_train[column])
        and pd.api.types.is_numeric_dtype(X_holdout[column])
    ]
    selected = [column for column in PREFERRED_DRIFT_FEATURES if column in numeric_columns]
    bureau_features = [
        column
        for column in numeric_columns
        if column.startswith("BUREAU_") and column not in selected
    ]
    selected.extend(bureau_features[: max_features - len(selected)])
    if len(selected) < max_features:
        selected.extend(
            [column for column in numeric_columns if column not in selected][
                : max_features - len(selected)
            ]
        )
    return selected[:max_features]


def build_drift_report(
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate PSI, missing-rate deltas, and summary-statistic shifts."""
    features = features or select_drift_features(X_train, X_holdout)
    rows = []
    for feature in features:
        train_values = X_train[feature]
        holdout_values = X_holdout[feature]
        train_missing_rate = float(train_values.isna().mean())
        holdout_missing_rate = float(holdout_values.isna().mean())
        train_mean = float(train_values.mean())
        holdout_mean = float(holdout_values.mean())
        train_median = float(train_values.median())
        holdout_median = float(holdout_values.median())
        psi = calculate_psi(train_values, holdout_values)
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "psi_interpretation": interpret_psi(psi),
                "train_missing_rate": train_missing_rate,
                "holdout_missing_rate": holdout_missing_rate,
                "missing_rate_delta": holdout_missing_rate - train_missing_rate,
                "train_mean": train_mean,
                "holdout_mean": holdout_mean,
                "mean_delta": holdout_mean - train_mean,
                "train_median": train_median,
                "holdout_median": holdout_median,
                "median_delta": holdout_median - train_median,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")


def save_drift_visual(drift_df: pd.DataFrame, config: dict) -> None:
    """Save a PSI bar chart."""
    output_path = ensure_parent_dir(config["visuals"]["psi_drift_report"])
    plot_df = drift_df.sort_values("psi", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["feature"], plot_df["psi"], color="#1B4F8A")
    ax.axvline(0.10, color="#B45309", linestyle="--", linewidth=1, label="Moderate threshold")
    ax.axvline(0.25, color="#991B1B", linestyle="--", linewidth=1, label="High threshold")
    ax.set_title("PSI Drift Simulation: Train vs Holdout")
    ax.set_xlabel("Population Stability Index")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_drift_markdown(drift_df: pd.DataFrame, config: dict) -> None:
    """Write the drift monitoring simulation report."""
    output_path = ensure_parent_dir(config["reports"]["drift_report_md"])
    report = f"""# Drift Monitoring Simulation

This report compares the project training split against the holdout split. It is a **simulation of monitoring expectations**, not live production monitoring.

PSI interpretation:

- PSI < 0.10: low shift
- 0.10 <= PSI < 0.25: moderate shift
- PSI >= 0.25: high shift

{drift_df.to_markdown(index=False, floatfmt=".4f")}

## Recommended Use

In production, the same checks should compare incoming application cohorts against a stable reference population, then be reviewed with performance, calibration, fairness, and manual-review outcome monitoring.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Generate train-vs-holdout drift simulation outputs."""
    config = load_config()
    X_train, X_holdout, _, _ = get_champion_holdout(config)
    features = select_drift_features(X_train, X_holdout)
    drift_df = build_drift_report(X_train, X_holdout, features)
    save_dataframe(drift_df, config["reports"]["drift_report_csv"])
    save_drift_visual(drift_df, config)
    write_drift_markdown(drift_df, config)
    print(f"Drift simulation saved to {config['reports']['drift_report_csv']}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        raise SystemExit(f"Data unavailable for drift monitoring simulation: {error}") from error
