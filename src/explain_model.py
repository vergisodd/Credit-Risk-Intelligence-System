"""Generate global and applicant-level SHAP explanations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.champion_model import get_champion_spec, load_champion_feature_data, load_champion_model
from src.config_loader import ensure_parent_dir, load_config, resolve_path
from src.model_utils import make_train_test_split, save_dataframe


FEATURE_INTERPRETATIONS = {
    "EXT_SOURCE_MEAN": "Lower external-source scores generally indicate a weaker external credit-risk signal.",
    "EXT_SOURCE_MIN": "The weakest external-source score can pull an applicant into a higher-review group.",
    "EXT_SOURCE_MAX": "The strongest external-source score can offset some weaker application attributes.",
    "EXT_SOURCE_1": "This external score is an externally derived risk signal used for ranking applicants.",
    "EXT_SOURCE_2": "This external score adds third-party risk information beyond income and loan amount.",
    "EXT_SOURCE_3": "This external score often captures repayment-risk signal not visible in simple affordability ratios.",
    "CODE_GENDER": "Gender is a sensitive demographic attribute and requires governance review before any deployment.",
    "NAME_EDUCATION_TYPE": "Education category may proxy socioeconomic patterns and should be reviewed for fairness impact.",
    "CREDIT_TERM_RATIO": "A higher annuity-to-credit relationship can indicate a shorter or heavier repayment structure.",
    "GOODS_CREDIT_RATIO": "This compares the financed goods price with the total requested credit.",
    "AMT_ANNUITY": "Higher required annuity payments can increase repayment burden.",
    "AMT_GOODS_PRICE": "The goods price helps contextualize the requested credit amount.",
    "CREDIT_INCOME_RATIO": "Higher requested credit relative to income can increase affordability risk.",
    "ANNUITY_INCOME_RATIO": "Higher annuity payments relative to income can pressure monthly cash flow.",
    "EMPLOYMENT_YEARS": "Employment tenure can provide stability signal, although it is not causal proof.",
    "DAYS_EMPLOYED_CLEAN": "Cleaned employment duration captures tenure after removing dataset sentinel values.",
    "BUREAU_LOAN_COUNT": "Total number of previous credit bureau loans; more loans can indicate credit dependence.",
    "BUREAU_ACTIVE_LOAN_COUNT": "Currently open bureau loans; high counts may indicate stretched repayment capacity.",
    "BUREAU_CLOSED_LOAN_COUNT": "Successfully closed loans can provide positive credit history signal.",
    "BUREAU_AVG_DAYS_CREDIT": "Average time since bureau credit lines were opened; more recent credit can indicate higher need.",
    "BUREAU_AVG_DAYS_CREDIT_ENDDATE": "Average days until bureau credit lines close; longer terms may indicate larger obligations.",
    "BUREAU_MAX_DAYS_OVERDUE": "Maximum days overdue on any bureau loan; any overdue history is a strong risk signal.",
    "BUREAU_MEAN_DAYS_OVERDUE": "Average overdue days across all bureau loans; even small values elevate risk.",
    "BUREAU_SUM_AMT_CREDIT_SUM": "Total credit exposure across all bureau loans; indicates overall debt burden.",
    "BUREAU_SUM_AMT_CREDIT_SUM_DEBT": "Total outstanding debt on bureau loans; directly captures unresolved obligations.",
    "BUREAU_SUM_AMT_CREDIT_SUM_OVERDUE": "Total overdue amount across bureau loans; even small overdue sums are material.",
    "BUREAU_ACTIVE_DEBT_RATIO": "Ratio of outstanding debt to total bureau credit; higher ratios indicate less repayment progress.",
    "BUREAU_PROLONGED_LOAN_COUNT": "Count of bureau loans with extended terms; prolongation often indicates repayment difficulty.",
    "BUREAU_CREDIT_ACTIVE_RATIO": "Share of bureau loans still active; high ratios may indicate difficulty closing obligations.",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    config = load_config()
    parser = argparse.ArgumentParser(description="Generate SHAP explainability outputs.")
    parser.add_argument("--sample-size", type=int, default=config["shap"]["sample_size"])
    parser.add_argument("--top-n", type=int, default=config["shap"]["top_n"])
    return parser.parse_args()


def to_dense_array(feature_matrix: Any) -> np.ndarray:
    """Convert sparse matrices to dense arrays for SHAP plotting."""
    if sparse.issparse(feature_matrix):
        return feature_matrix.toarray()
    return np.asarray(feature_matrix)


def strip_transformer_prefix(feature_name: str) -> str:
    """Remove ColumnTransformer prefixes from feature names."""
    if "__" in feature_name:
        return feature_name.split("__", maxsplit=1)[1]
    return feature_name


def get_transformer_columns(preprocessor: Any) -> tuple[list[str], list[str]]:
    """Extract original numeric and categorical columns from a fitted preprocessor."""
    numeric_features = []
    categorical_features = []
    for transformer_name, _, columns in preprocessor.transformers_:
        if transformer_name == "num":
            numeric_features = list(columns)
        elif transformer_name == "cat":
            categorical_features = list(columns)
    return numeric_features, categorical_features


def get_transformed_feature_names(preprocessor: Any) -> list[str]:
    """Get readable feature names after preprocessing."""
    raw_feature_names = preprocessor.get_feature_names_out()
    return [strip_transformer_prefix(str(feature_name)) for feature_name in raw_feature_names]


def map_to_original_feature(
    transformed_feature: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> str:
    """Map one-hot encoded feature names back to their source column."""
    if transformed_feature in numeric_features:
        return transformed_feature
    for categorical_feature in sorted(categorical_features, key=len, reverse=True):
        if transformed_feature.startswith(f"{categorical_feature}_"):
            return categorical_feature
    return transformed_feature


def get_feature_metadata(preprocessor: Any) -> pd.DataFrame:
    """Create transformed-to-original feature metadata."""
    transformed_features = get_transformed_feature_names(preprocessor)
    numeric_features, categorical_features = get_transformer_columns(preprocessor)
    feature_groups = [
        map_to_original_feature(feature, numeric_features, categorical_features)
        for feature in transformed_features
    ]
    return pd.DataFrame(
        {
            "transformed_feature": transformed_features,
            "feature": feature_groups,
        }
    )


def normalize_shap_values(shap_values: Any) -> np.ndarray:
    """Normalize common binary-classifier SHAP output shapes."""
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    return shap_values


def normalize_base_value(base_value: Any) -> float:
    """Return the positive-class base value for SHAP plots."""
    base_array = np.asarray(base_value)
    if base_array.ndim == 0:
        return float(base_array)
    return float(base_array.reshape(-1)[-1])


def calculate_shap_values(
    model_pipeline: Pipeline, X_sample: pd.DataFrame
) -> tuple[np.ndarray, float, np.ndarray, pd.DataFrame]:
    """Transform sample rows and calculate SHAP values for the model step."""
    try:
        import shap
    except ImportError as error:
        raise RuntimeError(
            "SHAP is not installed. Install dependencies before explaining models."
        ) from error

    preprocessor = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["model"]
    transformed_sample = to_dense_array(preprocessor.transform(X_sample))
    explainer = shap.TreeExplainer(estimator)
    shap_values = normalize_shap_values(explainer.shap_values(transformed_sample))
    base_value = normalize_base_value(explainer.expected_value)
    metadata = get_feature_metadata(preprocessor)
    return shap_values, base_value, transformed_sample, metadata


def aggregate_global_importance(
    feature_metadata: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """Aggregate transformed-feature SHAP values back to original features."""
    working = feature_metadata.copy()
    working["mean_abs_shap_value"] = np.abs(shap_values).mean(axis=0)
    working["mean_shap_value"] = shap_values.mean(axis=0)
    grouped = (
        working.groupby("feature", as_index=False)
        .agg(
            mean_abs_shap_value=("mean_abs_shap_value", "sum"),
            mean_shap_value=("mean_shap_value", "sum"),
            transformed_feature_count=("transformed_feature", "count"),
        )
        .sort_values("mean_abs_shap_value", ascending=False)
        .reset_index(drop=True)
    )
    grouped.insert(0, "rank", grouped.index + 1)
    grouped["direction"] = np.where(
        grouped["mean_shap_value"] >= 0,
        "increases risk on average",
        "decreases risk on average",
    )
    return grouped


def save_global_shap_chart(
    importance_df: pd.DataFrame,
    output_path: str,
    top_n: int,
    model_name: str = "Champion Model",
) -> None:
    """Save a horizontal global SHAP importance bar chart."""
    output_file = ensure_parent_dir(output_path)
    plot_df = importance_df.head(top_n).sort_values("mean_abs_shap_value", ascending=True)
    fig_height = max(6, top_n * 0.32)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap_value"], color="#1B4F8A")
    ax.set_title(f"Top {model_name} Features by Mean Absolute SHAP Value")
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_feature_interpretation(feature: str) -> str:
    """Return a one-sentence business interpretation for a feature."""
    return FEATURE_INTERPRETATIONS.get(
        feature,
        "This feature contributes predictive signal in the model but should not be interpreted as causal evidence.",
    )


def save_explainability_report(importance_df: pd.DataFrame, sample_size: int, config: dict) -> None:
    """Write the explainability report with global and individual methodology."""
    output_file = ensure_parent_dir(config["reports"]["explainability_report"])
    champion = get_champion_spec(config)
    top_features = importance_df.head(10).copy()
    table_lines = [
        "| Rank | Feature | Direction | Mean Abs SHAP | Interpretation |",
        "|---:|---|---|---:|---|",
    ]
    for _, row in top_features.iterrows():
        feature = row["feature"]
        table_lines.append(
            f"| {int(row['rank'])} | `{feature}` | {row['direction']} | "
            f"{row['mean_abs_shap_value']:.6f} | {create_feature_interpretation(feature)} |"
        )

    report = f"""# Explainability Report

## Global SHAP Findings

Global SHAP analysis was computed on {sample_size:,} stratified holdout applicants using the configured champion model: **{champion.model_name}**. One-hot encoded features are aggregated back to source feature names where possible. The top global features are:

{chr(10).join(table_lines)}

## Fairness Note

`CODE_GENDER` and `NAME_EDUCATION_TYPE` appear in the feature set and may contribute to model predictions. These variables require fairness review because they can directly encode or proxy protected and socioeconomic characteristics.

## Per-Applicant Explanation Methodology

Applicant-level explanations transform the applicant row through the fitted preprocessing pipeline, calculate SHAP values on the transformed feature vector, aggregate one-hot encoded values back to source feature names, and render a waterfall plot showing the strongest positive and negative contributors to that specific prediction. The generated reason-code table separates contributors increasing risk from contributors decreasing risk and includes raw applicant values where available.

## Limitations

Global feature importance reflects aggregate model behavior across the holdout population. SHAP values for individual applicants indicate feature contribution to that specific prediction, not causal credit risk factors and not legally sufficient adverse-action reasons. Sensitive attributes present in the feature set require governance review before any deployment in a regulated lending context under ECOA or equivalent frameworks.
"""
    output_file.write_text(report, encoding="utf-8")


def generate_applicant_reason_codes(
    pipeline: Pipeline,
    X_row: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return business-readable applicant-level SHAP reason codes."""
    try:
        import shap
    except ImportError as error:
        raise RuntimeError(
            "SHAP is not installed. Install dependencies before explaining applicants."
        ) from error

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    transformed = to_dense_array(preprocessor.transform(X_row))
    explainer = shap.TreeExplainer(estimator)
    shap_values = normalize_shap_values(explainer.shap_values(transformed))[0]
    metadata = get_feature_metadata(preprocessor)
    working = metadata.copy()
    working["shap_value"] = shap_values
    grouped = (
        working.groupby("feature", as_index=False)
        .agg(
            shap_value=("shap_value", "sum"),
            transformed_feature_count=("transformed_feature", "count"),
        )
        .assign(abs_shap=lambda df: df["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    grouped["direction"] = np.where(
        grouped["shap_value"] >= 0,
        "increases risk",
        "decreases risk",
    )
    raw_values = []
    for feature in grouped["feature"]:
        value = X_row.iloc[0][feature] if feature in X_row.columns else None
        raw_values.append("" if pd.isna(value) else str(value))
    grouped["raw_applicant_value"] = raw_values
    grouped["business_interpretation"] = grouped["feature"].apply(create_feature_interpretation)
    positive = grouped[grouped["shap_value"] > 0].head(top_n)
    negative = grouped[grouped["shap_value"] < 0].head(top_n)
    reason_codes = pd.concat([positive, negative], ignore_index=True)
    reason_codes.insert(0, "rank", range(1, len(reason_codes) + 1))
    return reason_codes[
        [
            "rank",
            "feature",
            "direction",
            "shap_value",
            "raw_applicant_value",
            "business_interpretation",
        ]
    ]


def generate_individual_explanation(
    pipeline: Pipeline,
    X_row: pd.DataFrame,
    feature_names: list[str],
    output_path: str | Path,
) -> dict[str, dict[str, float | str]]:
    """
    Generate a SHAP waterfall plot for one applicant and return top contributors.
    """
    try:
        import shap
    except ImportError as error:
        raise RuntimeError(
            "SHAP is not installed. Install dependencies before explaining applicants."
        ) from error

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    transformed = to_dense_array(preprocessor.transform(X_row))
    transformed_feature_names = feature_names or get_transformed_feature_names(preprocessor)
    explainer = shap.TreeExplainer(estimator)
    shap_values = normalize_shap_values(explainer.shap_values(transformed))
    base_value = normalize_base_value(explainer.expected_value)

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=transformed[0],
        feature_names=transformed_feature_names,
    )
    output_file = ensure_parent_dir(output_path)
    shap.plots.waterfall(
        explanation, max_display=load_config()["shap"]["waterfall_max_display"], show=False
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    top_indices = np.argsort(np.abs(shap_values[0]))[::-1][:5]
    top_features = {}
    for index in top_indices:
        feature = transformed_feature_names[index]
        contribution = float(shap_values[0][index])
        top_features[feature] = {
            "direction": "positive" if contribution >= 0 else "negative",
            "magnitude": abs(contribution),
            "shap_value": contribution,
            "actual_value": float(transformed[0][index]),
        }
    return top_features


def select_example_row(
    X_test: pd.DataFrame, probabilities: np.ndarray, lower: float | None, upper: float | None
) -> pd.DataFrame:
    """Select one holdout row matching a probability range, with nearest fallback."""
    mask = np.ones(len(probabilities), dtype=bool)
    if lower is not None:
        mask &= probabilities > lower
    if upper is not None:
        mask &= probabilities < upper
    matching_indices = np.flatnonzero(mask)
    if len(matching_indices) > 0:
        return X_test.iloc[[matching_indices[0]]]

    midpoint = upper if lower is None else lower if upper is None else (lower + upper) / 2
    nearest_index = int(np.argmin(np.abs(probabilities - midpoint)))
    return X_test.iloc[[nearest_index]]


def run_explainability(args: argparse.Namespace) -> None:
    """Generate global and three individual SHAP outputs."""
    config = load_config()
    champion = get_champion_spec(config)
    if args.sample_size <= 0 or args.top_n <= 0:
        raise ValueError("sample-size and top-n must be positive integers.")

    print(f"Loading champion pipeline: {champion.model_name}...")
    model_pipeline = load_champion_model(config)

    print("Preparing holdout data...")
    X, y = load_champion_feature_data(config)
    _, X_test, _, y_test = make_train_test_split(X, y, config)

    print("Sampling holdout rows for global SHAP...")
    sample_size = min(args.sample_size, len(X_test))
    X_sample, _, _, _ = train_test_split(
        X_test,
        y_test,
        train_size=sample_size,
        random_state=config["model"]["random_state"],
        stratify=y_test if sample_size >= y_test.nunique() else None,
    )

    print("Calculating global SHAP values...")
    shap_values, _, _, metadata = calculate_shap_values(model_pipeline, X_sample)
    importance_df = aggregate_global_importance(metadata, shap_values)

    print("Saving global SHAP outputs...")
    save_dataframe(importance_df.head(args.top_n), config["reports"]["shap_feature_importance"])
    save_global_shap_chart(
        importance_df,
        config["visuals"]["shap_feature_importance"],
        args.top_n,
        model_name=champion.model_name,
    )
    save_explainability_report(importance_df, sample_size, config)

    print("Generating individual applicant waterfall plots...")
    probabilities = model_pipeline.predict_proba(X_test)[:, 1]
    transformed_names = get_transformed_feature_names(model_pipeline.named_steps["preprocessor"])
    examples = config["shap"]["individual_examples"]
    example_specs = [
        (None, examples["low_max"], config["visuals"]["shap_individual_low_risk"]),
        (
            examples["medium_min"],
            examples["medium_max"],
            config["visuals"]["shap_individual_medium_risk"],
        ),
        (examples["high_min"], None, config["visuals"]["shap_individual_high_risk"]),
    ]
    reason_code_frames = []
    for label, (lower, upper, output_path) in zip(["low", "medium", "high"], example_specs):
        X_row = select_example_row(X_test, probabilities, lower, upper)
        generate_individual_explanation(
            model_pipeline, X_row, transformed_names, resolve_path(output_path)
        )
        reason_codes = generate_applicant_reason_codes(model_pipeline, X_row)
        reason_codes.insert(0, "example_risk_band", label)
        reason_codes.insert(
            1, "predicted_default_probability", float(model_pipeline.predict_proba(X_row)[:, 1][0])
        )
        reason_code_frames.append(reason_codes)
    save_dataframe(
        pd.concat(reason_code_frames, ignore_index=True),
        config["reports"]["applicant_reason_codes"],
    )

    print()
    print("Explainability complete.")


def main() -> None:
    """Parse arguments and run explainability."""
    args = parse_args()
    try:
        run_explainability(args)
    except Exception as error:
        print(f"Explainability failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
