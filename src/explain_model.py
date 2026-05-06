"""
Generate XGBoost explainability outputs for the Credit Risk Intelligence System.

This script:
- Loads the raw Home Credit application data
- Reuses the existing data cleaning and feature engineering workflow
- Loads the trained XGBoost sklearn Pipeline
- Applies the fitted preprocessing pipeline from the trained model
- Generates global SHAP feature importance on a sampled test set
- Generates built-in XGBoost feature importance
- Saves GitHub-friendly CSV, PNG, and Markdown outputs
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_cleaning import load_raw_data, prepare_features_and_target
from feature_engineering import add_domain_features


RAW_DATA_PATH = "data/raw/application_train.csv"
MODEL_PATH = "models/xgboost_credit_risk_model.joblib"

SHAP_IMPORTANCE_OUTPUT_PATH = "reports/shap_feature_importance.csv"
XGBOOST_IMPORTANCE_OUTPUT_PATH = "reports/xgboost_feature_importance.csv"
EXPLAINABILITY_REPORT_OUTPUT_PATH = "reports/explainability_report.md"

SHAP_VISUAL_OUTPUT_PATH = "visuals/shap_feature_importance_xgboost.png"
XGBOOST_VISUAL_OUTPUT_PATH = "visuals/xgboost_feature_importance.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_TOP_N = 20


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the explainability workflow.
    """
    parser = argparse.ArgumentParser(
        description="Generate SHAP and XGBoost feature importance outputs."
    )
    parser.add_argument(
        "--raw-data-path",
        default=RAW_DATA_PATH,
        help="Path to application_train.csv."
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to the trained XGBoost model pipeline."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of holdout rows to sample for SHAP."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top features to save in visuals."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for train/test split and SHAP sampling."
    )

    return parser.parse_args()


def load_xgboost_pipeline(model_path: str) -> Pipeline:
    """
    Load the trained XGBoost sklearn Pipeline from disk.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            "Trained XGBoost model artifact was not found. "
            f"Expected file: {model_path}. "
            "Run `python src/train_xgboost.py` before explainability. "
            "Model artifacts are intentionally not committed to GitHub."
        )

    model = joblib.load(path)

    if not isinstance(model, Pipeline):
        raise TypeError("Expected the saved XGBoost model to be an sklearn Pipeline.")

    if "preprocessor" not in model.named_steps or "model" not in model.named_steps:
        raise ValueError(
            "Expected pipeline steps named `preprocessor` and `model`."
        )

    return model


def prepare_holdout_data(
    raw_data_path: str,
    random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the same engineered holdout data used by model training.
    """
    path = Path(raw_data_path)

    if not path.exists():
        raise FileNotFoundError(
            "Raw Home Credit dataset was not found. "
            f"Expected file: {raw_data_path}. "
            "Download `application_train.csv` from Kaggle and place it in `data/raw/`."
        )

    df = load_raw_data(raw_data_path)
    X, y, _ = prepare_features_and_target(df)
    X = add_domain_features(X)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y
    )

    return X_test, y_test


def sample_holdout_data(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
    random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sample holdout rows for SHAP to keep runtime manageable.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")

    if sample_size >= len(X):
        return X, y

    stratify_target = y if sample_size >= y.nunique() else None

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        random_state=random_state,
        stratify=stratify_target
    )

    return X_sample, y_sample


def strip_transformer_prefix(feature_name: str) -> str:
    """
    Remove ColumnTransformer prefixes such as `num__` and `cat__`.
    """
    if "__" in feature_name:
        return feature_name.split("__", maxsplit=1)[1]

    return feature_name


def get_transformer_columns(preprocessor) -> tuple[list[str], list[str]]:
    """
    Extract original numeric and categorical column names from a fitted preprocessor.
    """
    numeric_features = []
    categorical_features = []

    for transformer_name, _, columns in preprocessor.transformers_:
        if transformer_name == "num":
            numeric_features = list(columns)
        elif transformer_name == "cat":
            categorical_features = list(columns)

    return numeric_features, categorical_features


def get_transformed_feature_names(preprocessor) -> list[str]:
    """
    Get readable feature names after preprocessing.
    """
    try:
        raw_feature_names = preprocessor.get_feature_names_out()
        return [
            strip_transformer_prefix(str(feature_name))
            for feature_name in raw_feature_names
        ]
    except AttributeError:
        numeric_features, categorical_features = get_transformer_columns(preprocessor)
        categorical_pipeline = preprocessor.named_transformers_["cat"]
        one_hot_encoder = categorical_pipeline.named_steps["onehot"]
        categorical_output_features = one_hot_encoder.get_feature_names_out(
            categorical_features
        )

        return list(numeric_features) + list(categorical_output_features)


def map_to_original_feature(
    transformed_feature: str,
    numeric_features: list[str],
    categorical_features: list[str]
) -> str:
    """
    Map one-hot encoded feature names back to their original feature group.
    """
    if transformed_feature in numeric_features:
        return transformed_feature

    for categorical_feature in sorted(categorical_features, key=len, reverse=True):
        if transformed_feature.startswith(f"{categorical_feature}_"):
            return categorical_feature

    return transformed_feature


def get_feature_metadata(preprocessor) -> pd.DataFrame:
    """
    Create metadata linking transformed features to original feature names.
    """
    transformed_features = get_transformed_feature_names(preprocessor)
    numeric_features, categorical_features = get_transformer_columns(preprocessor)

    feature_groups = [
        map_to_original_feature(
            transformed_feature=feature,
            numeric_features=numeric_features,
            categorical_features=categorical_features
        )
        for feature in transformed_features
    ]

    return pd.DataFrame(
        {
            "transformed_feature": transformed_features,
            "feature": feature_groups
        }
    )


def to_dense_array(feature_matrix):
    """
    Convert sparse matrices to dense arrays for SHAP TreeExplainer.
    """
    if sparse.issparse(feature_matrix):
        return feature_matrix.toarray()

    return feature_matrix


def aggregate_importance(
    feature_metadata: pd.DataFrame,
    importance_values: np.ndarray,
    value_column: str
) -> pd.DataFrame:
    """
    Aggregate transformed-feature importance back to original feature names.
    """
    importance_df = feature_metadata.copy()
    importance_df[value_column] = importance_values

    grouped_importance = (
        importance_df
        .groupby("feature", as_index=False)
        .agg(
            **{
                value_column: (value_column, "sum"),
                "transformed_feature_count": ("transformed_feature", "count")
            }
        )
        .sort_values(value_column, ascending=False)
        .reset_index(drop=True)
    )

    grouped_importance.insert(0, "rank", grouped_importance.index + 1)

    return grouped_importance


def calculate_shap_importance(
    xgboost_model,
    processed_sample,
    feature_metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP values for sampled holdout rows.
    """
    try:
        import shap
    except ImportError as error:
        raise RuntimeError(
            "SHAP is not installed in the active environment. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from error

    processed_sample_dense = to_dense_array(processed_sample)

    try:
        explainer = shap.TreeExplainer(xgboost_model)
        shap_values = explainer.shap_values(processed_sample_dense)
    except Exception as error:
        raise RuntimeError(
            "SHAP calculation failed. Try a smaller sample, for example "
            "`python src/explain_model.py --sample-size 250 --top-n 20`, "
            "and confirm that the installed SHAP and XGBoost versions are compatible."
        ) from error

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

    return aggregate_importance(
        feature_metadata=feature_metadata,
        importance_values=mean_abs_shap_values,
        value_column="mean_abs_shap_value"
    )


def calculate_xgboost_importance(
    xgboost_model,
    feature_metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract built-in XGBoost feature importance and aggregate to original features.
    """
    importance_values = np.asarray(xgboost_model.feature_importances_)

    if len(importance_values) != len(feature_metadata):
        raise ValueError(
            "XGBoost feature importance length does not match transformed feature count."
        )

    importance_df = aggregate_importance(
        feature_metadata=feature_metadata,
        importance_values=importance_values,
        value_column="xgboost_importance"
    )

    total_importance = importance_df["xgboost_importance"].sum()

    if total_importance > 0:
        importance_df["xgboost_importance_share"] = (
            importance_df["xgboost_importance"] / total_importance
        )
    else:
        importance_df["xgboost_importance_share"] = 0.0

    return importance_df


def save_importance_csv(
    importance_df: pd.DataFrame,
    output_path: str,
    top_n: int
) -> None:
    """
    Save top feature importance values as a CSV file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    importance_df.head(top_n).to_csv(output_file, index=False)


def save_importance_chart(
    importance_df: pd.DataFrame,
    value_column: str,
    title: str,
    xlabel: str,
    output_path: str,
    top_n: int
) -> None:
    """
    Save a horizontal bar chart for the top feature importance values.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plot_df = (
        importance_df
        .head(top_n)
        .sort_values(value_column, ascending=True)
    )

    fig_height = max(6, top_n * 0.32)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(plot_df["feature"], plot_df[value_column], color="#2C7A7B")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_markdown_table(
    importance_df: pd.DataFrame,
    value_column: str,
    top_n: int = 10
) -> str:
    """
    Create a compact Markdown table for the explainability report.
    """
    display_df = importance_df[["rank", "feature", value_column]].head(top_n).copy()

    value_label = value_column.replace("_", " ").title()
    value_label = value_label.replace("Shap", "SHAP").replace("Xgboost", "XGBoost")
    lines = [
        f"| Rank | Feature | {value_label} |",
        "|---:|---|---:|"
    ]

    for _, row in display_df.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['feature']} | {row[value_column]:.6f} |"
        )

    return "\n".join(lines)


def summarize_top_features(importance_df: pd.DataFrame, top_n: int = 5) -> str:
    """
    Create a readable comma-separated list of top feature names.
    """
    top_features = importance_df["feature"].head(top_n).tolist()

    return ", ".join(f"`{feature}`" for feature in top_features)


def save_explainability_report(
    shap_importance_df: pd.DataFrame,
    xgboost_importance_df: pd.DataFrame,
    sample_size: int,
    output_path: str
) -> None:
    """
    Save a Markdown report summarizing explainability outputs and limitations.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    shap_table = create_markdown_table(
        shap_importance_df,
        value_column="mean_abs_shap_value"
    )
    xgboost_table = create_markdown_table(
        xgboost_importance_df,
        value_column="xgboost_importance"
    )

    top_shap_features = summarize_top_features(shap_importance_df)
    top_xgboost_features = summarize_top_features(xgboost_importance_df)

    report = f"""# XGBoost Explainability Report

This report summarizes global feature importance for the trained XGBoost credit risk model in the Credit Risk Intelligence System.

Explainability in this project is used to understand broad model behavior and support business interpretation. It is not a substitute for credit policy, fairness review, or human judgment.

## Method

- Data source: `application_train.csv`
- Model: trained XGBoost sklearn Pipeline from `models/xgboost_credit_risk_model.joblib`
- Preprocessing: fitted preprocessing pipeline from the trained model
- SHAP sample size: {sample_size:,} holdout rows
- Feature grouping: one-hot encoded categorical features are grouped back to their original feature names

## Built-In XGBoost Importance vs. SHAP

Built-in XGBoost importance shows which transformed features the model uses heavily when building trees. It is fast and useful for a first-pass model inspection, but it can favor variables with many encoded categories.

SHAP importance estimates the average contribution size of each feature to model predictions across sampled holdout rows. In this report, SHAP values are summarized using mean absolute SHAP value and then grouped back to original feature names for readability.

## Top SHAP Features

SHAP importance is calculated as the mean absolute SHAP value across sampled holdout rows.
Higher values indicate a larger average contribution to model predictions.

{shap_table}

## Top Built-In XGBoost Features

Built-in XGBoost importance provides a fast secondary view of which features are used heavily by the model.

{xgboost_table}

## Risk Driver Interpretation

In this run, the strongest SHAP drivers include {top_shap_features}. The strongest built-in XGBoost importance drivers include {top_xgboost_features}.

External source variables are important because they summarize third-party or externally derived credit-risk signals. In this dataset, these variables often provide strong risk-ranking information beyond a single income, loan amount, or demographic field.

Affordability and credit burden ratios also matter because they describe the relationship between a requested loan and the applicant's financial capacity. Features such as credit-to-income, annuity-to-income, and loan-term ratios can help identify applicants whose repayment burden may be high relative to income or loan structure.

## Outputs

- `reports/shap_feature_importance.csv`
- `reports/xgboost_feature_importance.csv`
- `reports/explainability_report.md`
- `visuals/shap_feature_importance_xgboost.png`
- `visuals/xgboost_feature_importance.png`

## Interpretation Notes

- Global feature importance explains average model behavior, not individual applicant decisions.
- Feature importance is not causal evidence.
- SHAP is computed on a sampled holdout set to keep runtime manageable.
- Categorical variables are one-hot encoded before modeling and grouped back for readability.
- The current model uses only `application_train.csv`, so it does not include bureau, previous application, or payment history tables.
- Because default-class precision remains low, this model should support manual risk review rather than automatic rejection.
- Features such as gender, family status, occupation, and education require careful governance and fairness analysis before any real lending use.
"""

    output_file.write_text(report)


def run_explainability(args: argparse.Namespace) -> None:
    """
    Run the explainability workflow.
    """
    if args.top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    print("Loading trained XGBoost pipeline...")
    model_pipeline = load_xgboost_pipeline(args.model_path)
    preprocessor = model_pipeline.named_steps["preprocessor"]
    xgboost_model = model_pipeline.named_steps["model"]

    print("Preparing holdout data...")
    X_test, y_test = prepare_holdout_data(
        raw_data_path=args.raw_data_path,
        random_state=args.random_state
    )

    print("Sampling holdout rows for SHAP...")
    X_sample, _ = sample_holdout_data(
        X=X_test,
        y=y_test,
        sample_size=args.sample_size,
        random_state=args.random_state
    )

    print("Applying fitted preprocessing pipeline...")
    processed_sample = preprocessor.transform(X_sample)
    feature_metadata = get_feature_metadata(preprocessor)

    if processed_sample.shape[1] != len(feature_metadata):
        raise ValueError(
            "Transformed feature count does not match extracted feature names."
        )

    print("Calculating SHAP feature importance...")
    shap_importance_df = calculate_shap_importance(
        xgboost_model=xgboost_model,
        processed_sample=processed_sample,
        feature_metadata=feature_metadata
    )

    print("Calculating built-in XGBoost feature importance...")
    xgboost_importance_df = calculate_xgboost_importance(
        xgboost_model=xgboost_model,
        feature_metadata=feature_metadata
    )

    print("Saving explainability outputs...")
    save_importance_csv(
        shap_importance_df,
        SHAP_IMPORTANCE_OUTPUT_PATH,
        top_n=args.top_n
    )
    save_importance_csv(
        xgboost_importance_df,
        XGBOOST_IMPORTANCE_OUTPUT_PATH,
        top_n=args.top_n
    )

    save_importance_chart(
        importance_df=shap_importance_df,
        value_column="mean_abs_shap_value",
        title="Top XGBoost Features by Mean Absolute SHAP Value",
        xlabel="Mean Absolute SHAP Value",
        output_path=SHAP_VISUAL_OUTPUT_PATH,
        top_n=args.top_n
    )
    save_importance_chart(
        importance_df=xgboost_importance_df,
        value_column="xgboost_importance",
        title="Top XGBoost Features by Built-In Importance",
        xlabel="XGBoost Feature Importance",
        output_path=XGBOOST_VISUAL_OUTPUT_PATH,
        top_n=args.top_n
    )
    save_explainability_report(
        shap_importance_df=shap_importance_df,
        xgboost_importance_df=xgboost_importance_df,
        sample_size=len(X_sample),
        output_path=EXPLAINABILITY_REPORT_OUTPUT_PATH
    )

    print()
    print("Explainability complete.")
    print(f"Rows explained with SHAP: {len(X_sample):,}")
    print(f"SHAP importance saved to: {SHAP_IMPORTANCE_OUTPUT_PATH}")
    print(f"XGBoost importance saved to: {XGBOOST_IMPORTANCE_OUTPUT_PATH}")
    print(f"SHAP visual saved to: {SHAP_VISUAL_OUTPUT_PATH}")
    print(f"XGBoost visual saved to: {XGBOOST_VISUAL_OUTPUT_PATH}")
    print(f"Report saved to: {EXPLAINABILITY_REPORT_OUTPUT_PATH}")


def main() -> None:
    """
    Parse arguments and run explainability with clean error messages.
    """
    args = parse_args()

    try:
        run_explainability(args)
    except Exception as error:
        print()
        print(f"Explainability failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
