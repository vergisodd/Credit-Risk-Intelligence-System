"""
Data loading, validation, and cleaning utilities.

The cleaning layer keeps the feature matrix raw: no imputation, scaling, or
encoding happens here. Those operations belong inside model pipelines so that
they are fit only on training folds.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config_loader import load_config, resolve_path


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

DEFAULT_TARGET_COLUMN = "TARGET"
DEFAULT_ID_COLUMN = "SK_ID_CURR"
DEFAULT_MISSING_THRESHOLD = 0.40
FEATURE_ENGINEERING_SOURCE_COLUMNS = ["EXT_SOURCE_1"]


def load_raw_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the raw Home Credit application training dataset.
    """
    config = load_config()
    path = resolve_path(file_path or config["paths"]["raw_data"])

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset was not found at {path}. "
            "Download the Kaggle application training file before running this step."
        )

    LOGGER.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    LOGGER.info("Raw data shape: %s", df.shape)
    return df


def validate_data(df: pd.DataFrame, strict_shape: bool = True) -> None:
    """
    Validate key assumptions about the Home Credit training dataset.
    """
    if DEFAULT_TARGET_COLUMN not in df.columns:
        raise AssertionError("TARGET column is required for supervised training.")

    if DEFAULT_ID_COLUMN in df.columns and df[DEFAULT_ID_COLUMN].duplicated().any():
        raise AssertionError("SK_ID_CURR contains duplicate applicant identifiers.")

    observed_targets = set(df[DEFAULT_TARGET_COLUMN].dropna().unique())
    if not observed_targets.issubset({0, 1}):
        raise AssertionError("TARGET must contain only binary values 0 and 1.")

    default_rate = df[DEFAULT_TARGET_COLUMN].mean()
    if not 0.07 <= default_rate <= 0.10:
        raise AssertionError(
            "Default rate is outside the expected range "
            f"(observed {default_rate:.2%})."
        )

    if strict_shape:
        rows, columns = df.shape
        if not 300_000 <= rows <= 315_000 or not 115 <= columns <= 130:
            raise AssertionError(
                "Dataset shape is outside the expected Home Credit range "
                f"(observed {rows:,} rows and {columns:,} columns)."
            )


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a missing-value summary sorted by missing rate.
    """
    missing_summary = df.isnull().mean().sort_values(ascending=False).reset_index()
    missing_summary.columns = ["column", "missing_rate"]
    return missing_summary


def get_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = DEFAULT_MISSING_THRESHOLD,
    exclude_columns: List[str] | None = None,
) -> List[str]:
    """
    Identify columns with missing rates above the configured threshold.
    """
    exclude_columns = exclude_columns or []
    missing_summary = get_missing_summary(df)
    high_missing_columns = missing_summary.loc[
        missing_summary["missing_rate"] > threshold, "column"
    ].tolist()
    return [column for column in high_missing_columns if column not in exclude_columns]


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare raw feature matrix and target vector.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    columns_to_exclude = [target_column, *FEATURE_ENGINEERING_SOURCE_COLUMNS]
    if id_column in df.columns:
        columns_to_exclude.append(id_column)

    high_missing_columns = get_high_missing_columns(
        df=df,
        threshold=missing_threshold,
        exclude_columns=columns_to_exclude,
    )

    columns_to_drop = [column for column in [id_column] if column in df.columns]
    columns_to_drop.extend(high_missing_columns)

    LOGGER.info("Shape after high-missing/id drop: %s", df.drop(columns=columns_to_drop).shape)
    X = df.drop(columns=[target_column] + columns_to_drop)
    y = df[target_column].astype(int)

    LOGGER.info("Dropped %s columns during cleaning", len(columns_to_drop))
    LOGGER.info("Feature matrix shape after cleaning: %s", X.shape)
    LOGGER.info("Class distribution: %s", y.value_counts(normalize=True).to_dict())
    return X, y, columns_to_drop


def load_and_clean(
    file_path: str | Path | None = None,
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
    validate_schema: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load raw data, validate it, and return raw model features plus target.

    Returns
    -------
    tuple
        X, y, feature_names where X contains raw feature columns before
        preprocessing and feature_names is the list of retained original columns.
    """
    df = load_raw_data(file_path)
    validate_data(df, strict_shape=validate_schema)
    X, y, _ = prepare_features_and_target(df, missing_threshold=missing_threshold)
    feature_names = X.columns.tolist()
    LOGGER.info("Returning %s raw feature names", len(feature_names))
    return X, y, feature_names


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separate numeric and categorical feature names.
    """
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_features, categorical_features


if __name__ == "__main__":
    X_raw, y_raw, names = load_and_clean()
    numeric, categorical = get_feature_types(X_raw)
    print("Feature matrix shape:", X_raw.shape)
    print("Target shape:", y_raw.shape)
    print("Feature names:", len(names))
    print("Numeric features:", len(numeric))
    print("Categorical features:", len(categorical))
    print("Target distribution:")
    print(y_raw.value_counts(normalize=True))
