"""
Data cleaning utilities for the Credit Risk Intelligence System.

This module handles:
- Loading raw Home Credit data
- Summarizing missing values
- Dropping high-missing columns
- Separating features and target
- Detecting numeric and categorical columns
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd


DEFAULT_TARGET_COLUMN = "TARGET"
DEFAULT_ID_COLUMN = "SK_ID_CURR"


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw application training dataset.

    Parameters
    ----------
    file_path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(path)


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a missing-value summary table.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns and missing rates.
    """
    missing_summary = (
        df.isnull()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    missing_summary.columns = ["column", "missing_rate"]

    return missing_summary


def get_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.40,
    exclude_columns: List[str] | None = None
) -> List[str]:
    """
    Identify columns with missing rates above a selected threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    threshold : float
        Missing-rate cutoff. Default is 0.40.
    exclude_columns : list[str] or None
        Columns that should not be dropped even if they exceed the threshold.

    Returns
    -------
    list[str]
        List of high-missing columns.
    """
    if exclude_columns is None:
        exclude_columns = []

    missing_summary = get_missing_summary(df)

    high_missing_columns = missing_summary[
        missing_summary["missing_rate"] > threshold
    ]["column"].tolist()

    high_missing_columns = [
        column for column in high_missing_columns
        if column not in exclude_columns
    ]

    return high_missing_columns


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    missing_threshold: float = 0.40
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare feature matrix X and target vector y.

    This removes:
    - target column from X
    - ID column
    - columns with high missing rates

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    target_column : str
        Name of target column.
    id_column : str
        Name of ID column.
    missing_threshold : float
        Missing-rate cutoff for dropping columns.

    Returns
    -------
    tuple
        X, y, dropped_columns
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    columns_to_exclude = [target_column]

    if id_column in df.columns:
        columns_to_exclude.append(id_column)

    high_missing_columns = get_high_missing_columns(
        df=df,
        threshold=missing_threshold,
        exclude_columns=columns_to_exclude
    )

    columns_to_drop = []

    if id_column in df.columns:
        columns_to_drop.append(id_column)

    columns_to_drop.extend(high_missing_columns)

    X = df.drop(columns=[target_column] + columns_to_drop)
    y = df[target_column]

    return X, y, columns_to_drop


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separate numeric and categorical feature names.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    tuple
        numeric_features, categorical_features
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return numeric_features, categorical_features


if __name__ == "__main__":
    raw_file_path = "data/raw/application_train.csv"

    df = load_raw_data(raw_file_path)
    X, y, dropped_columns = prepare_features_and_target(df)
    numeric_features, categorical_features = get_feature_types(X)

    print("Raw data shape:", df.shape)
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Dropped columns:", len(dropped_columns))
    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))
    print()
    print("Target distribution:")
    print(y.value_counts(normalize=True))