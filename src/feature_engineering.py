"""
Feature engineering utilities for the Credit Risk Intelligence System.

This module creates domain-informed credit risk features such as:
- credit-to-income ratio
- annuity-to-income ratio
- income per family member
- applicant age
- employment length
- external source score averages
"""

from typing import List

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Safely divide two pandas Series.

    Replaces zero denominators with NaN to avoid infinite values.

    Parameters
    ----------
    numerator : pd.Series
        Numerator values.
    denominator : pd.Series
        Denominator values.

    Returns
    -------
    pd.Series
        Division result.
    """
    denominator_clean = denominator.replace(0, np.nan)

    return numerator / denominator_clean


def add_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-specific credit risk features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with added engineered features.
    """
    X = X.copy()

    if "AMT_CREDIT" in X.columns and "AMT_INCOME_TOTAL" in X.columns:
        X["CREDIT_INCOME_RATIO"] = safe_divide(
            X["AMT_CREDIT"],
            X["AMT_INCOME_TOTAL"]
        )

    if "AMT_ANNUITY" in X.columns and "AMT_INCOME_TOTAL" in X.columns:
        X["ANNUITY_INCOME_RATIO"] = safe_divide(
            X["AMT_ANNUITY"],
            X["AMT_INCOME_TOTAL"]
        )

    if "AMT_GOODS_PRICE" in X.columns and "AMT_CREDIT" in X.columns:
        X["GOODS_CREDIT_RATIO"] = safe_divide(
            X["AMT_GOODS_PRICE"],
            X["AMT_CREDIT"]
        )

    if "AMT_ANNUITY" in X.columns and "AMT_CREDIT" in X.columns:
        X["CREDIT_TERM_RATIO"] = safe_divide(
            X["AMT_ANNUITY"],
            X["AMT_CREDIT"]
        )

    if "AMT_INCOME_TOTAL" in X.columns and "CNT_FAM_MEMBERS" in X.columns:
        X["INCOME_PER_FAMILY_MEMBER"] = safe_divide(
            X["AMT_INCOME_TOTAL"],
            X["CNT_FAM_MEMBERS"]
        )

    if "DAYS_BIRTH" in X.columns:
        X["AGE_YEARS"] = np.abs(X["DAYS_BIRTH"]) / 365.25

    if "DAYS_EMPLOYED" in X.columns:
        X["DAYS_EMPLOYED_CLEAN"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)
        X["EMPLOYMENT_YEARS"] = np.abs(X["DAYS_EMPLOYED_CLEAN"]) / 365.25

    ext_source_columns = [
        column for column in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        if column in X.columns
    ]

    if ext_source_columns:
        X["EXT_SOURCE_MEAN"] = X[ext_source_columns].mean(axis=1)
        X["EXT_SOURCE_MIN"] = X[ext_source_columns].min(axis=1)
        X["EXT_SOURCE_MAX"] = X[ext_source_columns].max(axis=1)

    X = replace_infinite_values(X)

    return X


def replace_infinite_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Replace positive and negative infinite values with NaN.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    X = X.copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def get_engineered_feature_names(
    original_columns: List[str],
    engineered_df: pd.DataFrame
) -> List[str]:
    """
    Identify newly created engineered feature names.

    Parameters
    ----------
    original_columns : list[str]
        Feature names before engineering.
    engineered_df : pd.DataFrame
        Dataframe after feature engineering.

    Returns
    -------
    list[str]
        Names of newly engineered features.
    """
    original_columns_set = set(original_columns)

    engineered_features = [
        column for column in engineered_df.columns
        if column not in original_columns_set
    ]

    return engineered_features


if __name__ == "__main__":
    from data_cleaning import load_raw_data, prepare_features_and_target

    raw_file_path = "data/raw/application_train.csv"

    df = load_raw_data(raw_file_path)

    X, y, dropped_columns = prepare_features_and_target(df)

    original_columns = X.columns.tolist()

    X_engineered = add_domain_features(X)

    engineered_features = get_engineered_feature_names(
        original_columns=original_columns,
        engineered_df=X_engineered
    )

    print("Original feature shape:", X.shape)
    print("Engineered feature shape:", X_engineered.shape)
    print("New engineered features:", len(engineered_features))
    print()
    print("Engineered feature names:")
    for feature in engineered_features:
        print("-", feature)