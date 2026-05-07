"""
Domain-informed feature engineering for credit risk modelling.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

ENGINEERED_FEATURES = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "GOODS_CREDIT_RATIO",
    "CREDIT_TERM_RATIO",
    "INCOME_PER_FAMILY_MEMBER",
    "AGE_YEARS",
    "DAYS_EMPLOYED_CLEAN",
    "EMPLOYMENT_YEARS",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
    "EXT_SOURCE_MAX",
    "EXT_SOURCE_PRODUCT",
    "CREDIT_TO_INCOME_TERM",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Divide with zero-denominator protection.
    """
    return pd.Series(
        np.where(denominator != 0, numerator / denominator, np.nan),
        index=numerator.index,
    )


def add_credit_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add credit amount divided by applicant income."""
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        LOGGER.info("Adding CREDIT_INCOME_RATIO")
        df["CREDIT_INCOME_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    return df


def add_annuity_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add annuity payment divided by applicant income."""
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        LOGGER.info("Adding ANNUITY_INCOME_RATIO")
        df["ANNUITY_INCOME_RATIO"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    return df


def add_goods_credit_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add goods price divided by credit amount."""
    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(df.columns):
        LOGGER.info("Adding GOODS_CREDIT_RATIO")
        df["GOODS_CREDIT_RATIO"] = safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    return df


def add_credit_term_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add annuity amount divided by credit amount."""
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        LOGGER.info("Adding CREDIT_TERM_RATIO")
        df["CREDIT_TERM_RATIO"] = safe_divide(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    return df


def add_income_per_family_member(df: pd.DataFrame) -> pd.DataFrame:
    """Add income normalized by family member count."""
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        LOGGER.info("Adding INCOME_PER_FAMILY_MEMBER")
        df["INCOME_PER_FAMILY_MEMBER"] = safe_divide(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    return df


def add_age_years(df: pd.DataFrame) -> pd.DataFrame:
    """Convert applicant age from negative days to positive years."""
    if "DAYS_BIRTH" in df.columns:
        LOGGER.info("Adding AGE_YEARS")
        df["AGE_YEARS"] = np.abs(df["DAYS_BIRTH"]) / 365.25
    return df


def add_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean employment sentinel values and convert tenure to years."""
    if "DAYS_EMPLOYED" in df.columns:
        LOGGER.info("Adding employment duration features")
        df["DAYS_EMPLOYED_CLEAN"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        df["EMPLOYMENT_YEARS"] = np.abs(df["DAYS_EMPLOYED_CLEAN"]) / 365.25
    return df


def add_external_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add aggregate and interaction features from external source scores."""
    ext_source_columns = [
        column
        for column in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        if column in df.columns
    ]
    if ext_source_columns:
        LOGGER.info("Adding external source aggregate features")
        df["EXT_SOURCE_MEAN"] = df[ext_source_columns].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_source_columns].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_source_columns].max(axis=1)

    if {"EXT_SOURCE_1", "EXT_SOURCE_2"}.issubset(df.columns):
        LOGGER.info("Adding EXT_SOURCE_PRODUCT")
        df["EXT_SOURCE_PRODUCT"] = np.nanprod(
            df[["EXT_SOURCE_1", "EXT_SOURCE_2"]].to_numpy(dtype=float),
            axis=1,
        )
        missing_both = df[["EXT_SOURCE_1", "EXT_SOURCE_2"]].isna().all(axis=1)
        df.loc[missing_both, "EXT_SOURCE_PRODUCT"] = np.nan
    return df


def add_credit_to_income_term(df: pd.DataFrame) -> pd.DataFrame:
    """Add credit amount relative to income and estimated term burden."""
    required = {"AMT_CREDIT", "AMT_INCOME_TOTAL", "CREDIT_TERM_RATIO"}
    if required.issubset(df.columns):
        LOGGER.info("Adding CREDIT_TO_INCOME_TERM")
        denominator = df["AMT_INCOME_TOTAL"] * df["CREDIT_TERM_RATIO"]
        df["CREDIT_TO_INCOME_TERM"] = safe_divide(df["AMT_CREDIT"], denominator)
    return df


def replace_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace positive and negative infinite values with NaN.
    """
    return df.replace([np.inf, -np.inf], np.nan)


def add_all_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all credit-risk feature engineering steps.
    """
    df = X.copy()
    for feature_step in [
        add_credit_income_ratio,
        add_annuity_income_ratio,
        add_goods_credit_ratio,
        add_credit_term_ratio,
        add_income_per_family_member,
        add_age_years,
        add_employment_features,
        add_external_source_features,
        add_credit_to_income_term,
    ]:
        df = feature_step(df)

    df = replace_infinite_values(df)
    LOGGER.info("Feature matrix shape after engineering: %s", df.shape)
    return df


def add_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible alias for the original feature engineering function.
    """
    return add_all_features(X)


def get_engineered_feature_names(
    original_columns: List[str],
    engineered_df: pd.DataFrame,
) -> List[str]:
    """
    Identify newly created engineered feature names.
    """
    original_columns_set = set(original_columns)
    return [column for column in engineered_df.columns if column not in original_columns_set]


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean

    X_raw, _, original_features = load_and_clean()
    X_engineered = add_all_features(X_raw)
    engineered_features = get_engineered_feature_names(original_features, X_engineered)
    print("Original feature shape:", X_raw.shape)
    print("Engineered feature shape:", X_engineered.shape)
    print("New engineered features:", len(engineered_features))
    print("Engineered feature names:")
    for feature in engineered_features:
        print("-", feature)
