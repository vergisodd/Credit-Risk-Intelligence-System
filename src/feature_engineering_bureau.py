"""
Bureau feature engineering for credit risk modelling.

Aggregates bureau.csv credit history records per applicant (SK_ID_CURR).
Produces one row per applicant for left-joining to application_train features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import load_config, resolve_path


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

BUREAU_FEATURES = [
    "BUREAU_LOAN_COUNT",
    "BUREAU_ACTIVE_LOAN_COUNT",
    "BUREAU_CLOSED_LOAN_COUNT",
    "BUREAU_AVG_DAYS_CREDIT",
    "BUREAU_AVG_DAYS_CREDIT_ENDDATE",
    "BUREAU_MAX_DAYS_OVERDUE",
    "BUREAU_MEAN_DAYS_OVERDUE",
    "BUREAU_SUM_AMT_CREDIT_SUM",
    "BUREAU_SUM_AMT_CREDIT_SUM_DEBT",
    "BUREAU_SUM_AMT_CREDIT_SUM_OVERDUE",
    "BUREAU_ACTIVE_DEBT_RATIO",
    "BUREAU_PROLONGED_LOAN_COUNT",
    "BUREAU_CREDIT_ACTIVE_RATIO",
]


def load_bureau(file_path: str | Path | None = None) -> pd.DataFrame:
    """Load bureau.csv from configured path."""
    config = load_config()
    path = resolve_path(file_path or config["paths"]["bureau_data"])
    if not path.exists():
        raise FileNotFoundError(
            f"bureau.csv not found at {path}. "
            "Download from Kaggle Home Credit competition before running bureau features."
        )
    LOGGER.info("Loading bureau data from %s", path)
    df = pd.read_csv(path)
    LOGGER.info("Bureau data shape: %s", df.shape)
    return df


def aggregate_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bureau records to one row per SK_ID_CURR.

    All features are safe for use in supervised ML: they are computed from
    historical credit records and do not use the target variable.
    """
    grp = bureau.groupby("SK_ID_CURR")
    agg = pd.DataFrame(index=pd.Index(bureau["SK_ID_CURR"].unique(), name="SK_ID_CURR"))

    agg["BUREAU_LOAN_COUNT"] = grp["SK_ID_BUREAU"].count()
    agg["BUREAU_ACTIVE_LOAN_COUNT"] = (
        bureau[bureau["CREDIT_ACTIVE"] == "Active"]
        .groupby("SK_ID_CURR")["SK_ID_BUREAU"]
        .count()
    )
    agg["BUREAU_CLOSED_LOAN_COUNT"] = (
        bureau[bureau["CREDIT_ACTIVE"] == "Closed"]
        .groupby("SK_ID_CURR")["SK_ID_BUREAU"]
        .count()
    )
    agg["BUREAU_PROLONGED_LOAN_COUNT"] = (
        bureau[bureau["CNT_CREDIT_PROLONG"] > 0]
        .groupby("SK_ID_CURR")["SK_ID_BUREAU"]
        .count()
    )

    count_columns = [
        "BUREAU_ACTIVE_LOAN_COUNT",
        "BUREAU_CLOSED_LOAN_COUNT",
        "BUREAU_PROLONGED_LOAN_COUNT",
    ]
    agg[count_columns] = agg[count_columns].fillna(0)

    agg["BUREAU_AVG_DAYS_CREDIT"] = grp["DAYS_CREDIT"].mean()
    agg["BUREAU_AVG_DAYS_CREDIT_ENDDATE"] = grp["DAYS_CREDIT_ENDDATE"].mean()

    agg["BUREAU_MAX_DAYS_OVERDUE"] = grp["CREDIT_DAY_OVERDUE"].max()
    agg["BUREAU_MEAN_DAYS_OVERDUE"] = grp["CREDIT_DAY_OVERDUE"].mean()

    agg["BUREAU_SUM_AMT_CREDIT_SUM"] = grp["AMT_CREDIT_SUM"].sum()
    agg["BUREAU_SUM_AMT_CREDIT_SUM_DEBT"] = grp["AMT_CREDIT_SUM_DEBT"].sum()
    agg["BUREAU_SUM_AMT_CREDIT_SUM_OVERDUE"] = grp["AMT_CREDIT_SUM_OVERDUE"].sum()

    agg["BUREAU_ACTIVE_DEBT_RATIO"] = np.where(
        agg["BUREAU_SUM_AMT_CREDIT_SUM"] != 0,
        agg["BUREAU_SUM_AMT_CREDIT_SUM_DEBT"] / agg["BUREAU_SUM_AMT_CREDIT_SUM"],
        np.nan,
    )

    agg["BUREAU_CREDIT_ACTIVE_RATIO"] = np.where(
        agg["BUREAU_LOAN_COUNT"] != 0,
        agg["BUREAU_ACTIVE_LOAN_COUNT"] / agg["BUREAU_LOAN_COUNT"],
        np.nan,
    )

    non_ratio_columns = [
        column for column in BUREAU_FEATURES
        if column not in {"BUREAU_ACTIVE_DEBT_RATIO", "BUREAU_CREDIT_ACTIVE_RATIO"}
    ]
    agg[non_ratio_columns] = agg[non_ratio_columns].fillna(0)
    agg = agg.reset_index()
    LOGGER.info("Bureau aggregation shape: %s", agg.shape)
    return agg


def load_bureau_features(file_path: str | Path | None = None) -> pd.DataFrame:
    """Public interface: load and aggregate bureau features."""
    bureau = load_bureau(file_path)
    return aggregate_bureau_features(bureau)
