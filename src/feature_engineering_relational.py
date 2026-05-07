"""Optional relational feature engineering for Home Credit auxiliary tables."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config_loader import load_config, resolve_path


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

RELATIONAL_TABLE_PATH_KEYS = {
    "previous_application": "previous_application_data",
    "installments_payments": "installments_payments_data",
    "pos_cash_balance": "pos_cash_balance_data",
    "credit_card_balance": "credit_card_balance_data",
    "bureau_balance": "bureau_balance_data",
}

RELATIONAL_ZERO_FILL_KEYWORDS = (
    "_count",
    "_rate",
    "_total_",
    "_status_",
    "_record_",
    "_contract_",
    "_approved_",
    "_refused_",
    "_active_",
    "_completed_",
    "_underpayment_",
    "_late_",
)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while preserving NaN for zero denominators."""
    return pd.Series(
        np.where(denominator != 0, numerator / denominator, np.nan),
        index=numerator.index,
    ).replace([np.inf, -np.inf], np.nan)


def _require_columns(df: pd.DataFrame, columns: set[str], table_name: str) -> None:
    """Raise a clear error when a source table is missing expected columns."""
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {', '.join(missing)}")


def _read_configured_csv(config: dict, path_key: str, table_name: str) -> pd.DataFrame:
    """Read a configured CSV with a project-specific missing-file message."""
    path = resolve_path(config["paths"][path_key])
    if not path.exists():
        raise FileNotFoundError(
            f"{table_name}.csv not found at {path}. "
            "Download the Home Credit relational tables into data/raw/ before running this step."
        )
    LOGGER.info("Loading %s from %s", table_name, path)
    return pd.read_csv(path)


def _count_by_condition(
    df: pd.DataFrame,
    group_column: str,
    condition: pd.Series,
    value_column: str,
) -> pd.Series:
    """Count grouped records that satisfy a boolean condition."""
    return df.loc[condition].groupby(group_column)[value_column].count()


def aggregate_previous_application_features(previous_application: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous_application.csv to one row per applicant."""
    table_name = "previous_application"
    _require_columns(previous_application, {"SK_ID_CURR"}, table_name)

    df = previous_application.copy()
    group = df.groupby("SK_ID_CURR")
    agg = pd.DataFrame(index=pd.Index(df["SK_ID_CURR"].unique(), name="SK_ID_CURR"))
    count_column = "SK_ID_PREV" if "SK_ID_PREV" in df.columns else "SK_ID_CURR"

    agg["prev_app_count"] = group[count_column].count()
    if "NAME_CONTRACT_STATUS" in df.columns:
        status = df["NAME_CONTRACT_STATUS"].astype(str)
        agg["prev_app_approved_count"] = _count_by_condition(
            df, "SK_ID_CURR", status.eq("Approved"), count_column
        )
        agg["prev_app_refused_count"] = _count_by_condition(
            df, "SK_ID_CURR", status.eq("Refused"), count_column
        )
    else:
        agg["prev_app_approved_count"] = 0
        agg["prev_app_refused_count"] = 0

    agg[["prev_app_approved_count", "prev_app_refused_count"]] = agg[
        ["prev_app_approved_count", "prev_app_refused_count"]
    ].fillna(0)
    agg["prev_app_approval_rate"] = _safe_divide(
        agg["prev_app_approved_count"], agg["prev_app_count"]
    )
    agg["prev_app_refusal_rate"] = _safe_divide(
        agg["prev_app_refused_count"], agg["prev_app_count"]
    )

    if "AMT_APPLICATION" in df.columns:
        agg["prev_app_mean_amt_application"] = group["AMT_APPLICATION"].mean()
    if "AMT_CREDIT" in df.columns:
        agg["prev_app_mean_amt_credit"] = group["AMT_CREDIT"].mean()
    if {"AMT_CREDIT", "AMT_APPLICATION"}.issubset(df.columns):
        df["prev_app_credit_to_application_ratio"] = _safe_divide(
            df["AMT_CREDIT"], df["AMT_APPLICATION"]
        )
        agg["prev_app_credit_to_application_ratio_mean"] = group[
            "prev_app_credit_to_application_ratio"
        ].mean()
    if "DAYS_DECISION" in df.columns:
        agg["prev_app_mean_days_decision"] = group["DAYS_DECISION"].mean()
        recent = df["DAYS_DECISION"] >= -365
        agg["prev_app_recent_application_count"] = _count_by_condition(
            df, "SK_ID_CURR", recent, count_column
        ).fillna(0)

    return agg.reset_index().replace([np.inf, -np.inf], np.nan)


def load_previous_application_features(config: dict | None = None) -> pd.DataFrame:
    """Load and aggregate previous_application.csv."""
    config = config or load_config()
    df = _read_configured_csv(config, "previous_application_data", "previous_application")
    return aggregate_previous_application_features(df)


def aggregate_installments_payments_features(installments_payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments.csv to one row per applicant."""
    table_name = "installments_payments"
    _require_columns(installments_payments, {"SK_ID_CURR"}, table_name)

    df = installments_payments.copy()
    group = df.groupby("SK_ID_CURR")
    agg = pd.DataFrame(index=pd.Index(df["SK_ID_CURR"].unique(), name="SK_ID_CURR"))

    agg["inst_payment_count"] = group["SK_ID_CURR"].count()
    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(df.columns):
        delay = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
        df["inst_positive_payment_delay"] = delay.clip(lower=0)
        df["inst_is_late_payment"] = delay > 0
        agg["inst_late_payment_count"] = group["inst_is_late_payment"].sum()
        agg["inst_late_payment_rate"] = _safe_divide(
            agg["inst_late_payment_count"], agg["inst_payment_count"]
        )
        agg["inst_mean_payment_delay"] = group["inst_positive_payment_delay"].mean()
        agg["inst_max_payment_delay"] = group["inst_positive_payment_delay"].max()

    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(df.columns):
        df["inst_is_underpayment"] = df["AMT_PAYMENT"] < df["AMT_INSTALMENT"]
        df["inst_payment_ratio"] = _safe_divide(df["AMT_PAYMENT"], df["AMT_INSTALMENT"])
        agg["inst_underpayment_count"] = group["inst_is_underpayment"].sum()
        agg["inst_underpayment_rate"] = _safe_divide(
            agg["inst_underpayment_count"], agg["inst_payment_count"]
        )
        agg["inst_payment_ratio_mean"] = group["inst_payment_ratio"].mean()
        agg["inst_payment_ratio_min"] = group["inst_payment_ratio"].min()
        agg["inst_total_payment_amount"] = group["AMT_PAYMENT"].sum()

    return agg.reset_index().replace([np.inf, -np.inf], np.nan)


def load_installments_payments_features(config: dict | None = None) -> pd.DataFrame:
    """Load and aggregate installments_payments.csv."""
    config = config or load_config()
    df = _read_configured_csv(config, "installments_payments_data", "installments_payments")
    return aggregate_installments_payments_features(df)


def aggregate_pos_cash_balance_features(pos_cash_balance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance.csv to one row per applicant."""
    table_name = "POS_CASH_balance"
    _require_columns(pos_cash_balance, {"SK_ID_CURR"}, table_name)

    df = pos_cash_balance.copy()
    group = df.groupby("SK_ID_CURR")
    agg = pd.DataFrame(index=pd.Index(df["SK_ID_CURR"].unique(), name="SK_ID_CURR"))

    agg["pos_record_count"] = group["SK_ID_CURR"].count()
    if "SK_ID_PREV" in df.columns:
        agg["pos_contract_count"] = group["SK_ID_PREV"].nunique()
    if "SK_DPD" in df.columns:
        agg["pos_dpd_mean"] = group["SK_DPD"].mean()
        agg["pos_dpd_max"] = group["SK_DPD"].max()
    if "SK_DPD_DEF" in df.columns:
        agg["pos_dpd_def_mean"] = group["SK_DPD_DEF"].mean()
        agg["pos_dpd_def_max"] = group["SK_DPD_DEF"].max()
    if "NAME_CONTRACT_STATUS" in df.columns:
        status = df["NAME_CONTRACT_STATUS"].astype(str)
        agg["pos_active_count"] = _count_by_condition(
            df, "SK_ID_CURR", status.eq("Active"), "SK_ID_CURR"
        ).fillna(0)
        agg["pos_completed_count"] = _count_by_condition(
            df, "SK_ID_CURR", status.eq("Completed"), "SK_ID_CURR"
        ).fillna(0)

    return agg.reset_index().replace([np.inf, -np.inf], np.nan)


def load_pos_cash_balance_features(config: dict | None = None) -> pd.DataFrame:
    """Load and aggregate POS_CASH_balance.csv."""
    config = config or load_config()
    df = _read_configured_csv(config, "pos_cash_balance_data", "POS_CASH_balance")
    return aggregate_pos_cash_balance_features(df)


def aggregate_credit_card_balance_features(credit_card_balance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance.csv to one row per applicant."""
    table_name = "credit_card_balance"
    _require_columns(credit_card_balance, {"SK_ID_CURR"}, table_name)

    df = credit_card_balance.copy()
    group = df.groupby("SK_ID_CURR")
    agg = pd.DataFrame(index=pd.Index(df["SK_ID_CURR"].unique(), name="SK_ID_CURR"))

    agg["cc_record_count"] = group["SK_ID_CURR"].count()
    if "AMT_BALANCE" in df.columns:
        agg["cc_mean_balance"] = group["AMT_BALANCE"].mean()
        agg["cc_max_balance"] = group["AMT_BALANCE"].max()
    if "AMT_CREDIT_LIMIT_ACTUAL" in df.columns:
        agg["cc_mean_credit_limit"] = group["AMT_CREDIT_LIMIT_ACTUAL"].mean()
    if {"AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(df.columns):
        df["cc_utilization"] = _safe_divide(df["AMT_BALANCE"], df["AMT_CREDIT_LIMIT_ACTUAL"])
        agg["cc_utilization_mean"] = group["cc_utilization"].mean()
        agg["cc_utilization_max"] = group["cc_utilization"].max()
    if "SK_DPD" in df.columns:
        agg["cc_dpd_mean"] = group["SK_DPD"].mean()
        agg["cc_dpd_max"] = group["SK_DPD"].max()
    if "AMT_DRAWINGS_ATM_CURRENT" in df.columns:
        agg["cc_atm_drawings_mean"] = group["AMT_DRAWINGS_ATM_CURRENT"].mean()
    if {"AMT_PAYMENT_CURRENT", "AMT_INST_MIN_REGULARITY"}.issubset(df.columns):
        df["cc_payment_ratio"] = _safe_divide(
            df["AMT_PAYMENT_CURRENT"], df["AMT_INST_MIN_REGULARITY"]
        )
        agg["cc_payment_ratio_mean"] = group["cc_payment_ratio"].mean()

    return agg.reset_index().replace([np.inf, -np.inf], np.nan)


def load_credit_card_balance_features(config: dict | None = None) -> pd.DataFrame:
    """Load and aggregate credit_card_balance.csv."""
    config = config or load_config()
    df = _read_configured_csv(config, "credit_card_balance_data", "credit_card_balance")
    return aggregate_credit_card_balance_features(df)


def aggregate_bureau_balance_features(
    bureau_balance: pd.DataFrame,
    bureau: pd.DataFrame,
) -> pd.DataFrame:
    """Map bureau_balance.csv through bureau.csv and aggregate to applicant level."""
    table_name = "bureau_balance"
    _require_columns(bureau_balance, {"SK_ID_BUREAU", "STATUS"}, table_name)
    _require_columns(bureau, {"SK_ID_BUREAU", "SK_ID_CURR"}, "bureau")

    bb = bureau_balance.copy()
    bb["STATUS"] = bb["STATUS"].astype(str)
    bb["bureau_balance_is_late_status"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"])
    bb["bureau_balance_is_status_c"] = bb["STATUS"].eq("C")
    bb["bureau_balance_is_status_x"] = bb["STATUS"].eq("X")

    grouped = bb.groupby("SK_ID_BUREAU")
    bureau_level = pd.DataFrame(index=pd.Index(bb["SK_ID_BUREAU"].unique(), name="SK_ID_BUREAU"))
    bureau_level["bureau_balance_record_count"] = grouped["STATUS"].count()
    if "MONTHS_BALANCE" in bb.columns:
        bureau_level["bureau_balance_months_observed"] = grouped["MONTHS_BALANCE"].nunique()
    else:
        bureau_level["bureau_balance_months_observed"] = bureau_level["bureau_balance_record_count"]
    bureau_level["bureau_balance_late_status_count"] = grouped[
        "bureau_balance_is_late_status"
    ].sum()
    bureau_level["bureau_balance_status_c_count"] = grouped["bureau_balance_is_status_c"].sum()
    bureau_level["bureau_balance_status_x_count"] = grouped["bureau_balance_is_status_x"].sum()
    bureau_level = bureau_level.reset_index()

    mapping = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates("SK_ID_BUREAU")
    mapped = bureau_level.merge(mapping, on="SK_ID_BUREAU", how="inner")
    group = mapped.groupby("SK_ID_CURR")

    agg = pd.DataFrame(index=pd.Index(mapped["SK_ID_CURR"].unique(), name="SK_ID_CURR"))
    agg["bureau_balance_record_count"] = group["bureau_balance_record_count"].sum()
    agg["bureau_balance_months_observed_mean"] = group["bureau_balance_months_observed"].mean()
    agg["bureau_balance_late_status_count"] = group["bureau_balance_late_status_count"].sum()
    agg["bureau_balance_status_c_count"] = group["bureau_balance_status_c_count"].sum()
    agg["bureau_balance_status_x_count"] = group["bureau_balance_status_x_count"].sum()
    agg["bureau_balance_late_status_rate"] = _safe_divide(
        agg["bureau_balance_late_status_count"], agg["bureau_balance_record_count"]
    )

    return agg.reset_index().replace([np.inf, -np.inf], np.nan)


def load_bureau_balance_features(config: dict | None = None) -> pd.DataFrame:
    """Load and aggregate bureau_balance.csv using bureau.csv for applicant mapping."""
    config = config or load_config()
    bureau_balance = _read_configured_csv(config, "bureau_balance_data", "bureau_balance")
    bureau_path = resolve_path(config["paths"]["bureau_data"])
    if not bureau_path.exists():
        raise FileNotFoundError(
            f"bureau.csv not found at {bureau_path}. "
            "bureau_balance.csv must be joined through bureau.csv to map SK_ID_BUREAU to SK_ID_CURR."
        )
    bureau = pd.read_csv(bureau_path, usecols=["SK_ID_BUREAU", "SK_ID_CURR"])
    return aggregate_bureau_balance_features(bureau_balance, bureau)


RELATIONAL_LOADERS: dict[str, Callable[[dict | None], pd.DataFrame]] = {
    "previous_application": load_previous_application_features,
    "installments_payments": load_installments_payments_features,
    "pos_cash_balance": load_pos_cash_balance_features,
    "credit_card_balance": load_credit_card_balance_features,
    "bureau_balance": load_bureau_balance_features,
}


def _columns_to_zero_fill(columns: list[str]) -> list[str]:
    """Return aggregate columns where missing after joins means no observed records."""
    return [
        column
        for column in columns
        if column != "SK_ID_CURR"
        and any(keyword in column for keyword in RELATIONAL_ZERO_FILL_KEYWORDS)
    ]


def load_relational_features(
    config: dict,
    required_tables: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load available optional relational features and outer-join by applicant.

    Parameters
    ----------
    config:
        Parsed project configuration.
    required_tables:
        Optional list of table names that must be available. Supported names are
        previous_application, installments_payments, pos_cash_balance,
        credit_card_balance, and bureau_balance.
    """
    required = set(required_tables or [])
    unknown = sorted(required - set(RELATIONAL_LOADERS))
    if unknown:
        raise ValueError(f"Unknown relational table names: {', '.join(unknown)}")

    feature_frames = []
    missing_optional = []
    for table_name, loader in RELATIONAL_LOADERS.items():
        try:
            features = loader(config)
        except FileNotFoundError as error:
            if table_name in required:
                raise
            missing_optional.append(f"{table_name} ({error})")
            continue
        if features.empty:
            LOGGER.warning("%s produced no relational feature rows.", table_name)
            continue
        feature_frames.append(features)
        LOGGER.info("Loaded %s features with shape %s", table_name, features.shape)

    if missing_optional:
        LOGGER.info("Skipped missing optional relational tables: %s", "; ".join(missing_optional))

    if not feature_frames:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    relational = feature_frames[0]
    for frame in feature_frames[1:]:
        relational = relational.merge(frame, on="SK_ID_CURR", how="outer")

    zero_fill_columns = _columns_to_zero_fill(relational.columns.tolist())
    relational[zero_fill_columns] = relational[zero_fill_columns].fillna(0)
    LOGGER.info("Combined relational feature shape: %s", relational.shape)
    return relational.replace([np.inf, -np.inf], np.nan)
