import numpy as np
import pandas as pd
import pytest

from src.feature_engineering_relational import (
    aggregate_bureau_balance_features,
    aggregate_installments_payments_features,
    aggregate_previous_application_features,
    load_relational_features,
)


def test_previous_application_aggregation_rates():
    previous = pd.DataFrame(
        {
            "SK_ID_CURR": [100, 100, 101],
            "SK_ID_PREV": [1, 2, 3],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
            "AMT_APPLICATION": [100.0, 200.0, 400.0],
            "AMT_CREDIT": [80.0, 100.0, 400.0],
            "DAYS_DECISION": [-30, -500, -100],
        }
    )

    result = aggregate_previous_application_features(previous)
    applicant_100 = result.loc[result["SK_ID_CURR"] == 100].iloc[0]

    assert len(result) == 2
    assert applicant_100["prev_app_count"] == 2
    assert applicant_100["prev_app_approved_count"] == 1
    assert applicant_100["prev_app_refusal_rate"] == 0.5
    assert applicant_100["prev_app_recent_application_count"] == 1
    assert np.isclose(applicant_100["prev_app_credit_to_application_ratio_mean"], 0.65)


def test_installments_aggregation_late_and_underpayment_rates():
    installments = pd.DataFrame(
        {
            "SK_ID_CURR": [100, 100, 100, 101],
            "DAYS_INSTALMENT": [-10, -20, -30, -10],
            "DAYS_ENTRY_PAYMENT": [-9, -25, -28, -10],
            "AMT_INSTALMENT": [100.0, 100.0, 200.0, 50.0],
            "AMT_PAYMENT": [100.0, 80.0, 200.0, 50.0],
        }
    )

    result = aggregate_installments_payments_features(installments)
    applicant_100 = result.loc[result["SK_ID_CURR"] == 100].iloc[0]

    assert applicant_100["inst_payment_count"] == 3
    assert applicant_100["inst_late_payment_count"] == 2
    assert np.isclose(applicant_100["inst_late_payment_rate"], 2 / 3)
    assert applicant_100["inst_underpayment_count"] == 1
    assert np.isclose(applicant_100["inst_underpayment_rate"], 1 / 3)
    assert applicant_100["inst_max_payment_delay"] == 2


def test_bureau_balance_maps_to_applicant_level():
    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": [10, 10, 10, 11, 11, 12],
            "MONTHS_BALANCE": [-1, -2, -3, -1, -2, -1],
            "STATUS": ["0", "1", "C", "X", "2", "0"],
        }
    )
    bureau = pd.DataFrame(
        {
            "SK_ID_BUREAU": [10, 11, 12],
            "SK_ID_CURR": [100, 100, 101],
        }
    )

    result = aggregate_bureau_balance_features(bureau_balance, bureau)
    applicant_100 = result.loc[result["SK_ID_CURR"] == 100].iloc[0]

    assert len(result) == 2
    assert applicant_100["bureau_balance_record_count"] == 5
    assert applicant_100["bureau_balance_late_status_count"] == 2
    assert np.isclose(applicant_100["bureau_balance_late_status_rate"], 2 / 5)
    assert applicant_100["bureau_balance_status_c_count"] == 1
    assert applicant_100["bureau_balance_status_x_count"] == 1


def test_load_relational_features_returns_empty_when_optional_files_missing(tmp_path):
    config = {
        "paths": {
            "previous_application_data": str(tmp_path / "previous_application.csv"),
            "installments_payments_data": str(tmp_path / "installments_payments.csv"),
            "pos_cash_balance_data": str(tmp_path / "POS_CASH_balance.csv"),
            "credit_card_balance_data": str(tmp_path / "credit_card_balance.csv"),
            "bureau_balance_data": str(tmp_path / "bureau_balance.csv"),
            "bureau_data": str(tmp_path / "bureau.csv"),
        }
    }

    result = load_relational_features(config)

    assert result.empty
    assert list(result.columns) == ["SK_ID_CURR"]


def test_load_relational_features_required_table_raises_when_missing(tmp_path):
    config = {
        "paths": {
            "previous_application_data": str(tmp_path / "previous_application.csv"),
            "installments_payments_data": str(tmp_path / "installments_payments.csv"),
            "pos_cash_balance_data": str(tmp_path / "POS_CASH_balance.csv"),
            "credit_card_balance_data": str(tmp_path / "credit_card_balance.csv"),
            "bureau_balance_data": str(tmp_path / "bureau_balance.csv"),
            "bureau_data": str(tmp_path / "bureau.csv"),
        }
    }

    with pytest.raises(FileNotFoundError):
        load_relational_features(config, required_tables=["previous_application"])
