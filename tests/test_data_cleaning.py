import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import load_and_clean, validate_data


@pytest.fixture
def synthetic_application_df():
    rows = 200
    target = np.array([1] * 16 + [0] * (rows - 16))
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + rows),
            "TARGET": target,
            "AMT_INCOME_TOTAL": np.linspace(50000, 250000, rows),
            "AMT_CREDIT": np.linspace(100000, 700000, rows),
            "AMT_ANNUITY": np.linspace(5000, 40000, rows),
            "AMT_GOODS_PRICE": np.linspace(90000, 650000, rows),
            "DAYS_BIRTH": np.linspace(-25000, -7000, rows),
            "DAYS_EMPLOYED": np.linspace(-8000, -100, rows),
            "CNT_FAM_MEMBERS": np.tile([1, 2, 3, 4], rows // 4),
            "EXT_SOURCE_1": np.linspace(0.1, 0.9, rows),
            "EXT_SOURCE_2": np.linspace(0.2, 0.8, rows),
            "EXT_SOURCE_3": np.linspace(0.3, 0.7, rows),
            "CODE_GENDER": np.tile(["M", "F"], rows // 2),
            "NAME_EDUCATION_TYPE": np.tile(
                [
                    "Secondary / secondary special",
                    "Higher education",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree",
                ],
                rows // 5,
            ),
            "MOSTLY_MISSING": [np.nan] * 151 + list(range(49)),
        }
    )


def test_validate_data_raises_when_target_missing(synthetic_application_df):
    with pytest.raises(AssertionError, match="TARGET"):
        validate_data(synthetic_application_df.drop(columns=["TARGET"]), strict_shape=False)


def test_validate_data_raises_when_duplicate_ids_exist(synthetic_application_df):
    df = synthetic_application_df.copy()
    df.loc[1, "SK_ID_CURR"] = df.loc[0, "SK_ID_CURR"]
    with pytest.raises(AssertionError, match="duplicate"):
        validate_data(df, strict_shape=False)


def test_load_and_clean_returns_tuple_length_three(tmp_path, synthetic_application_df):
    csv_path = tmp_path / "application_train.csv"
    synthetic_application_df.to_csv(csv_path, index=False)
    result = load_and_clean(csv_path, validate_schema=False)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_load_and_clean_has_no_post_clean_high_missing_columns(tmp_path, synthetic_application_df):
    csv_path = tmp_path / "application_train.csv"
    synthetic_application_df.to_csv(csv_path, index=False)
    X, _, _ = load_and_clean(csv_path, validate_schema=False)
    assert not (X.isna().mean() > 0.50).any()


def test_load_and_clean_target_binary_and_expected_default_rate(tmp_path, synthetic_application_df):
    csv_path = tmp_path / "application_train.csv"
    synthetic_application_df.to_csv(csv_path, index=False)
    _, y, _ = load_and_clean(csv_path, validate_schema=False)
    assert set(y.unique()) <= {0, 1}
    assert 0.07 <= y.mean() <= 0.10
