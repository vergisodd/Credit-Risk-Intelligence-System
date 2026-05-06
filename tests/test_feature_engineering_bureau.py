import numpy as np
import pandas as pd
import pytest

from src.feature_engineering_bureau import (
    BUREAU_FEATURES,
    aggregate_bureau_features,
    load_bureau_features,
)


@pytest.fixture
def synthetic_bureau_df():
    n = 50
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.repeat([100, 101, 102, 103, 104], 10),
            "SK_ID_BUREAU": np.arange(n),
            "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], n),
            "DAYS_CREDIT": rng.integers(-2000, -100, n),
            "DAYS_CREDIT_ENDDATE": rng.integers(-500, 500, n, dtype=int).astype(float),
            "CREDIT_DAY_OVERDUE": rng.integers(0, 30, n),
            "AMT_CREDIT_SUM": rng.uniform(10000, 200000, n),
            "AMT_CREDIT_SUM_DEBT": rng.uniform(0, 100000, n),
            "AMT_CREDIT_SUM_OVERDUE": rng.uniform(0, 5000, n),
            "CNT_CREDIT_PROLONG": rng.integers(0, 3, n),
        }
    )


def test_load_bureau_features_raises_when_missing(tmp_path):
    missing_path = tmp_path / "bureau.csv"
    with pytest.raises(FileNotFoundError):
        load_bureau_features(missing_path)


def test_aggregate_bureau_features_has_expected_columns(synthetic_bureau_df):
    result = aggregate_bureau_features(synthetic_bureau_df)
    for feature in BUREAU_FEATURES:
        assert feature in result.columns


def test_active_debt_ratio_is_nan_when_credit_sum_zero(synthetic_bureau_df):
    df = synthetic_bureau_df.copy()
    df.loc[df["SK_ID_CURR"] == 100, "AMT_CREDIT_SUM"] = 0
    result = aggregate_bureau_features(df)
    ratio = result.loc[result["SK_ID_CURR"] == 100, "BUREAU_ACTIVE_DEBT_RATIO"].iloc[0]
    assert np.isnan(ratio)


def test_bureau_features_do_not_contain_inf(synthetic_bureau_df):
    result = aggregate_bureau_features(synthetic_bureau_df)
    assert not np.isinf(result[BUREAU_FEATURES]).any().any()


def test_output_has_one_row_per_applicant(synthetic_bureau_df):
    result = aggregate_bureau_features(synthetic_bureau_df)
    assert result["SK_ID_CURR"].nunique() == synthetic_bureau_df["SK_ID_CURR"].nunique()
    assert len(result) == synthetic_bureau_df["SK_ID_CURR"].nunique()
