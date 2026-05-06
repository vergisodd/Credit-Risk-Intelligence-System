import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import ENGINEERED_FEATURES, add_all_features


@pytest.fixture
def feature_df():
    return pd.DataFrame(
        {
            "AMT_CREDIT": [100000.0, 0.0, 300000.0],
            "AMT_INCOME_TOTAL": [50000.0, 0.0, 150000.0],
            "AMT_ANNUITY": [10000.0, 20000.0, 0.0],
            "AMT_GOODS_PRICE": [90000.0, 100000.0, 0.0],
            "CNT_FAM_MEMBERS": [2.0, 0.0, 3.0],
            "DAYS_BIRTH": [-12000, -20000, -9000],
            "DAYS_EMPLOYED": [-2000, 365243, -1000],
            "EXT_SOURCE_1": [0.2, np.nan, 0.8],
            "EXT_SOURCE_2": [0.6, 0.4, np.nan],
            "EXT_SOURCE_3": [0.3, 0.5, 0.7],
        }
    )


def test_add_all_features_has_all_engineered_features(feature_df):
    result = add_all_features(feature_df)
    for feature in ENGINEERED_FEATURES:
        assert feature in result.columns


def test_engineered_features_do_not_produce_inf(feature_df):
    result = add_all_features(feature_df)
    for feature in ENGINEERED_FEATURES:
        assert not np.isinf(result[feature]).any()


def test_division_by_zero_returns_nan(feature_df):
    result = add_all_features(feature_df)
    assert np.isnan(result.loc[1, "CREDIT_INCOME_RATIO"])
    assert np.isnan(result.loc[1, "INCOME_PER_FAMILY_MEMBER"])


def test_ext_source_mean_uses_non_null_values(feature_df):
    result = add_all_features(feature_df)
    expected = feature_df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    pd.testing.assert_series_equal(
        result["EXT_SOURCE_MEAN"],
        expected,
        check_names=False,
    )
