"""
Unit tests for the ETL and feature engineering pipeline.

Run:
    pytest tests/ -v
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.etl.clean_data import clean_data
from src.features.build_features import add_lag_features, encode_categoricals


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "Store": [1, 1, 1, 2, 2],
        "DayOfWeek": [1, 2, 3, 1, 2],
        "Date": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-01", "2015-01-02"]),
        "Sales": [1000, 0, 1200, 500, 600],
        "Customers": [100, 0, 120, 50, 60],
        "Open": [1, 0, 1, 1, 1],
        "Promo": [1, 0, 1, 0, 0],
        "StateHoliday": ["0", "a", "0", "0", "0"],
        "SchoolHoliday": [0, 0, 1, 0, 0],
    })


def test_clean_data_drops_closed_stores(sample_raw_df):
    result = clean_data(sample_raw_df)
    assert (result["Open"] == 1).all(), "clean_data should only keep rows where Open == 1"


def test_clean_data_normalizes_state_holiday(sample_raw_df):
    result = clean_data(sample_raw_df)
    assert "None" in result["StateHoliday"].values
    assert "0" not in result["StateHoliday"].values


def test_clean_data_adds_date_parts(sample_raw_df):
    result = clean_data(sample_raw_df)
    for col in ["Year", "Month", "Day", "WeekOfYear"]:
        assert col in result.columns


def test_clean_data_no_negative_sales_after_cleaning(sample_raw_df):
    result = clean_data(sample_raw_df)
    assert (result["Sales"] >= 0).all()


def test_add_lag_features_creates_expected_columns():
    df = pd.DataFrame({
        "Store": [1] * 10,
        "Date": pd.date_range("2015-01-01", periods=10),
        "Sales": list(range(100, 1100, 100)),
    })
    result = add_lag_features(df)
    assert "Sales_Lag_7" in result.columns
    assert "Sales_RollingMean_7" in result.columns
    # first 7 rows per store should have NaN lag (not enough history yet)
    assert result["Sales_Lag_7"].isna().sum() == 7


def test_encode_categoricals_one_hot_encodes_holiday():
    df = pd.DataFrame({"StateHoliday": ["None", "a", "b", "c"]})
    result = encode_categoricals(df)
    expected_cols = {"Holiday_None", "Holiday_a", "Holiday_b", "Holiday_c"}
    assert expected_cols.issubset(set(result.columns))


def test_clean_data_handles_empty_dataframe():
    empty_df = pd.DataFrame({
        "Store": [], "DayOfWeek": [], "Date": pd.to_datetime([]), "Sales": [],
        "Customers": [], "Open": [], "Promo": [], "StateHoliday": [], "SchoolHoliday": [],
    })
    result = clean_data(empty_df)
    assert len(result) == 0
