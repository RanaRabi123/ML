"""
Feature engineering for the sales forecasting model.
"""

import pandas as pd


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-store lag features: sales from 7 days ago, and a 7-day rolling average."""
    df = df.sort_values(["Store", "Date"]).copy()
    df["Sales_Lag_7"] = df.groupby("Store")["Sales"].shift(7)
    df["Sales_RollingMean_7"] = (
        df.groupby("Store")["Sales"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode StateHoliday; DayOfWeek/Month/Promo/SchoolHoliday are already numeric."""
    df = pd.get_dummies(df, columns=["StateHoliday"], prefix="Holiday")
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    df = encode_categoricals(df)
    # Drop rows where lag features are NaN (first 7 days of each store's history)
    df = df.dropna(subset=["Sales_Lag_7", "Sales_RollingMean_7"])
    return df


FEATURE_COLUMNS = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday",
    "Year", "Month", "Day", "WeekOfYear",
    "Sales_Lag_7", "Sales_RollingMean_7",
]
TARGET_COLUMN = "Sales"
