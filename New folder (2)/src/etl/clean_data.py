"""
ETL Script: Load raw Rossmann sales data, clean it, and save a processed version.

Run:
    python src/etl/clean_data.py
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/train.csv")
PROCESSED_PATH = Path("data/processed/clean_sales.csv")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV, forcing StateHoliday to string to avoid mixed-type warnings."""
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, dtype={"StateHoliday": str}, parse_dates=["Date"])
    logger.info(f"Loaded {len(df):,} rows")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules and basic sanity checks."""
    before = len(df)

    # Normalize StateHoliday: sometimes stored as '0' (string) vs 0 (int) in source data
    df["StateHoliday"] = df["StateHoliday"].replace({"0": "None", 0: "None"}).fillna("None")

    # Drop rows where store was closed AND sold nothing extra — these are not useful for
    # sales prediction since Sales is always 0 when Open == 0 (confirmed during EDA)
    df = df[df["Open"] == 1].copy()

    # Flag remaining anomalies: open but zero sales (data entry issue, not closure)
    anomaly_mask = df["Sales"] == 0
    logger.info(f"Found {anomaly_mask.sum()} rows with Open=1 but Sales=0 (kept, flagged)")
    df["is_anomaly"] = anomaly_mask

    # Feature: date parts, useful later for BI dashboard + model
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    logger.info(f"Rows before cleaning: {before:,} | after: {len(df):,}")
    return df


def main():
    df = load_raw_data(RAW_PATH)
    df_clean = clean_data(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"Saved cleaned data to {PROCESSED_PATH} ({len(df_clean):,} rows)")


if __name__ == "__main__":
    main()
