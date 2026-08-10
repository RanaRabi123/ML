"""
Train a sales forecasting model on the cleaned Rossmann data.

Run:
    python src/models/train_model.py
"""

import logging
import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features.build_features import build_feature_set, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed/clean_sales.csv")
MODEL_PATH = Path("data/processed/sales_model.joblib")
METRICS_PATH = Path("docs/model_metrics.json")


def main():
    logger.info("Loading cleaned data...")
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["Date"])

    logger.info("Building features (lag features, encoding)...")
    df = build_feature_set(df)

    feature_cols = [c for c in df.columns if c.startswith("Holiday_")] + [
        "Store", "DayOfWeek", "Promo", "SchoolHoliday",
        "Year", "Month", "Day", "WeekOfYear",
        "Sales_Lag_7", "Sales_RollingMean_7",
    ]

    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    # Time-aware split: train on earlier data, test on most recent 15% (mimics real forecasting)
    df_sorted = df.sort_values("Date")
    split_idx = int(len(df_sorted) * 0.85)
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    logger.info(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    mape = (abs((y_test - preds) / y_test.replace(0, 1))).mean() * 100

    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE_percent": round(mape, 2),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": feature_cols,
    }
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
