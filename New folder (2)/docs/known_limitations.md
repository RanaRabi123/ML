# Known Limitations

Being upfront about weaknesses is part of a professional engineering report.
Here is what this system does *not* do well yet, and what a production version
would need.

## Data Limitations

1. **`StateHoliday` had mixed types in the raw file** (string `"0"` and integer `0`
   both appeared). This was caught during EDA and normalized during cleaning —
   but it's a reminder that the source data isn't perfectly clean, and any
   pipeline consuming it needs defensive type handling.
2. **54 rows had `Open = 1` but `Sales = 0`.** These are likely data entry errors
   (a store marked open with no recorded sales) rather than genuine zero-sales
   days. They were kept and flagged (`is_anomaly` column) rather than silently
   dropped, since deleting data without justification is itself a risk.
3. **No store metadata** (store type, assortment, competition distance) was
   available in this dataset — only `train.csv` was provided. A real Rossmann
   BI system would join in `store.csv` for far stronger predictions (store type
   and competition distance are typically among the most predictive features
   in this dataset). This is the single biggest lever for improving model
   accuracy from here.

## Model Limitations

1. **MAPE of ~29.8%** means predictions can be off by a meaningful margin on
   individual days — acceptable for a first baseline, not yet accurate enough
   for financial planning. Next steps: add store metadata, tune hyperparameters,
   try ensembling multiple models.
2. **No confidence intervals.** The model returns a single point estimate. A
   production system should return a prediction range, not just one number.
3. **Trained once, not retrained automatically.** In production this would need
   a scheduled retraining pipeline (e.g. weekly), since sales patterns drift
   over time (seasonality, new competitors, economic conditions).

## Security & Auth Limitations (by design, for a capstone demo)

1. **Single hardcoded demo user** (`analyst` / `changeme123`) instead of a real
   user database. Production would use a proper Users table with per-user
   roles/permissions.
2. **JWT secret defaults to a dev value** if the `JWT_SECRET_KEY` environment
   variable isn't set. This is intentional for easy local demoing, but a
   production deployment must set a strong secret via a secrets manager
   (AWS Secrets Manager, HashiCorp Vault, etc.) and never fall back silently.
3. **No rate limiting** on the login endpoint, which in production would be
   vulnerable to brute-force password guessing. A production version would
   add rate limiting (e.g. via `slowapi` or an API gateway).
4. **No HTTPS in local dev.** In production this must sit behind TLS (e.g. via
   a reverse proxy like Nginx or a managed load balancer).

## Scalability Limitations (current form)

1. **Single-process API and dashboard** — fine for a demo/single-analyst use,
   not for concurrent load. See the "scaling to 1,000,000 users" discussion in
   the project report for how this would be redesigned.
2. **Data read from CSV, not a database.** Fine at ~850K rows for a demo;
   would need to move to a proper database/warehouse at real scale.
3. **Model loaded into memory on first request** — a cold-start delay on the
   first prediction after the API restarts.

## Testing Limitations

1. Tests cover ETL, feature engineering, and API auth/prediction logic (13
   tests, all passing — see `docs/testing_report.md`), but do not yet include
   load testing or tests for the Streamlit UI itself (UI testing frameworks
   like Selenium/Playwright could be added for that).
