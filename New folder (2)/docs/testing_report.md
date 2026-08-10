# Testing Report

## Summary

| Metric | Value |
|---|---|
| Total tests | 13 |
| Passed | 13 |
| Failed | 0 |
| Test framework | pytest |
| Coverage areas | ETL, feature engineering, API auth, API prediction |

Full run output is in `docs/test_run_output.txt`. Reproduce locally with:
```bash
pytest tests/ -v
```

## Test Breakdown

### ETL Tests (`tests/test_etl_and_features.py`)

| Test | What it verifies | Result |
|---|---|---|
| `test_clean_data_drops_closed_stores` | Closed-store rows (Open=0) are removed | PASSED |
| `test_clean_data_normalizes_state_holiday` | Mixed-type holiday values are unified to `"None"`/`a`/`b`/`c` | PASSED |
| `test_clean_data_adds_date_parts` | Year/Month/Day/WeekOfYear columns are generated | PASSED |
| `test_clean_data_no_negative_sales_after_cleaning` | No negative sales values slip through | PASSED |
| `test_add_lag_features_creates_expected_columns` | 7-day lag and rolling mean columns are created correctly, with expected NaNs on the first 7 days per store | PASSED |
| `test_encode_categoricals_one_hot_encodes_holiday` | One-hot encoding produces the expected holiday columns | PASSED |
| `test_clean_data_handles_empty_dataframe` | Pipeline doesn't crash on an empty input (edge case) | PASSED |

### API Tests (`tests/test_api.py`)

| Test | What it verifies | Result |
|---|---|---|
| `test_health_check` | `/health` responds 200 with status "ok" | PASSED |
| `test_login_success` | Valid credentials return a JWT access token | PASSED |
| `test_login_failure_wrong_password` | Wrong password is rejected with 401 | PASSED |
| `test_predict_requires_auth` | Prediction endpoint rejects requests with no token (401) | PASSED |
| `test_predict_with_valid_token` | A valid token allows a prediction, and the response is a positive sales number | PASSED |
| `test_predict_rejects_invalid_token` | A malformed/fake token is rejected (401) | PASSED |

## Manual / Integration Testing

In addition to automated tests, the full system was manually verified end-to-end:

1. Started the FastAPI server and confirmed `/health` responds.
2. Logged in via `/auth/login`, confirmed a JWT token is returned.
3. Called `/predict/sales` with the token — received a valid prediction (see screenshot).
4. Called `/predict/sales` with **no** token — confirmed 401 rejection (see screenshot/log).
5. Started the Streamlit dashboard, logged in through the UI, confirmed all four
   BI charts render with real aggregated data (see `docs/screenshot_dashboard.png`).
6. Used the "Predict" page in the dashboard to call the live API and received a
   correct prediction end-to-end through the UI (see `docs/screenshot_predict.png`).

## What Is Not Yet Tested

- **Load/performance testing** — no test yet simulates concurrent users hitting
  the API (would use a tool like Locust or k6 in a production project).
- **UI automated testing** — dashboard was tested manually and via screenshot
  capture, not with an automated UI test suite (e.g. Selenium).
- **Model drift / retraining tests** — no test currently verifies model
  performance stays within an acceptable range as new data arrives.

These gaps are also listed in `docs/known_limitations.md`.
