# Rossmann Sales Intelligence & Predictive Analytics System

An end-to-end Business Intelligence & Predictive Analytics system built on the
Rossmann Store Sales dataset (1,017,209 daily sales records across 1,115 stores,
Jan 2013 – Jul 2015).

The system has two halves:
1. **Business Intelligence** — an interactive Streamlit dashboard showing sales
   trends, day-of-week patterns, promo impact, and top-performing stores.
2. **Predictive Analytics** — an XGBoost regression model, served through a
   secured (JWT-authenticated) FastAPI backend, that forecasts a store's daily sales.

## Architecture

See `docs/architecture_diagram.png`. In short:

```
Raw CSV → ETL (clean_data.py) → Cleaned CSV
                                    ├── Feature Engineering → Model Training → Trained Model
                                    │                                              │
                                    └── Streamlit Dashboard  <──── REST/JWT ──── FastAPI
                                              ▲
                                          Analyst (login)
```

## Project Structure

```
rossmann_project/
├── data/
│   ├── raw/train.csv              # original dataset
│   └── processed/                 # cleaned data + trained model (generated)
├── src/
│   ├── etl/clean_data.py          # data cleaning pipeline
│   ├── features/build_features.py # feature engineering (lags, encoding)
│   ├── models/train_model.py      # model training + evaluation
│   └── api/main.py                # FastAPI service (auth + prediction)
├── dashboard/
│   ├── app.py                     # main Streamlit dashboard (login + BI)
│   └── pages/1_Predict.py         # prediction page (calls the API)
├── tests/                         # pytest unit + integration tests
├── docs/                          # architecture diagram, screenshots, reports
└── deployment/                    # Docker setup
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the data pipeline and train the model
```bash
python src/etl/clean_data.py
python src/models/train_model.py
```

### 3. Start the API (in one terminal)
```bash
uvicorn src.api.main:app --reload --port 8000
```
API docs available at `http://127.0.0.1:8000/docs`.

### 4. Start the dashboard (in another terminal)
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501`. Demo login: `analyst` / `changeme123`.

### 5. Run tests
```bash
pytest tests/ -v
```

## Model Performance

See `docs/model_metrics.json`. Current baseline model:
- **R²**: 0.70
- **MAE**: ~€1,266
- **MAPE**: ~29.8%

This is a reasonable first model. See `docs/known_limitations.md` for how it
could be improved.

## Security Notes

- API authentication uses JWT (JSON Web Tokens) with bcrypt-hashed passwords.
- The `/predict/sales` endpoint requires a valid token; `/health` does not
  (standard practice — load balancers/monitoring need to check health without credentials).
- Secrets (JWT signing key) are read from an environment variable, not hardcoded,
  so they can be swapped per-environment (dev/staging/prod) without code changes.

## Documentation

- `docs/architecture_diagram.png` — system architecture
- `docs/known_limitations.md` — known limitations and data quality issues
- `docs/testing_report.md` — test coverage and results
- `docs/model_metrics.json` — model evaluation metrics
- `docs/screenshot_*.png` — UI screenshots
