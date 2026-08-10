# Demo Video Script (5–10 minutes)

Record your screen while following this script. Suggested timing in [brackets].

## 1. Introduction [0:00 – 0:45]
"Hi, I'm [your name], and this is my capstone project: an End-to-End Business
Intelligence & Predictive Analytics System for retail sales, built on the
Rossmann Store Sales dataset — over 1 million daily sales records across
1,115 stores.

The system has two halves: a business intelligence dashboard for exploring
historical sales, and a machine learning model that forecasts future sales,
served through a secured API."

*(Show the architecture diagram — docs/architecture_diagram.png — on screen while saying this.)*

## 2. The Data Pipeline [0:45 – 1:45]
"Let's start with the data. I'm using the Rossmann dataset — daily sales,
customers, promotions, and holiday flags per store."

Run in terminal:
```bash
python src/etl/clean_data.py
```
"This cleaning step fixes a data quality issue I found during exploration —
mixed data types in the StateHoliday column — and flags 54 rows where a store
was marked open but recorded zero sales, rather than silently deleting them."

## 3. Model Training [1:45 – 2:45]
Run in terminal:
```bash
python src/models/train_model.py
```
"This trains an XGBoost model using lag features — sales from 7 days ago and
a 7-day rolling average — plus calendar and promotion features. I deliberately
used a time-ordered train/test split, not a random one, so the reported
accuracy reflects genuine forecasting performance, not data leakage.
The model achieves an R² of 0.70 on held-out data."

## 4. The API [2:45 – 4:00]
"Predictions are served through a FastAPI backend with JWT authentication."

Start the API: `uvicorn src.api.main:app --port 8000`

Show `/docs` in browser (or narrate if it doesn't render locally), then demonstrate:
- A login call to `/auth/login` returning a token
- A prediction call to `/predict/sales` using that token
- An attempt without a token, showing the 401 rejection

"This confirms the endpoint is genuinely secured, not just presented as secured."

## 5. The Dashboard [4:00 – 6:30]
Start the dashboard: `streamlit run dashboard/app.py`

- Log in with `analyst` / `changeme123`
- Walk through the KPIs, the sales trend chart, the day-of-week chart, and the
  promo impact chart — point out one real insight (e.g. "Sales are highest on
  Mondays and rise noticeably when a promotion is running")
- Go to the Predict page, fill in the form, and get a live prediction
  end-to-end through the UI

## 6. Testing & Known Limitations [6:30 – 8:00]
Run: `pytest tests/ -v`

"All 13 automated tests pass, covering the data cleaning, feature engineering,
and API security logic."

"I want to be upfront about this project's limitations too: the model's MAPE
is around 30%, which is a reasonable baseline but would improve significantly
with store metadata I didn't have access to. The current deployment is
single-instance, not yet load-tested, and uses a single demo user rather than
a real user database — all discussed in docs/known_limitations.md, along with
what a production version would need."

## 7. Scaling & Risk Discussion [8:00 – 9:30]
"For the expert-level questions: scaling this to a million users would mean
moving from CSV storage to a partitioned database or data warehouse,
containerizing and load-balancing the API and dashboard across multiple
instances, and adding proper identity management, rate limiting, and
CI/CD — all detailed in Section 9 of my project report.

The biggest technical risks I identified were data quality issues silently
corrupting results, time-series leakage in evaluation, and an unsecured
prediction endpoint — each addressed directly in this implementation, as
shown in Section 10 of the report."

## 8. Closing [9:30 – 10:00]
"That's my end-to-end BI and predictive analytics system — a working,
tested, documented pipeline from raw data to a secured, forecasting-capable
application. Thank you for watching."

---

**Tip:** Practice once without recording first so the terminal commands and
app navigation feel smooth — most of the "polish" in a demo video comes from
not fumbling clicks, not from editing.
