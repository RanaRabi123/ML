"""
Prediction page: calls the authenticated FastAPI backend to forecast sales.
"""

import requests
import streamlit as st

st.set_page_config(page_title="Predict Sales", page_icon="🔮")

API_URL = "http://127.0.0.1:8000"

if not st.session_state.get("authenticated"):
    st.warning("Please log in from the main dashboard page first.")
    st.stop()

st.title("🔮 Predict Store Sales")
st.caption("This form calls the secured FastAPI /predict/sales endpoint (JWT-authenticated).")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        store = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
        day_of_week = st.selectbox("Day of Week (1=Mon .. 7=Sun)", list(range(1, 8)), index=4)
        promo = st.selectbox("Promo running?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        school_holiday = st.selectbox("School holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        state_holiday = st.selectbox("State holiday type", ["None", "a", "b", "c"])
    with col2:
        year = st.number_input("Year", min_value=2013, max_value=2020, value=2015)
        month = st.number_input("Month", min_value=1, max_value=12, value=7)
        day = st.number_input("Day", min_value=1, max_value=31, value=31)
        week_of_year = st.number_input("Week of Year", min_value=1, max_value=53, value=31)
        sales_lag_7 = st.number_input("Sales 7 days ago (€)", min_value=0.0, value=5500.0)
        rolling_mean_7 = st.number_input("7-day average sales (€)", min_value=0.0, value=5600.0)

    submitted = st.form_submit_button("Predict Sales")

if submitted:
    payload = {
        "Store": store,
        "DayOfWeek": day_of_week,
        "Promo": promo,
        "SchoolHoliday": school_holiday,
        "Year": year,
        "Month": month,
        "Day": day,
        "WeekOfYear": week_of_year,
        "Sales_Lag_7": sales_lag_7,
        "Sales_RollingMean_7": rolling_mean_7,
        "StateHoliday": state_holiday,
    }
    try:
        # Step 1: login to get a fresh token
        login_resp = requests.post(
            f"{API_URL}/auth/login",
            data={"username": "analyst", "password": "changeme123"},
            timeout=5,
        )
        login_resp.raise_for_status()
        token = login_resp.json()["access_token"]

        # Step 2: call the protected prediction endpoint
        pred_resp = requests.post(
            f"{API_URL}/predict/sales",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
        pred_resp.raise_for_status()
        result = pred_resp.json()
        st.success(f"### Predicted Sales: €{result['predicted_sales']:,.2f}")
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not reach the API. Make sure it's running: "
            "`uvicorn src.api.main:app --port 8000`"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
