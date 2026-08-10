"""
Rossmann Sales Intelligence Dashboard (Streamlit)

Run:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="Rossmann Sales Intelligence", layout="wide", page_icon="📊")

DATA_PATH = Path("data/processed/clean_sales.csv")

# --- Very simple session-based auth (demo-grade; the real auth lives in the API/JWT layer) ---
VALID_USERS = {"analyst": "changeme123"}


def login_screen():
    st.title("🔐 Rossmann Sales Intelligence — Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if VALID_USERS.get(username) == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid username or password")
    st.caption("Demo credentials — username: analyst / password: changeme123")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df


def main_dashboard():
    df = load_data()

    st.sidebar.title(f"👋 Welcome, {st.session_state.get('username', 'user')}")
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

    st.sidebar.header("Filters")
    stores = sorted(df["Store"].unique())
    selected_stores = st.sidebar.multiselect("Select store(s)", stores, default=stores[:5])
    date_range = st.sidebar.date_input(
        "Date range",
        [df["Date"].min(), df["Date"].max()],
    )

    filtered = df[df["Store"].isin(selected_stores)]
    if len(date_range) == 2:
        filtered = filtered[
            (filtered["Date"] >= pd.Timestamp(date_range[0]))
            & (filtered["Date"] <= pd.Timestamp(date_range[1]))
        ]

    st.title("📊 Rossmann Sales Intelligence Dashboard")
    st.caption("Business Intelligence & Predictive Analytics System — Retail Sales")

    # --- KPI row ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"€{filtered['Sales'].sum():,.0f}")
    col2.metric("Avg Daily Sales", f"€{filtered['Sales'].mean():,.0f}")
    col3.metric("Total Customers", f"{filtered['Customers'].sum():,.0f}")
    col4.metric("Stores Selected", f"{filtered['Store'].nunique()}")

    st.divider()

    # --- Sales trend over time ---
    st.subheader("Sales Trend Over Time")
    trend = filtered.groupby("Date", as_index=False)["Sales"].sum()
    fig = px.line(trend, x="Date", y="Sales", title="Total Sales Over Time")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Sales by Day of Week")
        dow = filtered.groupby("DayOfWeek", as_index=False)["Sales"].mean()
        fig2 = px.bar(dow, x="DayOfWeek", y="Sales", title="Average Sales by Day of Week")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("Promo Impact on Sales")
        promo = filtered.groupby("Promo", as_index=False)["Sales"].mean()
        promo["Promo"] = promo["Promo"].map({0: "No Promo", 1: "Promo Running"})
        fig3 = px.bar(promo, x="Promo", y="Sales", title="Average Sales: Promo vs No Promo")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top 10 Stores by Total Sales")
    top_stores = (
        filtered.groupby("Store", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(10)
    )
    fig4 = px.bar(top_stores, x="Store", y="Sales", title="Top 10 Stores")
    st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.info(
        "💡 For sales **predictions** (forecasting future sales), see the '🔮 Predict' page "
        "in the sidebar navigation, which calls the authenticated FastAPI backend."
    )


def app():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login_screen()
    else:
        main_dashboard()


if __name__ == "__main__":
    app()
