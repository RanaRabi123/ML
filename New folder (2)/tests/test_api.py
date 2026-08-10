"""
Integration tests for the FastAPI service.

Run:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_login_success():
    resp = client.post("/auth/login", data={"username": "analyst", "password": "changeme123"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_failure_wrong_password():
    resp = client.post("/auth/login", data={"username": "analyst", "password": "wrong"})
    assert resp.status_code == 401


def test_predict_requires_auth():
    resp = client.post("/predict/sales", json={
        "Store": 1, "DayOfWeek": 5, "Promo": 1, "SchoolHoliday": 0,
        "Year": 2015, "Month": 7, "Day": 31, "WeekOfYear": 31,
        "Sales_Lag_7": 5000, "Sales_RollingMean_7": 5000, "StateHoliday": "None",
    })
    assert resp.status_code == 401


def test_predict_with_valid_token():
    login_resp = client.post("/auth/login", data={"username": "analyst", "password": "changeme123"})
    token = login_resp.json()["access_token"]

    resp = client.post(
        "/predict/sales",
        json={
            "Store": 1, "DayOfWeek": 5, "Promo": 1, "SchoolHoliday": 0,
            "Year": 2015, "Month": 7, "Day": 31, "WeekOfYear": 31,
            "Sales_Lag_7": 5000, "Sales_RollingMean_7": 5000, "StateHoliday": "None",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert "predicted_sales" in resp.json()
    assert resp.json()["predicted_sales"] > 0


def test_predict_rejects_invalid_token():
    resp = client.post(
        "/predict/sales",
        json={
            "Store": 1, "DayOfWeek": 5, "Promo": 1, "SchoolHoliday": 0,
            "Year": 2015, "Month": 7, "Day": 31, "WeekOfYear": 31,
            "Sales_Lag_7": 5000, "Sales_RollingMean_7": 5000, "StateHoliday": "None",
        },
        headers={"Authorization": "Bearer not.a.real.token"},
    )
    assert resp.status_code == 401
