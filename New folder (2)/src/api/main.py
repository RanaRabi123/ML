"""
FastAPI service exposing:
  - POST /auth/login       -> get a JWT token
  - POST /predict/sales    -> predict sales (requires valid JWT)
  - GET  /health           -> health check (no auth, used for monitoring/load balancers)

Run:
    uvicorn src.api.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive API documentation (auto-generated).
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from jose import jwt, JWTError
from passlib.context import CryptContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Config (in production these come from environment variables / a secrets manager,
#     NEVER hardcoded — this is called out explicitly in docs/known_limitations.md) ---
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-only-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

MODEL_PATH = Path("data/processed/sales_model.joblib")

app = FastAPI(
    title="Rossmann Sales Intelligence API",
    description="Predictive analytics API for retail sales forecasting.",
    version="1.0.0",
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# --- Fake user store (demo only). In production: a real Users table in a database. ---
FAKE_USERS_DB = {
    "analyst": {
        "username": "analyst",
        "hashed_password": pwd_context.hash("changeme123"),
    }
}

_model_bundle = None


def get_model():
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not trained yet. Run train_model.py first.")
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


class PredictRequest(BaseModel):
    Store: int = Field(..., example=1)
    DayOfWeek: int = Field(..., ge=1, le=7, example=5)
    Promo: int = Field(..., ge=0, le=1, example=1)
    SchoolHoliday: int = Field(..., ge=0, le=1, example=0)
    Year: int = Field(..., example=2015)
    Month: int = Field(..., ge=1, le=12, example=7)
    Day: int = Field(..., ge=1, le=31, example=31)
    WeekOfYear: int = Field(..., ge=1, le=53, example=31)
    Sales_Lag_7: float = Field(..., example=5500.0)
    Sales_RollingMean_7: float = Field(..., example=5600.0)
    StateHoliday: str = Field("None", example="None")


class PredictResponse(BaseModel):
    predicted_sales: float


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in FAKE_USERS_DB:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception


@app.get("/health")
def health():
    """Health check endpoint — used by load balancers / monitoring, no auth required."""
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/predict/sales", response_model=PredictResponse)
def predict_sales(req: PredictRequest, current_user: str = Depends(get_current_user)):
    try:
        bundle = get_model()
        model, feature_cols = bundle["model"], bundle["feature_cols"]

        row = {c: 0 for c in feature_cols}
        payload = req.dict()
        holiday_col = f"Holiday_{payload.pop('StateHoliday')}"
        if holiday_col in row:
            row[holiday_col] = 1
        for k, v in payload.items():
            if k in row:
                row[k] = v

        X = pd.DataFrame([row])[feature_cols]
        pred = model.predict(X)[0]
        return {"predicted_sales": round(float(pred), 2)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
 