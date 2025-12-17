import numpy as np
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse



# --------------------------------------------------
# Resolve base directory
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "rf_upi_fraud_model.joblib"
SCALER_PATH = BASE_DIR / "model" / "scaler.joblib"

# --------------------------------------------------
# Load model and scaler
# --------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="UPI Fraud Detection API",
    description="Predicts whether a UPI transaction is fraudulent using Random Forest",
    version="1.0"
)

# Serve static files under /static and index.html at root
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/")
def read_root():
    return FileResponse(BASE_DIR / "static" / "index.html")
# --------------------------------------------------
# Input schema
# --------------------------------------------------
class Transaction(BaseModel):
    trans_datetime: str
    category: int
    dob: str
    trans_amount: float
    state: int
    zip: int

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def detect_fraud(tx: Transaction):
    try:
        # Parse dates
        trans_datetime = pd.to_datetime(tx.trans_datetime)
        dob = pd.to_datetime(tx.dob)

        # Feature engineering (MUST match training):
        # training X has 8 features because the script reads train_dataset.csv
        # with index_col=0, so "trans_hour" is used as index and dropped.
        # Features used by the model (in order):
        # trans_day, trans_month, trans_year, category,
        # age, trans_amount, state, zip
        v1 = trans_datetime.day
        v2 = trans_datetime.month
        v3 = trans_datetime.year
        v4 = tx.category
        v5 = int(round((trans_datetime - dob).days / 365.25))  # age in years
        v6 = tx.trans_amount
        v7 = tx.state
        v8 = tx.zip

        # 8 features to match training
        features = np.array([[v1, v2, v3, v4, v5, v6, v7, v8]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Random Forest prediction
        fraud_prob = model.predict_proba(features_scaled)[0][1]

        # Risk banding:
        # < 0.5  -> safe / low risk
        # 0.5-0.7 -> medium risk
        # > 0.7  -> high risk
        if fraud_prob < 0.5:
            risk_level = "LOW_RISK"
            prediction = 0
            result_text = "VALID / SAFE TRANSACTION"
        elif fraud_prob < 0.7:
            risk_level = "MEDIUM_RISK"
            prediction = 1
            result_text = "MEDIUM RISK TRANSACTION"
        else:
            risk_level = "HIGH_RISK"
            prediction = 1
            result_text = "HIGH RISK / FRAUD TRANSACTION"

        return {
            "fraud_probability": float(fraud_prob),
            "fraud_prediction": prediction,
            "risk_level": risk_level,
            "result": result_text
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
