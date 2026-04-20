from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Real-Time Fraud Detection API")

# Load saved model and scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

THRESHOLD = 0.9


class TransactionInput(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "threshold": THRESHOLD
    }


@app.post("/predict")
def predict(transaction: TransactionInput):
    try:
        data = np.array(transaction.features).reshape(1, -1)
        data_scaled = scaler.transform(data)

        fraud_probability = float(model.predict_proba(data_scaled)[0][1])
        prediction = int(fraud_probability > THRESHOLD)

        if fraud_probability >= 0.9:
            risk_level = "high"
        elif fraud_probability >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "prediction": prediction,
            "fraud_probability": round(fraud_probability, 6),
            "risk_level": risk_level
        }

    except Exception as e:
        return {"error": str(e)}