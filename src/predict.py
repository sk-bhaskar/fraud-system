import joblib
import numpy as np

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_transaction(features):
    data = np.array(features).reshape(1, -1)
    data_scaled = scaler.transform(data)

    prediction = int(model.predict(data_scaled)[0])
    probability = float(model.predict_proba(data_scaled)[0][1])

    if probability >= 0.8:
        risk = "high"
    elif probability >= 0.4:
        risk = "medium"
    else:
        risk = "low"

    return {
        "prediction": prediction,
        "fraud_probability": probability,
        "risk_level": risk
    }