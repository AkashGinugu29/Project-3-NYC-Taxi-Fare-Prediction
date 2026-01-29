import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from features import add_features

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "taxi_fare_model.joblib")

app = FastAPI(title="NYC Taxi Fare Prediction API", version="1.0.0")

class PredictRequest(BaseModel):
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    pickup_datetime: str  # ISO string preferred

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run: python src/train.py")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        model = load_model()

        df = pd.DataFrame([req.model_dump()])
        df = add_features(df)

        X = df[["distance_km", "pickup_hour", "pickup_dayofweek"]]
        pred = float(model.predict(X)[0])

        return {"predicted_fare": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
