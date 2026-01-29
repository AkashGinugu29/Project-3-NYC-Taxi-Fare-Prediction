# NYC Taxi Fare Prediction (Scikit-learn + FastAPI)

An end-to-end regression project that predicts taxi fares using engineered distance and time-based features.
Includes training, evaluation, and a FastAPI inference service.

## Features
- Feature engineering (haversine distance + time features)
- Regression model training + evaluation (MAE, RMSE)
- Saved model artifact for reproducible inference
- FastAPI endpoint for predictions

## Quickstart
1) Install:
pip install -r requirements.txt

2) Train model:
python src/train.py

3) Run API:
uvicorn src.api:app --reload

## Endpoints
- GET /health
- POST /predict
