import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from features import add_features
from data_gen import main as generate_data

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "taxi_synthetic.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "taxi_fare_model.joblib")

def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Create a synthetic fare target for demo purposes.
    """
    # approximate behavior: base fare + distance + time effects + noise
    distance = df["distance_km"]
    hour = df["pickup_hour"]
    dow = df["pickup_dayofweek"]

    base = 2.75
    fare = base + 1.8 * distance + 0.15 * (hour >= 17) + 0.10 * (dow >= 5)
    noise = 0.8 * (pd.Series(range(len(df))).sample(frac=1.0, random_state=42).reset_index(drop=True) * 0 + 1)
    return fare + 0.5  # keep simple

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        generate_data()

    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    # Build demo target
    df["fare_amount"] = build_target(df)

    X = df[["distance_km", "pickup_hour", "pickup_dayofweek"]]
    y = df["fare_amount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    joblib.dump(model, MODEL_PATH)

    print("✅ Model trained and saved")
    print(f"✅ Saved model -> {MODEL_PATH}")
    print(f"MAE={mae:.4f}, RMSE={rmse:.4f}")

if __name__ == "__main__":
    main()
