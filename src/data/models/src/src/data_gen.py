import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def make_synthetic_taxi_data(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # NYC-ish latitude/longitude ranges
    pickup_lat = rng.uniform(40.63, 40.85, n)
    pickup_lon = rng.uniform(-74.05, -73.75, n)
    drop_lat = pickup_lat + rng.normal(0, 0.02, n)
    drop_lon = pickup_lon + rng.normal(0, 0.02, n)

    # random datetimes
    start = np.datetime64("2024-01-01")
    minutes = rng.integers(0, 60 * 24 * 365, n)
    pickup_dt = start + minutes.astype("timedelta64[m]")

    df = pd.DataFrame({
        "pickup_latitude": pickup_lat,
        "pickup_longitude": pickup_lon,
        "dropoff_latitude": drop_lat,
        "dropoff_longitude": drop_lon,
        "pickup_datetime": pickup_dt.astype(str),
    })

    return df

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = make_synthetic_taxi_data()
    out_path = os.path.join(DATA_DIR, "taxi_synthetic.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote synthetic dataset -> {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
