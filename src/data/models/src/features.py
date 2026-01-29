import numpy as np
import pandas as pd

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points (lat/lon) in kilometers.
    """
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns:
      pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, pickup_datetime
    """
    out = df.copy()

    # distance feature
    out["distance_km"] = haversine_km(
        out["pickup_latitude"],
        out["pickup_longitude"],
        out["dropoff_latitude"],
        out["dropoff_longitude"],
    )

    # time features
    dt = pd.to_datetime(out["pickup_datetime"])
    out["pickup_hour"] = dt.dt.hour
    out["pickup_dayofweek"] = dt.dt.dayofweek

    return out
