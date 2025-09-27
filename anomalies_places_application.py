# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.cluster import DBSCAN
import numpy as np
import redis

from redis_client import (
    save_user_data,
    get_user_data,
    save_model,
    load_model
)

app = FastAPI()

# radis database
r = redis.Redis(host='localhost', port=6379, db=0)

# how far two visits can be to belong to the same cluster
# 0.5 km in radians
EPS_KM = 100
EARTH_RADIUS_KM = 6371
EPS_RAD = EPS_KM / EARTH_RADIUS_KM

MIN_SAMPLES = 2  # min visits to form a cluster

class Visits(BaseModel):
    user_id: str
    lat: float
    lon: float

@app.post("/record")
def record_visit(visit: Visits):
    # store raw lat/lon
    user_data = get_user_data(visit.user_id)
    user_data.append([visit.lat, visit.lon])
    save_user_data(visit.user_id, user_data)

    # train DBSCAN if enough data
    if len(user_data) >= MIN_SAMPLES:
        arr = np.radians(np.array(user_data))  # convert degrees to radians
        model = DBSCAN(eps=EPS_RAD, min_samples=MIN_SAMPLES, metric="haversine")
        model.fit(arr)
        save_model(visit.user_id, model)

    return {"status": "recorded"}

@app.post("/predict")
def predict_visit(visit: Visits):
        model = load_model(visit.user_id)
        user_data = get_user_data(visit.user_id)

        if model is None or not user_data:
            return {"error": "Not enough data or model not trained yet."}

        # DBSCAN cannot "predict new data" → we check manually:
        # A point is normal if within eps of any known cluster point
        vector = np.radians(np.array([[visit.lat, visit.lon]]))
        arr = np.radians(np.array(user_data))

        # compute haversine distance manually
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(vector, arr, metric="haversine") * EARTH_RADIUS_KM

        if np.any(dists <= EPS_KM):
            return {"prediction": "normal", "value": 1}
        else:
            return {"prediction": "anomaly", "value": -1}


@app.get("/user_data/{user_id}")
def user_data(user_id: str):
    """Return all user visits with DBSCAN cluster/anomaly labels."""
    data = get_user_data(user_id)
    model = load_model(user_id)

    if not data:
        return {"error": "No data found for user."}

    points = []
    if model:
        arr = np.radians(np.array(data))
        labels = model.fit_predict(arr)  # get cluster labels
        for (lat, lon), label in zip(data, labels):
            points.append({
                "lat": lat,
                "lon": lon,
                "status": "anomaly" if label == -1 else f"cluster_{label}"
            })
    else:
        # no model trained yet
        points = [
            {"lat": lat, "lon": lon, "status": "unknown"}
            for lat, lon in data
        ]

    return {"user_id": user_id, "visits": points}


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
