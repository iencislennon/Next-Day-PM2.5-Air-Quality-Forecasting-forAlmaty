import os
import time

import pandas as pd
import requests
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")
LOCATIONS_URL = "https://api.openaq.org/v3/locations"
REQUESTS_PER_MINUTE = 60

headers = {
    "X-API-Key": API_KEY
}


def get_with_retry(url, params):
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:
            time.sleep(int(response.headers.get("X-Ratelimit-Reset", 5)) + 1)
            continue
        return response


# Step 1: find stations in the bbox that have a PM2.5 sensor
locations_params = {
    "parameters_id": "2",  # PM2.5
    "limit": "1000",
    "bbox": "76.75,43.15,77.05,43.40"
}

locations_response = get_with_retry(LOCATIONS_URL, locations_params)

if locations_response.status_code != 200:
    print(f"error {locations_response.status_code} - {locations_response.text}")
else:
    locations = locations_response.json()["results"]

    pm25_sensors = [
        (location["id"], location["name"], sensor["id"])
        for location in locations
        for sensor in location["sensors"]
        if sensor["parameter"]["name"] == "pm25"
    ]

    # Step 2: pull daily PM2.5 averages for each sensor, staying under the rate limit
    days_params = {
        "date_from": "2025-01-01",
        "date_to": "2026-01-01",
        "limit": "1000",
    }

    all_rows = []
    for location_id, location_name, sensor_id in pm25_sensors:
        url = f"https://api.openaq.org/v3/sensors/{sensor_id}/days"
        response = get_with_retry(url, days_params)
        if response.status_code == 200:
            for row in response.json()["results"]:
                row["location_id"] = location_id
                row["location_name"] = location_name
                all_rows.append(row)
        else:
            print(f"error {response.status_code} - {response.text}")
        time.sleep(60 / REQUESTS_PER_MINUTE)

    df = pd.json_normalize(all_rows)
    print(df.head())
    print(f"\n{len(df)} rows from {len(pm25_sensors)} sensors")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "Almaty_PM2.5.csv"), index=False)