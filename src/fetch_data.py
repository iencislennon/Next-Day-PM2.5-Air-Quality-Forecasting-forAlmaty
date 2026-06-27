import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")
LOCATIONS_URL = "https://api.openaq.org/v3/locations"
REQUESTS_PER_MINUTE = 60

headers = {
    "X-API-Key": API_KEY
}

DATE_RANGES = [
    ("2022-01-01", "2022-07-01"),
    ("2022-07-01", "2023-01-01"),
    ("2023-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-01-01"),
    ("2025-01-01", "2025-07-01"),
    ("2025-07-01", "2026-01-01"),
]


def get_with_retry(url, params):
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:
            time.sleep(int(response.headers.get("X-Ratelimit-Reset", 5)) + 1)
            continue
        return response

locations_params = {
    "parameters_id": "2",
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

    print(f"Found {len(pm25_sensors)} sensors, fetching data across {len(DATE_RANGES)} periods...")

    all_rows = []
    for location_id, location_name, sensor_id in pm25_sensors:
        url = f"https://api.openaq.org/v3/sensors/{sensor_id}/days"
        for date_from, date_to in DATE_RANGES:
            params = {
                "date_from": date_from,
                "date_to": date_to,
                "limit": "1000",
            }
            response = get_with_retry(url, params)
            if response.status_code == 200:
                results = response.json()["results"]
                for row in results:
                    row["location_id"] = location_id
                    row["location_name"] = location_name
                    all_rows.append(row)
                print(f"  [{date_from} → {date_to}] {location_name}: {len(results)} rows")
            else:
                print(f"  error {response.status_code} for {location_name} [{date_from} → {date_to}]: {response.text}")
            time.sleep(60 / REQUESTS_PER_MINUTE)

    df = pd.json_normalize(all_rows)

    df = df.drop_duplicates()
    
    print(f"\nTotal: {len(df)} rows from {len(pm25_sensors)} sensors (2022-2026)")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "Almaty_PM2.5.csv"), index=False)
    print(f"Saved to {data_dir}/Almaty_PM2.5.csv")