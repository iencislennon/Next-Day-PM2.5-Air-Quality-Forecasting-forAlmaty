# Almaty PM2.5 Prediction

Machine learning pipeline for predicting daily PM2.5 air quality in Almaty, Kazakhstan using historical sensor data from OpenAQ.

## Overview

Almaty has some of the worst air quality in Central Asia due to its bowl-shaped geography surrounded by mountains, a harsh continental climate, and coal heating in winter. This project builds and compares regression models to predict daily PM2.5 concentrations across 118 monitoring stations.

## Results

| Model | RMSE | R² |
|---|---|---|
| Ridge (baseline) | 17.84 µg/m³ | 0.55 |
| Random Forest | 15.08 µg/m³ | 0.68 |
| Gradient Boosting | 15.67 µg/m³ | 0.65 |

Random Forest performs best. Results are consistent with academic literature for PM2.5 models without meteorological features (typical R² = 0.55–0.70). Adding weather data (temperature, wind speed, humidity) could push R² above 0.80.

## Project Structure

```
.
├── data/
│   ├── raw/                    # Raw OpenAQ data (Almaty_PM2.5.csv)
│   └── processed/              # Cleaned dataset after preprocessing
├── models/
│   └── plots/                  # Evaluation plots per model
├── scripts/
│   └── train.py                # Entry point for training all models
└── src/
    ├── data_loader.py          # Data loading, preprocessing, train/test split
    ├── fetch_data.py           # OpenAQ API data collection
    ├── model.py                # Model pipeline definitions
    └── utils.py                # Shared utilities
```

## Data

Data is fetched from the [OpenAQ v3 API](https://api.openaq.org) — daily PM2.5 aggregates for Almaty stations from 2022 to 2026.

To fetch fresh data:
```bash
python -m src.fetch_data
```

This collects data in 6-month chunks to avoid API timeouts and saves to `data/raw/Almaty_PM2.5.csv`.

## Features

**Temporal**
- `month` — month of year (1–12)
- `day_of_year` — day of year (1–365)
- `season` — winter/spring/summer/autumn (label encoded)

**Lag features** (core predictors)
- `pm25_lag1` — previous day's PM2.5
- `pm25_lag7` — PM2.5 one week ago
- `pm25_rolling7` — 7-day rolling average of prior days

**Station context**
- `location_encoded` — target-encoded station mean PM2.5 (computed on train set only)
- `coverage.percentCoverage` — data completeness for that day
- `flagInfo.hasFlags` — whether quality flags were raised

## Preprocessing

- Rows with no measurements (fully null days) are dropped
- PM2.5 values above 300 µg/m³ are treated as sensor artifacts and removed (Almaty's historical maximum is ~256 µg/m³)
- `summary.sd` NaNs (single-reading days) are filled with 0
- Highly correlated summary statistics (q02, q25, q98, median) are dropped
- Location encoding is computed **after** train/test split to prevent data leakage

## Training

```bash
./venv/bin/python -m scripts.train
```

Trains Ridge, Random Forest, and Gradient Boosting models and logs MSE, RMSE, and R² for each.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your OpenAQ API key:
```
OPENAQ_API_KEY=your_key_here
```

## Limitations

- No meteorological features (wind speed, temperature, humidity) — these are the strongest predictors of PM2.5 in bowl-shaped cities like Almaty
- Station coverage is uneven — some stations have as few as 8–10 days of data
- Models are trained per-city, not per-station
