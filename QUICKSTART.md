# Quick Start Guide

This guide will help you get started with the Health Risk Prediction MLOps System.

## Project Overview

This project implements an end-to-end MLOps system for predicting health risks using:
1. **Wearable Health Device Data** - Fitness trackers, smartwatches
2. **Air Quality Sensor Data** - Pollution measurements from multiple cities
3. **Weather Data** - Meteorological data from various locations

The system uses **Federated Learning** to train models across distributed nodes (simulating hospitals/cities) while keeping data local.

## Initial Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Project Structure

The project should have the following structure:
```
PROJECT/
├── data/
│   ├── raw/              # Raw collected data
│   ├── processed/        # Preprocessed data
│   └── federated/        # Federated learning node data
├── notebooks/            # EDA and experimentation notebooks
├── src/                  # Source code
│   ├── data_ingestion/   # Data collection modules
│   ├── models/           # Model definitions (to be created)
│   ├── federated/        # Federated learning (to be created)
│   ├── mlops/            # MLOps pipeline (to be created)
│   ├── monitoring/       # Monitoring and drift detection (to be created)
│   └── dashboard/        # Dashboard backend (to be created)
├── models/               # Trained model artifacts
├── configs/              # Configuration files
└── requirements.txt      # Python dependencies
```

## Data Collection

### Collect All Data

The system can generate synthetic data for all three sources:

```bash
python src/data_ingestion/collect_data.py
```

This will:
- Generate wearable device data for 100 users across 5 federated nodes
- Create air quality data for 5 cities
- Generate weather data for 5 locations
- Save data to `data/raw/` and `data/federated/`

### Data Sources

**Wearable Data:**
- Features: heart_rate, steps, sleep_hours, calories, body_temperature, stress_level
- Health conditions: normal, at_risk, ill
- Distributed across 5 nodes (hospitals/cities)

**Air Quality Data:**
- Cities: New York, London, Tokyo, Delhi, Beijing
- Pollutants: PM2.5, PM10, NO2, O3, CO
- Calculates AQI and health risk levels

**Weather Data:**
- Features: temperature, humidity, pressure, wind_speed, visibility
- Weather conditions: normal, hot, cold, rainy

## Exploratory Data Analysis

Three EDA notebooks are available:

### 1. Wearable Device EDA
```bash
jupyter notebook notebooks/01_wearable_eda.ipynb
```
Analyzes patterns in wearable health device data, feature distributions, correlations, and health condition patterns.

### 2. Air Quality EDA
```bash
jupyter notebook notebooks/02_air_quality_eda.ipynb
```
Explores air quality patterns across cities, pollutant distributions, AQI trends, and health risk levels.

### 3. Weather Data EDA
```bash
jupyter notebook notebooks/03_weather_eda.ipynb
```
Examines weather patterns, correlations between metrics, and temporal trends.

## Next Steps

### Phase 1: Individual Models (Current Phase)
1. ✅ Data collection system
2. ✅ EDA notebooks
3. ⏳ Build models for each dataset:
   - Health risk prediction from wearable data
   - Health risk prediction from air quality
   - Health risk prediction from weather data

### Phase 2: Federated Learning
- Implement federated learning framework
- Train models across distributed nodes
- Aggregate model updates

### Phase 3: MLOps Pipeline
- CI/CD for ML models
- Docker containerization
- Kubernetes deployment
- Experiment tracking (MLflow/W&B)
- Model versioning

### Phase 4: Monitoring & Drift Detection
- Real-time model monitoring
- Data drift detection
- Model performance tracking
- Automated retraining triggers

### Phase 5: Dashboard
- Health authority dashboard (public health maps, alerts)
- Citizen dashboard (personal alerts, trends)

## Configuration

Edit `configs/config.yaml` to customize:
- Dataset parameters
- Federated learning settings
- MLOps tool configurations
- Dashboard settings

## Notes

- Currently using **simulated data** for demonstration
- To use real APIs, set `use_api=True` and provide API keys in the collector classes
- Data is generated deterministically using random seeds for reproducibility

## Troubleshooting

**Issue: Missing dependencies**
```bash
pip install -r requirements.txt
```

**Issue: Data not generated**
- Ensure you're in the project root directory
- Check that `configs/config.yaml` exists
- Verify write permissions in `data/` directory

**Issue: Import errors in notebooks**
- Ensure `src/` directory is in Python path (handled in notebooks)
- Run data collection first if files are missing


