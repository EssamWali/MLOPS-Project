# Complete Project Guide: Health Risk Prediction MLOps System

## 📋 Project Overview

This is an **end-to-end MLOps system** that predicts health risks using AI models trained on data from multiple sources:
- **Wearable health devices** (fitness trackers, smartwatches)
- **Air quality sensors** (pollution data from cities)
- **Weather data** (meteorological conditions)

The system uses **Federated Learning** to train models across distributed nodes (simulating hospitals or cities) while keeping data local and private.

---

## 🎯 Project Goals

1. **Collect** data from multiple distributed sources
2. **Train** AI models to predict health risks
3. **Use Federated Learning** to train without sharing raw data
4. **Deploy** using MLOps practices (Docker, CI/CD, monitoring)
5. **Monitor** model performance and detect data drift
6. **Visualize** results through interactive dashboards

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐          │
│  │ Wearable │  │ Air Quality  │  │   Weather   │          │
│  │ Devices  │  │   Sensors    │  │    Data     │          │
│  └────┬─────┘  └──────┬───────┘  └──────┬──────┘          │
│       │               │                  │                  │
└───────┼───────────────┼──────────────────┼──────────────────┘
        │               │                  │
        ▼               ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA INGESTION & PREPROCESSING                  │
└─────────────────────────────────────────────────────────────┘
        │               │                  │
        ▼               ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    INDIVIDUAL MODELS                         │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐          │
│  │ Wearable │  │ Air Quality  │  │   Weather   │          │
│  │  Model   │  │    Model     │  │    Model    │          │
│  │ (88.48%  │  │   (100% F1)  │  │  (100% F1)  │          │
│  │   F1)    │  │              │  │             │          │
│  └────┬─────┘  └──────┬───────┘  └──────┬──────┘          │
│       │               │                  │                  │
└───────┼───────────────┼──────────────────┼──────────────────┘
        │               │                  │
        └───────────────┼──────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │    MULTI-MODAL FUSION MODEL   │
        │  (Combines all three models)  │
        └───────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────┐           ┌──────────────────────┐
│  DASHBOARDS   │           │   FEDERATED LEARNING  │
│  (Authority   │           │   (Distributed        │
│   & Citizen)  │           │    Training)          │
└───────────────┘           └──────────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   MLOPS PIPELINE              │
        │  - MLflow Tracking            │
        │  - Docker Deployment          │
        │  - CI/CD Pipeline             │
        │  - Drift Detection            │
        └───────────────────────────────┘
```

---

## 📁 Project Structure

```
PROJECT/
├── data/
│   ├── raw/                    # Raw collected data
│   ├── processed/              # Preprocessed data
│   └── federated/              # Data per federated node
│
├── src/
│   ├── data_ingestion/         # Data collection modules
│   │   ├── wearable_data_generator.py
│   │   ├── air_quality_collector.py
│   │   ├── weather_collector.py
│   │   └── collect_data.py
│   │
│   ├── models/                 # Model definitions
│   │   ├── wearable_model.py
│   │   ├── air_quality_model.py
│   │   ├── weather_model.py
│   │   ├── multimodal_model.py
│   │   └── train_models.py
│   │
│   ├── federated/              # Federated learning
│   │   ├── federated_server.py
│   │   ├── federated_client.py
│   │   └── run_federated_learning.py
│   │
│   ├── mlops/                  # MLOps components
│   │   └── mlflow_tracking.py
│   │
│   ├── monitoring/             # Monitoring & drift detection
│   │   ├── drift_detector.py
│   │   └── check_drift.py
│   │
│   └── dashboard/              # Web dashboard
│       └── app.py
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_wearable_eda.ipynb
│   ├── 02_air_quality_eda.ipynb
│   ├── 03_weather_eda.ipynb
│   ├── 05_wearable_model_training.ipynb
│   ├── 06_air_quality_model_training.ipynb
│   ├── 07_weather_model_training.ipynb
│   └── 08_multimodal_model.ipynb
│
├── models/                     # Trained model files (.pkl)
│   ├── wearable_model_gradient_boosting.pkl
│   ├── air_quality_model_random_forest.pkl
│   └── weather_model_random_forest.pkl
│
├── reports/                    # Reports and documentation
│   ├── MODEL_TRAINING_SUMMARY.md
│   └── EVALUATION_REPORT.md
│
├── docker/                     # Docker files
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .github/workflows/          # CI/CD pipeline
│   └── mlops-pipeline.yml
│
├── configs/                    # Configuration files
│   └── config.yaml
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── QUICKSTART.md              # Quick start guide
└── PROJECT_GUIDE.md           # This file
```

---

## 🚀 How to Run the Project

### Step 1: Prerequisites

**Required Software:**
- Python 3.9 or higher
- pip (Python package manager)
- Jupyter Notebook (for running notebooks)
- Docker (optional, for containerization)
- Git (for version control)

**Operating System:**
- macOS, Linux, or Windows

### Step 2: Setup Environment

1. **Clone or navigate to the project directory:**
```bash
cd /Users/faiqahmed/Desktop/Semesters/Semester7/MLOPS/PROJECT
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note:** This installs all required packages including:
- ML libraries (scikit-learn, pandas, numpy)
- Deep learning (torch, tensorflow)
- Federated learning (flwr)
- Visualization (matplotlib, seaborn, plotly, dash)
- MLOps (mlflow, wandb, evidently)
- And more...

### Step 3: Collect Data

**Generate sample data for all three sources:**
```bash
python src/data_ingestion/collect_data.py
```

**What this does:**
- Generates wearable device data (3000 records across 5 nodes)
- Creates air quality data (150 records for 5 cities)
- Generates weather data (150 records for 5 locations)
- Saves data to `data/raw/` and `data/federated/`

**Expected output:**
```
============================================================
DATA COLLECTION STARTED
============================================================

1. Generating Wearable Health Device Data...
Saved 600 records to .../data/federated/wearable_node_0.csv
...
✓ Generated wearable data: 3000 records

2. Collecting Air Quality Data...
✓ Collected air quality data: 150 records

3. Collecting Weather Data...
✓ Collected weather data: 150 records

============================================================
DATA COLLECTION COMPLETE
============================================================
```

### Step 4: Explore Data (Optional but Recommended)

**Run EDA notebooks to understand the data:**
```bash
jupyter notebook notebooks/01_wearable_eda.ipynb
```

Or run all three EDA notebooks:
- `notebooks/01_wearable_eda.ipynb` - Explore wearable device data
- `notebooks/02_air_quality_eda.ipynb` - Explore air quality data
- `notebooks/03_weather_eda.ipynb` - Explore weather data

These notebooks show:
- Data distributions
- Correlations
- Patterns and trends
- Feature engineering

### Step 5: Train Individual Models

**Train all three models at once:**
```bash
python src/models/train_models.py
```

**What this does:**
1. Loads data for each source
2. Trains 3 model types per dataset (Random Forest, Gradient Boosting, Logistic Regression)
3. Evaluates each model
4. Selects the best model for each dataset
5. Saves trained models to `models/` directory

**Expected output:**
```
================================================================================
TRAINING ALL MODELS
================================================================================

MODEL 1: Wearable Health Device Model
...
Training RANDOM_FOREST...
Model Performance:
Accuracy: 0.8833
F1-Score: 0.8788
...

✓ Best wearable model (gradient_boosting) saved

MODEL 2: Air Quality Health Risk Model
...
✓ Best air quality model (random_forest) saved

MODEL 3: Weather Health Risk Model
...
✓ Best weather model (random_forest) saved
```

**Trained models will be saved:**
- `models/wearable_model_gradient_boosting.pkl`
- `models/air_quality_model_random_forest.pkl`
- `models/weather_model_random_forest.pkl`

**Or train individually using notebooks:**
```bash
jupyter notebook notebooks/05_wearable_model_training.ipynb
jupyter notebook notebooks/06_air_quality_model_training.ipynb
jupyter notebook notebooks/07_weather_model_training.ipynb
```

### Step 6: Use Multi-Modal Model (Optional)

**Test the multi-modal fusion model:**
```python
from src.models.multimodal_model import MultiModalHealthRiskModel
import pandas as pd

# Load data
wearable_df = pd.read_csv('data/raw/wearable_data.csv', parse_dates=['timestamp'])
air_quality_df = pd.read_csv('data/raw/air_quality_data.csv', parse_dates=['timestamp'])
weather_df = pd.read_csv('data/raw/weather_data.csv', parse_dates=['timestamp'])

# Initialize and load models
multimodal_model = MultiModalHealthRiskModel(strategy='ensemble')
multimodal_model.load_individual_models('models/')

# Make predictions
predictions, probabilities = multimodal_model.predict(
    df_wearable=wearable_df.head(100),
    df_air_quality=air_quality_df.head(30),
    df_weather=weather_df.head(30)
)

print(f"Predictions: {predictions}")
```

### Step 7: Run Federated Learning (Advanced)

**Federated learning simulates training across multiple nodes without sharing data.**

**Option A: Run on separate terminals (simulation)**

**Terminal 1 - Start Server:**
```bash
python src/federated/federated_server.py
```

**Terminal 2-N - Start Clients (run in separate terminals):**
```bash
# Client 0
python -c "from src.federated.federated_client import start_client; start_client(0, 'localhost:8080')"

# Client 1
python -c "from src.federated.federated_client import start_client; start_client(1, 'localhost:8080')"

# ... repeat for clients 2, 3, 4
```

**Note:** In production, each client would run on a different machine.

### Step 8: Launch Dashboard

**Start the interactive dashboard:**
```bash
python src/dashboard/app.py
```

**Open in browser:**
```
http://localhost:8050
```

**Dashboard Features:**
- **Health Authority Tab:**
  - Total users monitored
  - Active risk alerts
  - Cities monitored
  - Average risk level
  - Air quality by city chart
  - Health risk distribution
  - Time series trends
  - Interactive risk map

- **Citizen Tab:**
  - Personal risk level
  - Recommendations
  - Local air quality
  - Personal health trends
  - Individual metrics visualization

**To stop the dashboard:** Press `Ctrl+C` in the terminal

### Step 9: Monitor Data Drift

**Check for data drift:**
```bash
python src/monitoring/check_drift.py
```

**What this does:**
- Compares current data with reference (baseline) data
- Detects statistical differences in feature distributions
- Reports drifted features
- Saves drift report to `reports/`

**Expected output:**
```
============================================================
DATA DRIFT DETECTION
============================================================

1. Checking wearable device data drift...
  Drift detected: False
  Drifted features: 0/13

2. Checking air quality data drift...
  Drift detected: False
  Drifted features: 0/8

3. Checking weather data drift...
  Drift detected: False
  Drifted features: 0/5
```

### Step 10: View MLflow Experiments (Optional)

**Start MLflow UI:**
```bash
mlflow ui --backend-store-uri file:///$(pwd)/mlruns
```

**Open in browser:**
```
http://localhost:5000
```

**What you can see:**
- All training runs
- Model metrics (accuracy, F1-score, etc.)
- Model parameters
- Model artifacts
- Compare different runs

### Step 11: Deploy with Docker (Optional)

**Build Docker image:**
```bash
docker build -t health-risk-prediction -f docker/Dockerfile .
```

**Run with docker-compose:**
```bash
docker-compose -f docker/docker-compose.yml up
```

This starts:
- MLflow tracking server (port 5000)
- Model training service
- Federated learning server (port 8080)
- Dashboard (port 8050)

---

## 📊 Understanding the Components

### 1. Data Ingestion

**Purpose:** Collect data from multiple sources

**Files:**
- `src/data_ingestion/wearable_data_generator.py` - Generates wearable device data
- `src/data_ingestion/air_quality_collector.py` - Collects air quality data
- `src/data_ingestion/weather_collector.py` - Collects weather data

**Key Features:**
- Simulates data from distributed nodes
- Supports real API integration (with API keys)
- Configurable data generation

### 2. Individual Models

**Three separate models, each optimized for its data source:**

**Wearable Model:**
- **Best:** Gradient Boosting (88.48% F1-score)
- **Predicts:** Health conditions (normal, at_risk, ill)
- **Features:** Heart rate, steps, sleep, calories, temperature, stress

**Air Quality Model:**
- **Best:** Random Forest (100% F1-score)
- **Predicts:** Health risk levels (good, moderate, unhealthy, etc.)
- **Features:** PM2.5, PM10, NO2, O3, CO, AQI

**Weather Model:**
- **Best:** Random Forest (100% F1-score)
- **Predicts:** Health risk levels (low, moderate, high)
- **Features:** Temperature, humidity, pressure, wind, visibility

### 3. Multi-Modal Model

**Purpose:** Combines predictions from all three models

**Strategies:**
- **Ensemble Voting:** Majority vote across models
- **Weighted Average:** Weighted combination of probabilities

**Advantages:**
- More robust predictions
- Works even if one data source is missing
- Comprehensive risk assessment

### 4. Federated Learning

**Purpose:** Train models across distributed nodes without sharing raw data

**How it works:**
1. Each node (hospital/city) trains on local data
2. Nodes send model updates (not data) to central server
3. Server aggregates updates
4. Server sends aggregated model back to nodes
5. Process repeats for multiple rounds

**Benefits:**
- Privacy preservation
- Distributed computing
- No data centralization needed

### 5. MLOps Pipeline

**Components:**
- **MLflow:** Experiment tracking and model registry
- **Docker:** Containerization for deployment
- **CI/CD:** Automated testing and training (GitHub Actions)
- **Monitoring:** Performance tracking and drift detection

### 6. Dashboard

**Two views:**
- **Health Authority:** Population-level insights, alerts, trends
- **Citizen:** Personal health dashboard, individual alerts

**Technology:** Built with Dash (Plotly) for interactive visualizations

---

## 🔄 Typical Workflow

### Development Workflow

1. **Data Collection** → Collect/generate data
2. **EDA** → Explore and understand data
3. **Model Training** → Train individual models
4. **Evaluation** → Compare and select best models
5. **Multi-Modal** → Combine predictions
6. **Deployment** → Deploy with Docker/MLflow
7. **Monitoring** → Track performance and drift

### Production Workflow

1. **Federated Training** → Train across distributed nodes
2. **Model Registry** → Version and store models in MLflow
3. **Continuous Monitoring** → Detect drift and degradation
4. **Automated Retraining** → Retrain when needed
5. **Dashboard Updates** → Real-time visualization

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure you're in the project root directory
cd /Users/faiqahmed/Desktop/Semesters/Semester7/MLOPS/PROJECT

# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**2. Data Not Found**
```bash
# Run data collection first
python src/data_ingestion/collect_data.py
```

**3. Models Not Found**
```bash
# Train models first
python src/models/train_models.py
```

**4. Dashboard Won't Start**
```bash
# Check if port 8050 is available
# Or change port in src/dashboard/app.py:
# app.run_server(debug=True, port=8051, host='0.0.0.0')
```

**5. Federated Learning Issues**
```bash
# Make sure data files exist in data/federated/
ls data/federated/

# Check server is running before starting clients
```

---

## 📚 Additional Resources

**Documentation Files:**
- `README.md` - Project overview
- `QUICKSTART.md` - Quick start guide
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `reports/EVALUATION_REPORT.md` - Model evaluation
- `reports/MODEL_TRAINING_SUMMARY.md` - Training summary

**Configuration:**
- `configs/config.yaml` - All configuration settings

**Example Usage:**
- See notebooks in `notebooks/` directory for examples

---

## 🎓 Learning Path

**Beginner:**
1. Run data collection
2. Explore data with EDA notebooks
3. Train models
4. View dashboard

**Intermediate:**
1. Understand model code
2. Modify hyperparameters
3. Experiment with different models
4. Add new features

**Advanced:**
1. Set up federated learning
2. Deploy with Docker
3. Set up CI/CD pipeline
4. Implement custom monitoring

---

## ✅ Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Collect data: `python src/data_ingestion/collect_data.py`
- [ ] Train models: `python src/models/train_models.py`
- [ ] Launch dashboard: `python src/dashboard/app.py`
- [ ] View dashboard: Open http://localhost:8050
- [ ] (Optional) Check drift: `python src/monitoring/check_drift.py`
- [ ] (Optional) View MLflow: `mlflow ui`

---

## 🎯 Next Steps

After running the project:

1. **Explore the code** to understand implementation
2. **Modify models** to experiment with different approaches
3. **Add real data** by connecting to actual APIs
4. **Extend functionality** by adding new features
5. **Deploy** to production environment

---

**Questions?** Check the documentation files or explore the code!

**Happy Learning! 🚀**


