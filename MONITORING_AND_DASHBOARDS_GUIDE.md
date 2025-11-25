# Data Monitoring & Dashboards Guide

## 🎯 Overview

Your MLOps project includes **three layers of monitoring and visualization**:

1. **Drift Detection** - Automated data quality monitoring
2. **Dashboard** - Real-time health risk visualization
3. **MLflow** - Experiment tracking and model versioning

---

## 1. 📊 Drift Detection (Data Monitoring)

### What It Does
Detects when incoming data differs from training data distribution—a critical signal that model retraining is needed.

### How It Works

**Location**: `src/monitoring/check_drift.py` (runs automatically in CI/CD)

**Process**:
1. Compares reference data (training set) vs. current data (new incoming data)
2. Tests each feature for statistical drift using Kolmogorov-Smirnov test
3. Generates drift report with:
   - Which features drifted
   - Drift severity (p-value < 0.3 = drift detected)
   - Timestamp and recommendations

**Output Files**:
- `reports/drift_report.json` - Latest report (used by CI/CD)
- `reports/drift_report_YYYYMMDD_HHMMSS.json` - Timestamped archives (historical record)

### How to Use Locally

```powershell
# Activate venv
& D:\PROJECT\venv\Scripts\Activate.ps1

# Run drift detection
python src/monitoring/check_drift.py
```

**Example Output**:
```
================================================================================
DATA DRIFT DETECTION
================================================================================

1. Checking wearable device data drift...
  Drift detected: False
  Drifted features: 0/5

2. Checking air quality data drift...
  Drift detected: True
  Drifted features: 2/4

3. Checking weather data drift...
  Drift detected: False
  Drifted features: 0/3

================================================================================
DRIFT DETECTION COMPLETE
================================================================================

Report saved to: reports/drift_report_20251125_120000.json
Latest report: reports/drift_report.json

Summary: Drift detected in 1/3 datasets
```

### CI/CD Integration

Your workflow (`.github/workflows/mlops-pipeline.yml`) automatically:

```yaml
- name: Run drift detection
  run: |
    python src/monitoring/check_drift.py

- name: Upload drift report
  uses: actions/upload-artifact@v4
  with:
    name: drift-report
    path: reports/drift_report.json
```

**What happens**:
- Every push triggers drift detection
- Report is uploaded as CI artifact
- You can download and review it in GitHub Actions

### Interpreting Drift Reports

**Example drift_report.json**:
```json
{
  "timestamp": "2025-11-25T12:00:00",
  "results": {
    "wearable": {
      "drift_detected": false,
      "drifted_features": [],
      "num_drifted_features": 0,
      "total_features": 5
    },
    "air_quality": {
      "drift_detected": true,
      "drifted_features": ["pm25", "no2"],
      "num_drifted_features": 2,
      "total_features": 4,
      "p_values": {"pm25": 0.02, "no2": 0.15}
    }
  }
}
```

**Interpretation**:
- ✅ `drift_detected: false` → Data is stable, model predictions remain reliable
- ⚠️ `drift_detected: true` → Data has changed, consider retraining
- 🔴 Multiple features drifted → Urgent retraining recommended

---

## 2. 📈 Dashboard (Real-Time Visualization)

### What It Does

Interactive web dashboard with two views:
- **Health Authority View** - Public health overview, city-level risk maps, alerts
- **Citizen View** - Personal health alerts, individual trends, recommendations

### Architecture

**Location**: `src/dashboard/app.py`

**Tech Stack**:
- **Framework**: Dash (Plotly-based web framework)
- **Backend**: Python + Pandas
- **Models**: Loads trained `.pkl` models for real-time predictions
- **Visualizations**: Plotly interactive charts

### How to Run Locally

```powershell
# Activate venv
& D:\PROJECT\venv\Scripts\Activate.ps1

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the dashboard
python src/dashboard/app.py
```

**Output**:
```
Running on http://127.0.0.1:8050
WARNING in dash.dash: Trying to start a Dash app from "main" block...

Open http://localhost:8050 in your browser
```

### Dashboard Features

#### 📋 Health Authority Dashboard (Default View)

**Metrics Panel**:
- Total Users Under Monitoring
- Active Alerts
- Cities Covered
- Overall Risk Level

**Charts**:
1. **Air Quality by City** - Bar chart showing AQI levels across locations
2. **Health Risk Distribution** - Histogram of population risk levels
3. **Time Series Trends** - 30-day wearable health metrics trend
4. **Recent Alerts** - Table of high-risk users and high-AQI cities
5. **Interactive Risk Map** - Scatter plot of city locations with risk coloring

**Use Cases**:
- Monitor city-level air quality
- Identify outbreak hotspots
- Track population health trends
- Respond to high-risk alerts

#### 👤 Citizen Dashboard (Accessible via dropdown)

**Personal Metrics**:
- Your Risk Level (Low/Medium/High)
- Heart Rate & Activity Trends
- Local Air Quality
- Personalized Health Recommendations

**Features**:
- User selector dropdown (switch between users)
- Personal trend charts
- Context-aware health recommendations
- Real-time prediction updates

**Use Cases**:
- Citizens check personal health status
- Track personal trends over time
- Get location-based air quality alerts
- Receive personalized health advice

### How to Access in Production

Once deployed (e.g., via Docker/Kubernetes):

```bash
# Docker
docker run -p 8050:8050 health-risk-prediction:latest python src/dashboard/app.py

# Kubernetes
kubectl port-forward svc/dashboard-service 8050:8050
# Then open http://localhost:8050
```

### Dashboard Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                  Dashboard (Dash App)                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐      ┌──────────────────┐        │
│  │ Load Raw Data   │      │ Load Trained     │        │
│  │ from CSV files  │      │ Models (.pkl)    │        │
│  └────────┬────────┘      └────────┬─────────┘        │
│           │                        │                  │
│           └────────────┬───────────┘                  │
│                        │                              │
│              ┌─────────▼──────────┐                  │
│              │ Generate Predictions                  │
│              │ (Real-time scoring)                   │
│              └─────────┬──────────┘                  │
│                        │                              │
│         ┌──────────────┼──────────────┐              │
│         │              │              │              │
│    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐         │
│    │ Chart 1 │   │ Chart 2 │   │ Chart 3 │         │
│    │ (Plotly)│   │ (Plotly)│   │ (Plotly)│         │
│    └────┬────┘   └────┬────┘   └────┬────┘         │
│         │             │             │               │
│         └─────────────┼─────────────┘               │
│                       │                             │
│              ┌────────▼─────────┐                  │
│              │ Render HTML/JS   │                  │
│              │ Browser Display  │                  │
│              └──────────────────┘                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 3. 🔬 Experiment Tracking (MLflow)

### What It Does

Logs and tracks all model training experiments:
- Model hyperparameters
- Performance metrics
- Training data versions
- Model artifacts

### How to Use

**Location**: `src/mlops/mlflow_tracking.py`

**Start MLflow Server**:
```powershell
# Activate venv
& D:\PROJECT\venv\Scripts\Activate.ps1

# Start MLflow UI
mlflow ui
```

**Access at**: `http://localhost:5000`

**What You'll See**:
- List of all training experiments
- Metrics comparison across runs
- Best model metrics
- Parameter configurations
- Model artifacts

### Accessing in CI/CD

GitHub Actions automatically logs training metrics. Access them:

1. Go to GitHub repo → Actions
2. Click on latest workflow run
3. Download "trained-models" artifact to see model files
4. Download "drift-report" artifact to see data quality status

---

## 4. 🔄 Complete Monitoring Workflow

### Daily/Automated Flow (CI/CD)

```
Push to GitHub
    ↓
1. Tests run (validate code)
    ↓
2. Lint check (code quality)
    ↓
3. Models train (new data)
    ↓
4. Drift detection runs (data quality check) ← ⚠️ This is where drift warnings occur
    ↓
5. Docker image builds
    ↓
6. Deploy to production
    ↓
7. Dashboard updates with latest models
```

### Manual Monitoring

**Check data quality**:
```powershell
python src/monitoring/check_drift.py
cat reports/drift_report.json
```

**View dashboard**:
```powershell
python src/dashboard/app.py
# Open http://localhost:8050
```

**View experiment logs**:
```powershell
mlflow ui
# Open http://localhost:5000
```

---

## 5. ⚠️ Handling the Drift Warning

### The Issue You Encountered

```
drift-detection
No files were found with the provided path: reports/drift_report.json. 
No artifacts will be uploaded.
```

### Root Cause

The drift script was creating timestamped filenames (`drift_report_20251125_120000.json`) but CI/CD was looking for a fixed filename (`drift_report.json`).

### Solution ✅

**Fixed in `src/monitoring/check_drift.py`**:
- Now creates **both** timestamped AND fixed-name files
- `reports/drift_report.json` - Latest report (CI/CD uses this)
- `reports/drift_report_YYYYMMDD_HHMMSS.json` - Historical archive

### What to Do Now

1. **Commit the fix**:
```powershell
git add src/monitoring/check_drift.py
git commit -m "Fix: save drift report with both timestamped and fixed names"
git push origin main
```

2. **Monitor the CI run**: Check Actions to confirm artifact uploads successfully

3. **Review the reports**:
```powershell
# View latest drift report
Get-Content reports/drift_report.json | ConvertFrom-Json | ConvertTo-Json
```

---

## 6. 📋 Monitoring Best Practices

### For Your Project

| Task | Frequency | Command | Purpose |
|------|-----------|---------|---------|
| **Check drift** | After each data batch | `python src/monitoring/check_drift.py` | Detect data quality changes |
| **View dashboard** | Daily/Weekly | `python src/dashboard/app.py` | Monitor population health |
| **Review experiments** | Weekly | `mlflow ui` | Track model performance |
| **Retrain models** | When drift detected | Triggered by CI or manual | Keep models accurate |

### Alert Thresholds

Set up alerts when:
- ⚠️ Drift detected in >1 dataset → Manual review needed
- 🔴 Drift detected in >2 datasets → Immediate retraining recommended
- 📊 Dashboard shows high-risk spike → Investigate data source

---

## 7. 🚀 Next Steps

1. **Test the fix** - Run drift detection locally:
   ```powershell
   python src/monitoring/check_drift.py
   ```

2. **Verify CI/CD** - Push and check Actions for successful artifact upload

3. **Run dashboard** - Test visualization:
   ```powershell
   python src/dashboard/app.py
   ```

4. **Create monitoring schedule** - Decide how often to check drift, review dashboards, retrain

---

## 📚 Files Reference

| File | Purpose | Location |
|------|---------|----------|
| `check_drift.py` | Drift detection runner | `src/monitoring/check_drift.py` |
| `drift_detector.py` | Drift detection logic | `src/monitoring/drift_detector.py` |
| `app.py` | Dashboard application | `src/dashboard/app.py` |
| `mlflow_tracking.py` | Experiment logging | `src/mlops/mlflow_tracking.py` |
| CI/CD workflow | Automation pipeline | `.github/workflows/mlops-pipeline.yml` |

---

**Summary**: Your project has **three-layer monitoring** ready to use. The warning you saw is now fixed. Start by testing drift detection, then explore the dashboard!
