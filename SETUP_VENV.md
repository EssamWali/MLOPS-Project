# Virtual Environment Setup Guide

## Quick Setup (Run These Commands)

### 1. Activate Virtual Environment
```bash
cd /Users/faiqahmed/Desktop/Semesters/Semester7/MLOPS/PROJECT
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### 2. Install/Update Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter ipykernel
```

### 3. Register Jupyter Kernel with Virtual Environment
```bash
python -m ipykernel install --user --name=mlops_project --display-name "Python (MLOPS Project)"
```

### 4. Retrain Models (Important - Do This Once)
```bash
python src/data_ingestion/collect_data.py
python src/models/train_models.py
```

## Using Jupyter Notebooks

### Start Jupyter
```bash
# Make sure venv is activated first!
source venv/bin/activate
jupyter notebook
```

### Select Correct Kernel in Notebook
1. Open your notebook (e.g., `08_multimodal_model.ipynb`)
2. Click on **"Kernel"** → **"Change Kernel"** → Select **"Python (MLOPS Project)"**
3. This ensures the notebook uses your virtual environment

### If Kernel Doesn't Appear
```bash
source venv/bin/activate
python -m ipykernel install --user --name=mlops_project --display-name "Python (MLOPS Project)"
# Restart Jupyter
jupyter notebook
```

## Common Issues

### Issue: "ModuleNotFoundError" in Notebook
**Solution:** Make sure you selected the correct kernel (see above)

### Issue: "AttributeError: monotonic_cst" 
**Solution:** Retrain models with current environment:
```bash
source venv/bin/activate
python src/models/train_models.py
```

### Issue: Notebook uses wrong Python
**Solution:** Check kernel in notebook:
- Click **Kernel** → **Change Kernel** → Select **"Python (MLOPS Project)"**
- Or restart kernel: **Kernel** → **Restart**

## Verify Setup

### Check Python Path
```bash
source venv/bin/activate
which python
# Should show: .../PROJECT/venv/bin/python
```

### Check Installed Packages
```bash
source venv/bin/activate
pip list | grep scikit-learn
# Should show: scikit-learn 1.7.2
```

### Check Jupyter Kernel
```bash
source venv/bin/activate
jupyter kernelspec list
# Should show: mlops_project .../venv/share/jupyter/kernels/mlops_project
```

## Complete Setup Script

Run this once to set everything up:

```bash
cd /Users/faiqahmed/Desktop/Semesters/Semester7/MLOPS/PROJECT

# Activate venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter ipykernel

# Register kernel
python -m ipykernel install --user --name=mlops_project --display-name "Python (MLOPS Project)"

# Generate data
python src/data_ingestion/collect_data.py

# Train models
python src/models/train_models.py

# Start Jupyter
jupyter notebook
```

## Daily Workflow

Every time you open a new terminal:

```bash
cd /Users/faiqahmed/Desktop/Semesters/Semester7/MLOPS/PROJECT
source venv/bin/activate
jupyter notebook
```

**Remember:** Always activate the virtual environment before running Python scripts or Jupyter!




