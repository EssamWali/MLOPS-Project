#!/usr/bin/env python
"""Quick setup verification script"""

import sys
from pathlib import Path

print("=" * 60)
print("SETUP VERIFICATION")
print("=" * 60)

# Check Python path
print(f"\n1. Python Path:")
print(f"   {sys.executable}")
if 'venv' in sys.executable:
    print("   ✓ Using virtual environment")
else:
    print("   ✗ NOT using virtual environment!")
    print("   Please run: source venv/bin/activate")

# Check key packages
print(f"\n2. Package Versions:")
try:
    import sklearn
    print(f"   scikit-learn: {sklearn.__version__}")
except ImportError:
    print("   ✗ scikit-learn not installed")

try:
    import pandas as pd
    print(f"   pandas: {pd.__version__}")
except ImportError:
    print("   ✗ pandas not installed")

try:
    import numpy as np
    print(f"   numpy: {np.__version__}")
except ImportError:
    print("   ✗ numpy not installed")

# Check data files
print(f"\n3. Data Files:")
data_dir = Path(__file__).parent / "data" / "raw"
files_to_check = [
    "wearable_data.csv",
    "air_quality_data.csv",
    "weather_data.csv"
]
for file in files_to_check:
    filepath = data_dir / file
    if filepath.exists():
        size = filepath.stat().st_size / 1024  # KB
        print(f"   ✓ {file} ({size:.1f} KB)")
    else:
        print(f"   ✗ {file} - MISSING")

# Check models
print(f"\n4. Trained Models:")
models_dir = Path(__file__).parent / "models"
model_files = [
    "wearable_model_gradient_boosting.pkl",
    "air_quality_model_random_forest.pkl",
    "weather_model_gradient_boosting.pkl"
]
for model_file in model_files:
    modelpath = models_dir / model_file
    if modelpath.exists():
        size = modelpath.stat().st_size / 1024  # KB
        print(f"   ✓ {model_file} ({size:.1f} KB)")
    else:
        print(f"   ✗ {model_file} - MISSING")

# Check Jupyter kernel
print(f"\n5. Jupyter Kernel:")
try:
    from jupyter_client import kernelspec
    kernels = kernelspec.find_kernel_specs()
    if 'mlops_project' in kernels:
        print(f"   ✓ 'mlops_project' kernel registered")
    else:
        print(f"   ✗ 'mlops_project' kernel not found")
        print(f"   Available kernels: {list(kernels.keys())}")
except ImportError:
    print("   ✗ jupyter_client not installed")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nIf any items are missing, run:")
print("  source venv/bin/activate")
print("  pip install -r requirements.txt")
print("  python src/data_ingestion/collect_data.py")
print("  python src/models/train_models.py")
print("\nFor notebooks:")
print("  1. Make sure venv is activated")
print("  2. Start Jupyter: jupyter notebook")
print("  3. In notebook: Kernel → Change Kernel → Python (MLOPS Project)")




