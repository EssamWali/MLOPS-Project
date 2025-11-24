#!/usr/bin/env python
"""
Complete End-to-End Pipeline Test
Runs the entire project from data collection to dashboard
"""

import sys
import os
import io
import subprocess
import time
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def print_step(step_num, description):
    """Print a formatted step header"""
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {description}")
    print("=" * 70 + "\n")

def run_command(cmd, description, check=True):
    """Run a command and return success status"""
    print(f"[RUNNING] {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        print(f"[SUCCESS] {description}")
        if result.stdout:
            # Show last few lines if output is long
            lines = result.stdout.strip().split('\n')
            if len(lines) > 10:
                print("  Output (last 5 lines):")
                for line in lines[-5:]:
                    print(f"    {line}")
            else:
                print("  Output:")
                for line in lines:
                    print(f"    {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {description}")
        print(f"  Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] {description}: {str(e)}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size / 1024  # KB
        print(f"[OK] {description}: {filepath} ({size:.1f} KB)")
        return True
    else:
        print(f"[MISSING] {description}: {filepath}")
        return False

def main():
    """Run complete end-to-end pipeline"""
    print("\n" + "=" * 70)
    print("  COMPLETE END-TO-END PIPELINE TEST")
    print("  Health Risk Prediction MLOps System")
    print("=" * 70)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    results = {}
    
    # STEP 1: Data Collection
    print_step(1, "Data Collection")
    success = run_command(
        "python src/data_ingestion/collect_data.py",
        "Collecting data from all sources"
    )
    results['data_collection'] = success
    
    if success:
        # Verify data files
        print("\nVerifying data files:")
        check_file_exists("data/raw/wearable_data.csv", "Wearable data")
        check_file_exists("data/raw/air_quality_data.csv", "Air quality data")
        check_file_exists("data/raw/weather_data.csv", "Weather data")
    
    # STEP 2: Model Training
    print_step(2, "Model Training")
    success = run_command(
        "python src/models/train_models.py",
        "Training all models"
    )
    results['model_training'] = success
    
    if success:
        # Verify model files
        print("\nVerifying trained models:")
        check_file_exists("models/wearable_model_logistic_regression.pkl", "Wearable model")
        check_file_exists("models/air_quality_model_random_forest.pkl", "Air quality model")
        check_file_exists("models/weather_model_random_forest.pkl", "Weather model")
    
    # STEP 3: Multi-Modal Model Test
    print_step(3, "Multi-Modal Model Test")
    try:
        print("[RUNNING] Testing multi-modal model...")
        sys.path.insert(0, str(project_root / "src"))
        from models.multimodal_model import MultiModalHealthRiskModel
        import pandas as pd
        
        # Load data
        wearable_df = pd.read_csv("data/raw/wearable_data.csv", parse_dates=['timestamp'])
        air_quality_df = pd.read_csv("data/raw/air_quality_data.csv", parse_dates=['timestamp'])
        weather_df = pd.read_csv("data/raw/weather_data.csv", parse_dates=['timestamp'])
        
        # Initialize and test model
        multimodal_model = MultiModalHealthRiskModel(strategy='ensemble')
        multimodal_model.load_individual_models('models/')
        
        # Make predictions
        sample_size = min(10, len(wearable_df), len(air_quality_df), len(weather_df))
        predictions, probabilities = multimodal_model.predict(
            df_wearable=wearable_df.head(sample_size),
            df_air_quality=air_quality_df.head(sample_size),
            df_weather=weather_df.head(sample_size)
        )
        
        print(f"[SUCCESS] Multi-modal model test")
        print(f"  Generated {len(predictions)} predictions")
        print(f"  Prediction distribution: {pd.Series(predictions).value_counts().to_dict()}")
        results['multimodal'] = True
        
    except Exception as e:
        print(f"[FAILED] Multi-modal model test: {str(e)}")
        import traceback
        traceback.print_exc()
        results['multimodal'] = False
    
    # STEP 4: Drift Detection
    print_step(4, "Drift Detection")
    success = run_command(
        "python src/monitoring/check_drift.py",
        "Running drift detection",
        check=False  # Don't fail if drift is detected
    )
    results['drift_detection'] = success
    
    # STEP 5: Dashboard Verification
    print_step(5, "Dashboard Verification")
    try:
        print("[RUNNING] Testing dashboard initialization...")
        sys.path.insert(0, str(project_root / "src"))
        from dashboard.app import app
        
        if app is not None:
            print("[SUCCESS] Dashboard initialized")
            print("  Dashboard is ready to run")
            print("  To start: python src/dashboard/app.py")
            print("  Then open: http://localhost:8050")
            results['dashboard'] = True
        else:
            print("[FAILED] Dashboard not initialized")
            results['dashboard'] = False
            
    except Exception as e:
        print(f"[FAILED] Dashboard test: {str(e)}")
        import traceback
        traceback.print_exc()
        results['dashboard'] = False
    
    # Final Summary
    print("\n" + "=" * 70)
    print("  PIPELINE TEST SUMMARY")
    print("=" * 70)
    
    print("\nTest Results:")
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 70)
        print("  [SUCCESS] ALL STEPS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext Steps:")
        print("  1. Start dashboard: python src/dashboard/app.py")
        print("  2. View MLflow: mlflow ui")
        print("  3. Test federated learning (see FEDERATED_LEARNING_GUIDE.md)")
    else:
        print("\n" + "=" * 70)
        print("  [WARNING] SOME STEPS FAILED")
        print("=" * 70)
        print("\nPlease check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

