#!/usr/bin/env python
"""
End-to-End Project Test Script
Tests all components of the Health Risk Prediction MLOps System
"""

import sys
import subprocess
from pathlib import Path
import traceback
import os

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"▶ {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        if result.returncode == 0:
            print(f"[OK] {description} - SUCCESS")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 5:
                    print("  Output (last 5 lines):")
                    for line in lines[-5:]:
                        print(f"    {line}")
                else:
                    print("  Output:")
                    for line in lines:
                        print(f"    {line}")
            return True
        else:
            print(f"[FAIL] {description} - FAILED")
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"[FAIL] {description} - ERROR: {str(e)}")
        traceback.print_exc()
        return False

def check_file_exists(filepath, description):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size / 1024  # KB
        print(f"[OK] {description}: {filepath} ({size:.1f} KB)")
        return True
    else:
        print(f"[FAIL] {description}: {filepath} - MISSING")
        return False

def main():
    """Main test function"""
    print_section("HEALTH RISK PREDICTION MLOPS SYSTEM - END-TO-END TEST")
    
    # Change to project root
    project_root = Path(__file__).parent
    import os
    os.chdir(project_root)
    print(f"Working directory: {project_root}")
    
    results = {}
    
    # Step 1: Check Setup
    print_section("STEP 1: Setup Verification")
    
    print("Checking Python version...")
    print(f"  Python: {sys.version}")
    
    print("\nChecking key packages...")
    packages = ['pandas', 'numpy', 'sklearn', 'dash', 'flwr', 'mlflow']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  [OK] {pkg}: {version}")
        except ImportError:
            print(f"  [FAIL] {pkg}: NOT INSTALLED")
            results['setup'] = False
            return
    
    results['setup'] = True
    print("\n[OK] Setup verification complete")
    
    # Step 2: Data Collection
    print_section("STEP 2: Data Collection")
    
    data_collected = run_command(
        "python src/data_ingestion/collect_data.py",
        "Collecting data from all sources"
    )
    results['data_collection'] = data_collected
    
    if data_collected:
        # Verify data files
        print("\nVerifying data files...")
        data_files = [
            ("data/raw/wearable_data.csv", "Wearable data"),
            ("data/raw/air_quality_data.csv", "Air quality data"),
            ("data/raw/weather_data.csv", "Weather data"),
        ]
        for filepath, desc in data_files:
            check_file_exists(filepath, desc)
    
    # Step 3: Model Training
    print_section("STEP 3: Model Training")
    
    models_trained = run_command(
        "python src/models/train_models.py",
        "Training all models"
    )
    results['model_training'] = models_trained
    
    if models_trained:
        # Verify model files
        print("\nVerifying trained models...")
        model_files = [
            ("models/wearable_model_gradient_boosting.pkl", "Wearable model"),
            ("models/air_quality_model_random_forest.pkl", "Air quality model"),
            ("models/weather_model_random_forest.pkl", "Weather model"),
        ]
        for filepath, desc in model_files:
            check_file_exists(filepath, desc)
    
    # Step 4: Test Multi-Modal Model
    print_section("STEP 4: Multi-Modal Model Test")
    
    try:
        print("Testing multi-modal model import and basic functionality...")
        sys.path.insert(0, str(project_root / "src"))
        from models.multimodal_model import MultiModalHealthRiskModel
        import pandas as pd
        
        # Load a small sample of data
        wearable_df = pd.read_csv("data/raw/wearable_data.csv", parse_dates=['timestamp'])
        air_quality_df = pd.read_csv("data/raw/air_quality_data.csv", parse_dates=['timestamp'])
        weather_df = pd.read_csv("data/raw/weather_data.csv", parse_dates=['timestamp'])
        
        # Initialize model
        multimodal_model = MultiModalHealthRiskModel(strategy='ensemble')
        multimodal_model.load_individual_models('models/')
        
        # Make a prediction
        sample_size = min(5, len(wearable_df), len(air_quality_df), len(weather_df))
        predictions, probabilities = multimodal_model.predict(
            df_wearable=wearable_df.head(sample_size),
            df_air_quality=air_quality_df.head(sample_size),
            df_weather=weather_df.head(sample_size)
        )
        
        print(f"[OK] Multi-modal model test - SUCCESS")
        print(f"  Generated {len(predictions)} predictions")
        print(f"  Prediction distribution: {pd.Series(predictions).value_counts().to_dict()}")
        results['multimodal'] = True
        
    except Exception as e:
        print(f"[FAIL] Multi-modal model test - FAILED: {str(e)}")
        traceback.print_exc()
        results['multimodal'] = False
    
    # Step 5: Test Drift Detection
    print_section("STEP 5: Drift Detection Test")
    
    drift_test = run_command(
        "python src/monitoring/check_drift.py",
        "Running drift detection",
        check=False  # Don't fail if drift detected
    )
    results['drift_detection'] = drift_test
    
    # Step 6: Test Dashboard Import
    print_section("STEP 6: Dashboard Test")
    
    try:
        print("Testing dashboard import and initialization...")
        sys.path.insert(0, str(project_root / "src"))
        from dashboard.app import app
        
        # Check if app is properly initialized
        if app is not None:
            print("[OK] Dashboard app initialized successfully")
            print("  Note: To run the dashboard, use: python src/dashboard/app.py")
            print("  Then open: http://localhost:8050")
            results['dashboard'] = True
        else:
            print("[FAIL] Dashboard app not initialized")
            results['dashboard'] = False
            
    except Exception as e:
        print(f"[FAIL] Dashboard test - FAILED: {str(e)}")
        traceback.print_exc()
        results['dashboard'] = False
    
    # Step 7: Test Federated Learning Import
    print_section("STEP 7: Federated Learning Test")
    
    try:
        print("Testing federated learning imports...")
        sys.path.insert(0, str(project_root / "src"))
        from federated.federated_server import FederatedServer
        from federated.federated_client import FederatedClient
        
        print("[OK] Federated learning modules imported successfully")
        print("  Note: To run federated learning:")
        print("    1. Start server: python src/federated/federated_server.py")
        print("    2. Start clients: python src/federated/federated_client.py --node-id 0")
        results['federated'] = True
        
    except Exception as e:
        print(f"[FAIL] Federated learning test - FAILED: {str(e)}")
        traceback.print_exc()
        results['federated'] = False
    
    # Final Summary
    print_section("TEST SUMMARY")
    
    print("\nTest Results:")
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 70)
        print("  [SUCCESS] ALL TESTS PASSED! Project is working correctly.")
        print("=" * 70)
        print("\nNext Steps:")
        print("  1. Run dashboard: python src/dashboard/app.py")
        print("  2. View MLflow: mlflow ui")
        print("  3. Test federated learning (see FEDERATED_LEARNING_GUIDE.md)")
        print("  4. Deploy with Docker: docker-compose -f docker/docker-compose.yml up")
    else:
        print("\n" + "=" * 70)
        print("  [WARNING] SOME TESTS FAILED. Please check the errors above.")
        print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

