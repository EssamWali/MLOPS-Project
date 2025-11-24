"""
Training script for all individual models
"""

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Fix Windows encoding issues
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.air_quality_model import AirQualityHealthRiskModel
from models.wearable_model import WearableHealthRiskModel
from models.weather_model import WeatherHealthRiskModel


def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_all_models():
    """Train all three models"""
    config = load_config()

    # Create models directory
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Create reports directory
    reports_dir = Path(__file__).parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    all_results = {}

    print("=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)

    # 1. Train Wearable Model
    print("\n" + "=" * 80)
    print("MODEL 1: Wearable Health Device Model")
    print("=" * 80)

    wearable_data_path = (
        Path(__file__).parent.parent.parent / "data" / "raw" / "wearable_data.csv"
    )
    if wearable_data_path.exists():
        df_wearable = pd.read_csv(wearable_data_path, parse_dates=["timestamp"])

        # Train with different model types
        model_types = ["random_forest", "gradient_boosting", "logistic_regression"]
        best_model = None
        best_score = 0
        best_type = None

        for model_type in model_types:
            print(f"\n--- Training {model_type.upper()} ---")
            model = WearableHealthRiskModel(model_type=model_type)
            metrics = model.train(df_wearable, target_col="health_condition")

            # Use .get() to avoid KeyErrors, defaulting to 0.0
            f1 = metrics.get("f1_score", 0.0)
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_type = model_type

            # Use .get() to safely pull all expected metrics
            all_results[f"wearable_{model_type}"] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1_score": f1,
                "roc_auc": metrics.get("roc_auc", 0.0),
            }

        # Save best model
        if best_model:
            # 🟢 FIX: Pass ONLY the directory path. The model's save method will handle the filename.
            best_model.save(str(models_dir))

            # Define the full path for logging/summary purposes
            model_path_filename = models_dir / f"wearable_model_{best_type}.pkl"

            print(
                f"\n[OK] Best wearable model ({best_type}) saved to {model_path_filename}"
            )
            all_results["wearable_best"] = {
                "model_type": best_type,
                "f1_score": best_score,
                "model_path": str(model_path_filename),
            }
    else:
        print(f"✗ Wearable data not found at {wearable_data_path}")

    # 2. Train Air Quality Model
    print("\n" + "=" * 80)
    print("MODEL 2: Air Quality Health Risk Model")
    print("=" * 80)

    air_quality_path = (
        Path(__file__).parent.parent.parent / "data" / "raw" / "air_quality_data.csv"
    )
    if air_quality_path.exists():
        df_air = pd.read_csv(air_quality_path, parse_dates=["timestamp"])

        # Train with different model types
        model_types = ["random_forest", "gradient_boosting", "logistic_regression"]
        best_model = None
        best_score = 0
        best_type = None

        for model_type in model_types:
            print(f"\n--- Training {model_type.upper()} ---")
            model = AirQualityHealthRiskModel(model_type=model_type)
            metrics = model.train(df_air, target_col="health_risk_level")

            f1 = metrics.get("f1_score", 0.0)
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_type = model_type

            # Use .get() to safely pull all expected metrics
            all_results[f"air_quality_{model_type}"] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1_score": f1,
                "roc_auc": metrics.get("roc_auc", 0.0),
            }

        # Save best model
        if best_model:
            # 🟢 FIX: Pass ONLY the directory path.
            best_model.save(str(models_dir))

            # Define the full path for logging/summary purposes
            model_path_filename = models_dir / f"air_quality_model_{best_type}.pkl"

            print(
                f"\n[OK] Best air quality model ({best_type}) saved to {model_path_filename}"
            )
            all_results["air_quality_best"] = {
                "model_type": best_type,
                "f1_score": best_score,
                "model_path": str(model_path_filename),
            }
    else:
        print(f"✗ Air quality data not found at {air_quality_path}")

    # 3. Train Weather Model
    print("\n" + "=" * 80)
    print("MODEL 3: Weather Health Risk Model")
    print("=" * 80)

    weather_path = (
        Path(__file__).parent.parent.parent / "data" / "raw" / "weather_data.csv"
    )
    if weather_path.exists():
        df_weather = pd.read_csv(weather_path, parse_dates=["timestamp"])

        # Train with different model types
        model_types = ["random_forest", "gradient_boosting", "logistic_regression"]
        best_model = None
        best_score = 0
        best_type = None

        for model_type in model_types:
            print(f"\n--- Training {model_type.upper()} ---")
            model = WeatherHealthRiskModel(model_type=model_type)
            metrics = model.train(
                df_weather, target_col=None
            )  # Derives risk from weather

            f1 = metrics.get("f1_score", 0.0)
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_type = model_type

            # Use .get() to safely pull all expected metrics
            all_results[f"weather_{model_type}"] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1_score": f1,
                "roc_auc": metrics.get("roc_auc", 0.0),
            }

        # Save best model
        if best_model:
            # 🟢 FIX: Pass ONLY the directory path.
            best_model.save(str(models_dir))

            # Define the full path for logging/summary purposes
            model_path_filename = models_dir / f"weather_model_{best_type}.pkl"

            print(
                f"\n[OK] Best weather model ({best_type}) saved to {model_path_filename}"
            )
            all_results["weather_best"] = {
                "model_type": best_type,
                "f1_score": best_score,
                "model_path": str(model_path_filename),
            }
    else:
        print(f"✗ Weather data not found at {weather_path}")

    # Save results summary
    results_path = (
        reports_dir
        / f"model_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {results_path}")

    # Print summary
    print("\nModel Performance Summary:")
    print("-" * 80)
    for key, value in all_results.items():
        if "best" in key:
            print(f"{key.replace('_', ' ').title()}:")
            print(f"  Model Type: {value.get('model_type', 'N/A')}")
            print(f"  F1-Score: {value.get('f1_score', 0):.4f}")
            print(f"  Path: {value.get('model_path', 'N/A')}")
            print()

    return all_results


if __name__ == "__main__":
    train_all_models()
# Replace the entire content of src/models/train_models.py with the code above.
