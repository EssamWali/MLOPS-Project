"""
Script to check for data drift
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.drift_detector import DataDriftDetector


def check_all_drift():
    """Check drift for all datasets"""

    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    reports_dir = Path(__file__).parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    drift_results = {}

    print("=" * 80)
    print("DATA DRIFT DETECTION")
    print("=" * 80)

    # Check if data files exist
    wearable_ref = data_dir / "wearable_data.csv"
    air_quality_ref = data_dir / "air_quality_data.csv"
    weather_ref = data_dir / "weather_data.csv"

    data_files_exist = all(
        [wearable_ref.exists(), air_quality_ref.exists(), weather_ref.exists()]
    )

    if not data_files_exist:
        print("\n[WARNING] Data files not found (expected in CI environment)")
        print("   Creating empty drift report for CI/CD compatibility...")
        drift_results = {
            "wearable": {"drift_detected": False, "status": "Data unavailable"},
            "air_quality": {"drift_detected": False, "status": "Data unavailable"},
            "weather": {"drift_detected": False, "status": "Data unavailable"},
        }
    else:
        # Check wearable data drift
        print("\n1. Checking wearable device data drift...")
        if wearable_ref.exists():
            ref_data = pd.read_csv(wearable_ref, parse_dates=["timestamp"])
            # For simulation, use subset as "current" data
            curr_data = ref_data.tail(100).copy()
            ref_data = ref_data.head(len(ref_data) - 100).copy()

            detector = DataDriftDetector(ref_data, threshold=0.7)
            results = detector.detect_drift(curr_data)

            drift_results["wearable"] = results
            print(f"  Drift detected: {results['drift_detected']}")
            print(
                f"  Drifted features: {results['num_drifted_features']}/{results['total_features']}"
            )

        # Check air quality data drift
        print("\n2. Checking air quality data drift...")
        if air_quality_ref.exists():
            ref_data = pd.read_csv(air_quality_ref, parse_dates=["timestamp"])
            curr_data = ref_data.tail(20).copy()
            ref_data = ref_data.head(len(ref_data) - 20).copy()

            detector = DataDriftDetector(ref_data, threshold=0.7)
            results = detector.detect_drift(curr_data)

            drift_results["air_quality"] = results
            print(f"  Drift detected: {results['drift_detected']}")
            print(
                f"  Drifted features: {results['num_drifted_features']}/{results['total_features']}"
            )

        # Check weather data drift
        print("\n3. Checking weather data drift...")
        if weather_ref.exists():
            ref_data = pd.read_csv(weather_ref, parse_dates=["timestamp"])
            curr_data = ref_data.tail(20).copy()
            ref_data = ref_data.head(len(ref_data) - 20).copy()

            detector = DataDriftDetector(ref_data, threshold=0.7)
            results = detector.detect_drift(curr_data)

            drift_results["weather"] = results
            print(f"  Drift detected: {results['drift_detected']}")
            print(
                f"  Drifted features: {results['num_drifted_features']}/{results['total_features']}"
            )

    # Save results
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "data_available": data_files_exist,
        "results": drift_results,
    }

    # Save both timestamped and fixed-name versions for CI/CD compatibility
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path_timestamped = reports_dir / f"drift_report_{timestamp_str}.json"
    report_path_latest = reports_dir / "drift_report.json"  # For CI/CD artifact upload

    with open(report_path_timestamped, "w") as f:
        json.dump(drift_report, f, indent=2, default=str)

    # Also save as fixed filename for CI/CD to find
    with open(report_path_latest, "w") as f:
        json.dump(drift_report, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("DRIFT DETECTION COMPLETE")
    print("=" * 80)
    print(f"\nReport saved to: {report_path_timestamped}")
    print(f"Latest report: {report_path_latest}")

    # Summary
    total_drift = sum(1 for r in drift_results.values() if r["drift_detected"])
    print(f"\nSummary: Drift detected in {total_drift}/{len(drift_results)} datasets")

    return drift_report


if __name__ == "__main__":
    check_all_drift()
