"""
Script to check for data drift
"""

import sys
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

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
    
    # Check wearable data drift
    print("\n1. Checking wearable device data drift...")
    wearable_ref = data_dir / "wearable_data.csv"
    wearable_curr = data_dir / "wearable_data.csv"  # In production, this would be new data
    
    if wearable_ref.exists():
        ref_data = pd.read_csv(wearable_ref, parse_dates=['timestamp'])
        # For simulation, use subset as "current" data
        curr_data = ref_data.tail(100).copy()
        ref_data = ref_data.head(len(ref_data) - 100).copy()
        
        detector = DataDriftDetector(ref_data, threshold=0.7)
        results = detector.detect_drift(curr_data)
        
        drift_results['wearable'] = results
        print(f"  Drift detected: {results['drift_detected']}")
        print(f"  Drifted features: {results['num_drifted_features']}/{results['total_features']}")
    
    # Check air quality data drift
    print("\n2. Checking air quality data drift...")
    air_quality_ref = data_dir / "air_quality_data.csv"
    
    if air_quality_ref.exists():
        ref_data = pd.read_csv(air_quality_ref, parse_dates=['timestamp'])
        curr_data = ref_data.tail(20).copy()
        ref_data = ref_data.head(len(ref_data) - 20).copy()
        
        detector = DataDriftDetector(ref_data, threshold=0.7)
        results = detector.detect_drift(curr_data)
        
        drift_results['air_quality'] = results
        print(f"  Drift detected: {results['drift_detected']}")
        print(f"  Drifted features: {results['num_drifted_features']}/{results['total_features']}")
    
    # Check weather data drift
    print("\n3. Checking weather data drift...")
    weather_ref = data_dir / "weather_data.csv"
    
    if weather_ref.exists():
        ref_data = pd.read_csv(weather_ref, parse_dates=['timestamp'])
        curr_data = ref_data.tail(20).copy()
        ref_data = ref_data.head(len(ref_data) - 20).copy()
        
        detector = DataDriftDetector(ref_data, threshold=0.7)
        results = detector.detect_drift(curr_data)
        
        drift_results['weather'] = results
        print(f"  Drift detected: {results['drift_detected']}")
        print(f"  Drifted features: {results['num_drifted_features']}/{results['total_features']}")
    
    # Save results
    drift_report = {
        'timestamp': datetime.now().isoformat(),
        'results': drift_results
    }
    
    report_path = reports_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(drift_report, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("DRIFT DETECTION COMPLETE")
    print("=" * 80)
    print(f"\nReport saved to: {report_path}")
    
    # Summary
    total_drift = sum(1 for r in drift_results.values() if r['drift_detected'])
    print(f"\nSummary: Drift detected in {total_drift}/{len(drift_results)} datasets")
    
    return drift_report


if __name__ == "__main__":
    check_all_drift()


