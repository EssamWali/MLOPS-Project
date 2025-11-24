#!/usr/bin/env python
"""
Verify that all services are running and accessible
"""

import sys
import io
import requests
import time
import subprocess
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_service(url, name, timeout=5):
    """Check if a service is accessible"""
    print(f"[CHECKING] {name} at {url}...")
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"[SUCCESS] {name} is running and accessible")
            return True
        else:
            print(f"[WARNING] {name} returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[NOT RUNNING] {name} is not accessible (connection refused)")
        return False
    except requests.exceptions.Timeout:
        print(f"[TIMEOUT] {name} did not respond in {timeout} seconds")
        return False
    except Exception as e:
        print(f"[ERROR] {name}: {str(e)}")
        return False

def main():
    """Verify all services"""
    print("\n" + "=" * 70)
    print("  SERVICE VERIFICATION")
    print("=" * 70 + "\n")
    
    results = {}
    
    # Check Dashboard
    print("1. Dashboard (http://localhost:8050)")
    results['dashboard'] = check_service("http://localhost:8050", "Dashboard")
    print()
    
    # Check MLflow
    print("2. MLflow UI (http://localhost:5000)")
    results['mlflow'] = check_service("http://localhost:5000", "MLflow UI")
    print()
    
    # Summary
    print("=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)
    
    for service, running in results.items():
        status = "[RUNNING]" if running else "[NOT RUNNING]"
        print(f"  {service:15s}: {status}")
    
    all_running = all(results.values())
    
    if not all_running:
        print("\n[INFO] To start services:")
        print("  Dashboard: python src/dashboard/app.py")
        print("  MLflow:    mlflow ui --backend-store-uri file:///mlruns --port 5000")
    
    return 0 if all_running else 1

if __name__ == "__main__":
    sys.exit(main())

