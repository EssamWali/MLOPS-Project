#!/usr/bin/env python
"""
Start All Services Script
Starts dashboard, MLflow, and provides instructions for federated learning
"""

import sys
import io
import subprocess
import time
import webbrowser
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def start_dashboard():
    """Start the dashboard"""
    print("[STARTING] Dashboard on http://localhost:8050...")
    try:
        # Start dashboard in background
        process = subprocess.Popen(
            [sys.executable, "src/dashboard/app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)  # Wait for server to start
        
        # Check if it's running
        import requests
        try:
            response = requests.get("http://localhost:8050", timeout=2)
            if response.status_code == 200:
                print("[SUCCESS] Dashboard is running at http://localhost:8050")
                print("[INFO] Opening dashboard in browser...")
                webbrowser.open("http://localhost:8050")
                return process
            else:
                print(f"[WARNING] Dashboard returned status {response.status_code}")
                return process
        except:
            print("[INFO] Dashboard process started (check manually at http://localhost:8050)")
            return process
    except Exception as e:
        print(f"[ERROR] Failed to start dashboard: {str(e)}")
        return None

def check_mlflow():
    """Check if MLflow is running"""
    print("\n[CHECKING] MLflow UI...")
    try:
        import requests
        response = requests.get("http://localhost:5000", timeout=2)
        if response.status_code == 200:
            print("[SUCCESS] MLflow UI is running at http://localhost:5000")
            print("[INFO] Opening MLflow in browser...")
            webbrowser.open("http://localhost:5000")
            return True
        else:
            print(f"[WARNING] MLflow returned status {response.status_code}")
            return False
    except:
        print("[INFO] MLflow is not running")
        print("[INFO] To start MLflow: mlflow ui --backend-store-uri file:///mlruns --port 5000")
        return False

def main():
    """Start all services"""
    print("\n" + "=" * 70)
    print("  STARTING ALL SERVICES")
    print("=" * 70 + "\n")
    
    # Start Dashboard
    dashboard_process = start_dashboard()
    
    # Check MLflow
    mlflow_running = check_mlflow()
    
    # Summary
    print("\n" + "=" * 70)
    print("  SERVICES STATUS")
    print("=" * 70)
    
    print(f"\nDashboard: {'[RUNNING]' if dashboard_process else '[NOT STARTED]'}")
    print("  URL: http://localhost:8050")
    
    print(f"\nMLflow: {'[RUNNING]' if mlflow_running else '[NOT RUNNING]'}")
    print("  URL: http://localhost:5000")
    if not mlflow_running:
        print("  To start: mlflow ui --backend-store-uri file:///mlruns --port 5000")
    
    print("\nFederated Learning:")
    print("  To run federated learning, use separate terminals:")
    print("    Terminal 1: python -m src.federated.federated_server")
    print("    Terminal 2: python -m src.federated.federated_client --node-id 0")
    
    print("\n" + "=" * 70)
    print("  SERVICES STARTED")
    print("=" * 70)
    print("\n[INFO] Press Ctrl+C to stop all services")
    
    try:
        if dashboard_process:
            dashboard_process.wait()
    except KeyboardInterrupt:
        print("\n[STOPPING] Services...")
        if dashboard_process:
            dashboard_process.terminate()
        print("[DONE] Services stopped")

if __name__ == "__main__":
    main()

