#!/usr/bin/env python
"""
Test Federated Learning - Quick Test
Runs a minimal federated learning test
"""

import sys
import io
import time
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def test_federated_imports():
    """Test that federated learning modules can be imported"""
    print("[TESTING] Federated Learning Module Imports...")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test server import
        from federated.federated_server import FederatedServer
        print("[OK] FederatedServer imported successfully")
        
        # Test client import
        from federated.federated_client import FederatedClient, start_client
        print("[OK] FederatedClient imported successfully")
        
        # Test server initialization
        server = FederatedServer(num_clients=2, rounds=2)
        print("[OK] FederatedServer initialized successfully")
        print(f"  - Expected clients: {server.num_clients}")
        print(f"  - Rounds: {server.rounds}")
        
        return True
    except Exception as e:
        print(f"[FAILED] Import test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_federated_simulation():
    """Test federated learning simulation script"""
    print("\n[TESTING] Federated Learning Simulation Script...")
    try:
        # Check if data files exist
        data_dir = Path(__file__).parent / "data" / "federated"
        required_files = [f"wearable_node_{i}.csv" for i in range(5)]
        
        missing_files = []
        for file in required_files:
            if not (data_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"[WARNING] Missing data files: {missing_files}")
            print("[INFO] Run data collection first: python src/data_ingestion/collect_data.py")
            return False
        
        print("[OK] All required data files exist")
        print("[INFO] To run federated learning:")
        print("  1. Start server: python -m src.federated.federated_server")
        print("  2. Start clients: python -m src.federated.federated_client --node-id 0")
        print("  Or use: python src/federated/run_federated_learning.py")
        
        return True
    except Exception as e:
        print(f"[FAILED] Simulation test: {str(e)}")
        return False

def main():
    """Run federated learning tests"""
    print("\n" + "=" * 70)
    print("  FEDERATED LEARNING TEST")
    print("=" * 70 + "\n")
    
    results = {}
    
    # Test imports
    results['imports'] = test_federated_imports()
    
    # Test simulation setup
    results['simulation'] = test_federated_simulation()
    
    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    
    for test, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test:15s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n[SUCCESS] Federated learning components are ready")
        print("\n[INFO] To actually run federated learning:")
        print("  Option 1: Use the simulation script")
        print("    python src/federated/run_federated_learning.py --num-clients 3 --rounds 3")
        print("\n  Option 2: Run manually (in separate terminals)")
        print("    Terminal 1: python -m src.federated.federated_server")
        print("    Terminal 2: python -m src.federated.federated_client --node-id 0")
        print("    Terminal 3: python -m src.federated.federated_client --node-id 1")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

