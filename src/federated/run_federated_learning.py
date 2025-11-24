"""
Script to run federated learning simulation
Starts server and multiple clients
"""

import subprocess
import sys
import time
from pathlib import Path
import multiprocessing


def run_server(rounds=10, num_clients=5):
    """Run federated learning server"""
    from federated_server import start_federated_server
    start_federated_server(num_clients=num_clients, rounds=rounds)


def run_client(node_id, server_address="localhost:8080"):
    """Run a federated learning client"""
    from federated_client import start_client
    
    data_path = Path(__file__).parent.parent.parent / "data" / "federated" / f"wearable_node_{node_id}.csv"
    start_client(node_id=node_id, server_address=server_address, data_path=data_path)


def simulate_federated_learning(num_clients=5, rounds=10):
    """
    Simulate federated learning with server and multiple clients
    
    Args:
        num_clients: Number of client nodes
        rounds: Number of federated learning rounds
    """
    print("=" * 80)
    print("FEDERATED LEARNING SIMULATION")
    print("=" * 80)
    print(f"Number of clients: {num_clients}")
    print(f"Number of rounds: {rounds}")
    print()
    
    # Note: In a real scenario, clients would run on different machines
    # For simulation, we'll run them sequentially
    print("Starting federated learning server...")
    print("Note: In production, clients would connect from different nodes")
    print()
    
    # Start server (in a separate process)
    server_process = multiprocessing.Process(
        target=run_server,
        args=(rounds, num_clients)
    )
    server_process.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Start clients (in separate processes)
    client_processes = []
    for node_id in range(num_clients):
        print(f"Starting client {node_id}...")
        client_proc = multiprocessing.Process(
            target=run_client,
            args=(node_id, "localhost:8080")
        )
        client_proc.start()
        client_processes.append(client_proc)
        time.sleep(1)
    
    # Wait for all processes
    server_process.join()
    for client_proc in client_processes:
        client_proc.join()
    
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run federated learning simulation")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    
    args = parser.parse_args()
    
    simulate_federated_learning(num_clients=args.num_clients, rounds=args.rounds)


