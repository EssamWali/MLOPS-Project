"""
Federated Learning Server
Coordinates federated learning across multiple nodes
"""

import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Optional
from flwr.server.strategy import FedAvg
import logging

logging.basicConfig(level=logging.INFO)


class FederatedServer:
    """Federated learning server"""
    
    def __init__(self, num_clients: int = 5, rounds: int = 10, 
                 min_fit_clients: int = 3, min_available_clients: int = 3):
        """
        Initialize federated server
        
        Args:
            num_clients: Number of expected clients
            rounds: Number of federated learning rounds
            min_fit_clients: Minimum number of clients for fitting
            min_available_clients: Minimum number of available clients
        """
        self.num_clients = num_clients
        self.rounds = rounds
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        
        # Define strategy
        self.strategy = FedAvg(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=self._fit_config,
            on_evaluate_config_fn=self._evaluate_config,
            fraction_fit=1.0,  # Use all available clients
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=self._aggregate_metrics
        )
    
    def _fit_config(self, rnd: int) -> Dict:
        """Return configuration for fit round"""
        return {
            "round": rnd,
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.01
        }
    
    def _evaluate_config(self, rnd: int) -> Dict:
        """Return configuration for evaluate round"""
        return {"round": rnd}
    
    def _aggregate_metrics(self, metrics: List[Tuple[int, Dict]]) -> Dict:
        """Aggregate metrics from multiple clients"""
        if not metrics:
            return {}
        
        # Aggregate accuracy and f1_score
        accuracies = []
        f1_scores = []
        
        for num_samples, metric_dict in metrics:
            if 'accuracy' in metric_dict:
                accuracies.append(metric_dict['accuracy'])
            if 'f1_score' in metric_dict:
                f1_scores.append(metric_dict['f1_score'])
        
        aggregated = {}
        if accuracies:
            aggregated['accuracy'] = np.mean(accuracies)
            aggregated['accuracy_std'] = np.std(accuracies)
        if f1_scores:
            aggregated['f1_score'] = np.mean(f1_scores)
            aggregated['f1_score_std'] = np.std(f1_scores)
        
        return aggregated
    
    def start(self, server_address: str = "0.0.0.0:8080"):
        """
        Start the federated learning server
        
        Args:
            server_address: Server address (format: "host:port")
        """
        print(f"Starting federated learning server on {server_address}")
        print(f"Expected clients: {self.num_clients}")
        print(f"Rounds: {self.rounds}")
        print(f"Min fit clients: {self.min_fit_clients}")
        print(f"Min available clients: {self.min_available_clients}")
        
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=self.rounds),
            strategy=self.strategy
        )


def start_federated_server(num_clients: int = 5, rounds: int = 10,
                          server_address: str = "0.0.0.0:8080"):
    """
    Convenience function to start federated server
    
    Args:
        num_clients: Number of expected clients
        rounds: Number of federated learning rounds
        server_address: Server address
    """
    server = FederatedServer(
        num_clients=num_clients,
        rounds=rounds
    )
    server.start(server_address)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    num_clients = 5
    rounds = 10
    
    if len(sys.argv) > 1:
        num_clients = int(sys.argv[1])
    if len(sys.argv) > 2:
        rounds = int(sys.argv[2])
    
    start_federated_server(num_clients=num_clients, rounds=rounds)


