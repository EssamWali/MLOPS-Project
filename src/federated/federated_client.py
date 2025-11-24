"""
Federated Learning Client
Simulates a client/node in federated learning (e.g., hospital or city)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd

from ..models.wearable_model import WearableHealthRiskModel

# Note: torch imports removed - we use sklearn models, not PyTorch


class FederatedClient(fl.client.NumPyClient):
    """Federated learning client for wearable device data"""

    def __init__(self, node_id: int, data_path: Path, model_type="random_forest"):
        """
        Initialize federated client

        Args:
            node_id: Unique identifier for this client node
            data_path: Path to node's data file
            model_type: Type of model to use
        """
        self.node_id = node_id
        self.data_path = data_path
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Load data
        self._load_data()

        # Initialize model
        self._initialize_model()

    def _load_data(self):
        """Load node-specific data"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path, parse_dates=["timestamp"])

        # Prepare features and target
        model = WearableHealthRiskModel(model_type=self.model_type)
        X = model.prepare_features(df)
        y = df["health_condition"].copy()

        # Split into train and test
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        self.label_encoder = le

        print(
            f"Node {self.node_id}: Loaded {len(self.X_train)} training samples, "
            f"{len(self.X_test)} test samples"
        )

    def _initialize_model(self):
        """Initialize the model"""
        self.model = WearableHealthRiskModel(model_type=self.model_type)

        # Train on local data
        df = pd.read_csv(self.data_path, parse_dates=["timestamp"])
        self.model.train(df, target_col="health_condition", test_size=0.2)

    '''def get_parameters(self, config) -> List[np.ndarray]:
        """Return model parameters as NumPy arrays"""
        if self.model_type == 'random_forest':
            # For Random Forest, we extract feature importances as parameters
            if hasattr(self.model.model, 'feature_importances_'):
                return [self.model.model.feature_importances_]
            return [np.array([0])]
        
        elif self.model_type == 'gradient_boosting':
            if hasattr(self.model.model, 'feature_importances_'):
                return [self.model.model.feature_importances_]
            return [np.array([0])]
        
        elif self.model_type == 'logistic_regression':
            # For Logistic Regression, return coefficients
            if hasattr(self.model.model, 'coef_'):
                return [self.model.model.coef_.flatten(), 
                       self.model.model.intercept_]
            return [np.array([0]), np.array([0])]
        
        return [np.array([0])]'''

    # In src/federated/federated_client.py, locate and replace the get_parameters method:

    def get_parameters(self, config) -> List[np.ndarray]:
        """Return model parameters as NumPy arrays, using dummy zeros if model is untrained."""

        # Check for trained model weights/coefficients
        if (
            hasattr(self.model.model, "coef_")
            and self.model_type == "logistic_regression"
        ):
            # For Logistic Regression, return coefficients (flattened) and intercept
            # This is the actual expected shape (10 features + 1 bias for 1 class)
            return [self.model.model.coef_.flatten(), self.model.model.intercept_]

        # Fallback for feature-importance models (Random Forest/Gradient Boosting).
        # We assume 10 features based on your model design.
        elif hasattr(self.model.model, "feature_importances_"):
            return [self.model.model.feature_importances_]

        # 🟢 FINAL FALLBACK: If the model hasn't been trained yet (first round/crash),
        # return a dummy structure that MATCHES the expected full size (10 weights + 1 bias).
        # This prevents the server from crashing on an invalid list length.
        else:
            if self.model_type == "logistic_regression":
                # Expected output: [10 weights, 1 bias]
                return [np.zeros(10, dtype=np.float32), np.zeros(1, dtype=np.float32)]
            elif self.model_type in ["random_forest", "gradient_boosting"]:
                # Expected output: [10 feature importances]
                return [np.zeros(10, dtype=np.float32)]

            # Default safest fallback
            return [np.zeros(1, dtype=np.float32)]

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data

        Args:
            parameters: Global model parameters from server
            config: Configuration dictionary

        Returns:
            Updated parameters, number of samples, and metrics
        """
        # Update local model with global parameters if applicable
        # (For tree-based models, this is simplified)

        # Retrain on local data
        df = pd.read_csv(self.data_path, parse_dates=["timestamp"])
        metrics = self.model.train(df, target_col="health_condition", test_size=0.2)

        # Return updated parameters
        updated_params = self.get_parameters(config)
        num_samples = len(self.X_train)

        # Return metrics
        fit_metrics = {
            "node_id": self.node_id,
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
        }

        return updated_params, num_samples, fit_metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local test data

        Args:
            parameters: Model parameters
            config: Configuration dictionary

        Returns:
            Loss, number of samples, and metrics
        """
        # Evaluate on local test data
        predictions, probabilities = self.model.predict(
            pd.DataFrame(self.X_test, columns=self.model.feature_names)
        )

        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(
            self.y_test, self.label_encoder.transform(predictions)
        )

        # Calculate loss (1 - accuracy for simplicity)
        loss = 1.0 - accuracy
        num_samples = len(self.X_test)

        eval_metrics = {"node_id": self.node_id, "accuracy": accuracy}

        return float(loss), num_samples, eval_metrics


def start_client(
    node_id: int,
    server_address: str = "localhost:8080",
    data_path: Optional[Path] = None,
    model_type: str = "random_forest",
):
    """
    Start a federated learning client

    Args:
        node_id: Unique node identifier
        server_address: Server address (format: "host:port")
        data_path: Path to node's data file
        model_type: Type of model to use
    """
    if data_path is None:
        data_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "federated"
            / f"wearable_node_{node_id}.csv"
        )

    client = FederatedClient(node_id, data_path, model_type)

    print(f"Starting client {node_id}...")
    # Use new Flower API (to_client() method)
    try:
        fl.client.start_client(server_address=server_address, client=client.to_client())
    except AttributeError:
        # Fallback to deprecated API if to_client() not available
        fl.client.start_numpy_client(server_address=server_address, client=client)
