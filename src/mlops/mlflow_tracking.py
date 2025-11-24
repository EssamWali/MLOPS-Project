"""
MLflow Experiment Tracking
Tracks experiments, models, and metrics
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class MLflowTracker:
    """MLflow experiment tracking wrapper"""

    def __init__(
        self,
        experiment_name: str = "health-risk-prediction",
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (default: file-based)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local file storage
            tracking_uri = str(Path(__file__).parent.parent.parent / "mlruns")
            mlflow.set_tracking_uri(f"file://{tracking_uri}")

        # Set or create experiment
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except:
            mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        print(f"MLflow tracking initialized: {tracking_uri}")

    def log_model_training(
        self,
        model,
        model_name: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        feature_names: Optional[list] = None,
        training_data_path: Optional[str] = None,
    ):
        """
        Log model training run

        Args:
            model: Trained model object
            model_name: Name of the model
            metrics: Dictionary of metrics to log
            parameters: Dictionary of parameters/hyperparameters
            feature_names: List of feature names
            training_data_path: Path to training data
        """
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(parameters)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Log feature names if provided
            if feature_names:
                mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")

            # Log training data path if provided
            if training_data_path:
                mlflow.log_param("training_data", training_data_path)

            print(f"✓ Logged model training: {model_name}")
            print(f"  Run ID: {mlflow.active_run().info.run_id}")
            print(f"  Metrics: {metrics}")

    def log_prediction(
        self,
        model_name: str,
        predictions: np.ndarray,
        input_data: pd.DataFrame,
        output_path: Optional[str] = None,
    ):
        """
        Log predictions

        Args:
            model_name: Name of the model
            predictions: Model predictions
            input_data: Input data used for prediction
            output_path: Path to save predictions
        """
        with mlflow.start_run(run_name=f"{model_name}_prediction"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("num_predictions", len(predictions))

            # Save and log predictions
            if output_path:
                pred_df = pd.DataFrame(
                    {
                        "prediction": predictions,
                        **{col: input_data[col].values for col in input_data.columns},
                    }
                )
                pred_df.to_csv(output_path, index=False)
                mlflow.log_artifact(output_path)

    def register_model(
        self, run_id: str, model_name: str, model_version: str = "latest"
    ):
        """
        Register model in MLflow model registry

        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            model_version: Model version
        """
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, model_name)
        print(f"✓ Registered model: {model_name} (version: {model_version})")

    def load_model(self, model_name: str, version: Optional[int] = None):
        """
        Load model from MLflow registry

        Args:
            model_name: Name of the registered model
            version: Model version (None for latest)

        Returns:
            Loaded model
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"

        model = mlflow.sklearn.load_model(model_uri)
        print(f"✓ Loaded model: {model_name} (version: {version or 'latest'})")
        return model

    def search_runs(self, filter_string: Optional[str] = None, max_results: int = 100):
        """
        Search previous runs

        Args:
            filter_string: MLflow filter string
            max_results: Maximum number of results

        Returns:
            DataFrame of runs
        """
        runs = mlflow.search_runs(filter_string=filter_string, max_results=max_results)
        return runs

    def get_best_run(self, metric: str = "f1_score", ascending: bool = False):
        """
        Get best run based on a metric

        Args:
            metric: Metric name to optimize
            ascending: True for ascending order

        Returns:
            Best run info
        """
        runs = mlflow.search_runs(max_results=100)

        if metric not in runs.columns:
            print(f"Metric '{metric}' not found in runs")
            return None

        best_run = runs.sort_values(by=metric, ascending=ascending).iloc[0]
        return best_run


def track_model_training(
    model,
    model_type: str,
    dataset_type: str,
    metrics: Dict[str, float],
    parameters: Dict[str, Any],
):
    """
    Convenience function to track model training

    Args:
        model: Trained model
        model_type: Type of model (e.g., 'random_forest')
        dataset_type: Type of dataset (e.g., 'wearable')
        metrics: Training metrics
        parameters: Model parameters
    """
    tracker = MLflowTracker()

    model_name = f"{dataset_type}_{model_type}"

    # Get feature names if available
    feature_names = None
    if hasattr(model, "feature_names"):
        feature_names = model.feature_names

    tracker.log_model_training(
        model=model.model if hasattr(model, "model") else model,
        model_name=model_name,
        metrics=metrics,
        parameters=parameters,
        feature_names=feature_names,
    )
