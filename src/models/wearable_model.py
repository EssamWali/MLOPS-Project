"""
Wearable Health Risk Model Definition and Feature Engineering
Used by both centralized training and federated clients.
"""

import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Import available sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class WearableHealthRiskModel:
    """
    Handles data processing, feature engineering, and model management
    for the Wearable Health Risk Prediction module.
    """

    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = self._get_model()
        self.label_encoder = LabelEncoder()
        # Define feature names explicitly for consistency during inference
        self.feature_names = [
            "heart_rate",
            "steps",
            "sleep_hours",
            "calories",
            "body_temperature",
            "stress_level",
            "hr_rolling_mean",
            "steps_rolling_mean",
            "day_of_week",
            "time_of_day",
        ]

    def _get_model(self):
        """Initializes and returns the specified model."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=self.random_state
            )
        # Default and best performing model from the summary
        return RandomForestClassifier(n_estimators=100, random_state=self.random_state)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates time-series features essential for the prediction model.
        """
        if df.empty:
            return pd.DataFrame(columns=self.feature_names)

        # 1. Ensure timestamp is datetime and set as index
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp", drop=True).sort_index()

        # 2. Extract Temporal Features
        df["day_of_week"] = df.index.dayofweek
        df["time_of_day"] = df.index.hour

        # 3. Create Interaction Feature
        df["steps_per_hr"] = df["steps"] / (df["heart_rate"].replace(0, 1))

        # 4. Drift Detection Features (Rolling Window on time-series)
        rolling_window = 7

        # 🟢 FIX: Use an integer window (window=7) for stability across fragmented FL data
        df["hr_rolling_mean"] = (
            df["heart_rate"]
            .rolling(window=rolling_window)
            .mean()
            .fillna(df["heart_rate"])
        )
        df["steps_rolling_mean"] = (
            df["steps"].rolling(window=rolling_window).mean().fillna(df["steps"])
        )

        # Select final features
        X = df[self.feature_names]

        # Handle NaNs created by rolling window at the start of the series by filling
        X = X.fillna(X.mean())

        return X

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for feature_engineering for use in client interface."""
        return self.feature_engineering(df)

    # In src/models/wearable_model.py, replace the entire 'train' method:

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "health_condition",
        test_size: float = 0.2,
    ) -> Dict[str, float]:
        """
        Trains the model and evaluates performance.
        Returns evaluation metrics including precision, recall, f1_score, and roc_auc.
        """
        X = self.feature_engineering(df)
        y = df[target_col].copy()

        if y.empty:
            print("Warning: Target data is empty after feature preparation.")
            return {}

        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y_encoded,
            )
        except ValueError as e:
            print(f"Training failed during splitting due to data imbalance: {e}")
            return {}

        # Train the model
        self.model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # Determine number of classes present in the test set
        n_classes = len(self.label_encoder.classes_)

        # 🟢 FIX: CONDITIONAL ROC_AUC CALCULATION
        if n_classes <= 2:
            # If 2 classes (binary), use y_proba[:, 1] (probability of the positive class)
            # This requires y_proba to be a 1D array, which solves the ValueError.
            # We use try/except as a fail-safe for the rare case where the model output is not two columns.
            try:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1], average="weighted")
            except ValueError:
                # Fallback for unexpected dimensionality (e.g., only one class in the training split)
                roc_auc = 0.0
        else:
            # If > 2 classes (multi-class), use the standard One-vs-Rest (OVR) method
            roc_auc = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted"
            )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc,  # Use the conditionally calculated score
        }

        return metrics

    def predict(self, X: pd.DataFrame) -> Tuple[List[str], List[float]]:
        # ... (Method remains unchanged from your previous version)
        if "timestamp" in X.columns:
            X_processed = self.feature_engineering(X)
        else:
            X_processed = X

        X_processed = X_processed[self.feature_names]

        y_pred_encoded = self.model.predict(X_processed)
        y_proba = self.model.predict_proba(X_processed)

        predictions = self.label_encoder.inverse_transform(y_pred_encoded)

        return predictions.tolist(), np.max(y_proba, axis=1).tolist()

    # 💾 FIX: Corrected Save/Load methods to only use the directory argument and build the path internally
    def save(self, directory: str):
        """Saves the trained model and label encoder to the specified directory."""
        os.makedirs(directory, exist_ok=True)  # Ensure directory exists

        model_filename = f"wearable_model_{self.model_type}.pkl"
        encoder_filename = "wearable_label_encoder.pkl"

        joblib.dump(self.model, os.path.join(directory, model_filename))
        joblib.dump(self.label_encoder, os.path.join(directory, encoder_filename))

    def load(self, directory: str):
        """Loads a pre-trained model and label encoder."""
        model_filename = f"wearable_model_{self.model_type}.pkl"
        encoder_filename = "wearable_label_encoder.pkl"

        self.model = joblib.load(os.path.join(directory, model_filename))
        self.label_encoder = joblib.load(os.path.join(directory, encoder_filename))


# Replace the entire content of src/models/wearable_model.py with the code above.
