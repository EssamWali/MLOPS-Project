"""
Health Risk Prediction Model for Air Quality Data
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class AirQualityHealthRiskModel:
    """Model for predicting health risks from air quality data"""

    def __init__(self, model_type="random_forest"):
        """
        Initialize the model

        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'logistic_regression'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None

        # Initialize model based on type
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000, random_state=42, multi_class="multinomial"
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def prepare_features(self, df):
        """
        Prepare features from raw air quality data

        Args:
            df: DataFrame with air quality data

        Returns:
            Feature matrix X
        """
        df_processed = df.copy()

        # Time-based features
        df_processed["hour"] = df_processed["timestamp"].dt.hour
        df_processed["day_of_week"] = df_processed["timestamp"].dt.dayofweek
        df_processed["is_weekend"] = (df_processed["day_of_week"] >= 5).astype(int)

        # Extract pollutant features
        pollutant_features = ["pm25", "pm10", "no2", "o3", "co", "aqi"]
        pollutant_features = [
            col for col in pollutant_features if col in df_processed.columns
        ]

        # Create additional features
        if "pm25" in df_processed.columns and "pm10" in df_processed.columns:
            df_processed["pm_ratio"] = df_processed["pm25"] / (df_processed["pm10"] + 1)

        if "no2" in df_processed.columns and "o3" in df_processed.columns:
            df_processed["no2_o3_ratio"] = df_processed["no2"] / (
                df_processed["o3"] + 1
            )

        # AQI categories (already calculated, but can use as feature)
        # Pollution index (weighted combination)
        if all(col in df_processed.columns for col in ["pm25", "pm10", "no2"]):
            df_processed["pollution_index"] = (
                df_processed["pm25"] * 0.4
                + df_processed["pm10"] * 0.3
                + df_processed["no2"] * 0.3
            )

        # Select final features
        feature_cols = pollutant_features + ["hour", "day_of_week", "is_weekend"]

        # Add engineered features if they exist
        if "pm_ratio" in df_processed.columns:
            feature_cols.append("pm_ratio")
        if "no2_o3_ratio" in df_processed.columns:
            feature_cols.append("no2_o3_ratio")
        if "pollution_index" in df_processed.columns:
            feature_cols.append("pollution_index")

        # Remove any columns that don't exist
        feature_cols = [col for col in feature_cols if col in df_processed.columns]

        X = df_processed[feature_cols].copy()
        self.feature_names = feature_cols

        return X

    def train(self, df, target_col="health_risk_level", test_size=0.2):
        """
        Train the model

        Args:
            df: Training DataFrame
            target_col: Name of target column
            test_size: Fraction of data to use for testing

        Returns:
            dict: Training metrics
        """
        # Prepare features and target
        X = self.prepare_features(df)
        y = df[target_col].copy()

        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Scale features (if needed for logistic regression)
        if self.model_type == "logistic_regression":
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # ROC AUC (multi-class)
        try:
            roc_auc = roc_auc_score(
                y_test, y_pred_proba, multi_class="ovr", average="weighted"
            )
        except:
            roc_auc = None

        # Get unique classes in test set for classification report
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        class_names = [self.label_encoder.classes_[i] for i in unique_classes]

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "classification_report": classification_report(
                y_test, y_pred, labels=unique_classes, target_names=class_names
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC-AUC: {roc_auc:.4f}")

        print("\nClassification Report:")
        print(metrics["classification_report"])

        return metrics

    def predict(self, df):
        """
        Predict health risks for new data

        Args:
            df: DataFrame with air quality data

        Returns:
            Predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = self.prepare_features(df)

        # Scale if needed
        if self.model_type == "logistic_regression":
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Handle potential version compatibility issues
        try:
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
        except (AttributeError, TypeError) as e:
            error_msg = str(e)
            if "monotonic_cst" in error_msg or "attribute" in error_msg.lower():
                raise ValueError(
                    f"Model version incompatibility detected. Please retrain the model:\n"
                    f"  python src/models/train_models.py\n"
                    f"Original error: {error_msg}"
                )
            else:
                raise

        # Decode labels
        predictions_decoded = self.label_encoder.inverse_transform(predictions)

        return predictions_decoded, probabilities

    # In src/models/air_quality_model.py (Locate and replace the save method)

    # In src/models/air_quality_model.py, find the save and load methods and replace them:

    # 💾 FIX: Corrected Save/Load methods to only use the directory argument and build the path internally
    def save(self, directory: str):
        """Saves the trained model and label encoder to the specified directory."""
        os.makedirs(directory, exist_ok=True)

        model_filename = f"air_quality_model_{self.model_type}.pkl"
        encoder_filename = "air_quality_label_encoder.pkl"

        joblib.dump(self.model, os.path.join(directory, model_filename))
        joblib.dump(self.label_encoder, os.path.join(directory, encoder_filename))

    def load(self, directory: str):
        """Loads a pre-trained model and label encoder."""
        model_filename = f"air_quality_model_{self.model_type}.pkl"
        encoder_filename = "air_quality_label_encoder.pkl"

        self.model = joblib.load(os.path.join(directory, model_filename))
        self.label_encoder = joblib.load(os.path.join(directory, encoder_filename))
