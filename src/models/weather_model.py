"""
Health Risk Prediction Model for Weather Data
Predicts health risk levels based on weather conditions
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


class WeatherHealthRiskModel:
    """Model for predicting health risks from weather data"""

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
        Prepare features from raw weather data

        Args:
            df: DataFrame with weather data

        Returns:
            Feature matrix X
        """
        df_processed = df.copy()

        # Time-based features
        df_processed["hour"] = df_processed["timestamp"].dt.hour
        df_processed["day_of_week"] = df_processed["timestamp"].dt.dayofweek
        df_processed["is_weekend"] = (df_processed["day_of_week"] >= 5).astype(int)
        df_processed["month"] = df_processed["timestamp"].dt.month

        # Extract weather features
        weather_features = [
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "visibility",
        ]
        weather_features = [
            col for col in weather_features if col in df_processed.columns
        ]

        # Create additional features
        # Heat index approximation
        if "temperature" in df_processed.columns and "humidity" in df_processed.columns:
            T = df_processed["temperature"]
            H = df_processed["humidity"]
            # Simplified heat index
            df_processed["heat_index"] = (
                -8.78469475556
                + 1.61139411 * T
                + 2.33854883889 * H
                + -0.14611605 * T * H
                + -0.012308094 * T**2
                + -0.0164248277778 * H**2
                + 0.002211732 * T**2 * H
                + 0.00072546 * T * H**2
                + -0.000003582 * T**2 * H**2
            )

        # Comfort index (temperature and humidity combination)
        if "temperature" in df_processed.columns and "humidity" in df_processed.columns:
            # Ideal temp range: 18-24°C, ideal humidity: 40-60%
            temp_comfort = np.where(
                (df_processed["temperature"] >= 18)
                & (df_processed["temperature"] <= 24),
                1,
                0,
            )
            humidity_comfort = np.where(
                (df_processed["humidity"] >= 40) & (df_processed["humidity"] <= 60),
                1,
                0,
            )
            df_processed["comfort_index"] = (temp_comfort + humidity_comfort) / 2

        # Weather condition encoding (if exists)
        if "weather_condition" in df_processed.columns:
            weather_encoder = LabelEncoder()
            df_processed["weather_condition_encoded"] = weather_encoder.fit_transform(
                df_processed["weather_condition"]
            )

        # Select final features
        feature_cols = weather_features + ["hour", "day_of_week", "is_weekend", "month"]

        # Add engineered features if they exist
        if "heat_index" in df_processed.columns:
            feature_cols.append("heat_index")
        if "comfort_index" in df_processed.columns:
            feature_cols.append("comfort_index")
        if "weather_condition_encoded" in df_processed.columns:
            feature_cols.append("weather_condition_encoded")

        # Remove any columns that don't exist
        feature_cols = [col for col in feature_cols if col in df_processed.columns]

        X = df_processed[feature_cols].copy()
        self.feature_names = feature_cols

        return X

    def _derive_health_risk(self, df):
        """
        Derive health risk levels from weather conditions
        This creates a target variable if it doesn't exist
        """
        risk_levels = []

        for idx, row in df.iterrows():
            temp = row.get("temperature", 20)
            humidity = row.get("humidity", 50)
            wind = row.get("wind_speed", 5)
            condition = row.get("weather_condition", "normal")

            # Determine risk based on weather conditions
            risk_score = 0

            # Temperature extremes
            if temp < 0 or temp > 35:
                risk_score += 2
            elif temp < 5 or temp > 30:
                risk_score += 1

            # High humidity
            if humidity > 80:
                risk_score += 1
            elif humidity < 20:
                risk_score += 1

            # Extreme weather
            if condition in ["cold", "hot"]:
                risk_score += 1
            elif condition == "rainy":
                risk_score += 0.5

            # Convert score to risk level
            if risk_score >= 3:
                risk = "high"
            elif risk_score >= 1.5:
                risk = "moderate"
            else:
                risk = "low"

            risk_levels.append(risk)

        return risk_levels

    def train(self, df, target_col=None, test_size=0.2):
        """
        Train the model

        Args:
            df: Training DataFrame
            target_col: Name of target column (if None, derives from weather)
            test_size: Fraction of data to use for testing

        Returns:
            dict: Training metrics
        """
        df_processed = df.copy()

        # Derive target if not provided
        if target_col is None or target_col not in df_processed.columns:
            print("Deriving health risk levels from weather conditions...")
            df_processed["health_risk_level"] = self._derive_health_risk(df_processed)
            target_col = "health_risk_level"

        # Prepare features and target
        X = self.prepare_features(df_processed)
        y = df_processed[target_col].copy()

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

        print("\nModel Performance:")
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
            df: DataFrame with weather data

        Returns:
            Predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = self.prepare_features(df)

        # Ensure X is numpy array and has correct shape
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Ensure feature names match (in case of version differences)
        # Fill missing features with 0 if needed
        if (
            hasattr(self.model, "feature_names_in_")
            and self.model.feature_names_in_ is not None
        ):
            # Model expects specific features
            expected_features = list(self.model.feature_names_in_)
            if self.feature_names and len(X[0]) != len(expected_features):
                # Try to align features
                X_df = pd.DataFrame(
                    X,
                    columns=(
                        self.feature_names[: len(X[0])] if self.feature_names else None
                    ),
                )
                # Add missing columns with zeros
                for feat in expected_features:
                    if feat not in X_df.columns:
                        X_df[feat] = 0
                # Reorder to match expected order
                X_df = X_df[[f for f in expected_features if f in X_df.columns]]
                X = X_df.values

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
            # If model is incompatible, provide helpful error message
            error_msg = str(e)
            if "monotonic_cst" in error_msg or "attribute" in error_msg.lower():
                raise ValueError(
                    f"Model version incompatibility detected. The model was trained with a different "
                    f"version of scikit-learn. Please retrain the model by running:\n"
                    f"  python src/models/train_models.py\n"
                    f"Original error: {error_msg}"
                )
            else:
                raise

        # Decode labels
        predictions_decoded = self.label_encoder.inverse_transform(predictions)

        return predictions_decoded, probabilities

    # In src/models/weather_model.py, find the save and load methods and replace them:

    # 💾 FIX: Corrected Save/Load methods to only use the directory argument and build the path internally
    def save(self, directory: str):
        """Saves the trained model and label encoder to the specified directory."""
        os.makedirs(directory, exist_ok=True)

        model_filename = f"weather_model_{self.model_type}.pkl"
        encoder_filename = "weather_label_encoder.pkl"

        joblib.dump(self.model, os.path.join(directory, model_filename))
        joblib.dump(self.label_encoder, os.path.join(directory, encoder_filename))

    def load(self, directory: str):
        """Loads a pre-trained model and label encoder."""
        model_filename = f"weather_model_{self.model_type}.pkl"
        encoder_filename = "weather_label_encoder.pkl"

        self.model = joblib.load(os.path.join(directory, model_filename))
        self.label_encoder = joblib.load(os.path.join(directory, encoder_filename))
