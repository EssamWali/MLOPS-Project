"""
Multi-Modal Health Risk Prediction Model
Combines predictions from wearable, air quality, and weather models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from .wearable_model import WearableHealthRiskModel
from .air_quality_model import AirQualityHealthRiskModel
from .weather_model import WeatherHealthRiskModel


class MultiModalHealthRiskModel:
    """Multi-modal model that combines predictions from all three data sources"""
    
    def __init__(self, strategy='ensemble'):
        """
        Initialize multi-modal model
        
        Args:
            strategy: 'ensemble' (voting), 'stacking', 'weighted_average'
        """
        self.strategy = strategy
        self.wearable_model = None
        self.air_quality_model = None
        self.weather_model = None
        self.meta_classifier = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_individual_models(self, models_dir):
        """Load the trained individual models"""
        models_path = Path(models_dir)
        
        # Load wearable model
        wearable_path = models_path / "wearable_model_gradient_boosting.pkl"
        if wearable_path.exists():
            self.wearable_model = WearableHealthRiskModel.load(str(wearable_path))
            print(f"✓ Loaded wearable model from {wearable_path}")
        else:
            print(f"✗ Wearable model not found at {wearable_path}")
        
        # Load air quality model (try different possible filenames)
        air_quality_paths = [
            models_path / "air_quality_model_random_forest.pkl",
            models_path / "air_quality_model_gradient_boosting.pkl",
            models_path / "air_quality_model_logistic_regression.pkl"
        ]
        air_quality_loaded = False
        for air_quality_path in air_quality_paths:
            if air_quality_path.exists():
                self.air_quality_model = AirQualityHealthRiskModel.load(str(air_quality_path))
                print(f"✓ Loaded air quality model from {air_quality_path}")
                air_quality_loaded = True
                break
        if not air_quality_loaded:
            print(f"✗ Air quality model not found. Tried: {[str(p) for p in air_quality_paths]}")
        
        # Load weather model (try different possible filenames)
        weather_paths = [
            models_path / "weather_model_gradient_boosting.pkl",
            models_path / "weather_model_random_forest.pkl",
            models_path / "weather_model_logistic_regression.pkl"
        ]
        weather_loaded = False
        for weather_path in weather_paths:
            if weather_path.exists():
                self.weather_model = WeatherHealthRiskModel.load(str(weather_path))
                print(f"✓ Loaded weather model from {weather_path}")
                weather_loaded = True
                break
        if not weather_loaded:
            print(f"✗ Weather model not found. Tried: {[str(p) for p in weather_paths]}")
    
    def _get_base_predictions(self, df_wearable, df_air_quality, df_weather):
        """
        Get predictions from all individual models
        
        Returns:
            tuple: (predictions_dict, probabilities_dict)
        """
        predictions = {}
        probabilities = {}
        
        # Get wearable predictions
        if self.wearable_model and df_wearable is not None and len(df_wearable) > 0:
            try:
                pred, prob = self.wearable_model.predict(df_wearable)
                predictions['wearable'] = pred
                probabilities['wearable'] = prob
            except Exception as e:
                print(f"Warning: Could not get wearable predictions: {e}")
                predictions['wearable'] = None
                probabilities['wearable'] = None
        
        # Get air quality predictions
        if self.air_quality_model and df_air_quality is not None and len(df_air_quality) > 0:
            try:
                pred, prob = self.air_quality_model.predict(df_air_quality)
                predictions['air_quality'] = pred
                probabilities['air_quality'] = prob
            except Exception as e:
                print(f"Warning: Could not get air quality predictions: {e}")
                predictions['air_quality'] = None
                probabilities['air_quality'] = None
        
        # Get weather predictions
        if self.weather_model and df_weather is not None and len(df_weather) > 0:
            try:
                pred, prob = self.weather_model.predict(df_weather)
                predictions['weather'] = pred
                probabilities['weather'] = prob
            except Exception as e:
                print(f"Warning: Could not get weather predictions: {e}")
                predictions['weather'] = None
                probabilities['weather'] = None
        
        return predictions, probabilities
    
    def _normalize_risk_levels(self, predictions_dict):
        """
        Normalize different risk level formats to a common scale
        Converts all predictions to: low, moderate, high
        """
        normalized = {}
        
        # Mapping from different formats to common format
        wearable_mapping = {
            'normal': 'low',
            'at_risk': 'moderate',
            'ill': 'high'
        }
        
        air_quality_mapping = {
            'good': 'low',
            'moderate': 'low',
            'unhealthy_sensitive': 'moderate',
            'unhealthy': 'high',
            'very_unhealthy': 'high',
            'hazardous': 'high'
        }
        
        weather_mapping = {
            'low': 'low',
            'moderate': 'moderate',
            'high': 'high'
        }
        
        if 'wearable' in predictions_dict and predictions_dict['wearable'] is not None:
            normalized['wearable'] = np.array([
                wearable_mapping.get(pred, 'moderate') for pred in predictions_dict['wearable']
            ])
        
        if 'air_quality' in predictions_dict and predictions_dict['air_quality'] is not None:
            normalized['air_quality'] = np.array([
                air_quality_mapping.get(pred, 'moderate') for pred in predictions_dict['air_quality']
            ])
        
        if 'weather' in predictions_dict and predictions_dict['weather'] is not None:
            normalized['weather'] = np.array([
                weather_mapping.get(pred, 'moderate') for pred in predictions_dict['weather']
            ])
        
        return normalized
    
    def predict_ensemble(self, df_wearable=None, df_air_quality=None, df_weather=None):
        """
        Predict using ensemble voting
        
        Args:
            df_wearable: Wearable device data
            df_air_quality: Air quality data
            df_weather: Weather data
            
        Returns:
            Final predictions and probabilities
        """
        predictions_dict, probabilities_dict = self._get_base_predictions(
            df_wearable, df_air_quality, df_weather
        )
        
        normalized_predictions = self._normalize_risk_levels(predictions_dict)
        
        # Determine number of samples (use first available)
        num_samples = None
        for key, preds in normalized_predictions.items():
            if preds is not None:
                num_samples = len(preds)
                break
        
        if num_samples is None:
            raise ValueError("No valid predictions from any model")
        
        # Combine predictions using majority voting
        final_predictions = []
        final_probabilities = []
        
        risk_levels = ['low', 'moderate', 'high']
        
        for i in range(num_samples):
            votes = {'low': 0, 'moderate': 0, 'high': 0}
            probs_sum = {'low': 0, 'moderate': 0, 'high': 0}
            count = 0
            
            # Collect votes from each model
            for key, preds in normalized_predictions.items():
                if preds is not None and i < len(preds):
                    vote = preds[i]
                    votes[vote] += 1
                    count += 1
                    
                    # Add probability if available
                    if key in probabilities_dict and probabilities_dict[key] is not None:
                        prob = probabilities_dict[key][i]
                        if hasattr(prob, '__iter__'):
                            # Handle different probability formats
                            for j, level in enumerate(risk_levels):
                                if j < len(prob):
                                    probs_sum[level] += prob[j]
            
            # Majority vote
            final_pred = max(votes, key=votes.get)
            final_predictions.append(final_pred)
            
            # Average probabilities
            if count > 0:
                avg_probs = [probs_sum[level] / count for level in risk_levels]
                final_probabilities.append(avg_probs)
            else:
                final_probabilities.append([1/3, 1/3, 1/3])
        
        return np.array(final_predictions), np.array(final_probabilities)
    
    def predict_weighted(self, df_wearable=None, df_air_quality=None, df_weather=None,
                        weights={'wearable': 0.5, 'air_quality': 0.3, 'weather': 0.2}):
        """
        Predict using weighted average of probabilities
        
        Args:
            df_wearable: Wearable device data
            df_air_quality: Air quality data
            df_weather: Weather data
            weights: Dictionary of weights for each model type
            
        Returns:
            Final predictions and probabilities
        """
        predictions_dict, probabilities_dict = self._get_base_predictions(
            df_wearable, df_air_quality, df_weather
        )
        
        normalized_predictions = self._normalize_risk_levels(predictions_dict)
        
        # Determine number of samples
        num_samples = None
        for key, preds in normalized_predictions.items():
            if preds is not None:
                num_samples = len(preds)
                break
        
        if num_samples is None:
            raise ValueError("No valid predictions from any model")
        
        risk_levels = ['low', 'moderate', 'high']
        final_predictions = []
        final_probabilities = []
        
        for i in range(num_samples):
            weighted_probs = np.zeros(len(risk_levels))
            total_weight = 0
            
            # Weight probabilities from each model
            for key in ['wearable', 'air_quality', 'weather']:
                if key in normalized_predictions and normalized_predictions[key] is not None:
                    if i < len(normalized_predictions[key]):
                        weight = weights.get(key, 0)
                        
                        if key in probabilities_dict and probabilities_dict[key] is not None:
                            prob = probabilities_dict[key][i]
                            if hasattr(prob, '__iter__'):
                                # Map probabilities to common risk levels
                                if len(prob) >= len(risk_levels):
                                    weighted_probs += np.array(prob[:len(risk_levels)]) * weight
                                    total_weight += weight
                                else:
                                    # Create probabilities from prediction
                                    pred = normalized_predictions[key][i]
                                    prob_vec = np.zeros(len(risk_levels))
                                    prob_vec[risk_levels.index(pred)] = 1.0
                                    weighted_probs += prob_vec * weight
                                    total_weight += weight
            
            # Normalize probabilities
            if total_weight > 0:
                weighted_probs = weighted_probs / total_weight
            else:
                weighted_probs = np.array([1/3, 1/3, 1/3])
            
            final_probabilities.append(weighted_probs)
            final_predictions.append(risk_levels[np.argmax(weighted_probs)])
        
        return np.array(final_predictions), np.array(final_probabilities)
    
    def predict(self, df_wearable=None, df_air_quality=None, df_weather=None, **kwargs):
        """
        Main prediction method
        
        Args:
            df_wearable: Wearable device data
            df_air_quality: Air quality data
            df_weather: Weather data
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            Final predictions and probabilities
        """
        if self.strategy == 'ensemble':
            return self.predict_ensemble(df_wearable, df_air_quality, df_weather)
        elif self.strategy == 'weighted_average':
            weights = kwargs.get('weights', {'wearable': 0.5, 'air_quality': 0.3, 'weather': 0.2})
            return self.predict_weighted(df_wearable, df_air_quality, df_weather, weights)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def evaluate(self, df_wearable, df_air_quality, df_weather, y_true):
        """
        Evaluate multi-modal model
        
        Args:
            df_wearable: Wearable device data
            df_air_quality: Air quality data
            df_weather: Weather data
            y_true: True labels (normalized to low/moderate/high)
            
        Returns:
            Evaluation metrics
        """
        y_pred, y_pred_proba = self.predict(df_wearable, df_air_quality, df_weather)
        
        # Encode labels
        all_labels = np.unique(np.concatenate([y_true, y_pred]))
        y_true_encoded = self.label_encoder.fit_transform(y_true)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted')
        recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted')
        f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
        
        try:
            roc_auc = roc_auc_score(y_true_encoded, y_pred_proba, 
                                   multi_class='ovr', average='weighted')
        except:
            roc_auc = None
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': classification_report(
                y_true_encoded, y_pred_encoded,
                target_names=self.label_encoder.classes_
            ),
            'confusion_matrix': confusion_matrix(y_true_encoded, y_pred_encoded)
        }
        
        return metrics
    
    def save(self, filepath):
        """Save the multi-modal model"""
        model_data = {
            'strategy': self.strategy,
            'wearable_model': self.wearable_model,
            'air_quality_model': self.air_quality_model,
            'weather_model': self.weather_model,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, filepath)
        print(f"Multi-modal model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, models_dir=None):
        """Load the multi-modal model"""
        model_data = joblib.load(filepath)
        
        instance = cls(strategy=model_data['strategy'])
        instance.wearable_model = model_data.get('wearable_model')
        instance.air_quality_model = model_data.get('air_quality_model')
        instance.weather_model = model_data.get('weather_model')
        instance.label_encoder = model_data.get('label_encoder', LabelEncoder())
        
        # If models not included, load them separately
        if models_dir and (instance.wearable_model is None or 
                          instance.air_quality_model is None or 
                          instance.weather_model is None):
            instance.load_individual_models(models_dir)
        
        return instance


