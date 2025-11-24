"""
Data Drift Detection
Detects distribution shifts in data over time
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from evidently.metrics import DataDriftTable
    from evidently.report import Report
    from evidently import ColumnMapping
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("Warning: Evidently not available. Using basic drift detection.")


class DataDriftDetector:
    """Detects data drift in features"""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.7):
        """
        Initialize drift detector
        
        Args:
            reference_data: Reference dataset (baseline)
            threshold: Drift detection threshold (0-1)
        """
        self.reference_data = reference_data.copy()
        self.threshold = threshold
        self.feature_columns = None
        self._prepare_features()
    
    def _prepare_features(self):
        """Prepare feature columns for drift detection"""
        # Select numeric columns
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID and target columns
        exclude_cols = ['user_id', 'node_id', 'health_condition', 'health_risk_level', 
                       'timestamp', 'date', 'city']
        self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]
    
    def detect_drift_evidently(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect drift using Evidently AI
        
        Args:
            current_data: Current dataset to compare
            
        Returns:
            Drift detection results
        """
        if not EVIDENTLY_AVAILABLE:
            return self.detect_drift_basic(current_data)
        
        try:
            # Prepare data
            ref_data = self.reference_data[self.feature_columns].copy()
            curr_data = current_data[self.feature_columns].copy()
            
            # Remove any columns that don't exist in both
            common_cols = list(set(ref_data.columns) & set(curr_data.columns))
            ref_data = ref_data[common_cols]
            curr_data = curr_data[common_cols]
            
            # Create column mapping
            column_mapping = ColumnMapping()
            column_mapping.numerical_features = common_cols
            
            # Create drift report
            data_drift_table = DataDriftTable()
            data_drift_table.calculate(ref_data, curr_data, column_mapping)
            
            # Get drift results
            drift_metrics = data_drift_table.get_result()
            
            # Extract drift information
            drift_detected = drift_metrics.metrics.number_of_drifted_features > 0
            drifted_features = []
            
            for feature in common_cols:
                if hasattr(drift_metrics.metrics, 'drift_by_columns'):
                    feature_drift = drift_metrics.metrics.drift_by_columns.get(feature)
                    if feature_drift and feature_drift.drift_detected:
                        drifted_features.append({
                            'feature': feature,
                            'drift_score': feature_drift.drift_score,
                            'statistical_test': feature_drift.stattest_name
                        })
            
            results = {
                'drift_detected': drift_detected,
                'num_drifted_features': len(drifted_features),
                'total_features': len(common_cols),
                'drift_ratio': len(drifted_features) / len(common_cols) if common_cols else 0,
                'drifted_features': drifted_features,
                'threshold': self.threshold
            }
            
            return results
            
        except Exception as e:
            print(f"Error in Evidently drift detection: {e}")
            return self.detect_drift_basic(current_data)
    
    def detect_drift_basic(self, current_data: pd.DataFrame) -> Dict:
        """
        Basic drift detection using statistical tests
        
        Args:
            current_data: Current dataset to compare
            
        Returns:
            Drift detection results
        """
        from scipy import stats
        
        drifted_features = []
        
        # Check each feature
        for feature in self.feature_columns:
            if feature not in current_data.columns:
                continue
            
            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()
            
            if len(ref_values) < 10 or len(curr_values) < 10:
                continue
            
            # Kolmogorov-Smirnov test for distribution difference
            try:
                statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                
                # Consider drift if p-value < 0.05
                if p_value < 0.05:
                    # Calculate drift score (normalized)
                    drift_score = 1 - p_value
                    
                    # Calculate mean shift
                    mean_shift = abs(ref_values.mean() - curr_values.mean()) / (ref_values.std() + 1e-10)
                    
                    if drift_score > self.threshold or mean_shift > 2:
                        drifted_features.append({
                            'feature': feature,
                            'drift_score': drift_score,
                            'p_value': p_value,
                            'mean_shift': mean_shift,
                            'statistical_test': 'KS_test'
                        })
            except Exception as e:
                # Skip features that cause errors
                continue
        
        drift_detected = len(drifted_features) > 0
        total_features = len(self.feature_columns)
        
        results = {
            'drift_detected': drift_detected,
            'num_drifted_features': len(drifted_features),
            'total_features': total_features,
            'drift_ratio': len(drifted_features) / total_features if total_features > 0 else 0,
            'drifted_features': drifted_features,
            'threshold': self.threshold
        }
        
        return results
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect drift (uses Evidently if available, otherwise basic method)
        
        Args:
            current_data: Current dataset to compare
            
        Returns:
            Drift detection results
        """
        if EVIDENTLY_AVAILABLE:
            return self.detect_drift_evidently(current_data)
        else:
            return self.detect_drift_basic(current_data)
    
    def update_reference(self, new_reference_data: pd.DataFrame):
        """Update reference data with new baseline"""
        self.reference_data = new_reference_data.copy()
        self._prepare_features()
        print("Reference data updated")


class ModelPerformanceMonitor:
    """Monitors model performance over time"""
    
    def __init__(self, baseline_metrics: Dict[str, float], threshold: float = 0.1):
        """
        Initialize performance monitor
        
        Args:
            baseline_metrics: Baseline performance metrics
            threshold: Performance degradation threshold (relative)
        """
        self.baseline_metrics = baseline_metrics.copy()
        self.threshold = threshold
        self.performance_history = []
    
    def check_performance(self, current_metrics: Dict[str, float]) -> Dict:
        """
        Check if performance has degraded
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Performance check results
        """
        alerts = []
        degradations = {}
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            
            # Calculate relative change
            if baseline_value > 0:
                relative_change = (baseline_value - current_value) / baseline_value
            else:
                relative_change = 0
            
            degradations[metric_name] = {
                'baseline': baseline_value,
                'current': current_value,
                'change': baseline_value - current_value,
                'relative_change': relative_change
            }
            
            # Check if degradation exceeds threshold
            if relative_change > self.threshold:
                alerts.append({
                    'metric': metric_name,
                    'baseline': baseline_value,
                    'current': current_value,
                    'degradation': relative_change * 100,
                    'severity': 'high' if relative_change > 0.2 else 'medium'
                })
        
        # Store in history
        self.performance_history.append({
            'metrics': current_metrics,
            'timestamp': pd.Timestamp.now(),
            'alerts': alerts
        })
        
        results = {
            'performance_degradation': len(alerts) > 0,
            'num_alerts': len(alerts),
            'alerts': alerts,
            'degradations': degradations,
            'threshold': self.threshold
        }
        
        return results
    
    def get_performance_trend(self, metric_name: str) -> pd.DataFrame:
        """
        Get performance trend for a specific metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            DataFrame with trend data
        """
        trend_data = []
        
        for entry in self.performance_history:
            if metric_name in entry['metrics']:
                trend_data.append({
                    'timestamp': entry['timestamp'],
                    metric_name: entry['metrics'][metric_name],
                    'alerts': len(entry['alerts'])
                })
        
        if trend_data:
            return pd.DataFrame(trend_data)
        else:
            return pd.DataFrame()


def monitor_data_drift(reference_data_path: str, current_data_path: str,
                      threshold: float = 0.7) -> Dict:
    """
    Convenience function to monitor data drift
    
    Args:
        reference_data_path: Path to reference data
        current_data_path: Path to current data
        threshold: Drift threshold
        
    Returns:
        Drift detection results
    """
    ref_data = pd.read_csv(reference_data_path)
    curr_data = pd.read_csv(current_data_path)
    
    detector = DataDriftDetector(ref_data, threshold=threshold)
    results = detector.detect_drift(curr_data)
    
    return results


