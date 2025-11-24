# Individual Model Training Summary

This document summarizes the training results for all three individual models in the Health Risk Prediction System.

## Overview

Three separate models were trained to predict health risks from different data sources:
1. **Wearable Device Model** - Predicts health conditions from fitness tracker data
2. **Air Quality Model** - Predicts health risk levels from air pollution data
3. **Weather Model** - Predicts health risk levels from weather conditions

## Model 1: Wearable Health Device Model

### Purpose
Predicts health conditions (normal, at_risk, ill) from wearable device metrics including:
- Heart rate
- Steps taken
- Sleep hours
- Calories burned
- Body temperature
- Stress level

### Training Results

| Model Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------|----------|-----------|--------|----------|---------|
| Random Forest | 0.8833 | 0.8778 | 0.8833 | 0.8788 | 0.9360 |
| **Gradient Boosting** | **0.8883** | **0.8838** | **0.8883** | **0.8848** | **0.9356** |
| Logistic Regression | 0.8900 | 0.8837 | 0.8900 | 0.8846 | 0.9432 |

### Best Model: Gradient Boosting
- **F1-Score**: 0.8848
- **Accuracy**: 0.8883
- **Saved Location**: `models/wearable_model_gradient_boosting.pkl`

### Performance by Class
- **Normal**: Excellent performance (precision: 0.91, recall: 0.95)
- **Ill**: Excellent performance (precision: 1.00, recall: 0.91)
- **At Risk**: Moderate performance (precision: 0.70, recall: 0.60)

### Key Insights
- Model performs best at detecting normal and ill conditions
- "At risk" class is more challenging to predict accurately
- All three model types showed similar performance, with Gradient Boosting slightly edging out

## Model 2: Air Quality Health Risk Model

### Purpose
Predicts health risk levels (good, moderate, unhealthy_sensitive, unhealthy, very_unhealthy, hazardous) from air quality data including:
- PM2.5, PM10
- NO2, O3, CO
- AQI (Air Quality Index)
- Temporal features

### Training Results

| Model Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------|----------|-----------|--------|----------|---------|
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | - |
| **Gradient Boosting** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | - |
| Logistic Regression | 0.7667 | 0.8514 | 0.7667 | 0.7414 | - |

### Best Model: Random Forest
- **F1-Score**: 1.0000 (Perfect score)
- **Accuracy**: 1.0000
- **Saved Location**: `models/air_quality_model_random_forest.pkl`

### Performance by Class
All risk levels achieved perfect classification:
- **Good**: 100% precision and recall
- **Moderate**: 100% precision and recall
- **Unhealthy**: 100% precision and recall
- **Very Unhealthy**: 100% precision and recall

### Key Insights
- Tree-based models (Random Forest and Gradient Boosting) achieved perfect classification
- High correlation between air quality metrics and health risk levels
- Simpler dataset with clear patterns enabled perfect separation

## Model 3: Weather Health Risk Model

### Purpose
Predicts health risk levels (low, moderate, high) from weather data including:
- Temperature
- Humidity
- Pressure
- Wind speed
- Visibility
- Weather conditions

### Training Results

| Model Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------|----------|-----------|--------|----------|---------|
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Gradient Boosting** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

### Best Model: Random Forest
- **F1-Score**: 1.0000 (Perfect score)
- **Accuracy**: 1.0000
- **ROC-AUC**: 1.0000
- **Saved Location**: `models/weather_model_random_forest.pkl`

### Performance by Class
All risk levels achieved perfect classification:
- **Low**: 100% precision and recall
- **Moderate**: 100% precision and recall
- **High**: 100% precision and recall

### Key Insights
- All model types achieved perfect classification
- Weather-derived risk levels showed clear separation
- The heuristic-based risk derivation creates predictable patterns

## Model Architecture Details

### Feature Engineering
Each model includes:
- **Time-based features**: hour, day_of_week, is_weekend, month
- **Engineered features**: ratios, deviations, combined indices
- **Normalized/scaled features**: appropriate preprocessing per model type

### Model Types Evaluated
1. **Random Forest**: Ensemble of decision trees
   - Hyperparameters: n_estimators=100, max_depth=10
   - Good for handling non-linear relationships
   
2. **Gradient Boosting**: Sequential ensemble
   - Hyperparameters: n_estimators=100, max_depth=5, learning_rate=0.1
   - Excellent for complex patterns
   
3. **Logistic Regression**: Linear classifier
   - Hyperparameters: max_iter=1000
   - Simple, interpretable baseline

### Preprocessing
- **StandardScaler**: Applied for Logistic Regression models
- **LabelEncoder**: Used for encoding categorical targets
- **Train/Test Split**: 80/20 split with stratification

## Saved Models

All trained models are serialized using joblib and saved in the `models/` directory:

1. `wearable_model_gradient_boosting.pkl` - Best wearable model
2. `air_quality_model_random_forest.pkl` - Best air quality model
3. `weather_model_random_forest.pkl` - Best weather model

Each saved model includes:
- Trained model object
- Scaler (if applicable)
- Label encoder
- Feature names
- Model type metadata

## Next Steps: Multi-Modal Model

The next phase will combine all three models into a unified multi-modal health risk prediction system that:
1. Takes inputs from all three data sources simultaneously
2. Leverages ensemble methods or fusion techniques
3. Provides comprehensive health risk predictions
4. Handles missing data from any source gracefully

## Usage Examples

### Loading and Using Models

```python
from models.wearable_model import WearableHealthRiskModel
from models.air_quality_model import AirQualityHealthRiskModel
from models.weather_model import WeatherHealthRiskModel

# Load models
wearable_model = WearableHealthRiskModel.load('models/wearable_model_gradient_boosting.pkl')
air_model = AirQualityHealthRiskModel.load('models/air_quality_model_random_forest.pkl')
weather_model = WeatherHealthRiskModel.load('models/weather_model_random_forest.pkl')

# Make predictions
predictions, probabilities = wearable_model.predict(df_wearable)
```

## Notes

- Models were trained on simulated/synthetic data
- Perfect scores on air quality and weather models may indicate overfitting or simple data patterns
- For production use, additional validation and testing on real-world data is recommended
- Model performance may vary with larger, more diverse datasets


