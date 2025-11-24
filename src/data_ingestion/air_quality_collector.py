"""
Air Quality Data Collector
Collects real-time air quality data from OpenAQ or simulates it
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Optional
import time


class AirQualityCollector:
    """Collect or simulate air quality sensor data"""

    def __init__(self, use_api: bool = False, api_key: Optional[str] = None):
        self.use_api = use_api
        self.api_key = api_key
        self.openaq_base_url = "https://api.openaq.org/v2/"

    def get_openaq_data(
        self,
        cities: List[str],
        parameters: List[str] = ["pm25", "pm10", "no2", "o3"],
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch air quality data from OpenAQ API

        Args:
            cities: List of city names
            parameters: List of pollutants to fetch
            days: Number of days of historical data
        """
        if not self.use_api:
            print("API mode disabled, generating simulated data...")
            return self.simulate_air_quality(cities, parameters, days)

        all_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        for city in cities:
            try:
                # OpenAQ API endpoint
                url = f"{self.openaq_base_url}measurements"
                params = {
                    "date_from": start_date.isoformat(),
                    "date_to": end_date.isoformat(),
                    "limit": 1000,
                    "page": 1,
                }

                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # Process and extract relevant data
                    # (Implementation depends on OpenAQ API response structure)
                else:
                    print(f"API request failed for {city}, using simulation")
                    city_data = self.simulate_city_air_quality(city, parameters, days)
                    all_data.append(city_data)

            except Exception as e:
                print(f"Error fetching data for {city}: {e}")
                city_data = self.simulate_city_air_quality(city, parameters, days)
                all_data.append(city_data)

            time.sleep(0.5)  # Rate limiting

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def simulate_city_air_quality(
        self, city: str, parameters: List[str], days: int
    ) -> pd.DataFrame:
        """Simulate air quality data for a city"""
        # Baseline AQI values by city (realistic approximations)
        city_baselines = {
            "New York": {"pm25": 12, "pm10": 20, "no2": 25, "o3": 50, "co": 0.5},
            "London": {"pm25": 10, "pm10": 18, "no2": 30, "o3": 60, "co": 0.4},
            "Tokyo": {"pm25": 15, "pm10": 22, "no2": 20, "o3": 55, "co": 0.3},
            "Delhi": {"pm25": 120, "pm10": 200, "no2": 80, "o3": 70, "co": 2.0},
            "Beijing": {"pm25": 80, "pm10": 140, "no2": 60, "o3": 65, "co": 1.5},
        }

        baseline = city_baselines.get(city, city_baselines["New York"])

        dates = [datetime.now() - timedelta(days=d) for d in range(days, 0, -1)]

        data = []
        for date in dates:
            # Add daily variation and trends
            day_of_week = date.weekday()
            hour = date.hour if len(dates) <= 24 else 12  # Use noon for daily data

            # Weekday vs weekend patterns
            weekday_mult = 1.1 if day_of_week < 5 else 0.9

            # Hourly patterns (higher during rush hours)
            if hour in [7, 8, 17, 18]:
                hour_mult = 1.2
            elif hour in [0, 1, 2, 3]:
                hour_mult = 0.8
            else:
                hour_mult = 1.0

            row = {
                "city": city,
                "timestamp": date,
                "latitude": np.random.normal(40.7, 0.1),  # Approximate
                "longitude": np.random.normal(-74.0, 0.1),
            }

            for param in parameters:
                if param in baseline:
                    base_value = baseline[param]
                    # Add noise and variation
                    value = base_value * weekday_mult * hour_mult
                    value += np.random.normal(0, base_value * 0.2)
                    value = max(0, value)  # Can't be negative
                    row[param] = round(value, 2)

            # Calculate AQI (simplified)
            row["aqi"] = self.calculate_aqi(row.get("pm25", 0))
            row["health_risk_level"] = self.aqi_to_risk_level(row["aqi"])

            data.append(row)

        return pd.DataFrame(data)

    def simulate_air_quality(
        self, cities: List[str], parameters: List[str], days: int
    ) -> pd.DataFrame:
        """Generate simulated air quality data for multiple cities"""
        all_data = []
        for city in cities:
            city_data = self.simulate_city_air_quality(city, parameters, days)
            all_data.append(city_data)

        return pd.concat(all_data, ignore_index=True)

    def calculate_aqi(self, pm25: float) -> int:
        """Calculate simplified AQI from PM2.5"""
        # Simplified AQI calculation
        if pm25 <= 12:
            return int(50 * pm25 / 12)
        elif pm25 <= 35.4:
            return int(50 + 50 * (pm25 - 12) / 23.4)
        elif pm25 <= 55.4:
            return int(100 + 50 * (pm25 - 35.4) / 20)
        elif pm25 <= 150.4:
            return int(150 + 100 * (pm25 - 55.4) / 95)
        else:
            return min(500, int(250 + 250 * (pm25 - 150.4) / 150))

    def aqi_to_risk_level(self, aqi: int) -> str:
        """Convert AQI to health risk level"""
        if aqi <= 50:
            return "good"
        elif aqi <= 100:
            return "moderate"
        elif aqi <= 150:
            return "unhealthy_sensitive"
        elif aqi <= 200:
            return "unhealthy"
        elif aqi <= 300:
            return "very_unhealthy"
        else:
            return "hazardous"

    def save_to_csv(self, data: pd.DataFrame, filepath: str):
        """Save air quality data to CSV"""
        data.to_csv(filepath, index=False)
        print(f"Saved {len(data)} air quality records to {filepath}")
