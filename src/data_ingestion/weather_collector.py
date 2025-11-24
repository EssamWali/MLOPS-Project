"""
Weather Data Collector
Collects weather data from OpenWeatherMap API or simulates it
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Optional
import time


class WeatherCollector:
    """Collect or simulate weather data"""

    def __init__(self, use_api: bool = False, api_key: Optional[str] = None):
        self.use_api = use_api
        self.api_key = api_key
        self.weather_base_url = "https://api.openweathermap.org/data/2.5/"

        # City coordinates for simulation
        self.city_coords = {
            "New York": {"lat": 40.7128, "lon": -74.0060},
            "London": {"lat": 51.5074, "lon": -0.1278},
            "Tokyo": {"lat": 35.6762, "lon": 139.6503},
            "Delhi": {"lat": 28.6139, "lon": 77.2090},
            "Beijing": {"lat": 39.9042, "lon": 116.4074},
        }

    def get_weather_data(self, cities: List[str], days: int = 30) -> pd.DataFrame:
        """
        Fetch weather data from OpenWeatherMap API or simulate

        Args:
            cities: List of city names
            days: Number of days of historical data
        """
        if not self.use_api or not self.api_key:
            print("API mode disabled, generating simulated weather data...")
            return self.simulate_weather_data(cities, days)

        all_data = []
        for city in cities:
            try:
                coords = self.city_coords.get(city)
                if not coords:
                    print(f"Coordinates not found for {city}, using simulation")
                    city_data = self.simulate_city_weather(city, days)
                    all_data.append(city_data)
                    continue

                # Historical weather API (requires subscription)
                # Using current weather as fallback
                url = f"{self.weather_base_url}weather"
                params = {
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                    "appid": self.api_key,
                    "units": "metric",
                }

                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # Process current weather (for real-time use)
                    # For historical, would need different endpoint
                else:
                    city_data = self.simulate_city_weather(city, days)
                    all_data.append(city_data)

            except Exception as e:
                print(f"Error fetching weather for {city}: {e}")
                city_data = self.simulate_city_weather(city, days)
                all_data.append(city_data)

            time.sleep(0.5)  # Rate limiting

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def simulate_city_weather(self, city: str, days: int) -> pd.DataFrame:
        """Simulate realistic weather data for a city"""
        # Seasonal base temperatures by city (approximate averages)
        city_climates = {
            "New York": {"temp": 15, "humidity": 65, "pressure": 1013},
            "London": {"temp": 12, "humidity": 75, "pressure": 1015},
            "Tokyo": {"temp": 18, "humidity": 70, "pressure": 1012},
            "Delhi": {"temp": 28, "humidity": 55, "pressure": 1008},
            "Beijing": {"temp": 14, "humidity": 60, "pressure": 1016},
        }

        climate = city_climates.get(city, city_climates["New York"])
        coords = self.city_coords.get(city, {"lat": 40.7, "lon": -74.0})

        # Generate dates
        start_date = datetime.now() - timedelta(days=days)
        dates = []

        # Generate hourly data or daily data
        if days <= 7:
            # Hourly for recent data
            for i in range(days * 24):
                dates.append(start_date + timedelta(hours=i))
        else:
            # Daily for longer periods
            for i in range(days):
                dates.append(start_date + timedelta(days=i))

        data = []
        for date in dates:
            # Temperature varies by time of day and day of year
            hour = date.hour
            day_of_year = date.timetuple().tm_yday

            # Daily temperature cycle
            temp_base = climate["temp"]
            temp_variation = 5 * np.sin(2 * np.pi * hour / 24)

            # Seasonal variation
            seasonal_var = 10 * np.sin(2 * np.pi * (day_of_year - 81) / 365)

            temperature = temp_base + temp_variation + seasonal_var
            temperature += np.random.normal(0, 3)

            # Humidity (inverse relationship with temperature, generally)
            humidity_base = climate["humidity"]
            humidity = humidity_base - (temperature - temp_base) * 0.5
            humidity += np.random.normal(0, 10)
            humidity = max(20, min(100, humidity))

            # Pressure (relatively stable with small variations)
            pressure = climate["pressure"] + np.random.normal(0, 10)
            pressure = max(980, min(1040, pressure))

            # Wind speed (higher during day, lower at night)
            wind_speed = 3 + 2 * np.sin(2 * np.pi * hour / 24)
            wind_speed += np.random.normal(0, 2)
            wind_speed = max(0, min(30, wind_speed))

            # Visibility (inverse relationship with humidity)
            visibility = 10 - (humidity - 50) / 10
            visibility += np.random.normal(0, 2)
            visibility = max(0, min(20, visibility))

            # Weather condition (simplified)
            if humidity > 80 and temperature < temp_base:
                condition = "rainy"
            elif temperature < temp_base - 5:
                condition = "cold"
            elif temperature > temp_base + 10:
                condition = "hot"
            else:
                condition = "normal"

            data.append(
                {
                    "city": city,
                    "timestamp": date,
                    "latitude": coords["lat"],
                    "longitude": coords["lon"],
                    "temperature": round(temperature, 2),
                    "humidity": round(humidity, 2),
                    "pressure": round(pressure, 2),
                    "wind_speed": round(wind_speed, 2),
                    "visibility": round(visibility, 2),
                    "weather_condition": condition,
                }
            )

        return pd.DataFrame(data)

    def simulate_weather_data(self, cities: List[str], days: int) -> pd.DataFrame:
        """Generate simulated weather data for multiple cities"""
        all_data = []
        for city in cities:
            city_data = self.simulate_city_weather(city, days)
            all_data.append(city_data)

        return pd.concat(all_data, ignore_index=True)

    def save_to_csv(self, data: pd.DataFrame, filepath: str):
        """Save weather data to CSV"""
        data.to_csv(filepath, index=False)
        print(f"Saved {len(data)} weather records to {filepath}")
