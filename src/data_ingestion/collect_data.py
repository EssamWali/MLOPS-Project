"""
Main script to collect data from all sources
Simulates multi-node data collection
"""

import os
import sys
import io
import yaml
import pandas as pd
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_ingestion.wearable_data_generator import WearableDataGenerator
from data_ingestion.air_quality_collector import AirQualityCollector
from data_ingestion.weather_collector import WeatherCollector


def load_config():
    """Load configuration from config file"""
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def collect_all_data(config):
    """Collect data from all sources"""
    
    # Create data directories
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    federated_dir = Path(__file__).parent.parent.parent / "data" / "federated"
    federated_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DATA COLLECTION STARTED")
    print("=" * 60)
    
    # 1. Collect Wearable Health Device Data
    print("\n1. Generating Wearable Health Device Data...")
    wearable_gen = WearableDataGenerator(seed=42)
    
    # Generate data for federated learning (distributed across nodes)
    wearable_config = config["datasets"]["wearable"]
    num_nodes = config["federated_learning"]["num_clients"]
    num_users = 100
    
    node_data = wearable_gen.generate_federated_dataset(
        num_users=num_users,
        num_nodes=num_nodes,
        days_per_user=30
    )
    
    # Save per-node data (simulating distributed storage)
    for node_id, df in node_data.items():
        node_file = federated_dir / f"wearable_node_{node_id}.csv"
        wearable_gen.save_to_csv(df, str(node_file))
    
    # Combine all wearable data for centralized analysis
    all_wearable = pd.concat(list(node_data.values()), ignore_index=True)
    wearable_file = data_dir / "wearable_data.csv"
    wearable_gen.save_to_csv(all_wearable, str(wearable_file))
    
    print(f"[OK] Generated wearable data: {len(all_wearable)} records")
    
    # 2. Collect Air Quality Data
    print("\n2. Collecting Air Quality Data...")
    air_quality_config = config["datasets"]["air_quality"]
    
    air_collector = AirQualityCollector(use_api=False)  # Set to True with API key
    air_quality_data = air_collector.simulate_air_quality(
        cities=air_quality_config["cities"],
        parameters=air_quality_config["pollutants"],
        days=30
    )
    
    air_quality_file = data_dir / "air_quality_data.csv"
    air_collector.save_to_csv(air_quality_data, str(air_quality_file))
    print(f"[OK] Collected air quality data: {len(air_quality_data)} records")
    
    # 3. Collect Weather Data
    print("\n3. Collecting Weather Data...")
    weather_config = config["datasets"]["weather"]
    
    weather_collector = WeatherCollector(use_api=False)  # Set to True with API key
    weather_data = weather_collector.simulate_weather_data(
        cities=weather_config["locations"],
        days=30
    )
    
    weather_file = data_dir / "weather_data.csv"
    weather_collector.save_to_csv(weather_data, str(weather_file))
    print(f"[OK] Collected weather data: {len(weather_data)} records")
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nData saved to:")
    print(f"  - Raw data: {data_dir}")
    print(f"  - Federated data: {federated_dir}")


if __name__ == "__main__":
    config = load_config()
    collect_all_data(config)

