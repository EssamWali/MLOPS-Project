"""
Data Generator for Wearable Health Device Data
Simulates data from fitness trackers and health monitors
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List


class WearableDataGenerator:
    """Generate synthetic wearable health device data"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_user_data(
        self,
        user_id: int,
        days: int = 30,
        start_date: datetime = None,
        health_condition: str = "normal"
    ) -> pd.DataFrame:
        """
        Generate wearable data for a single user
        
        Args:
            user_id: Unique user identifier
            days: Number of days of data to generate
            start_date: Start date for data generation
            health_condition: "normal", "at_risk", "ill"
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
            
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Base values depend on health condition
        condition_multipliers = {
            "normal": {"hr": 1.0, "steps": 1.0, "sleep": 1.0},
            "at_risk": {"hr": 1.15, "steps": 0.85, "sleep": 0.9},
            "ill": {"hr": 1.3, "steps": 0.6, "sleep": 0.75}
        }
        
        mult = condition_multipliers.get(health_condition, condition_multipliers["normal"])
        
        data = []
        for date in dates:
            # Heart rate: 60-100 bpm normal, varies by time of day
            hour = date.hour
            base_hr = 70 + 10 * np.sin(2 * np.pi * hour / 24)
            heart_rate = max(60, min(120, 
                np.random.normal(base_hr * mult["hr"], 5)))
            
            # Steps: more during day, less at night
            if 6 <= hour <= 22:
                base_steps = np.random.normal(8000 * mult["steps"], 2000)
            else:
                base_steps = np.random.normal(500 * mult["steps"], 200)
            steps = max(0, base_steps)
            
            # Sleep hours: 6-9 hours typical
            sleep_hours = max(4, min(12, 
                np.random.normal(7.5 * mult["sleep"], 1.5)))
            
            # Calories: correlates with steps and heart rate
            calories = steps * 0.04 + (heart_rate - 60) * 10
            calories += np.random.normal(0, 50)
            calories = max(1000, calories)
            
            # Body temperature: 36.1-37.2°C normal
            base_temp = 36.6
            if health_condition == "ill":
                base_temp += np.random.normal(0.8, 0.3)
            body_temp = max(36.0, min(38.5, 
                np.random.normal(base_temp, 0.2)))
            
            # Stress level: 0-100 scale
            stress = min(100, max(0, 
                np.random.normal(30 + (100 - heart_rate) * 0.5, 15)))
            
            data.append({
                "user_id": user_id,
                "timestamp": date,
                "heart_rate": round(heart_rate, 1),
                "steps": int(steps),
                "sleep_hours": round(sleep_hours, 2),
                "calories": round(calories, 1),
                "body_temperature": round(body_temp, 2),
                "stress_level": round(stress, 1),
                "health_condition": health_condition
            })
            
        return pd.DataFrame(data)
    
    def generate_federated_dataset(
        self,
        num_users: int = 100,
        num_nodes: int = 5,
        days_per_user: int = 30
    ) -> Dict[int, pd.DataFrame]:
        """
        Generate data distributed across multiple nodes (simulating hospitals/cities)
        
        Returns:
            Dictionary mapping node_id to DataFrame
        """
        users_per_node = num_users // num_nodes
        node_data = {}
        
        for node_id in range(num_nodes):
            node_df = pd.DataFrame()
            
            # Assign health conditions: 70% normal, 20% at_risk, 10% ill
            for i in range(users_per_node):
                user_id = node_id * users_per_node + i
                rand = random.random()
                
                if rand < 0.7:
                    condition = "normal"
                elif rand < 0.9:
                    condition = "at_risk"
                else:
                    condition = "ill"
                
                user_df = self.generate_user_data(
                    user_id=user_id,
                    days=days_per_user,
                    health_condition=condition
                )
                user_df["node_id"] = node_id
                node_df = pd.concat([node_df, user_df], ignore_index=True)
            
            node_data[node_id] = node_df
            
        return node_data
    
    def save_to_csv(self, data: pd.DataFrame, filepath: str):
        """Save generated data to CSV"""
        data.to_csv(filepath, index=False)
        print(f"Saved {len(data)} records to {filepath}")


