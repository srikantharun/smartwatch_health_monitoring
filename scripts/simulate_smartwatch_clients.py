#!/usr/bin/env python3
"""
Simulate smartwatch clients sending health data to the monitoring API.

This script reads data from CSV files and sends it to the API endpoint,
mimicking real-time data transmission from smartwatches.
"""

import requests
import pandas as pd
import time
import argparse
import random
import json
import logging
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smartwatch_client")

class SmartwatchClient:
    """Simulates a smartwatch client sending health data to a server."""
    
    def __init__(self, user_id, api_url, data_file=None):
        """
        Initialize the smartwatch client.
        
        Args:
            user_id: User identifier
            api_url: URL of the API endpoint
            data_file: CSV file with user data (optional)
        """
        self.user_id = user_id
        self.api_url = api_url
        self.data = None
        self.last_sync_index = -1
        
        if data_file and os.path.exists(data_file):
            self.load_data(data_file)
        else:
            logger.warning(f"Data file for {user_id} not found or not specified")
    
    def load_data(self, data_file):
        """Load user data from CSV file."""
        try:
            self.data = pd.read_csv(data_file)
            logger.info(f"Loaded {len(self.data)} records for {self.user_id}")
            
            # Verify this data is for the correct user
            if 'user_id' in self.data.columns and self.user_id != self.data['user_id'].iloc[0]:
                logger.warning(f"Data file contains records for {self.data['user_id'].iloc[0]}, not {self.user_id}")
            
            # Reset sync index
            self.last_sync_index = -1
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.data = None
    
    def get_next_reading(self):
        """Get the next reading to send."""
        if self.data is None or self.last_sync_index >= len(self.data) - 1:
            return None
        
        self.last_sync_index += 1
        reading = self.data.iloc[self.last_sync_index]
        return reading
    
    def format_reading(self, reading):
        """Format the reading data for API transmission."""
        if reading is None:
            return None
        
        # Create a features dictionary with the feature columns
        features = {}
        for i in range(1, 6):  # Assuming features 1-5
            feature_name = f"feature{i}"
            if feature_name in reading:
                features[feature_name] = float(reading[feature_name])
        
        # Format for API
        data = {
            "features": features,
            "metadata": {
                "user_id": self.user_id,
                "timestamp": reading["timestamp"] if "timestamp" in reading else datetime.now().isoformat(),
                "device_type": "smartwatch",
                "device_id": f"watch_{self.user_id}"
            }
        }
        
        # Include actual value if available
        if "target" in reading:
            data["actual_value"] = float(reading["target"])
        
        return data
    
    def send_reading(self):
        """Send the next reading to the API."""
        reading = self.get_next_reading()
        if reading is None:
            logger.info(f"No more readings available for {self.user_id}")
            return False
        
        data = self.format_reading(reading)
        if not data:
            logger.warning(f"Failed to format reading for {self.user_id}")
            return False
        
        try:
            # Send prediction request
            response = requests.post(
                f"{self.api_url}/predictions/predict",
                json=data
            )
            
            if response.status_code != 200:
                logger.error(f"Error sending prediction: {response.status_code} - {response.text}")
                return False
            
            result = response.json()
            logger.info(f"Prediction for {self.user_id}: {result['prediction']:.2f}")
            
            # If we have the actual value, send it as well after a short delay
            if "actual_value" in data:
                time.sleep(0.5)  # Small delay
                actual_data = {
                    "prediction_id": result.get("id", None),
                    "actual_value": data["actual_value"]
                }
                
                if actual_data["prediction_id"]:
                    actual_response = requests.post(
                        f"{self.api_url}/predictions/update-actual-value",
                        json=actual_data
                    )
                    
                    if actual_response.status_code == 200:
                        logger.info(f"Updated actual value for {self.user_id}")
                    else:
                        logger.warning(f"Failed to update actual value: {actual_response.status_code}")
            
            return True
        except Exception as e:
            logger.error(f"Error in send_reading: {e}")
            return False

def simulate_clients(api_url, data_dir, num_readings=None, interval=60):
    """
    Simulate multiple smartwatch clients sending data.
    
    Args:
        api_url: URL of the API endpoint
        data_dir: Directory containing user data files
        num_readings: Number of readings to send (None for all available)
        interval: Time interval between readings in seconds
    """
    # Find user data files
    user_files = {}
    for file in os.listdir(data_dir):
        if file.endswith("_data.csv") and file.startswith("user"):
            user_id = file.split("_")[0]
            user_files[user_id] = os.path.join(data_dir, file)
    
    if not user_files:
        logger.error(f"No user data files found in {data_dir}")
        return
    
    logger.info(f"Found {len(user_files)} user data files")
    
    # Create clients
    clients = {}
    for user_id, file_path in user_files.items():
        clients[user_id] = SmartwatchClient(user_id, api_url, file_path)
    
    # Simulate sending readings
    reading_count = 0
    active_clients = set(clients.keys())
    
    while active_clients and (num_readings is None or reading_count < num_readings):
        # Choose a random client to send data
        user_id = random.choice(list(active_clients))
        client = clients[user_id]
        
        # Send reading
        success = client.send_reading()
        if not success:
            active_clients.remove(user_id)
            logger.info(f"Client {user_id} has no more data to send")
        
        reading_count += 1
        if num_readings is not None:
            logger.info(f"Sent {reading_count}/{num_readings} readings")
        else:
            logger.info(f"Sent {reading_count} readings")
        
        # Sleep between readings
        if active_clients and (num_readings is None or reading_count < num_readings):
            # Add some randomness to the interval
            sleep_time = interval * (0.8 + 0.4 * random.random())
            logger.info(f"Waiting {sleep_time:.1f} seconds before next reading")
            time.sleep(sleep_time)
    
    logger.info(f"Simulation complete. Sent {reading_count} readings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate smartwatch clients sending health data")
    parser.add_argument("--api-url", default="http://localhost:8000/api/v1", help="API base URL")
    parser.add_argument("--data-dir", default="../data", help="Directory containing user data files")
    parser.add_argument("--readings", type=int, default=None, help="Number of readings to send (default: all available)")
    parser.add_argument("--interval", type=int, default=60, help="Time interval between readings in seconds")
    
    args = parser.parse_args()
    simulate_clients(args.api_url, args.data_dir, args.readings, args.interval)