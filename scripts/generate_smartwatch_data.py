#!/usr/bin/env python3
"""
Generate simulated smartwatch health data for multiple users over a week period.

Feature descriptions:
- feature1: Heart rate variability (HRV) - measures the variation in time between heartbeats
- feature2: Average daily steps normalized (0-1 scale from sedentary to very active)
- feature3: Sleep quality score (0-1 scale)
- feature4: Resting heart rate (normalized)
- feature5: Skin temperature variation (normalized)

Target: Stress level score (0-10, where 10 is highest stress)
"""

import pandas as pd
import numpy as np
import datetime
import os
import argparse

# Set random seed for reproducibility
np.random.seed(42)

# User profiles to create realistic patterns
USER_PROFILES = {
    "user1": {  # Active user with good sleep, low stress
        "name": "Alex",
        "hrv_base": 65,  # Higher HRV is typically better
        "hrv_var": 10,
        "steps_base": 0.7,  # Active
        "steps_var": 0.2,
        "sleep_base": 0.8,  # Good sleep
        "sleep_var": 0.1,
        "heart_rate_base": 0.4,  # Lower RHR is better
        "heart_rate_var": 0.1,
        "skin_temp_base": 0.5,
        "skin_temp_var": 0.1,
        "stress_base": 3,  # Low stress
        "stress_var": 2,
        "weekday_stress_modifier": 1.2,  # Slightly more stressed on weekdays
        "weekend_stress_modifier": 0.8,  # Less stressed on weekends
    },
    "user2": {  # Sedentary user with poor sleep, high stress
        "name": "Blake",
        "hrv_base": 40,
        "hrv_var": 8,
        "steps_base": 0.3,  # Sedentary
        "steps_var": 0.15,
        "sleep_base": 0.4,  # Poor sleep
        "sleep_var": 0.15,
        "heart_rate_base": 0.7,  # Higher RHR
        "heart_rate_var": 0.1,
        "skin_temp_base": 0.6,
        "skin_temp_var": 0.15,
        "stress_base": 7,  # High stress
        "stress_var": 1.5,
        "weekday_stress_modifier": 1.3,
        "weekend_stress_modifier": 0.9,
    },
    "user3": {  # Moderate activity, variable sleep, moderate stress
        "name": "Casey",
        "hrv_base": 55,
        "hrv_var": 12,
        "steps_base": 0.5,  # Moderate activity
        "steps_var": 0.25,
        "sleep_base": 0.6,  # Variable sleep
        "sleep_var": 0.2,
        "heart_rate_base": 0.5,
        "heart_rate_var": 0.12,
        "skin_temp_base": 0.5,
        "skin_temp_var": 0.1,
        "stress_base": 5,  # Moderate stress
        "stress_var": 2.5,
        "weekday_stress_modifier": 1.4,
        "weekend_stress_modifier": 0.7,
    },
    "user4": {  # Very active user, good sleep but high stress
        "name": "Dana",
        "hrv_base": 70,
        "hrv_var": 8,
        "steps_base": 0.85,  # Very active
        "steps_var": 0.1,
        "sleep_base": 0.75,  # Good sleep
        "sleep_var": 0.1,
        "heart_rate_base": 0.3,  # Athletic low heart rate
        "heart_rate_var": 0.08,
        "skin_temp_base": 0.45,
        "skin_temp_var": 0.05,
        "stress_base": 6,  # Higher stress despite good metrics (work-related)
        "stress_var": 2,
        "weekday_stress_modifier": 1.5,
        "weekend_stress_modifier": 0.6,
    },
    "user5": {  # Elderly user with health concerns
        "name": "Elliot",
        "hrv_base": 35,
        "hrv_var": 5,
        "steps_base": 0.25,  # Limited mobility
        "steps_var": 0.1,
        "sleep_base": 0.5,  # Disrupted sleep
        "sleep_var": 0.15,
        "heart_rate_base": 0.6,
        "heart_rate_var": 0.15,
        "skin_temp_base": 0.55,
        "skin_temp_var": 0.12,
        "stress_base": 4,  # Moderate stress
        "stress_var": 1.5,
        "weekday_stress_modifier": 1.1,
        "weekend_stress_modifier": 0.9,
    }
}

def normalize_feature(value, min_val=0, max_val=1):
    """Normalize value to range 0-1."""
    return max(0, min(1, value))

def generate_data_for_user(user_id, days=7, readings_per_day=24):
    """
    Generate health data for a single user over specified days.
    
    Args:
        user_id: User identifier
        days: Number of days to generate data for
        readings_per_day: Number of readings per day
        
    Returns:
        DataFrame with generated health data
    """
    if user_id not in USER_PROFILES:
        raise ValueError(f"User ID {user_id} not found in profiles")
    
    profile = USER_PROFILES[user_id]
    total_readings = days * readings_per_day
    
    # Initialize arrays for features
    timestamps = []
    hrv_values = []
    step_values = []
    sleep_values = []
    heart_rate_values = []
    skin_temp_values = []
    stress_values = []
    
    # Generate data
    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    for day in range(days):
        current_date = start_date + datetime.timedelta(days=day)
        is_weekend = current_date.weekday() >= 5  # 5 and 6 are Saturday and Sunday
        
        # Apply weekend/weekday modifier
        stress_modifier = profile["weekend_stress_modifier"] if is_weekend else profile["weekday_stress_modifier"]
        
        for hour in range(readings_per_day):
            # Create timestamp
            timestamp = current_date + datetime.timedelta(hours=hour)
            timestamps.append(timestamp)
            
            # Time of day effect (0-1 scale, peaks mid-day)
            time_factor = 1 - abs((hour - 12) / 12)  # 1 at noon, lower at night/morning
            
            # Generate feature values with realistic patterns
            
            # Heart Rate Variability (HRV) - higher during rest/sleep
            rest_factor = 1 - time_factor  # Higher at night
            hrv = profile["hrv_base"] + profile["hrv_var"] * np.random.randn() + (10 * rest_factor)
            hrv = max(20, min(100, hrv))  # Limit range
            hrv_norm = (hrv - 20) / 80  # Normalize to 0-1 scale
            hrv_values.append(hrv_norm)
            
            # Steps - higher during day, lower at night
            step_base = profile["steps_base"] * time_factor
            if hour < 7 or hour > 22:  # Very few steps during sleep hours
                step_base *= 0.1
            steps = normalize_feature(step_base + profile["steps_var"] * np.random.randn())
            step_values.append(steps)
            
            # Sleep quality - only meaningful for night readings (higher during night)
            sleep_time_factor = 0.1
            if hour < 8:  # Morning hours
                sleep_time_factor = 1 - (hour / 8)
            elif hour >= 22:  # Night hours
                sleep_time_factor = (hour - 21) / 3
            
            sleep = normalize_feature(profile["sleep_base"] * sleep_time_factor + profile["sleep_var"] * np.random.randn())
            sleep_values.append(sleep)
            
            # Resting heart rate - lower during rest, higher during activity
            rhr = profile["heart_rate_base"] + (0.2 * time_factor) + profile["heart_rate_var"] * np.random.randn()
            rhr = normalize_feature(rhr)
            heart_rate_values.append(rhr)
            
            # Skin temperature - slight variations
            skin_temp = profile["skin_temp_base"] + profile["skin_temp_var"] * np.random.randn()
            skin_temp = normalize_feature(skin_temp)
            skin_temp_values.append(skin_temp)
            
            # Stress level - complex interaction of factors
            # Factors that reduce stress: high HRV, good sleep, low heart rate
            # Factors that increase stress: high activity without recovery, poor sleep
            
            stress_factors = [
                (1 - hrv_norm) * 3,  # Low HRV increases stress
                (1 - sleep) * 2,     # Poor sleep increases stress
                rhr * 2,             # High heart rate increases stress
                (1 - steps) * 0.5,   # Being sedentary slightly increases stress
                skin_temp * 0.5      # Higher skin temp slightly increases stress
            ]
            
            # Combine with base level and random variation
            stress = (profile["stress_base"] + sum(stress_factors) / len(stress_factors)) * stress_modifier
            stress += profile["stress_var"] * np.random.randn()
            stress = max(0, min(10, stress))  # Cap between 0-10
            stress_values.append(stress)
    
    # Create DataFrame
    data = {
        "user_id": [user_id] * total_readings,
        "user_name": [profile["name"]] * total_readings,
        "timestamp": timestamps,
        "feature1": hrv_values,         # Heart Rate Variability (normalized)
        "feature2": step_values,        # Steps (normalized)
        "feature3": sleep_values,       # Sleep quality (normalized)
        "feature4": heart_rate_values,  # Resting heart rate (normalized)
        "feature5": skin_temp_values,   # Skin temperature variation (normalized)
        "target": stress_values         # Stress level (0-10)
    }
    
    return pd.DataFrame(data)

def generate_all_user_data(output_path="data", days=7, readings_per_day=24):
    """
    Generate health data for all users and save to CSV files.
    
    Args:
        output_path: Directory to save output files
        days: Number of days to generate data for
        readings_per_day: Number of readings per day
    """
    all_data = []
    reference_data = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate data for each user
    for user_id in USER_PROFILES.keys():
        print(f"Generating data for {user_id}...")
        user_data = generate_data_for_user(user_id, days, readings_per_day)
        
        # Save individual user data
        user_file = os.path.join(output_path, f"{user_id}_data.csv")
        user_data.to_csv(user_file, index=False)
        print(f"Saved user data to {user_file}")
        
        # Add to all data
        all_data.append(user_data)
        
        # Add a subset to reference data (first 24 hours)
        reference_data.append(user_data.iloc[:readings_per_day])
    
    # Combine all user data
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_file = os.path.join(output_path, "current_data.csv")
    combined_data.to_csv(combined_file, index=False)
    print(f"Saved combined data to {combined_file}")
    
    # Create reference data (for model training and drift detection)
    reference_combined = pd.concat(reference_data, ignore_index=True)
    reference_file = os.path.join(output_path, "reference_data.csv")
    reference_combined.to_csv(reference_file, index=False)
    print(f"Saved reference data to {reference_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simulated smartwatch health data")
    parser.add_argument("--output", default="../data", help="Output directory for data files")
    parser.add_argument("--days", type=int, default=7, help="Number of days to generate data for")
    parser.add_argument("--readings", type=int, default=24, help="Number of readings per day")
    
    args = parser.parse_args()
    generate_all_user_data(args.output, args.days, args.readings)