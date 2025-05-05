#!/usr/bin/env python3
"""
Train a model to predict stress levels from smartwatch health data.

This script reads reference data, trains a simple regression model,
and saves it to a pickle file for use by the monitoring application.
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(data_path="../data/reference_data.csv", output_path="../data/model.pkl"):
    """
    Train a model on the reference data and save it to disk.
    
    Args:
        data_path: Path to reference data CSV
        output_path: Path to save model pickle file
    
    Returns:
        dict: Dictionary of model metrics
    """
    print(f"Loading reference data from {data_path}")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load the reference data
    data = pd.read_csv(data_path)
    
    # Get feature columns and target
    feature_cols = [col for col in data.columns if col.startswith('feature')]
    if 'target' not in data.columns:
        raise ValueError("Target column 'target' not found in data")
    
    X = data[feature_cols]
    y = data['target']
    
    print(f"Training model on {len(data)} samples with {len(feature_cols)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    # Use Ridge regression to handle potential multicollinearity in health metrics
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print feature importance (coefficients)
    print("\nFeature importance:")
    for feature, coef in zip(feature_cols, model.coef_):
        print(f"{feature}: {coef:.4f}")
    
    # Print evaluation metrics
    print("\nModel performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {output_path}")
    
    return {
        "model_type": "Ridge Regression",
        "n_samples": len(data),
        "n_features": len(feature_cols),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "feature_importance": dict(zip(feature_cols, model.coef_.tolist()))
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to predict stress from smartwatch data")
    parser.add_argument("--data", default="../data/reference_data.csv", help="Path to reference data CSV")
    parser.add_argument("--output", default="../data/model.pkl", help="Path to save model pickle file")
    
    args = parser.parse_args()
    train_model(args.data, args.output)