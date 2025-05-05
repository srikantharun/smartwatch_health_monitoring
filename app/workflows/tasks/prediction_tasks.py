import pandas as pd
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from prefect import task
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.base import PredictionRecord
from app.services.model_service import model_service

logger = logging.getLogger(__name__)

@task(name="get_batch_prediction_data")
def get_batch_prediction_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Get batch data for predictions.
    
    Args:
        data_path: Path to the data file or None to use default
        
    Returns:
        DataFrame: Data for batch predictions
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading batch prediction data from {data_path}")
        return pd.read_csv(data_path)
    
    # Use current data if available
    current_data_path = "./data/current_data.csv"
    if os.path.exists(current_data_path):
        logger.info(f"Loading batch prediction data from {current_data_path}")
        return pd.read_csv(current_data_path)
    
    # No data available, generate some dummy data
    logger.warning("No data available for batch predictions, generating dummy data")
    return model_service._create_dummy_data(n_samples=50)

@task(name="make_batch_predictions")
def make_batch_predictions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make batch predictions using the model service.
    
    Args:
        data: Input data for batch predictions
        
    Returns:
        DataFrame: Data with predictions
    """
    logger.info(f"Making batch predictions for {len(data)} samples")
    
    # Get feature columns
    feature_cols = [col for col in data.columns if col.startswith("feature")]
    
    # Make predictions
    predictions = []
    for _, row in data.iterrows():
        features = {col: row[col] for col in feature_cols}
        prediction = model_service.predict(features)
        predictions.append(prediction)
    
    # Add predictions to dataframe
    data["prediction"] = predictions
    
    return data

@task(name="save_batch_predictions")
def save_batch_predictions(data: pd.DataFrame) -> None:
    """
    Save batch predictions to the database.
    
    Args:
        data: Data with predictions
    """
    logger.info(f"Saving {len(data)} batch predictions to database")
    
    # Get feature columns
    feature_cols = [col for col in data.columns if col.startswith("feature")]
    
    # Create database session
    db = SessionLocal()
    try:
        # Save predictions
        for _, row in data.iterrows():
            features = {col: float(row[col]) for col in feature_cols}
            prediction = float(row["prediction"])
            actual = float(row["target"]) if "target" in row else None
            
            # Create prediction record
            prediction_record = PredictionRecord(
                feature_values=features,
                prediction=prediction,
                actual=actual,
                model_version=model_service.model_version
            )
            
            db.add(prediction_record)
        
        db.commit()
        logger.info("Batch predictions saved successfully")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving batch predictions: {e}")
        raise
    finally:
        db.close()

@task(name="export_batch_predictions")
def export_batch_predictions(data: pd.DataFrame, output_path: str = "./data/batch_predictions.csv") -> str:
    """
    Export batch predictions to a CSV file.
    
    Args:
        data: Data with predictions
        output_path: Path to save the CSV file
        
    Returns:
        str: Path to the exported file
    """
    logger.info(f"Exporting {len(data)} batch predictions to {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    data["timestamp"] = datetime.now()
    data.to_csv(output_path, index=False)
    
    return output_path