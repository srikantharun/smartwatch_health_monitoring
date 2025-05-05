from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, List, Optional

from app.db.session import get_db
from app.schemas.prediction import (
    PredictionInput, 
    PredictionResult, 
    ActualValueInput,
    HistoricalPrediction, 
    PredictionsList,
    PredictionMetadata
)
from app.db.base import PredictionRecord
from app.services.model_service import model_service

import logging
import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/predict", response_model=PredictionResult)
async def predict(
    input_data: PredictionInput,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Make prediction with ML model and log it for monitoring.
    
    Args:
        input_data: Features for prediction
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        PredictionResult: Model prediction
    """
    try:
        # Extract features and metadata
        features = input_data.features
        metadata = input_data.metadata
        
        # Make prediction
        prediction = model_service.predict(features)
        
        # Create metadata object if provided
        prediction_metadata = None
        if metadata:
            prediction_metadata = PredictionMetadata(
                user_id=metadata.get("user_id"),
                device_id=metadata.get("device_id"),
                device_type=metadata.get("device_type"),
                location=metadata.get("location"),
                timestamp=metadata.get("timestamp")
            )
        
        # Create prediction record
        prediction_record = model_service.save_prediction(
            db=db,
            features=features,
            prediction=prediction,
            metadata=metadata
        )
        
        return PredictionResult(
            id=prediction_record.id,
            prediction=prediction,
            model_version=model_service.model_version,
            timestamp=datetime.datetime.now(),
            metadata=prediction_metadata
        )
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-actual-value")
async def update_actual_value(
    actual_data: ActualValueInput,
    db: Session = Depends(get_db)
):
    """
    Update prediction record with actual value.
    
    Args:
        actual_data: Actual value data
        db: Database session
        
    Returns:
        Dict: Success message
    """
    try:
        # Get prediction record
        prediction_record = db.query(PredictionRecord).filter(
            PredictionRecord.id == actual_data.prediction_id
        ).first()
        
        if not prediction_record:
            raise HTTPException(status_code=404, detail="Prediction record not found")
        
        # Update actual value
        prediction_record.actual = actual_data.actual_value
        
        db.commit()
        db.refresh(prediction_record)
        
        return {"message": "Actual value updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating actual value: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions", response_model=PredictionsList)
async def get_predictions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get historical predictions.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        PredictionsList: List of historical prediction records
    """
    try:
        # Query predictions
        predictions = db.query(PredictionRecord).order_by(
            PredictionRecord.timestamp.desc()
        ).offset(skip).limit(limit).all()
        
        # Get total count
        total = db.query(PredictionRecord).count()
        
        # Convert to response model
        result = []
        for p in predictions:
            # Create metadata object if available
            prediction_metadata = None
            if hasattr(p, 'metadata') and p.metadata:
                prediction_metadata = PredictionMetadata(
                    user_id=p.metadata.get("user_id"),
                    device_id=p.metadata.get("device_id"),
                    device_type=p.metadata.get("device_type"),
                    location=p.metadata.get("location"),
                    timestamp=p.metadata.get("timestamp")
                )
            
            result.append(
                HistoricalPrediction(
                    id=p.id,
                    timestamp=p.timestamp,
                    features=p.feature_values,
                    prediction=p.prediction,
                    actual=p.actual,
                    model_version=p.model_version,
                    metadata=prediction_metadata
                )
            )
        
        return PredictionsList(predictions=result, total=total)
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))