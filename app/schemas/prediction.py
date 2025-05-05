from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, List, Union, Any
import datetime

class PredictionMetadata(BaseModel):
    """
    Schema for prediction metadata.
    """
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    device_type: Optional[str] = None
    location: Optional[str] = None
    timestamp: Optional[datetime.datetime] = None
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class PredictionInput(BaseModel):
    """
    Schema for prediction input data.
    """
    features: Dict[str, Union[float, int, str]] = Field(
        ..., 
        description="Dictionary of feature names and values"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata about the device and user"
    )
    
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "features": {
                    "feature1": 0.65,  # Heart Rate Variability (normalized)
                    "feature2": 0.7,   # Activity level (normalized)
                    "feature3": 0.8,   # Sleep quality (normalized)
                    "feature4": 0.4,   # Resting heart rate (normalized)
                    "feature5": 0.5    # Skin temperature (normalized)
                },
                "metadata": {
                    "user_id": "user1",
                    "device_id": "smartwatch_123",
                    "device_type": "smartwatch",
                    "timestamp": "2023-05-05T12:00:00"
                }
            }
        }
    )

class PredictionResult(BaseModel):
    """
    Schema for prediction result.
    """
    id: Optional[int] = None
    prediction: float
    model_version: str
    timestamp: datetime.datetime
    metadata: Optional[PredictionMetadata] = None
    
    model_config = ConfigDict(protected_namespaces=())

class ActualValueInput(BaseModel):
    """
    Schema for submitting actual values for past predictions.
    """
    prediction_id: int
    actual_value: float

class HistoricalPrediction(BaseModel):
    """
    Schema for historical prediction record.
    """
    id: int
    timestamp: datetime.datetime
    features: Dict[str, Any]
    prediction: float
    actual: Optional[float] = None
    model_version: str
    metadata: Optional[PredictionMetadata] = None
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class PredictionsList(BaseModel):
    """
    Schema for list of predictions.
    """
    predictions: List[HistoricalPrediction]
    total: int