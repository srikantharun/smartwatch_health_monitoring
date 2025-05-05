from sqlalchemy import Column, Integer, Float, String, JSON, DateTime
import datetime
from sqlalchemy.sql import func

from app.db.session import Base

class PredictionRecord(Base):
    """Model for storing prediction records."""
    
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Input features
    feature_values = Column(JSON, nullable=False)
    
    # Prediction outputs
    prediction = Column(Float, nullable=False)
    
    # Actual target value (if available)
    actual = Column(Float, nullable=True)
    
    # Metadata
    model_version = Column(String, nullable=False)
    metadata = Column(JSON, nullable=True)

class ModelMetrics(Base):
    """Model for storing model performance metrics."""
    
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Metrics
    rmse = Column(Float)
    mae = Column(Float)
    r2 = Column(Float)
    
    # Metadata
    model_version = Column(String, nullable=False)
    data_size = Column(Integer)

class DataDriftMetrics(Base):
    """Model for storing data drift metrics."""
    
    __tablename__ = "data_drift_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Drift metrics
    feature_name = Column(String, nullable=False)
    drift_score = Column(Float)
    is_drift_detected = Column(Integer)  # 0=False, 1=True
    
    # Statistical properties
    current_mean = Column(Float, nullable=True)
    reference_mean = Column(Float, nullable=True)
    current_std = Column(Float, nullable=True)
    reference_std = Column(Float, nullable=True)
    
    # Metadata
    model_version = Column(String, nullable=False)