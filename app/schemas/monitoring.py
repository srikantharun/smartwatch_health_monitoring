from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional, Union
import datetime

class MetricValue(BaseModel):
    """
    Schema for a single metric value.
    """
    name: str
    value: float
    
class ModelMetricsData(BaseModel):
    """
    Schema for model performance metrics.
    """
    timestamp: datetime.datetime
    metrics: List[MetricValue]
    model_version: str
    data_size: int
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class FeatureDriftData(BaseModel):
    """
    Schema for feature drift data.
    """
    feature_name: str
    drift_score: float
    is_drift_detected: bool
    current_mean: Optional[float] = None
    reference_mean: Optional[float] = None
    current_std: Optional[float] = None
    reference_std: Optional[float] = None
    
class DriftReport(BaseModel):
    """
    Schema for drift monitoring report.
    """
    timestamp: datetime.datetime
    model_version: str
    features_drift: List[FeatureDriftData]
    dataset_drift: bool
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class MonitoringReport(BaseModel):
    """
    Schema for monitoring report response.
    """
    report_id: str
    report_type: str
    html_path: str  # Path to the HTML report
    timestamp: datetime.datetime
    
    model_config = ConfigDict(from_attributes=True)