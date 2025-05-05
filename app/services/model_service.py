import pickle
import pandas as pd
import numpy as np
import logging
import os
import datetime
import uuid
from typing import Dict, List, Tuple, Any, Optional

from sqlalchemy.orm import Session
from app.db.base import PredictionRecord, ModelMetrics, DataDriftMetrics
from app.core.config import get_settings

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import *

logger = logging.getLogger(__name__)
settings = get_settings()

class ModelService:
    """Service for model prediction and monitoring."""
    
    def __init__(self):
        self.model = None
        self.model_version = "0.1.0"  # Default version
        self.reference_data = None
        self.feature_columns = None
        self._load_model()
        self._load_reference_data()
    
    def _load_model(self):
        """Load ML model from disk."""
        try:
            model_path = settings.MODEL_PATH
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}. Using dummy model.")
                # Create a dummy model for demonstration
                from sklearn.linear_model import LinearRegression
                self.model = LinearRegression()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Use dummy model if loading fails
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
    
    def _load_reference_data(self):
        """Load reference data for monitoring."""
        try:
            data_path = settings.REFERENCE_DATA_PATH
            if os.path.exists(data_path):
                self.reference_data = pd.read_csv(data_path)
                logger.info(f"Reference data loaded from {data_path}")
                # Get feature columns, excluding the target column
                self.feature_columns = [col for col in self.reference_data.columns if col != 'target']
            else:
                logger.warning(f"Reference data not found at {data_path}. Using dummy data.")
                # Create dummy reference data
                self.reference_data = self._create_dummy_data()
                self.feature_columns = [f"feature{i}" for i in range(1, 4)]
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            # Use dummy data if loading fails
            self.reference_data = self._create_dummy_data()
            self.feature_columns = [f"feature{i}" for i in range(1, 4)]
    
    def _create_dummy_data(self, n_samples=100):
        """Create dummy data for demonstration purposes."""
        np.random.seed(42)
        data = {}
        for i in range(1, 4):
            col_name = f"feature{i}"
            data[col_name] = np.random.normal(0, 1, n_samples)
        
        # Create a simple target variable with noise
        X = np.column_stack([data[f] for f in data])
        weights = np.random.rand(3)
        target = X @ weights + np.random.normal(0, 0.1, n_samples)
        data["target"] = target
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, features_dict: Dict) -> pd.DataFrame:
        """
        Preprocess input features for model prediction.
        
        Args:
            features_dict: Dictionary of feature values
            
        Returns:
            DataFrame: Processed input data
        """
        # Create a new DataFrame with only required feature columns
        processed_df = pd.DataFrame()
        
        # Filter out non-feature columns (like metadata fields)
        model_features = [col for col in self.feature_columns if col.startswith('feature')]
        
        # Ensure all required feature columns are present
        for col in model_features:
            if col in features_dict:
                processed_df[col] = [float(features_dict[col])]
            else:
                processed_df[col] = [0.0]  # Default value for missing features
        
        # Ensure correct column order
        processed_df = processed_df[model_features]
        
        return processed_df
    
    def predict(self, features: Dict) -> float:
        """
        Make prediction using the loaded model.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            float: Model prediction
        """
        try:
            # Preprocess input data
            input_df = self.preprocess_data(features)
            
            # Make prediction
            prediction = float(self.model.predict(input_df)[0])
            
            return prediction
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def save_prediction(self, db: Session, features: Dict, prediction: float, metadata: Optional[Dict] = None, actual: Optional[float] = None) -> PredictionRecord:
        """
        Save prediction record to database.
        
        Args:
            db: Database session
            features: Dictionary of feature values
            prediction: Model prediction value
            metadata: Optional metadata about the prediction (user, device, etc.)
            actual: Actual target value (if available)
            
        Returns:
            PredictionRecord: Saved prediction record
        """
        prediction_record = PredictionRecord(
            feature_values=features,
            prediction=prediction,
            actual=actual,
            model_version=self.model_version,
            metadata=metadata
        )
        
        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)
        
        return prediction_record
    
    def get_recent_predictions(self, db: Session, limit: int = 100) -> pd.DataFrame:
        """
        Get recent predictions from database for monitoring.
        
        Args:
            db: Database session
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame: Recent prediction records
        """
        # Query recent predictions
        records = db.query(PredictionRecord).order_by(
            PredictionRecord.timestamp.desc()
        ).limit(limit).all()
        
        if not records:
            logger.warning("No prediction records found in database")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for record in records:
            row = record.feature_values.copy()
            row['prediction'] = record.prediction
            if record.actual is not None:
                row['target'] = record.actual
            row['timestamp'] = record.timestamp
            data.append(row)
            
        current_data = pd.DataFrame(data)
        
        return current_data
    
    def generate_model_performance_report(self, db: Session) -> Tuple[str, str]:
        """
        Generate model performance report using Evidently.
        
        Args:
            db: Database session
            
        Returns:
            Tuple[str, str]: Report ID and file path
        """
        # Get recent predictions with actual values
        query = db.query(PredictionRecord).filter(
            PredictionRecord.actual.isnot(None)
        ).order_by(PredictionRecord.timestamp.desc()).limit(100)
        
        records = query.all()
        
        if not records:
            logger.warning("No records with actual values found for performance monitoring")
            return None, None
            
        # Create DataFrames for reference and current data
        reference_data = self.reference_data.copy()
        
        # Create current data from database records
        current_data = []
        for record in records:
            row = record.feature_values.copy()
            row['prediction'] = record.prediction
            row['target'] = record.actual
            current_data.append(row)
            
        current_df = pd.DataFrame(current_data)
        
        # Generate report
        model_report = Report(metrics=[
            RegressionPreset(),
        ])
        
        model_report.run(reference_data=reference_data, current_data=current_df)
        
        # Save report to file
        os.makedirs("app/templates/reports", exist_ok=True)
        report_id = f"model_performance_{uuid.uuid4()}"
        report_path = f"app/templates/reports/{report_id}.html"
        model_report.save_html(report_path)
        
        # Save metrics to database
        try:
            metrics = {}
            for metric in model_report.as_dict()['metrics']:
                if metric['metric'] == 'ColumnQuantileMetric':
                    metrics['rmse'] = metric['result']['current']['quantile_values']['0.5']
                elif metric['metric'] == 'RegressionQualityMetric':
                    metrics['mae'] = metric['result']['current']['mean_abs_error']
                    metrics['r2'] = metric['result']['current']['r2_score']
            
            if metrics:
                metric_record = ModelMetrics(
                    rmse=metrics.get('rmse'),
                    mae=metrics.get('mae'),
                    r2=metrics.get('r2'),
                    model_version=self.model_version,
                    data_size=len(current_df)
                )
                
                db.add(metric_record)
                db.commit()
        except Exception as e:
            logger.error(f"Error saving metrics to database: {e}")
        
        return report_id, report_path
    
    def generate_data_drift_report(self, db: Session) -> Tuple[str, str]:
        """
        Generate data drift report using Evidently.
        
        Args:
            db: Database session
            
        Returns:
            Tuple[str, str]: Report ID and file path
        """
        # Get recent predictions
        current_df = self.get_recent_predictions(db, limit=100)
        
        if current_df.empty:
            logger.warning("No data available for drift detection")
            return None, None
            
        # Make sure reference data and current data have the same columns
        reference_data = self.reference_data.copy()
        
        # Keep only common columns
        common_cols = list(set(reference_data.columns).intersection(set(current_df.columns)))
        
        if not common_cols:
            logger.warning("No common columns found between reference and current data")
            return None, None
            
        reference_data = reference_data[common_cols]
        current_df = current_df[common_cols]
        
        # Generate drift report
        drift_report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        drift_report.run(reference_data=reference_data, current_data=current_df)
        
        # Save report to file
        os.makedirs("app/templates/reports", exist_ok=True)
        report_id = f"data_drift_{uuid.uuid4()}"
        report_path = f"app/templates/reports/{report_id}.html"
        drift_report.save_html(report_path)
        
        # Save drift metrics to database
        try:
            drift_data = drift_report.as_dict()
            
            for metric in drift_data['metrics']:
                if metric['metric'] == 'DatasetDriftMetric':
                    dataset_drift = metric['result']['dataset_drift']
                    
                    # Get drift metrics for each feature
                    for feature in metric['result']['drift_by_columns']:
                        feature_name = feature['column_name']
                        drift_score = feature['drift_score']
                        is_drift = feature['drift_detected']
                        
                        # Create database record
                        drift_record = DataDriftMetrics(
                            feature_name=feature_name,
                            drift_score=drift_score,
                            is_drift_detected=1 if is_drift else 0,
                            model_version=self.model_version
                        )
                        
                        # Add statistical properties if available
                        for stat in metric['result']['statistical_properties']:
                            if stat['column_name'] == feature_name:
                                drift_record.current_mean = stat['current']['mean']
                                drift_record.reference_mean = stat['reference']['mean']
                                drift_record.current_std = stat['current']['std']
                                drift_record.reference_std = stat['reference']['std']
                                break
                        
                        db.add(drift_record)
                    
                    break
            
            db.commit()
        except Exception as e:
            logger.error(f"Error saving drift metrics to database: {e}")
        
        return report_id, report_path

# Singleton instance
model_service = ModelService()