import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from prefect import task
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.db.session import SessionLocal
from app.db.base import PredictionRecord, ModelMetrics, DataDriftMetrics
from app.services.model_service import model_service
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, DataQualityPreset
from evidently.metrics import *

logger = logging.getLogger(__name__)

@task(name="get_reference_data")
def get_reference_data() -> pd.DataFrame:
    """
    Get reference data for monitoring.
    
    Returns:
        DataFrame: Reference data
    """
    return model_service.reference_data.copy()

@task(name="get_current_data")
def get_current_data(limit: int = 100) -> pd.DataFrame:
    """
    Get current data from database for monitoring.
    
    Args:
        limit: Maximum number of records to retrieve
        
    Returns:
        DataFrame: Current data
    """
    # Create database session
    db = SessionLocal()
    try:
        return model_service.get_recent_predictions(db, limit=limit)
    finally:
        db.close()

@task(name="generate_data_quality_metrics")
def generate_data_quality_metrics(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict:
    """
    Generate data quality metrics using Evidently.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        
    Returns:
        Dict: Data quality metrics
    """
    logger.info("Generating data quality metrics")
    
    # Check if current data is available
    if current_data.empty:
        logger.warning("No current data available for data quality metrics")
        return {"error": "No current data available"}
    
    # Prepare data
    # Make sure reference data and current data have the same columns
    common_cols = list(set(reference_data.columns).intersection(set(current_data.columns)))
    if not common_cols:
        logger.warning("No common columns found between reference and current data")
        return {"error": "No common columns found"}
        
    reference_data_subset = reference_data[common_cols]
    current_data_subset = current_data[common_cols]
    
    # Generate report
    data_quality_report = Report(metrics=[
        DataQualityPreset(),
    ])
    
    try:
        data_quality_report.run(reference_data=reference_data_subset, current_data=current_data_subset)
        report_dict = data_quality_report.as_dict()
        
        # Extract metrics
        metrics = {}
        for metric in report_dict["metrics"]:
            metric_name = metric["metric"]
            if metric_name == "DatasetMissingValuesMetric":
                metrics["missing_values_rate"] = metric["result"]["current"]["share_of_missing_values"]
            elif metric_name == "DatasetDuplicatedRowsMetric":
                metrics["duplicated_rows_rate"] = metric["result"]["current"]["share_of_duplicated_rows"]
        
        logger.info(f"Data quality metrics generated: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error generating data quality metrics: {e}")
        return {"error": str(e)}

@task(name="generate_data_drift_metrics")
def generate_data_drift_metrics(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Tuple[Dict, str]:
    """
    Generate data drift metrics using Evidently.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        
    Returns:
        Tuple[Dict, str]: Data drift metrics and report path
    """
    logger.info("Generating data drift metrics")
    
    # Check if current data is available
    if current_data.empty:
        logger.warning("No current data available for data drift metrics")
        return {"error": "No current data available"}, ""
    
    # Prepare data
    # Make sure reference data and current data have the same columns
    common_cols = list(set(reference_data.columns).intersection(set(current_data.columns)))
    if not common_cols:
        logger.warning("No common columns found between reference and current data")
        return {"error": "No common columns found"}, ""
        
    reference_data_subset = reference_data[common_cols]
    current_data_subset = current_data[common_cols]
    
    # Generate report
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    try:
        drift_report.run(reference_data=reference_data_subset, current_data=current_data_subset)
        report_dict = drift_report.as_dict()
        
        # Save report to file
        os.makedirs("app/templates/reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"app/templates/reports/data_drift_{timestamp}.html"
        drift_report.save_html(report_path)
        
        # Extract metrics
        metrics = {"timestamp": datetime.now().isoformat()}
        dataset_drift = False
        feature_metrics = []
        
        for metric in report_dict["metrics"]:
            if metric["metric"] == "DatasetDriftMetric":
                dataset_drift = metric["result"]["dataset_drift"]
                metrics["dataset_drift"] = dataset_drift
                
                # Get drift metrics for each feature
                for feature in metric["result"]["drift_by_columns"]:
                    feature_name = feature["column_name"]
                    drift_score = feature["drift_score"]
                    is_drift = feature["drift_detected"]
                    
                    feature_metrics.append({
                        "feature_name": feature_name,
                        "drift_score": drift_score,
                        "is_drift_detected": is_drift
                    })
                
        metrics["features"] = feature_metrics
        
        # Save drift metrics to database
        db = SessionLocal()
        try:
            for feature in feature_metrics:
                # Create database record
                drift_record = DataDriftMetrics(
                    feature_name=feature["feature_name"],
                    drift_score=feature["drift_score"],
                    is_drift_detected=1 if feature["is_drift_detected"] else 0,
                    model_version=model_service.model_version
                )
                
                db.add(drift_record)
            
            db.commit()
            logger.info("Data drift metrics saved to database")
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving drift metrics to database: {e}")
        finally:
            db.close()
        
        logger.info(f"Data drift metrics generated: dataset_drift={dataset_drift}")
        return metrics, report_path
    except Exception as e:
        logger.error(f"Error generating data drift metrics: {e}")
        return {"error": str(e)}, ""

@task(name="generate_model_performance_metrics")
def generate_model_performance_metrics(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Tuple[Dict, str]:
    """
    Generate model performance metrics using Evidently.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        
    Returns:
        Tuple[Dict, str]: Model performance metrics and report path
    """
    logger.info("Generating model performance metrics")
    
    # Check if current data contains target column and predictions
    if current_data.empty or "target" not in current_data.columns or "prediction" not in current_data.columns:
        logger.warning("Current data missing target or prediction columns")
        return {"error": "Current data missing target or prediction columns"}, ""
    
    # Generate report
    model_report = Report(metrics=[
        RegressionPreset(),
    ])
    
    try:
        model_report.run(reference_data=reference_data, current_data=current_data)
        report_dict = model_report.as_dict()
        
        # Save report to file
        os.makedirs("app/templates/reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"app/templates/reports/model_performance_{timestamp}.html"
        model_report.save_html(report_path)
        
        # Extract metrics
        metrics = {"timestamp": datetime.now().isoformat()}
        
        for metric in report_dict["metrics"]:
            if metric["metric"] == "RegressionQualityMetric":
                result = metric["result"]["current"]
                metrics.update({
                    "rmse": result["rmse"],
                    "mae": result["mean_abs_error"],
                    "mape": result["mean_abs_perc_error"],
                    "r2": result["r2_score"],
                    "model_version": model_service.model_version,
                    "data_size": len(current_data)
                })
                break
        
        # Save metrics to database
        db = SessionLocal()
        try:
            metric_record = ModelMetrics(
                rmse=metrics.get("rmse"),
                mae=metrics.get("mae"),
                r2=metrics.get("r2"),
                model_version=model_service.model_version,
                data_size=len(current_data)
            )
            
            db.add(metric_record)
            db.commit()
            logger.info("Model performance metrics saved to database")
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving model metrics to database: {e}")
        finally:
            db.close()
        
        logger.info(f"Model performance metrics generated: {metrics}")
        return metrics, report_path
    except Exception as e:
        logger.error(f"Error generating model performance metrics: {e}")
        return {"error": str(e)}, ""

@task(name="export_metrics_for_grafana")
def export_metrics_for_grafana(
    data_quality_metrics: Dict,
    data_drift_metrics: Dict,
    model_performance_metrics: Dict,
    output_path: str = "./data/metrics_export.json"
) -> str:
    """
    Export metrics in a format suitable for Grafana.
    
    Args:
        data_quality_metrics: Data quality metrics
        data_drift_metrics: Data drift metrics
        model_performance_metrics: Model performance metrics
        output_path: Path to save the metrics
        
    Returns:
        str: Path to the exported metrics
    """
    logger.info("Exporting metrics for Grafana")
    
    # Combine metrics
    combined_metrics = {
        "timestamp": datetime.now().isoformat(),
        "data_quality": data_quality_metrics,
        "data_drift": data_drift_metrics,
        "model_performance": model_performance_metrics
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(combined_metrics, f, indent=2)
    
    logger.info(f"Metrics exported to {output_path}")
    return output_path