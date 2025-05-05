from prefect import flow
import pandas as pd
import logging
from typing import Dict, Optional, Tuple

from app.workflows.tasks.monitoring_tasks import (
    get_reference_data,
    get_current_data,
    generate_data_quality_metrics,
    generate_data_drift_metrics,
    generate_model_performance_metrics,
    export_metrics_for_grafana
)

logger = logging.getLogger(__name__)

@flow(name="Data Monitoring Pipeline")
def data_monitoring_pipeline(
    limit: int = 100,
    export_path: str = "./data/monitoring_metrics.json"
) -> Dict:
    """
    Run data monitoring pipeline.
    
    Args:
        limit: Maximum number of records to process
        export_path: Path to export metrics
        
    Returns:
        Dict: Monitoring metrics
    """
    logger.info("Starting data monitoring pipeline")
    
    # Get reference and current data
    reference_data = get_reference_data()
    current_data = get_current_data(limit=limit)
    
    # Generate data quality metrics
    data_quality_metrics = generate_data_quality_metrics(reference_data, current_data)
    
    # Generate data drift metrics
    data_drift_metrics, drift_report_path = generate_data_drift_metrics(reference_data, current_data)
    
    logger.info(f"Data monitoring pipeline completed, drift report saved to {drift_report_path}")
    
    # Return metrics
    return {
        "data_quality_metrics": data_quality_metrics,
        "data_drift_metrics": data_drift_metrics,
        "drift_report_path": drift_report_path
    }

@flow(name="Model Monitoring Pipeline")
def model_monitoring_pipeline(
    limit: int = 100,
    export_path: str = "./data/model_metrics.json"
) -> Dict:
    """
    Run model monitoring pipeline.
    
    Args:
        limit: Maximum number of records to process
        export_path: Path to export metrics
        
    Returns:
        Dict: Monitoring metrics
    """
    logger.info("Starting model monitoring pipeline")
    
    # Get reference and current data
    reference_data = get_reference_data()
    current_data = get_current_data(limit=limit)
    
    # Generate model performance metrics
    model_metrics, performance_report_path = generate_model_performance_metrics(reference_data, current_data)
    
    logger.info(f"Model monitoring pipeline completed, performance report saved to {performance_report_path}")
    
    # Return metrics
    return {
        "model_performance_metrics": model_metrics,
        "performance_report_path": performance_report_path
    }

@flow(name="Full Monitoring Pipeline")
def full_monitoring_pipeline(
    limit: int = 100,
    export_path: str = "./data/all_metrics.json"
) -> Dict:
    """
    Run complete monitoring pipeline.
    
    Args:
        limit: Maximum number of records to process
        export_path: Path to export metrics
        
    Returns:
        Dict: All monitoring metrics
    """
    logger.info("Starting full monitoring pipeline")
    
    # Run data monitoring pipeline
    data_monitoring_results = data_monitoring_pipeline(limit=limit)
    
    # Run model monitoring pipeline
    model_monitoring_results = model_monitoring_pipeline(limit=limit)
    
    # Export metrics for Grafana
    export_path = export_metrics_for_grafana(
        data_quality_metrics=data_monitoring_results["data_quality_metrics"],
        data_drift_metrics=data_monitoring_results["data_drift_metrics"],
        model_performance_metrics=model_monitoring_results["model_performance_metrics"],
        output_path=export_path
    )
    
    logger.info(f"Full monitoring pipeline completed, metrics exported to {export_path}")
    
    # Return all metrics
    return {
        **data_monitoring_results,
        **model_monitoring_results,
        "export_path": export_path
    }

if __name__ == "__main__":
    # Run the pipeline
    full_monitoring_pipeline()