from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, List, Optional

from app.db.session import get_db
from app.schemas.monitoring import (
    ModelMetricsData,
    DriftReport,
    MonitoringReport
)
from app.services.model_service import model_service
from app.db.base import ModelMetrics, DataDriftMetrics

# Import Prefect workflows
from app.workflows.monitoring_pipeline import (
    data_monitoring_pipeline,
    model_monitoring_pipeline,
    full_monitoring_pipeline
)
from app.workflows.prediction_pipeline import batch_prediction_pipeline
from app.workflows.scheduled_pipelines import run_all_pipelines

import logging
import os
import asyncio
import json
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/monitor-model", response_model=MonitoringReport)
async def generate_model_performance_report(
    use_prefect: bool = False,
    db: Session = Depends(get_db)
):
    """
    Generate model performance monitoring report.
    
    Args:
        use_prefect: Whether to use Prefect for report generation
        db: Database session
        
    Returns:
        MonitoringReport: Report information
    """
    try:
        if use_prefect:
            # Use Prefect for report generation
            logger.info("Using Prefect for model performance report generation")
            # Create a background task to run the Prefect flow
            asyncio.create_task(run_prefect_model_monitoring())
            
            return MonitoringReport(
                report_id=f"prefect_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                report_type="model_performance",
                html_path="/reports/latest_model_performance.html",
                timestamp=datetime.now()
            )
        else:
            # Use direct service call
            report_id, report_path = model_service.generate_model_performance_report(db)
            
            if not report_id:
                raise HTTPException(
                    status_code=404, 
                    detail="Not enough data with actual values to generate report"
                )
            
            # Get HTML file path for frontend
            html_path = f"/reports/{os.path.basename(report_path)}"
            
            return MonitoringReport(
                report_id=report_id,
                report_type="model_performance",
                html_path=html_path,
                timestamp=model_service.reference_data.timestamp[0] if hasattr(model_service.reference_data, 'timestamp') else None
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating model performance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitor-target", response_model=MonitoringReport)
async def generate_target_drift_report(
    use_prefect: bool = False,
    db: Session = Depends(get_db)
):
    """
    Generate target drift monitoring report.
    
    Args:
        use_prefect: Whether to use Prefect for report generation
        db: Database session
        
    Returns:
        MonitoringReport: Report information
    """
    try:
        if use_prefect:
            # Use Prefect for report generation
            logger.info("Using Prefect for data drift report generation")
            # Create a background task to run the Prefect flow
            asyncio.create_task(run_prefect_data_monitoring())
            
            return MonitoringReport(
                report_id=f"prefect_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                report_type="data_drift",
                html_path="/reports/latest_data_drift.html",
                timestamp=datetime.now()
            )
        else:
            # Use direct service call
            report_id, report_path = model_service.generate_data_drift_report(db)
            
            if not report_id:
                raise HTTPException(
                    status_code=404, 
                    detail="Not enough data to generate drift report"
                )
            
            # Get HTML file path for frontend
            html_path = f"/reports/{os.path.basename(report_path)}"
            
            return MonitoringReport(
                report_id=report_id,
                report_type="data_drift",
                html_path=html_path,
                timestamp=model_service.reference_data.timestamp[0] if hasattr(model_service.reference_data, 'timestamp') else None
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating data drift report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report/{report_id}", response_class=HTMLResponse)
async def get_report(report_id: str):
    """
    Get HTML report by report ID.
    
    Args:
        report_id: Report ID
        
    Returns:
        HTMLResponse: HTML report
    """
    try:
        report_path = f"app/templates/reports/{report_id}.html"
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        with open(report_path, "r") as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=List[ModelMetricsData])
async def get_model_metrics(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get model performance metrics history.
    
    Args:
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        List[ModelMetricsData]: List of model metrics records
    """
    try:
        # Query metrics
        metrics = db.query(ModelMetrics).order_by(
            ModelMetrics.timestamp.desc()
        ).limit(limit).all()
        
        # Convert to response model
        result = []
        for m in metrics:
            metric_values = [
                {"name": "rmse", "value": m.rmse},
                {"name": "mae", "value": m.mae},
                {"name": "r2", "value": m.r2}
            ]
            
            result.append(
                ModelMetricsData(
                    timestamp=m.timestamp,
                    metrics=metric_values,
                    model_version=m.model_version,
                    data_size=m.data_size
                )
            )
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/run-batch-predictions")
async def run_batch_predictions():
    """
    Run batch predictions using Prefect.
    
    Returns:
        JSONResponse: Status message
    """
    try:
        # Create a background task to run the Prefect flow
        asyncio.create_task(run_prefect_batch_predictions())
        
        return JSONResponse(
            status_code=202,
            content={"message": "Batch prediction pipeline started"}
        )
    except Exception as e:
        logger.error(f"Error starting batch prediction pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/run-all-pipelines")
async def trigger_all_pipelines():
    """
    Run all monitoring pipelines using Prefect.
    
    Returns:
        JSONResponse: Status message
    """
    try:
        # Create a background task to run all Prefect flows
        asyncio.create_task(run_prefect_all_pipelines())
        
        return JSONResponse(
            status_code=202,
            content={"message": "All monitoring pipelines started"}
        )
    except Exception as e:
        logger.error(f"Error starting monitoring pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prefect-status")
async def get_prefect_status():
    """
    Get Prefect server status.
    
    Returns:
        JSONResponse: Status information
    """
    try:
        import prefect
        from prefect.client import get_client
        
        client = get_client()
        
        try:
            # Try to get healthcheck
            healthcheck = await client.api_healthcheck()
            
            # Get flow runs
            flow_runs = []
            return JSONResponse(
                content={
                    "status": "running",
                    "api_version": prefect.__version__,
                    "api_healthcheck": healthcheck,
                    "ui_url": "http://localhost:4200",
                }
            )
        except Exception:
            return JSONResponse(
                content={
                    "status": "not_running",
                    "api_version": prefect.__version__,
                    "message": "Prefect API is not available. Please start the Prefect server."
                }
            )
    except Exception as e:
        logger.error(f"Error checking Prefect status: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

async def run_prefect_model_monitoring():
    """Run model monitoring pipeline using Prefect."""
    try:
        logger.info("Running model monitoring pipeline with Prefect")
        result = await asyncio.to_thread(model_monitoring_pipeline)
        logger.info(f"Model monitoring pipeline completed: {result}")
        
        # Create a symbolic link to the latest report
        if "performance_report_path" in result and result["performance_report_path"]:
            try:
                latest_link = "app/templates/reports/latest_model_performance.html"
                if os.path.exists(latest_link):
                    os.remove(latest_link)
                os.symlink(
                    os.path.abspath(result["performance_report_path"]), 
                    os.path.abspath(latest_link)
                )
            except Exception as e:
                logger.error(f"Error creating symbolic link to latest report: {e}")
    except Exception as e:
        logger.error(f"Error running model monitoring pipeline: {e}")

async def run_prefect_data_monitoring():
    """Run data monitoring pipeline using Prefect."""
    try:
        logger.info("Running data monitoring pipeline with Prefect")
        result = await asyncio.to_thread(data_monitoring_pipeline)
        logger.info(f"Data monitoring pipeline completed: {result}")
        
        # Create a symbolic link to the latest report
        if "drift_report_path" in result and result["drift_report_path"]:
            try:
                latest_link = "app/templates/reports/latest_data_drift.html"
                if os.path.exists(latest_link):
                    os.remove(latest_link)
                os.symlink(
                    os.path.abspath(result["drift_report_path"]), 
                    os.path.abspath(latest_link)
                )
            except Exception as e:
                logger.error(f"Error creating symbolic link to latest report: {e}")
    except Exception as e:
        logger.error(f"Error running data monitoring pipeline: {e}")

async def run_prefect_batch_predictions():
    """Run batch prediction pipeline using Prefect."""
    try:
        logger.info("Running batch prediction pipeline with Prefect")
        result = await asyncio.to_thread(batch_prediction_pipeline)
        logger.info(f"Batch prediction pipeline completed: {result}")
    except Exception as e:
        logger.error(f"Error running batch prediction pipeline: {e}")

async def run_prefect_all_pipelines():
    """Run all pipelines using Prefect."""
    try:
        logger.info("Running all pipelines with Prefect")
        result = await asyncio.to_thread(run_all_pipelines)
        logger.info(f"All pipelines completed")
    except Exception as e:
        logger.error(f"Error running all pipelines: {e}")