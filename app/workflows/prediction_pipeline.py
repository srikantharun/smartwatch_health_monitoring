from prefect import flow
import pandas as pd
import logging
from typing import Optional

from app.workflows.tasks.prediction_tasks import (
    get_batch_prediction_data,
    make_batch_predictions,
    save_batch_predictions,
    export_batch_predictions
)

logger = logging.getLogger(__name__)

@flow(name="Batch Prediction Pipeline")
def batch_prediction_pipeline(
    data_path: Optional[str] = None,
    export_path: str = "./data/batch_predictions.csv"
) -> str:
    """
    Run batch prediction pipeline.
    
    Args:
        data_path: Path to the input data file
        export_path: Path to export predictions
        
    Returns:
        str: Path to the exported predictions
    """
    logger.info("Starting batch prediction pipeline")
    
    # Get batch prediction data
    data = get_batch_prediction_data(data_path)
    
    # Make batch predictions
    predictions = make_batch_predictions(data)
    
    # Save batch predictions to database
    save_batch_predictions(predictions)
    
    # Export batch predictions
    output_path = export_batch_predictions(predictions, export_path)
    
    logger.info(f"Batch prediction pipeline completed, predictions saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Run the pipeline
    batch_prediction_pipeline()