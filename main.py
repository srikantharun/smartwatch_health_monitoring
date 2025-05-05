from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import uvicorn
import os
import subprocess
import threading
import time

from app.api.api import api_router
from app.core.config import get_settings
from app.db.init_db import init_db
from app.db.session import get_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount static files
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/reports", StaticFiles(directory="app/templates/reports"), name="reports")

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Root endpoint that returns the dashboard HTML page.
    
    Args:
        request: FastAPI request object
        
    Returns:
        HTMLResponse: Dashboard HTML
    """
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": settings.PROJECT_NAME}
    )

def setup_prefect():
    """Setup Prefect configuration"""
    try:
        logger.info("Setting up Prefect")
        
        # Create Prefect directory if it doesn't exist
        os.makedirs(".prefect", exist_ok=True)
        
        # Try to setup Prefect scheduled deployments
        try:
            from app.workflows.scheduled_pipelines import create_scheduled_pipelines
            logger.info("Creating scheduled Prefect pipelines")
            create_scheduled_pipelines()
            logger.info("Prefect pipelines scheduled successfully")
        except Exception as e:
            logger.error(f"Error setting up Prefect scheduled pipelines: {e}")
        
    except Exception as e:
        logger.error(f"Error setting up Prefect: {e}")

def start_prefect_server():
    """Start Prefect server in a separate thread"""
    try:
        logger.info("Starting Prefect server in background")
        
        # Function to run in separate thread
        def run_server():
            try:
                # Check if Prefect server is already running
                try:
                    result = subprocess.run(
                        ["prefect", "server", "version"],
                        capture_output=True,
                        text=True
                    )
                    if "not running" not in result.stderr:
                        logger.info("Prefect server is already running")
                        return
                except Exception:
                    pass
                
                # Start Prefect server
                process = subprocess.Popen(
                    ["prefect", "server", "start"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logger.info(f"Prefect server started with PID: {process.pid}")
                
                # Wait for server to start
                time.sleep(5)
                logger.info("Prefect server should be available at: http://localhost:4200")
            except Exception as e:
                logger.error(f"Error starting Prefect server: {e}")
        
        # Start in a separate thread
        thread = threading.Thread(target=run_server)
        thread.daemon = True
        thread.start()
        
    except Exception as e:
        logger.error(f"Error starting Prefect server thread: {e}")

# Initialize database and other resources on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and create tables on application startup."""
    logger.info("Starting ML Monitoring application")
    db = next(get_db())
    init_db(db)
    logger.info("Database initialized")
    
    # Ensure reports directory exists
    os.makedirs("app/templates/reports", exist_ok=True)
    
    # Check for model and reference data
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(settings.MODEL_PATH):
        logger.warning(f"Model file not found at {settings.MODEL_PATH}")
    
    if not os.path.exists(settings.REFERENCE_DATA_PATH):
        logger.warning(f"Reference data not found at {settings.REFERENCE_DATA_PATH}")
    
    # Setup Prefect in a separate thread to avoid blocking startup
    prefect_thread = threading.Thread(target=setup_prefect)
    prefect_thread.daemon = True
    prefect_thread.start()
    
    # Start Prefect server if needed
    start_prefect_server()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)