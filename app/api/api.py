from fastapi import APIRouter
from app.api.routes import predictions, monitoring

# Create main API router
api_router = APIRouter()

# Include route modules
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])