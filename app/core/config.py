from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseModel):
    PROJECT_NAME: str = "ML Monitoring with FastAPI and Evidently AI"
    PROJECT_DESCRIPTION: str = "A FastAPI application for ML model monitoring using Evidently AI"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database settings
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "ml_monitoring")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    
    # For development, use SQLite
    USE_SQLITE: bool = os.getenv("USE_SQLITE", "True").lower() == "true"
    SQLITE_URL: str = "sqlite:///./ml_monitoring.db"
    
    # Data paths
    REFERENCE_DATA_PATH: str = os.getenv("REFERENCE_DATA_PATH", "./data/reference_data.csv")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./data/model.pkl")
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        if self.USE_SQLITE:
            return self.SQLITE_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

def get_settings() -> Settings:
    return Settings()