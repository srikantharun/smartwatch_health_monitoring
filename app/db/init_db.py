import logging
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.session import engine

logger = logging.getLogger(__name__)

def init_db(db: Session) -> None:
    """
    Initialize database tables.
    
    Args:
        db: Database session
    """
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def create_initial_data(db: Session) -> None:
    """
    Create initial data for the database.
    
    Args:
        db: Database session
    """
    # This function can be used to populate the database with initial data
    # if needed for development or testing purposes
    pass