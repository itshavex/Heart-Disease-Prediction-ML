"""
Centralized logging configuration for production FastAPI backend.
"""
import logging
import sys
from app.config import settings

def setup_logger():
    # Get the root logger
    logger = logging.getLogger()
    
    # Parse log level from config
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Set the standard production format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
    )
    
    # Configure console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    return logger

# Initialize the global logger instance
setup_logger()
logger = logging.getLogger(__name__)
