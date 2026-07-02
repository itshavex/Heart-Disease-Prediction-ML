"""
Centralized logging configuration for production FastAPI backend.
"""
import logging
import sys

def setup_logger():
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Set the standard production format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
    )
    
    # Configure console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    return logger

# Initialize the global logger instance
setup_logger()
logger = logging.getLogger(__name__)
