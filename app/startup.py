"""
Application startup configuration and model loading logic.
"""
import logging
import os
import sys
import joblib

# Ensure src/ is in the path so we can cleanly reuse the preprocessing pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataset
from src.feature_engineering import perform_feature_engineering

from app.middleware import RequestLoggingMiddleware
from app.exceptions import (
    global_exception_handler, 
    http_exception_handler, 
    validation_exception_handler
)

logger = logging.getLogger(__name__)

# Global state for loaded model artifacts (singleton style)
model_registry = {}

def load_artifacts():
    """
    Load necessary ML models from the file system into memory.
    """
    logger.info("Loading model artifacts into memory...")
    try:
        # Load the best-performing trained model (Logistic Regression)
        model_path = os.path.join("models", "logistic_regression.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
            
        model = joblib.load(model_path)
        model_registry['model'] = model
        logger.info(f"Model successfully loaded from {model_path}")
        
        # Since the preprocessor was not saved as an explicit .pkl artifact in the original pipeline,
        # we dynamically restore the exact fitted state of the preprocessor using the original logic.
        logger.info("Restoring preprocessing pipeline state...")
        df = load_dataset()
        target_col = "Target Label" if "Target Label" in df.columns else "target"
        
        df_preprocessed, preprocessor = preprocess_dataset(df, target_col=target_col)
        df_final = perform_feature_engineering(df_preprocessed, target_col=target_col)
        
        # Extract the exact feature names that the model was trained on
        final_features = df_final.drop(columns=[target_col]).columns.tolist()
        
        model_registry['preprocessor'] = preprocessor
        model_registry['final_features'] = final_features
        logger.info("Preprocessing artifacts and feature schemas loaded successfully.")
        
    except Exception as e:
        logger.error(f"Failed to load artifacts during startup: {e}", exc_info=True)
        raise

def register_middleware(app: FastAPI):
    """Register all application middlewares."""
    logger.info("Registering API middleware...")
    app.add_middleware(RequestLoggingMiddleware)
    
    # Enable CORS for localhost development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://localhost:8000", "http://127.0.0.1", "http://127.0.0.1:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def register_exception_handlers(app: FastAPI):
    """Register robust, production-grade exception handlers."""
    logger.info("Registering global exception handlers...")
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
