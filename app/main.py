"""
FastAPI application core execution orchestrator.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.startup import load_artifacts

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="Research-grade Machine Learning API for heart disease prediction.",
    contact={
        "name": "Shashwat Tiwari",
        "url": "https://github.com/itshavex"
    }
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event handler
@app.on_event("startup")
async def startup_event():
    """Execute startup tasks like loading the ML models exactly once."""
    logger.info("Initializing API and loading models...")
    load_artifacts()
    logger.info("Startup complete. Ready to serve predictions.")

# Register routes
app.include_router(router)
