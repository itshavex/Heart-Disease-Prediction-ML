"""
API endpoint routing definitions.
"""
import logging
from fastapi import APIRouter

from app.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/", tags=["General"])
async def root():
    """Root endpoint verifying API is alive."""
    return {"message": "Welcome to the Heart Disease Prediction ML API"}

@router.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint for system monitoring."""
    return HealthResponse(status="healthy", version="1.0.0")
