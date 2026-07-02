"""
API endpoint routing definitions.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends

from app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from app.predictor import predict
from app.security import verify_api_key

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

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"], dependencies=[Depends(verify_api_key)])
async def predict_heart_disease(request: PredictionRequest):
    """
    Generate a heart disease prediction using the trained Machine Learning pipeline.
    
    Accepts raw patient features, handles preprocessing dynamically, and returns a binary prediction and probability score.
    """
    logger.info("Received prediction request via API.")
    
    # model_dump(by_alias=True) ensures that keys with spaces (like "cholst pain") are mapped correctly
    payload = request.model_dump(by_alias=True)
    
    result = predict(payload)
    
    if result.get("status") == "error":
        logger.error(f"Internal prediction error: {result.get('message')}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {result.get('message')}")
        
    return PredictionResponse(
        status=result["status"],
        prediction=result["prediction"],
        probability=result["probability"]
    )
