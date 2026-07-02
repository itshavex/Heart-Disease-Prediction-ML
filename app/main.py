"""
FastAPI application core execution orchestrator.
"""
from fastapi import FastAPI
from app.logger import logger
from app.api.routes import router
from app.startup import load_artifacts, register_middleware, register_exception_handlers

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

# Register application configurations
register_middleware(app)
register_exception_handlers(app)

# Startup event handler
@app.on_event("startup")
async def startup_event():
    """Execute startup tasks like loading the ML models exactly once."""
    logger.info("Initializing API and loading models...")
    load_artifacts()
    logger.info("Startup complete. Ready to serve predictions.")

# Register routes
app.include_router(router)
