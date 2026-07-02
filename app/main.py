"""
FastAPI application core execution orchestrator.
"""
from fastapi import FastAPI
from app.config import settings
from app.logger import logger
from app.api.routes import router
from app.startup import load_artifacts, register_middleware, register_exception_handlers

# Initialize FastAPI app dynamically via environment config
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Research-grade Machine Learning API for heart disease prediction.",
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
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
