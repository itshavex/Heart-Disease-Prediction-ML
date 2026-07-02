"""
Global exception handlers to secure the API and format error outputs.
"""
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Gracefully handle Pydantic payload formatting violations."""
    logger.warning(f"Payload validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "status": "error", 
            "message": "Invalid request payload. Please verify your data bounds and data types.", 
            "details": exc.errors()
        },
    )

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle explicit REST HTTPException raises safely."""
    logger.warning(f"HTTP exception on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )

async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all unhandled exceptions (e.g. TypeError, ConnectionResetError).
    The traceback is logged securely on the backend, but completely hidden from the client JSON.
    """
    logger.error(f"Unhandled internal server error on {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error", 
            "message": "An internal server error occurred. Please contact the administrator."
        },
    )
