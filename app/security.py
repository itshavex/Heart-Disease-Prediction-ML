"""
API authentication and security dependency module.
"""
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.config import settings
from app.logger import logger

# Declare the API Key header scheme (not auto-erroring so we can provide a custom JSON response)
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency to verify the presence and validity of the x-api-key header.
    Rejects missing or invalid keys with HTTP 401 Unauthorized.
    """
    if not api_key:
        logger.warning("Authentication failed: Missing x-api-key header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key in headers."
        )
        
    if api_key != settings.api_key:
        logger.warning("Authentication failed: Invalid API Key provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key."
        )
        
    return api_key
