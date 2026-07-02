"""
Production middleware utilities.
"""
import time
import uuid
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Generate X-Request-ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Inject tracking header
            response.headers["X-Request-ID"] = request_id
            
            # Calculate metrics
            process_time = (time.time() - start_time) * 1000
            
            # Standardize request log format
            logger.info(f"{request.method} {request.url.path} | Status {response.status_code} | {process_time:.2f} ms | RequestID: {request_id}")
            return response
            
        except Exception as e:
            # Catch critical middleware-level failures and log them securely before the exception handler picks them up
            process_time = (time.time() - start_time) * 1000
            logger.error(f"{request.method} {request.url.path} | Status 500 | {process_time:.2f} ms | RequestID: {request_id}")
            raise e
