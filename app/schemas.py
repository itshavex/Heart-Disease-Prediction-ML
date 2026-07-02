"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    version: str
