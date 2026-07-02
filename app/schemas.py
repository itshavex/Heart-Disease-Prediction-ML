"""
Pydantic schemas for request/response validation.
"""
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class HealthResponse(BaseModel):
    """Schema for the API health check response."""
    status: str = Field(..., description="Current status of the API (e.g., healthy)")
    version: str = Field(..., description="Current version of the API")


class PredictionRequest(BaseModel):
    """
    Schema for a single patient's raw feature inputs.
    We use aliases for fields with spaces to seamlessly map to the original dataset schema.
    """
    model_config = ConfigDict(populate_by_name=True)

    age: float = Field(
        ..., 
        gt=0, 
        description="Age of the patient in years"
    )
    sex: int = Field(
        ..., 
        description="Sex of the patient (1 = Male, 0 = Female)"
    )
    cholst_pain: int = Field(
        ..., 
        alias="cholst pain",
        description="Chest pain type (Typical angina, Atypical angina, Non-anginal, Asymptomatic)"
    )
    RestBP: float = Field(
        ..., 
        gt=0, 
        description="Resting blood pressure (in mm Hg on admission to the hospital)"
    )
    CholL: float = Field(
        ..., 
        gt=0, 
        description="Serum cholesterol in mg/dl"
    )
    FastBS: int = Field(
        ..., 
        description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)"
    )
    RESTECG: int = Field(
        ..., 
        description="Resting electrocardiographic results (0, 1, or 2)"
    )
    Thalach: float = Field(
        ..., 
        gt=0, 
        description="Maximum heart rate achieved during stress test"
    )
    EXAng: int = Field(
        ..., 
        description="Exercise induced angina (1 = yes; 0 = no)"
    )
    OLDPeak: float = Field(
        ..., 
        ge=0, 
        description="ST depression induced by exercise relative to rest"
    )
    SLOPE: int = Field(
        ..., 
        description="The slope of the peak exercise ST segment (0, 1, or 2)"
    )


class PredictionResponse(BaseModel):
    """Schema for the prediction output."""
    status: str = Field(..., description="Status of the prediction operation (success or error)")
    prediction: Optional[int] = Field(None, description="The binary prediction (1 = Disease, 0 = No Disease)")
    probability: Optional[float] = Field(None, description="The probability score associated with the positive class")
    message: Optional[str] = Field(None, description="Error message if status is error")
