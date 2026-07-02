"""
schemas.py
──────────
Pydantic models for request validation and response serialisation.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request schemas ────────────────────────────────────────────────────────────

class HouseFeatures(BaseModel):
    """
    Input features for a single house prediction.
    All values reflect the California Housing dataset conventions.
    """
    MedInc: float = Field(
        ..., gt=0, description="Median income of the block group (tens of thousands USD)",
        examples=[8.3252]
    )
    HouseAge: float = Field(
        ..., ge=0, le=100, description="Median house age in years",
        examples=[41.0]
    )
    AveRooms: float = Field(
        ..., gt=0, description="Average number of rooms per household",
        examples=[6.98]
    )
    AveBedrms: float = Field(
        ..., gt=0, description="Average number of bedrooms per household",
        examples=[1.02]
    )
    Population: float = Field(
        ..., gt=0, description="Block group population",
        examples=[322.0]
    )
    AveOccup: float = Field(
        ..., gt=0, description="Average number of household members",
        examples=[2.56]
    )
    Latitude: float = Field(
        ..., ge=32.0, le=42.0, description="Block group latitude",
        examples=[37.88]
    )
    Longitude: float = Field(
        ..., ge=-125.0, le=-114.0, description="Block group longitude",
        examples=[-122.23]
    )

    @field_validator("AveBedrms")
    @classmethod
    def bedrooms_le_rooms(cls, v: float, info) -> float:
        rooms = info.data.get("AveRooms")
        if rooms is not None and v > rooms:
            raise ValueError("AveBedrms cannot exceed AveRooms")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.98,
                "AveBedrms": 1.02,
                "Population": 322.0,
                "AveOccup": 2.56,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }
    }


class BatchPredictRequest(BaseModel):
    houses: List[HouseFeatures] = Field(
        ..., min_length=1, max_length=1000,
        description="List of houses to predict (max 1000 per request)"
    )


# ── Response schemas ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    predicted_price_usd: float = Field(
        description="Predicted median house value in USD"
    )
    predicted_price_100k: float = Field(
        description="Predicted value in $100k units (native model output)"
    )
    model_name: str
    confidence_note: str = Field(
        default="Point estimate only. See /health for model metrics.",
    )


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_name: str
    trained_at: str
    metrics: dict
    feature_names: List[str]
    version: str = "1.0.0"
