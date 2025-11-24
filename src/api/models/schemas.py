# src/api/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class SentimentLabel(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral" 
    POSITIVE = "positive"

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Texte à analyser")

class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="Liste de textes à analyser")

class SentimentPrediction(BaseModel):
    text: str
    sentiment: SentimentLabel
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Dict[SentimentLabel, float]

class BatchSentimentResponse(BaseModel):
    predictions: List[SentimentPrediction]
    statistics: Dict[str, Any]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    model_accuracy: Optional[float] = None
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None