# src/api/routes/health.py
from fastapi import APIRouter
from ...api.models.schemas import HealthResponse
from ...api.utils.model_loader import sentiment_model
from datetime import datetime

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Vérifie l'état de l'API et du modèle
    """
    model_info = sentiment_model.get_model_info()
    
    return HealthResponse(
        status="healthy" if sentiment_model.is_loaded else "unhealthy",
        model_loaded=sentiment_model.is_loaded,
        model_type=model_info.get("model_type"),
        model_accuracy=model_info.get("accuracy"),
        timestamp=datetime.now().isoformat()
    )