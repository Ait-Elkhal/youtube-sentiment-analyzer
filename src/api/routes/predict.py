# src/api/routes/predict.py
from fastapi import APIRouter, HTTPException
from typing import List
import time
from ...api.models.schemas import (
    SentimentRequest, BatchSentimentRequest, 
    SentimentPrediction, BatchSentimentResponse,
    ErrorResponse
)
from ...api.utils.model_loader import sentiment_model

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post(
    "/single",
    response_model=SentimentPrediction,
    responses={500: {"model": ErrorResponse}}
)
async def predict_sentiment(request: SentimentRequest):
    """
    Analyse le sentiment d'un seul commentaire
    """
    if not sentiment_model.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        result = sentiment_model.predict_single(request.text)
        
        return SentimentPrediction(
            text=request.text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@router.post(
    "/batch", 
    response_model=BatchSentimentResponse,
    responses={500: {"model": ErrorResponse}}
)
async def predict_batch_sentiment(request: BatchSentimentRequest):
    """
    Analyse le sentiment d'un lot de commentaires (max 100)
    """
    if not sentiment_model.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 commentaires par requête")
    
    try:
        result = sentiment_model.predict_batch(request.texts)
        
        # Convertir les prédictions en objets Pydantic
        predictions = [
            SentimentPrediction(
                text=pred["text"],
                sentiment=pred["sentiment"],
                confidence=pred["confidence"],
                probabilities=pred["probabilities"]
            )
            for pred in result["predictions"]
        ]
        
        return BatchSentimentResponse(
            predictions=predictions,
            statistics=result["statistics"],
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")