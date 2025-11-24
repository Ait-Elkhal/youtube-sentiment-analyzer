# app.py - YouTube Sentiment Analysis by HardyZona
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration pour Hugging Face Spaces
class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/trained/best_sentiment_model.joblib")
    VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "models/trained/tfidf_vectorizer.joblib")
    METRICS_PATH = os.getenv("METRICS_PATH", "models/trained/model_metrics.joblib")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 7860))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Modèles Pydantic pour la validation des données
class CommentRequest(BaseModel):
    text: str

class BatchCommentRequest(BaseModel):
    texts: List[str]

class SentimentPrediction(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class BatchSentimentResponse(BaseModel):
    predictions: List[SentimentPrediction]
    statistics: Dict[str, Any]
    processing_time: float
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    vectorizer_type: Optional[str] = None
    timestamp: str
    version: str
    author: str = "HardyZona"

# Gestion des modèles ML
class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.metrics = None
        self.is_loaded = False
        self.load_time = None
        
    def load_models(self):
        """Charge les modèles entraînés et les métriques"""
        try:
            logger.info("🔄 Chargement des modèles ML par HardyZona...")
            
            # Vérifier l'existence des fichiers
            required_files = [
                Config.VECTORIZER_PATH,
                Config.MODEL_PATH,
                Config.METRICS_PATH
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Fichier manquant: {file_path}")
                logger.info(f"✅ Fichier trouvé: {file_path}")
            
            # Charger le vectoriseur TF-IDF
            self.vectorizer = joblib.load(Config.VECTORIZER_PATH)
            logger.info(f"✅ Vectoriseur chargé: {type(self.vectorizer).__name__}")
            
            # Charger le modèle de sentiment
            self.model = joblib.load(Config.MODEL_PATH)
            logger.info(f"✅ Modèle chargé: {type(self.model).__name__}")
            
            # Charger les métriques
            self.metrics = joblib.load(Config.METRICS_PATH)
            logger.info(f"✅ Métriques chargées: {len(self.metrics)} métriques disponibles")
            
            self.is_loaded = True
            self.load_time = datetime.now()
            
            logger.info("🎯 Tous les modèles sont chargés et prêts pour l'analyse!")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
            self.is_loaded = False
            raise
    
    def predict_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """Prédit le sentiment pour une liste de textes"""
        if not self.is_loaded:
            raise RuntimeError("Modèles non chargés")
        
        if not texts:
            return self._empty_response()
        
        start_time = time.time()
        
        try:
            # Vectorisation des textes
            texts_tfidf = self.vectorizer.transform(texts)
            
            # Prédictions
            predictions = self.model.predict(texts_tfidf)
            
            # Probabilités
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(texts_tfidf)
            else:
                probabilities = np.ones((len(texts), 3)) * 0.33
            
            # Traitement des résultats
            results = []
            label_mapping = {-1: "negative", 0: "neutral", 1: "positive"}
            sentiment_counts = {"negative": 0, "neutral": 0, "positive": 0}
            
            for i, text in enumerate(texts):
                sentiment = label_mapping.get(predictions[i], "neutral")
                confidence = float(np.max(probabilities[i]))
                
                prob_dict = self._get_probabilities(probabilities[i])
                
                result = {
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "probabilities": prob_dict
                }
                
                results.append(result)
                sentiment_counts[sentiment] += 1
            
            # Calcul des statistiques
            statistics = self._calculate_statistics(results, sentiment_counts)
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Analyse terminée: {len(texts)} commentaires en {processing_time:.3f}s")
            
            return {
                "predictions": results,
                "statistics": statistics,
                "processing_time": round(processing_time, 4),
                "model_info": self._get_model_info()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction: {e}")
            raise
    
    def _get_probabilities(self, probabilities: np.ndarray) -> Dict[str, float]:
        """Extrait les probabilités selon la shape du tableau"""
        if probabilities.shape[0] == 3:
            return {
                "negative": float(probabilities[0]),
                "neutral": float(probabilities[1]),
                "positive": float(probabilities[2])
            }
        else:
            return {
                "negative": 0.33,
                "neutral": 0.33,
                "positive": 0.34
            }
    
    def _calculate_statistics(self, results: List[Dict], counts: Dict[str, int]) -> Dict[str, Any]:
        """Calcule les statistiques détaillées"""
        total = len(results)
        
        if total == 0:
            return self._empty_statistics()
        
        distribution = {
            sentiment: {
                "count": count,
                "percentage": round(count / total * 100, 2)
            }
            for sentiment, count in counts.items()
        }
        
        confidences = [r["confidence"] for r in results]
        
        return {
            "total_comments": total,
            "sentiment_distribution": distribution,
            "average_confidence": round(float(np.mean(confidences)), 4),
            "dominant_sentiment": max(counts.items(), key=lambda x: x[1])[0],
            "analysis_by": "HardyZona"
        }
    
    def _empty_response(self) -> Dict[str, Any]:
        """Retourne une réponse vide"""
        return {
            "predictions": [],
            "statistics": self._empty_statistics(),
            "processing_time": 0,
            "model_info": self._get_model_info()
        }
    
    def _empty_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques vides"""
        return {
            "total_comments": 0,
            "sentiment_distribution": {
                "negative": {"count": 0, "percentage": 0},
                "neutral": {"count": 0, "percentage": 0},
                "positive": {"count": 0, "percentage": 0}
            },
            "average_confidence": 0,
            "dominant_sentiment": "neutral",
            "analysis_by": "HardyZona"
        }
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle"""
        return {
            "model_type": type(self.model).__name__ if self.model else None,
            "vectorizer_type": type(self.vectorizer).__name__ if self.vectorizer else None,
            "model_loaded": self.is_loaded,
            "author": "HardyZona",
            "version": "2.0.0",
            "description": "YouTube Sentiment Analysis API"
        }

# Initialisation de l'application FastAPI
app = FastAPI(
    title="YouTube Sentiment Analysis by HardyZona",
    description="""
    🎯 API d'analyse de sentiment des commentaires YouTube
    
    **Développée par HardyZona** dans le cadre du module Virtualisation & Cloud Computing.
    
    ## Fonctionnalités
    
    - ✅ Analyse de sentiment en temps réel
    - ✅ Support des batches de commentaires
    - ✅ Modèle ML optimisé (TF-IDF + Logistic Regression)
    - ✅ Statistiques détaillées
    - ✅ Déploiement Hugging Face Spaces
    
    ## Auteur
    **HardyZona** - Étudiant en INDIA
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Instance globale de l'analyseur
sentiment_analyzer = SentimentAnalyzer()

# Événements de l'application
@app.on_event("startup")
async def startup_event():
    """Charge les modèles au démarrage"""
    logger.info("🚀 Démarrage de l'API YouTube Sentiment Analysis by HardyZona...")
    try:
        sentiment_analyzer.load_models()
        logger.info("✅ API prête à recevoir des requêtes!")
    except Exception as e:
        logger.error(f"❌ Échec du démarrage: {e}")

# Routes de l'API
@app.get("/", include_in_schema=False)
async def root():
    """Endpoint racine"""
    return {
        "message": "🎯 YouTube Sentiment Analysis API by HardyZona",
        "version": "2.0.0",
        "author": "HardyZona",
        "status": "operational" if sentiment_analyzer.is_loaded else "degraded",
        "model_loaded": sentiment_analyzer.is_loaded,
        "timestamp": datetime.now().isoformat(),
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérifie l'état de santé de l'API"""
    return HealthResponse(
        status="healthy" if sentiment_analyzer.is_loaded else "unhealthy",
        model_loaded=sentiment_analyzer.is_loaded,
        model_type=type(sentiment_analyzer.model).__name__ if sentiment_analyzer.model else None,
        vectorizer_type=type(sentiment_analyzer.vectorizer).__name__ if sentiment_analyzer.vectorizer else None,
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        author="HardyZona"
    )

@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(request: BatchCommentRequest):
    """Analyse le sentiment d'un lot de commentaires"""
    if not sentiment_analyzer.is_loaded:
        raise HTTPException(status_code=503, detail="Service unavailable - ML models not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided in request")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 comments per request")
    
    try:
        result = sentiment_analyzer.predict_sentiment(request.texts)
        return BatchSentimentResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Erreur de prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/single")
async def predict_single_sentiment(request: CommentRequest):
    """Analyse le sentiment d'un seul commentaire"""
    if not sentiment_analyzer.is_loaded:
        raise HTTPException(status_code=503, detail="Service unavailable - ML models not loaded")
    
    try:
        result = sentiment_analyzer.predict_sentiment([request.text])
        return result["predictions"][0] if result["predictions"] else {}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Point d'entrée pour Hugging Face
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")