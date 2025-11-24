# src/api/config.py
import os
from pathlib import Path

# Configuration de l'API
API_CONFIG = {
    "title": "YouTube Sentiment Analysis API",
    "description": "API pour l'analyse de sentiment des commentaires YouTube",
    "version": "1.0.0",
    "debug": True,
    "host": "0.0.0.0",
    "port": 8000
}

# Chemins des modèles
MODEL_PATHS = {
    "model": Path("models/trained/best_sentiment_model.joblib"),
    "vectorizer": Path("models/trained/tfidf_vectorizer.joblib"),
    "metrics": Path("models/trained/model_metrics.joblib")
}

# Vérifier que les modèles existent
for name, path in MODEL_PATHS.items():
    if not path.exists():
        raise FileNotFoundError(f"Fichier modèle manquant: {path}")