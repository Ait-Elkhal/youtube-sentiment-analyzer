# src/api/utils/model_loader.py
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, Any, List
import time
from ...api.config import MODEL_PATHS

class SentimentModel:
    """Classe pour charger et utiliser le modèle de sentiment"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.metrics = None
        self.is_loaded = False
        
    def load_models(self) -> bool:
        """Charge le modèle et le vectoriseur"""
        try:
            print("📦 Chargement des modèles...")
            
            # Charger le modèle
            self.model = joblib.load(MODEL_PATHS["model"])
            print(f"✅ Modèle chargé: {type(self.model).__name__}")
            
            # Charger le vectoriseur
            self.vectorizer = joblib.load(MODEL_PATHS["vectorizer"])
            print(f"✅ Vectoriseur chargé: {self.vectorizer.__class__.__name__}")
            
            # Charger les métriques
            self.metrics = joblib.load(MODEL_PATHS["metrics"])
            print(f"✅ Métriques chargées - Accuracy: {self.metrics.get('test_accuracy', 'N/A')}")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            self.is_loaded = False
            return False
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Prédit le sentiment d'un seul texte"""
        if not self.is_loaded:
            raise RuntimeError("Modèle non chargé")
        
        start_time = time.time()
        
        # Vectorisation
        text_tfidf = self.vectorizer.transform([text])
        
        # Prédiction
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        # Mapping des labels
        label_mapping = {-1: "negative", 0: "neutral", 1: "positive"}
        sentiment = label_mapping[prediction]
        
        # Probabilités par sentiment
        prob_dict = {
            "negative": float(probabilities[0]),
            "neutral": float(probabilities[1]), 
            "positive": float(probabilities[2])
        }
        
        confidence = max(probabilities)
        processing_time = time.time() - start_time
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": prob_dict,
            "processing_time": processing_time
        }
    
    def predict_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Prédit le sentiment d'une liste de textes"""
        if not self.is_loaded:
            raise RuntimeError("Modèle non chargé")
        
        start_time = time.time()
        
        # Vectorisation par lot
        texts_tfidf = self.vectorizer.transform(texts)
        
        # Prédictions par lot
        predictions = self.model.predict(texts_tfidf)
        probabilities = self.model.predict_proba(texts_tfidf)
        
        # Traitement des résultats
        results = []
        label_mapping = {-1: "negative", 0: "neutral", 1: "positive"}
        
        for i, text in enumerate(texts):
            sentiment = label_mapping[predictions[i]]
            confidence = max(probabilities[i])
            prob_dict = {
                "negative": float(probabilities[i][0]),
                "neutral": float(probabilities[i][1]),
                "positive": float(probabilities[i][2])
            }
            
            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": prob_dict
            })
        
        # Statistiques
        sentiments = [result["sentiment"] for result in results]
        sentiment_counts = {
            "negative": sentiments.count("negative"),
            "neutral": sentiments.count("neutral"), 
            "positive": sentiments.count("positive")
        }
        
        total = len(results)
        statistics = {
            "total_comments": total,
            "sentiment_distribution": {
                sentiment: {"count": count, "percentage": count/total*100} 
                for sentiment, count in sentiment_counts.items()
            },
            "average_confidence": np.mean([r["confidence"] for r in results])
        }
        
        processing_time = time.time() - start_time
        
        return {
            "predictions": results,
            "statistics": statistics,
            "processing_time": processing_time
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_type": type(self.model).__name__,
            "accuracy": self.metrics.get("test_accuracy"),
            "feature_dimension": self.vectorizer.transform(["test"]).shape[1],
            "training_date": self.metrics.get("training_date")
        }

# Instance globale du modèle
sentiment_model = SentimentModel()