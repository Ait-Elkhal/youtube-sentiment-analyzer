# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from .config import API_CONFIG
from .utils.model_loader import sentiment_model
from .routes import predict, health

# Création de l'application FastAPI
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"]
)

# Configuration CORS pour l'extension Chrome
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "chrome-extension://*",  # Extensions Chrome
        "moz-extension://*",     # Extensions Firefox
        "http://localhost",      # Développement local
        "http://localhost:8000", # API locale
        "https://*.youtube.com"  # YouTube
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)
# Inclusion des routes
app.include_router(predict.router)
app.include_router(health.router)

@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage de l'API"""
    print("🚀 Démarrage de l'API YouTube Sentiment Analysis...")
    success = sentiment_model.load_models()
    if success:
        print("✅ API prête à recevoir des requêtes!")
    else:
        print("❌ Erreur lors du chargement des modèles")

@app.get("/")
async def root():
    """Endpoint racine avec documentation"""
    return {
        "message": "YouTube Sentiment Analysis API",
        "version": API_CONFIG["version"],
        "endpoints": {
            "documentation": "/docs",
            "health": "/health/",
            "single_prediction": "/predict/single",
            "batch_prediction": "/predict/batch"
        }
    }

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Gestionnaire d'erreurs personnalisé"""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "details": str(exc)}
    )

# Pour l'exécution directe
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["debug"]
    )