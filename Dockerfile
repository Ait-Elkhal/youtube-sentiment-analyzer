# Dockerfile - HardyZona YouTube Sentiment API
FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements d'abord (optimisation cache Docker)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Créer la structure de dossiers
RUN mkdir -p models/trained

# Copier les modèles entraînés
COPY models/trained/ ./models/trained/

# Copier l'application
COPY app.py .

# Exposer le port (standard Hugging Face)
EXPOSE 7860

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Commande de démarrage
CMD ["python", "app.py"]