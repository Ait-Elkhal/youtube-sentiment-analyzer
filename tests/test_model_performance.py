import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def test_model_performance():
    print("🧪 TEST DES PERFORMANCES DU MODÈLE")
    print("=" * 50)
    
    # Charger le modèle et les métriques
    model = joblib.load('models/trained/best_sentiment_model.joblib')
    vectorizer = joblib.load('models/trained/tfidf_vectorizer.joblib')
    metrics = joblib.load('models/trained/model_metrics.joblib')
    
    print("✅ Modèles chargés avec succès")
    print(f"🤖 Modèle: {type(model).__name__}")
    print(f"🔤 Vectoriseur: {type(vectorizer).__name__}")
    
    # Afficher les métriques
    print("\n📊 MÉTRIQUES D'ENTRAÎNEMENT:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.4f}")
    
    # Charger les données de test
    try:
        test_data = pd.read_csv('data/processed/test.csv')
        print(f"\n📁 Données de test: {len(test_data)} échantillons")
        
        # Préparer les données
        X_test = vectorizer.transform(test_data['text'])
        y_test = test_data['label']
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques de test
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"🎯 Accuracy sur test set: {accuracy:.4f}")
        print(f"📈 F1-score sur test set: {f1:.4f}")
        
        # Seuils de performance
        if accuracy >= 0.75:
            print("✅ PERFORMANCE: Acceptable")
        elif accuracy >= 0.80:
            print("✅ PERFORMANCE: Bonne")
        else:
            print("⚠️  PERFORMANCE: À améliorer")
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")

if __name__ == "__main__":
    test_model_performance()
