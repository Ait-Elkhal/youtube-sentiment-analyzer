import requests
import json
import time

def test_hardyzona_complete():
    API_URL = "https://HARDYZONA-youtube-sentiment-analyzer.hf.space"
    
    print("🎯 TEST COMPLET API HARDYZONA")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1. 🩺 TEST SANTÉ...")
    try:
        start = time.time()
        response = requests.get(f"{API_URL}/health", timeout=15)
        health_time = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SANTÉ: {data['status']}")
            print(f"   🤖 Modèle: {data.get('model_type', 'N/A')}")
            print(f"   📍 Chargé: {data.get('model_loaded', False)}")
            print(f"   ⚡ Temps: {health_time:.2f}s")
            print(f"   👤 Auteur: {data.get('author', 'HardyZona')}")
        else:
            print(f"   ❌ Erreur santé: {response.status_code}")
            return False
    except Exception as e:
        print(f"   💥 Health échoué: {e}")
        return False
    
    # Test 2: Batch Prediction
    print("\n2. 🔥 TEST PRÉDICTION BATCH...")
    test_comments = [
        "Incroyable ! Cette API fonctionne parfaitement !",
        "Super travail HardyZona, très impressionnant ! 🚀",
        "Le modèle ML est très précis, bon travail !",
        "Je suis impressionné par la performance !",
        "Déploiement cloud réussi, félicitations !",
        "FastAPI + Hugging Face = combo gagnant !",
        "L'analyse de sentiment est très précise",
        "Excellent projet MLOps complet !",
        "Bravo pour ce déploiement réussi !",
        "HardyZona a fait du excellent travail ! 👏"
    ]
    
    try:
        start = time.time()
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"texts": test_comments},
            timeout=30
        )
        prediction_time = time.time() - start
        
        if response.status_code == 200:
            results = response.json()
            stats = results['statistics']
            
            print(f"   ✅ PRÉDICTION RÉUSSIE!")
            print(f"   📊 {stats['total_comments']} commentaires analysés")
            print(f"   ⚡ Temps: {results['processing_time']}s")
            print(f"   📈 RÉPARTITION:")
            print(f"      • Positif: {stats['sentiment_distribution']['positive']['percentage']}%")
            print(f"      • Négatif: {stats['sentiment_distribution']['negative']['percentage']}%") 
            print(f"      • Neutre: {stats['sentiment_distribution']['neutral']['percentage']}%")
            print(f"      • Confiance: {stats['average_confidence']:.2f}")
            
            # Afficher quelques résultats
            print(f"\n   🔍 EXEMPLES:")
            for i, pred in enumerate(results['predictions'][:3]):
                emoji = "😊" if pred['sentiment'] == 'positive' else "😞" if pred['sentiment'] == 'negative' else "😐"
                print(f"      {i+1}. {emoji} {pred['sentiment'].upper()}: {pred['text'][:40]}...")
                
            return True
            
        else:
            print(f"   ❌ Prédiction échouée: {response.status_code}")
            print(f"   Message: {response.text}")
            return False
            
    except Exception as e:
        print(f"   💥 Prédiction échouée: {e}")
        return False

if __name__ == "__main__":
    print("""
    🚀 YouTube Sentiment Analysis - HardyZona
    ⚡ Test de déploiement Hugging Face Spaces
    """)
    
    success = test_hardyzona_complete()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 DÉPLOIEMENT RÉUSSI!")
        print("✅ Phase 6: DÉPLOIEMENT CLOUD TERMINÉE!")
        print("🌐 API: https://HARDYZONA-youtube-sentiment-analyzer.hf.space")
        print("📚 Docs: /docs")
        print("👤 Par: HardyZona - INDIA ENSAM Rabat")
    else:
        print("❌ PROBLEME DÉTECTÉ!")
        print("🔧 Vérifie les logs Hugging Face")
    print("=" * 60)
