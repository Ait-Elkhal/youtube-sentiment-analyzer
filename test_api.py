# test_api.py
import requests
import json

# URL de base de l'API
BASE_URL = "http://localhost:8000"

def test_health():
    """Teste l'endpoint health"""
    response = requests.get(f"{BASE_URL}/health/")
    print("🔍 Health Check:")
    print(json.dumps(response.json(), indent=2))

def test_single_prediction():
    """Teste la prédiction simple"""
    test_data = {
        "text": "This movie is absolutely amazing and wonderful!"
    }
    
    response = requests.post(f"{BASE_URL}/predict/single", json=test_data)
    print("\n🎯 Single Prediction:")
    print(json.dumps(response.json(), indent=2))

def test_batch_prediction():
    """Teste la prédiction par lot"""
    test_data = {
        "texts": [
            "I love this product!",
            "This is okay, nothing special",
            "I hate this, it's terrible",
            "The quality is good but expensive",
            "Not sure how I feel about this"
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=test_data)
    print("\n📦 Batch Prediction:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("🧪 TEST DE L'API YOUTUBE SENTIMENT ANALYSIS")
    print("=" * 50)
    
    try:
        test_health()
        test_single_prediction() 
        test_batch_prediction()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("Assurez-vous que l'API est démarrée (python run_api.py)")