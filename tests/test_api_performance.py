import requests
import time
import json

def test_api_performance():
    API_URL = "https://hardyzona-youtube-sentiment-analyzer.hf.space"
    
    print("🧪 TEST DE PERFORMANCE API HARDYZONA")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. 🩺 TEST HEALTH CHECK...")
    start_time = time.time()
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        health_time = time.time() - start_time
        print(f"   ✅ Temps réponse: {health_time:.3f}s")
        print(f"   📊 Status: {response.json()}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return
    
    # Test 2: Performance avec différents batch sizes
    test_cases = [
        {"name": "Petit batch", "size": 5},
        {"name": "Batch moyen", "size": 20},
        {"name": "Gros batch", "size": 50}
    ]
    
    test_comments = [
        "Super vidéo ! Très instructive" for _ in range(50)
    ]
    
    print("\n2. ⚡ TEST PERFORMANCE BATCH...")
    for test_case in test_cases:
        print(f"\n   📦 {test_case['name']} ({test_case['size']} commentaires)")
        
        batch_texts = test_comments[:test_case['size']]
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_URL}/predict/batch",
                json={"texts": batch_texts},
                timeout=30
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"      ✅ Succès: {processing_time:.3f}s")
                print(f"      📊 Temps processing API: {data['processing_time']}s")
                
                # Vérification des critères de performance
                if processing_time < 2.0:
                    print("      🎯 PERFORMANCE: Excellente")
                elif processing_time < 5.0:
                    print("      ✅ PERFORMANCE: Acceptable")
                else:
                    print("      ⚠️  PERFORMANCE: Lente")
            else:
                print(f"      ❌ Erreur HTTP: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Erreur: {e}")
    
    # Test 3: Robustesse avec données edge cases
    print("\n3. 🛡️ TEST ROBUSTESSE...")
    edge_cases = [
        {"name": "Texte vide", "texts": [""]},
        {"name": "Texte très long", "texts": ["x" * 500]},
        {"name": "Emojis", "texts": ["👍❤️🔥🎉"]},
        {"name": "Mix langues", "texts": ["Hello! Bonjour! مرحبا"]}
    ]
    
    for case in edge_cases:
        try:
            response = requests.post(
                f"{API_URL}/predict/batch",
                json={"texts": case['texts']},
                timeout=10
            )
            if response.status_code == 200:
                print(f"   ✅ {case['name']}: Géré avec succès")
            else:
                print(f"   ⚠️  {case['name']}: Erreur {response.status_code}")
        except Exception as e:
            print(f"   ❌ {case['name']}: {e}")

if __name__ == "__main__":
    test_api_performance()
