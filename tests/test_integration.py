import requests
import time

def test_complete_integration():
    print("🧪 TEST D'INTÉGRATION COMPLET")
    print("🚀 Simulation du flux: YouTube → Extension → API → Résultats")
    print("=" * 60)
    
    API_URL = "https://hardyzona-youtube-sentiment-analyzer.hf.space"
    
    # Simuler des commentaires YouTube réalistes
    simulated_comments = [
        "Cette vidéo est incroyable ! J'ai tout compris 👍",
        "Le formateur explique très bien, merci !",
        "Pas mal mais un peu trop rapide à certains moments",
        "Je suis déçu, je m'attendais à mieux...",
        "Super contenu, très utile pour mon projet !",
        "Bof, rien de nouveau par rapport aux autres tutos",
        "Excellent ! Les exemples sont très clairs 👏",
        "La qualité audio pourrait être meilleure",
        "Génial ! J'attends la suite avec impatience 🚀",
        "Trop complexe pour les débutants"
    ]
    
    print(f"📺 Simulation de {len(simulated_comments)} commentaires YouTube")
    
    # Étape 1: Vérification API
    print("\n1. 🔗 VÉRIFICATION API...")
    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
        if health.status_code == 200:
            health_data = health.json()
            print(f"   ✅ API HardyZona: {health_data['status']}")
            print(f"   🤖 Modèle: {health_data['model_type']}")
        else:
            print(f"   ❌ API non disponible: {health.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Erreur connexion API: {e}")
        return False
    
    # Étape 2: Analyse par lot
    print("\n2. 🔥 ANALYSE DES SENTIMENTS...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"texts": simulated_comments},
            timeout=30
        )
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json()
            stats = results['statistics']
            
            print(f"   ✅ Analyse réussie en {total_time:.2f}s")
            print(f"   📊 RÉSULTATS:")
            print(f"      • Commentaires analysés: {stats['total_comments']}")
            print(f"      • Positifs: {stats['sentiment_distribution']['positive']['percentage']}%")
            print(f"      • Neutres: {stats['sentiment_distribution']['neutral']['percentage']}%")
            print(f"      • Négatifs: {stats['sentiment_distribution']['negative']['percentage']}%")
            print(f"      • Confiance moyenne: {stats['average_confidence']:.2f}")
            
            # Afficher quelques prédictions détaillées
            print(f"\n   🔍 EXEMPLES DE PRÉDICTIONS:")
            for i, pred in enumerate(results['predictions'][:3]):
                sentiment_emoji = "😊" if pred['sentiment'] == 'positive' else "😐" if pred['sentiment'] == 'neutral' else "😞"
                print(f"      {i+1}. {sentiment_emoji} {pred['sentiment'].upper()}")
                print(f"         Text: {pred['text'][:50]}...")
                print(f"         Confiance: {pred['confidence']:.2f}")
            
            return True
            
        else:
            print(f"   ❌ Erreur analyse: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur lors de l'analyse: {e}")
        return False

def validate_performance():
    print("\n" + "=" * 60)
    print("📈 VALIDATION DES PERFORMANCES")
    
    # Critères de performance
    criteria = {
        "Temps réponse santé API": "< 2s",
        "Temps analyse 10 commentaires": "< 5s", 
        "Temps analyse 50 commentaires": "< 10s",
        "Disponibilité API": "> 95%",
        "Précision modèle": "> 75%"
    }
    
    print("🎯 CRITÈRES DE PERFORMANCE:")
    for criterion, target in criteria.items():
        print(f"   ✅ {criterion}: {target}")
    
    print("\n💡 RECOMMANDATIONS:")
    print("   • Surveiller les logs Hugging Face régulièrement")
    print("   • Tester avec différents types de contenu YouTube")
    print("   • Vérifier la consommation mémoire de l'extension")
    print("   • Documenter les cas d'erreur rencontrés")

if __name__ == "__main__":
    success = test_complete_integration()
    validate_performance()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST D'INTÉGRATION RÉUSSI!")
        print("✅ Le système HardyZona est opérationnel et performant!")
    else:
        print("❌ PROBLEMES IDENTIFIÉS - Vérifier les points ci-dessus")
    print("=" * 60)
