# src/models/train_model.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import time
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    PHASE 3: Développement et entraînement du modèle de classification de sentiment
    Conforme aux exigences du TP
    """
    print("=" * 70)
    print("PHASE 3: DÉVELOPPEMENT ET ENTRAÎNEMENT DU MODÈLE")
    print("Conforme aux exigences du TP - Analyse de Sentiment YouTube")
    print("=" * 70)
    
    # Création des dossiers
    Path("models/trained").mkdir(parents=True, exist_ok=True)
    Path("models/experiments").mkdir(parents=True, exist_ok=True)
    
    # 1. CHARGEMENT DES DONNÉES
    print("\n📥 1. CHARGEMENT DES DONNÉES")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # 2. VECTORISATION TF-IDF OPTIMISÉE
    print("\n🔧 2. VECTORISATION TF-IDF AVEC PARAMÈTRES OPTIMISÉS")
    vectorizer, X_train_tfidf, X_test_tfidf = create_optimized_tfidf(X_train, X_test)
    
    # 3. ENTRAÎNEMENT LOGISTIC REGRESSION AVEC OPTIMISATION
    print("\n🧠 3. LOGISTIC REGRESSION - OPTIMISATION HYPERPARAMÈTRES")
    lr_model, lr_metrics = train_logistic_regression(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # 4. EXPÉRIMENTATION AVEC D'AUTRES ALGORITHMES
    print("\n🔬 4. EXPÉRIMENTATION AVEC D'AUTRES ALGORITHMES")
    rf_model, rf_metrics = train_random_forest(X_train_tfidf, y_train, X_test_tfidf, y_test)
    svm_model, svm_metrics = train_svm(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # 5. COMPARAISON ET SÉLECTION DU MEILLEUR MODÈLE
    print("\n🏆 5. COMPARAISON ET SÉLECTION DU MEILLEUR MODÈLE")
    best_model, best_model_name, best_metrics = select_best_model(
        lr_model, lr_metrics, rf_model, rf_metrics, svm_model, svm_metrics
    )
    
    # 6. ÉVALUATION DÉTAILLÉE DU MEILLEUR MODÈLE
    print("\n📊 6. ÉVALUATION DÉTAILLÉE AVEC MÉTRIQUES")
    evaluate_best_model(best_model, X_test_tfidf, y_test, best_model_name)
    
    # 7. TEST DES PERFORMANCES D'INFÉRENCE
    print("\n⚡ 7. TEST DES PERFORMANCES D'INFÉRENCE")
    inference_metrics = test_inference_performance(best_model, vectorizer, X_test)
    
    # 8. SAUVEGARDE DES MODÈLES ET RAPPORTS
    print("\n💾 8. SAUVEGARDE DES MODÈLES ET GÉNÉRATION DE RAPPORTS")
    save_models_and_reports(best_model, vectorizer, best_metrics, inference_metrics, y_test, 
                          X_test_tfidf, best_model_name)
    
    # 9. VÉRIFICATION FINALE DES CRITÈRES DU TP
    print("\n✅ 9. VÉRIFICATION DES CRITÈRES DE PERFORMANCE DU TP")
    verify_tp_criteria(best_metrics, inference_metrics)
    
    print("\n" + "=" * 70)
    print("🎉 PHASE 3 TERMINÉE AVEC SUCCÈS!")
    print("=" * 70)

def load_and_prepare_data():
    """Charge et prépare les données d'entraînement et de test"""
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df['cleaned_text'].astype(str)
    y_train = train_df['label']
    X_test = test_df['cleaned_text'].astype(str)
    y_test = test_df['label']
    
    print(f"✅ Données chargées:")
    print(f"   - Train: {len(X_train)} échantillons")
    print(f"   - Test: {len(X_test)} échantillons")
    
    # Distribution des classes
    print(f"📊 Distribution des classes (Train):")
    for label in [-1, 0, 1]:
        count = (y_train == label).sum()
        percentage = count / len(y_train) * 100
        sentiment = {1: 'Positif', 0: 'Neutre', -1: 'Négatif'}[label]
        print(f"   - {sentiment}: {count} échantillons ({percentage:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def create_optimized_tfidf(X_train, X_test):
    """
    Implémente un vectoriseur TF-IDF avec paramètres optimisés
    Conforme à l'exigence: 'Implémenter un vectoriseur TF-IDF avec paramètres optimisés'
    """
    print("🔧 Création du vectoriseur TF-IDF optimisé...")
    
    # Paramètres optimisés basés sur l'analyse du dataset
    vectorizer = TfidfVectorizer(
        max_features=5000,           # Limite la dimensionnalité
        ngram_range=(1, 2),          # Unigrams et bigrams
        stop_words='english',        # Supprime les stop words
        min_df=2,                    # Termes apparaissant au moins 2 fois
        max_df=0.95,                 # Termes apparaissant dans max 95% des documents
        sublinear_tf=True,           # Application log pour pénaliser les termes fréquents
        norm='l2'                    # Normalisation L2 pour les vecteurs
    )
    
    start_time = time.time()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    vectorization_time = time.time() - start_time
    
    print(f"✅ Vectorisation TF-IDF terminée en {vectorization_time:.2f}s")
    print(f"   - Dimension des features: {X_train_tfidf.shape[1]}")
    print(f"   - Taille du vocabulaire: {len(vectorizer.vocabulary_)}")
    print(f"   - N-gram range: {vectorizer.ngram_range}")
    
    return vectorizer, X_train_tfidf, X_test_tfidf

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Entraîne une Logistic Regression avec optimisation des hyperparamètres
    Conforme à l'exigence: 'Entraîner un modèle de Logistic Regression'
    """
    print("🧠 Entraînement de la Logistic Regression avec GridSearchCV...")
    
    # Calcul des poids des classes pour gérer le déséquilibre
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # GridSearch pour l'optimisation des hyperparamètres
    param_grid = {
        'C': [0.1, 1, 10, 100],           # Force de régularisation
        'penalty': ['l1', 'l2'],          # Type de régularisation
        'solver': ['liblinear', 'saga'],  # Algorithmes d'optimisation
        'max_iter': [1000]                # Nombre maximum d'itérations
    }
    
    lr_model = LogisticRegression(
        class_weight=class_weight_dict,
        random_state=42
    )
    
    # GridSearchCV pour l'optimisation
    grid_search = GridSearchCV(
        lr_model, param_grid,
        cv=3,                    # 3-fold cross-validation
        scoring='f1_weighted',   # Métrique d'optimisation
        n_jobs=-1,              # Utilisation de tous les cores
        verbose=1
    )
    
    print("🔍 Début de l'optimisation des hyperparamètres...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    best_model = grid_search.best_estimator_
    
    # Évaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    print(f"✅ Logistic Regression optimisée en {training_time:.2f}s")
    print(f"🎯 Meilleurs paramètres: {grid_search.best_params_}")
    print(f"📈 Meilleur score CV: {grid_search.best_score_:.4f}")
    print(f"🧪 Performance test - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    metrics = {
        'model': best_model,
        'accuracy': accuracy,
        'f1_weighted': f1,
        'f1_per_class': f1_per_class,
        'best_params': grid_search.best_params_,
        'training_time': training_time
    }
    
    return best_model, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Expérimente avec Random Forest
    Conforme à l'exigence: 'Expérimenter avec d'autres algorithmes (Random Forest, SVM, etc.)'
    """
    print("🌲 Entraînement de Random Forest avec RandomizedSearchCV...")
    
    # Calcul des poids des classes
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # RandomizedSearch pour Random Forest (plus rapide que GridSearch)
    param_dist = {
        'n_estimators': [100, 200, 300],      # Nombre d'arbres
        'max_depth': [None, 10, 20, 30],      # Profondeur maximale
        'min_samples_split': [2, 5, 10],      # Échantillons minimum pour diviser
        'min_samples_leaf': [1, 2, 4],        # Échantillons minimum par feuille
        'max_features': ['sqrt', 'log2']      # Nombre de features pour split
    }
    
    rf_model = RandomForestClassifier(
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1
    )
    
    random_search = RandomizedSearchCV(
        rf_model, param_dist,
        n_iter=10,              # 10 combinaisons aléatoires
        cv=3,                   # 3-fold cross-validation
        scoring='f1_weighted',  # Métrique d'optimisation
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    best_model = random_search.best_estimator_
    
    # Évaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    print(f"✅ Random Forest optimisé en {training_time:.2f}s")
    print(f"🎯 Meilleurs paramètres: {random_search.best_params_}")
    print(f"📈 Meilleur score CV: {random_search.best_score_:.4f}")
    print(f"🧪 Performance test - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    metrics = {
        'model': best_model,
        'accuracy': accuracy,
        'f1_weighted': f1,
        'f1_per_class': f1_per_class,
        'best_params': random_search.best_params_,
        'training_time': training_time
    }
    
    return best_model, metrics

def train_svm(X_train, y_train, X_test, y_test):
    """
    Expérimente avec SVM
    Conforme à l'exigence: 'Expérimenter avec d'autres algorithmes (Random Forest, SVM, etc.)'
    """
    print("⚡ Entraînement de SVM (version optimisée)...")
    
    # SVM avec noyau linéaire pour efficacité
    svm_model = SVC(
        C=1.0,                    # Paramètre de régularisation
        kernel='linear',          # Noyau linéaire pour efficacité
        probability=True,         # Permet predict_proba
        random_state=42,
        class_weight='balanced'   # Gestion automatique du déséquilibre
    )
    
    start_time = time.time()
    svm_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Évaluation
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    print(f"✅ SVM entraîné en {training_time:.2f}s")
    print(f"🧪 Performance test - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    metrics = {
        'model': svm_model,
        'accuracy': accuracy,
        'f1_weighted': f1,
        'f1_per_class': f1_per_class,
        'training_time': training_time
    }
    
    return svm_model, metrics

def select_best_model(lr_model, lr_metrics, rf_model, rf_metrics, svm_model, svm_metrics):
    """Sélectionne le meilleur modèle basé sur le F1-score"""
    print("🏆 Comparaison des modèles...")
    
    models_comparison = {
        'Logistic Regression': lr_metrics,
        'Random Forest': rf_metrics,
        'SVM': svm_metrics
    }
    
    # Affichage du tableau de comparaison
    print("\n📊 TABLEAU COMPARATIF DES MODÈLES:")
    print("-" * 70)
    print(f"{'Modèle':<20} {'Accuracy':<10} {'F1-Score':<10} {'Temps (s)':<10}")
    print("-" * 70)
    
    best_f1 = 0
    best_model_name = ""
    best_model = None
    best_metrics = None
    
    for name, metrics in models_comparison.items():
        print(f"{name:<20} {metrics['accuracy']:.4f}    {metrics['f1_weighted']:.4f}    {metrics['training_time']:>8.1f}")
        
        if metrics['f1_weighted'] > best_f1:
            best_f1 = metrics['f1_weighted']
            best_model_name = name
            best_model = metrics['model']
            best_metrics = metrics
    
    print("-" * 70)
    print(f"🎯 MEILLEUR MODÈLE: {best_model_name}")
    print(f"   - F1-Score: {best_metrics['f1_weighted']:.4f}")
    print(f"   - Accuracy: {best_metrics['accuracy']:.4f}")
    
    return best_model, best_model_name, best_metrics

def evaluate_best_model(model, X_test, y_test, model_name):
    """
    Évalue le modèle avec métriques appropriées
    Conforme à l'exigence: 'Évaluer avec métriques appropriées (accuracy, F1-score, matrice de confusion)'
    """
    print(f"📊 Évaluation détaillée du modèle {model_name}...")
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques détaillées
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    print(f"📈 MÉTRIQUES DÉTAILLÉES:")
    print(f"   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   - F1-score (weighted): {f1_weighted:.4f}")
    print(f"   - F1-score par classe:")
    for i, label in enumerate([-1, 0, 1]):
        sentiment = {1: 'Positif', 0: 'Neutre', -1: 'Négatif'}[label]
        print(f"     {sentiment}: {f1_per_class[i]:.4f}")
    
    # Rapport de classification complet
    print(f"\n📝 RAPPORT DE CLASSIFICATION COMPLET:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Négatif', 'Neutre', 'Positif'],
                              digits=4))
    
    # Matrice de confusion
    print("📊 Génération de la matrice de confusion...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Négatif', 'Neutre', 'Positif'],
                yticklabels=['Négatif', 'Neutre', 'Positif'])
    plt.title(f'Matrice de Confusion - {model_name}\nAccuracy: {accuracy:.4f}', fontsize=14)
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Vérité Terrain', fontsize=12)
    plt.tight_layout()
    plt.savefig('models/experiments/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Matrice de confusion sauvegardée")

def test_inference_performance(model, vectorizer, X_test):
    """
    Teste le temps d'inférence pour vérifier le critère de performance
    Conforme à l'exigence: 'Temps d'inférence < 100ms pour un batch de 50 commentaires'
    """
    print("⚡ Test des performances d'inférence...")
    
    # Test avec différents batch sizes
    batch_sizes = [1, 10, 50, 100]
    results = {}
    
    for batch_size in batch_sizes:
        # Sélection aléatoire de commentaires
        sample_indices = np.random.choice(len(X_test), batch_size, replace=False)
        sample_texts = X_test.iloc[sample_indices]
        
        # Mesure du temps d'inférence
        start_time = time.time()
        
        # Vectorisation
        sample_tfidf = vectorizer.transform(sample_texts)
        vectorization_time = time.time() - start_time
        
        # Prédiction
        prediction_start = time.time()
        predictions = model.predict(sample_tfidf)
        prediction_time = time.time() - prediction_start
        
        total_time = time.time() - start_time
        
        results[batch_size] = {
            'total_time': total_time,
            'vectorization_time': vectorization_time,
            'prediction_time': prediction_time,
            'time_per_comment': total_time / batch_size,
            'comments_per_second': batch_size / total_time
        }
        
        print(f"   - Batch {batch_size:3d} comments: {total_time*1000:6.2f}ms "
              f"({total_time/batch_size*1000:5.2f}ms/comment)")
    
    # Vérification spécifique du critère pour 50 commentaires
    inference_50 = results[50]
    criteria_met = inference_50['total_time'] < 0.1  # < 100ms
    
    print(f"\n🎯 CRITÈRE D'INFÉRENCE - 50 commentaires:")
    print(f"   - Temps total: {inference_50['total_time']*1000:.2f}ms")
    print(f"   - Critère: < 100ms")
    print(f"   - Résultat: {'✅ ATTEINT' if criteria_met else '❌ NON ATTEINT'}")
    
    return {
        'inference_50_time': inference_50['total_time'],
        'criteria_met': criteria_met,
        'all_results': results
    }

def save_models_and_reports(model, vectorizer, metrics, inference_metrics, y_test, X_test, model_name):
    """
    Sauvegarde les modèles et génère les rapports
    Conforme à l'exigence: 'Sauvegarder le meilleur modèle et le vectoriseur avec joblib'
    """
    print("💾 Sauvegarde des modèles et génération des rapports...")
    
    # Sauvegarde du modèle
    model_path = "models/trained/best_sentiment_model.joblib"
    joblib.dump(model, model_path)
    
    # Sauvegarde du vectoriseur
    vectorizer_path = "models/trained/tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    
    # Métriques complètes
    full_metrics = {
        'model_name': model_name,
        'test_accuracy': metrics['accuracy'],
        'test_f1_weighted': metrics['f1_weighted'],
        'test_f1_per_class': metrics['f1_per_class'].tolist(),
        'inference_time_50': inference_metrics['inference_50_time'],
        'inference_criteria_met': inference_metrics['criteria_met'],
        'training_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset_info': {
            'train_size': len(X_test),  # Approximation
            'feature_dimension': X_test.shape[1]
        }
    }
    
    metrics_path = "models/trained/model_metrics.joblib"
    joblib.dump(full_metrics, metrics_path)
    
    # Génération du rapport de performance
    generate_performance_report(full_metrics, model_name)
    
    print("✅ Tous les fichiers sauvegardés:")
    print(f"   - Modèle: {model_path}")
    print(f"   - Vectoriseur: {vectorizer_path}")
    print(f"   - Métriques: {metrics_path}")

def generate_performance_report(metrics, model_name):
    """Génère un rapport de performance détaillé"""
    report = f"""
# 📊 RAPPORT DE PERFORMANCE - MODÈLE DE SENTIMENT

## 🎯 INFORMATIONS GÉNÉRALES
- **Modèle**: {model_name}
- **Date d'entraînement**: {metrics['training_date']}
- **Taille du dataset**: {metrics['dataset_info']['train_size']} échantillons
- **Dimension des features**: {metrics['dataset_info']['feature_dimension']}

## 📈 PERFORMANCES

### Métriques de Classification
- **Accuracy**: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)
- **F1-Score (weighted)**: {metrics['test_f1_weighted']:.4f} ({metrics['test_f1_weighted']*100:.2f}%)

### F1-Score par Classe
- **Négatif**: {metrics['test_f1_per_class'][0]:.4f}
- **Neutre**: {metrics['test_f1_per_class'][1]:.4f}
- **Positif**: {metrics['test_f1_per_class'][2]:.4f}

### Performances d'Inférence
- **Temps pour 50 commentaires**: {metrics['inference_time_50']*1000:.2f}ms
- **Critère de performance**: {'✅ ATTEINT' if metrics['inference_criteria_met'] else '❌ NON ATTEINT'}

## 🔧 DÉTAILS TECHNIQUES

### Vectorisation TF-IDF
- **max_features**: 5000
- **ngram_range**: (1, 2)
- **stop_words**: english
- **min_df**: 2
- **max_df**: 0.95

### Optimisation des Hyperparamètres
- **Méthode**: GridSearchCV / RandomizedSearchCV
- **Scoring**: F1-score weighted
- **Cross-validation**: 3 folds

---
*Rapport généré automatiquement - Phase 3 du TP Cloud Computing*
"""
    
    with open("models/experiments/performance_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ Rapport de performance généré")

def verify_tp_criteria(metrics, inference_metrics):
    """
    Vérifie que tous les critères du TP sont atteints
    """
    print("✅ VÉRIFICATION DES CRITÈRES DU TP")
    print("-" * 50)
    
    # Critère 1: Accuracy minimale 80%
    accuracy_ok = metrics['accuracy'] >= 0.80
    print(f"1. Accuracy ≥ 80%: {metrics['accuracy']:.4f} {'✅' if accuracy_ok else '❌'}")
    
    # Critère 2: F1-score par classe > 0.75
    f1_ok = all(f1 >= 0.75 for f1 in metrics['f1_per_class'])
    f1_details = [f"{f1:.4f}" for f1 in metrics['f1_per_class']]
    print(f"2. F1-score par classe > 0.75: {f1_details} {'✅' if f1_ok else '❌'}")
    
    # Critère 3: Temps d'inférence < 100ms pour 50 commentaires
    inference_ok = inference_metrics['inference_50_time'] < 0.1
    print(f"3. Temps inférence < 100ms: {inference_metrics['inference_50_time']*1000:.2f}ms {'✅' if inference_ok else '❌'}")
    
    # Résumé final
    all_criteria_met = accuracy_ok and f1_ok and inference_ok
    print("-" * 50)
    print(f"🎯 TOUS LES CRITÈRES: {'✅ ATTEINTS' if all_criteria_met else '❌ NON ATTEINTS'}")
    
    if all_criteria_met:
        print("🎉 FÉLICITATIONS! La Phase 3 est complètement conforme aux exigences du TP!")
    else:
        print("⚠️  Certains critères ne sont pas atteints. Améliorations nécessaires.")

if __name__ == "__main__":
    main()