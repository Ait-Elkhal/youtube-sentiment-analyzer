import pandas as pd
import os
import requests
from pathlib import Path

def download_reddit_dataset():
    """
    Télécharge le dataset Reddit depuis GitHub
    """
    # URL du dataset
    url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
    
    # Chemins des dossiers
    raw_data_path = Path("data/raw")
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    # Chemin complet du fichier
    file_path = raw_data_path / "reddit.csv"
    
    print("📥 Téléchargement du dataset Reddit...")
    
    try:
        # Télécharger le fichier
        response = requests.get(url)
        response.raise_for_status()  # Vérifier les erreurs HTTP
        
        # Sauvegarder le fichier
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Dataset téléchargé avec succès: {file_path}")
        
        # Charger et analyser le dataset
        df = pd.read_csv(file_path)
        
        # Renommer les colonnes pour la cohérence
        df = df.rename(columns={'clean_comment': 'text', 'category': 'label'})
        
        # Supprimer les lignes avec du texte manquant
        initial_count = len(df)
        df = df.dropna(subset=['text'])
        final_count = len(df)
        removed_count = initial_count - final_count
        
        print(f"📊 Statistiques du dataset:")
        print(f"   - Commentaires initiaux: {initial_count}")
        print(f"   - Commentaires après nettoyage: {final_count}")
        print(f"   - Commentaires supprimés (NaN): {removed_count}")
        
        # Distribution des labels
        print(f"   - Distribution des labels:")
        label_distribution = df['label'].value_counts().sort_index()
        for label, count in label_distribution.items():
            sentiment = {1: 'Positif', 0: 'Neutre', -1: 'Négatif'}.get(label, label)
            print(f"     {sentiment} ({label}): {count} échantillons ({count/len(df)*100:.1f}%)")
        
        # Vérifier la taille minimale
        min_samples = 300
        adequate_sizes = all(count >= min_samples for count in label_distribution)
        if adequate_sizes:
            print(f"✅ Taille adéquate (au moins {min_samples} par classe)")
        else:
            print(f"⚠️  Certaines classes ont moins de {min_samples} échantillons")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        return None

def validate_dataset(df):
    """
    Valide la structure et la qualité du dataset
    """
    print("\n🔍 Validation du dataset...")
    
    # Vérifier les colonnes requises
    required_columns = ['text', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Colonnes manquantes: {missing_columns}")
        return False
    else:
        print("✅ Toutes les colonnes requises sont présentes")
    
    # Vérifier les valeurs manquantes
    missing_values = df.isnull().sum()
    print("📋 Valeurs manquantes par colonne:")
    for col, count in missing_values.items():
        print(f"   - {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Vérifier les types de données
    print("📝 Types de données:")
    print(f"   - text: {df['text'].dtype}")
    print(f"   - label: {df['label'].dtype}")
    
    # Vérifier la longueur des textes
    text_lengths = df['text'].str.len()
    print("📏 Statistiques de longueur de texte:")
    print(f"   - Moyenne: {text_lengths.mean():.1f} caractères")
    print(f"   - Médiane: {text_lengths.median():.1f} caractères")
    print(f"   - Min: {text_lengths.min()} caractères")
    print(f"   - Max: {text_lengths.max()} caractères")
    
    return True

if __name__ == "__main__":
    # Télécharger le dataset
    df = download_reddit_dataset()
    
    if df is not None:
        # Valider le dataset
        validate_dataset(df)
        
        # Sauvegarder la version standardisée
        output_path = Path("data/raw/reddit_standardized.csv")
        df.to_csv(output_path, index=False)
        print(f"\n💾 Dataset standardisé sauvegardé: {output_path}")
        
        # Aperçu des données
        print("\n👀 Aperçu des premières lignes:")
        print(df.head())
        
        print("\n🎯 Exemples par sentiment:")
        for label in [-1, 0, 1]:
            sentiment_name = {1: 'Positif', 0: 'Neutre', -1: 'Négatif'}[label]
            sample = df[df['label'] == label].iloc[0]['text']
            print(f"   {sentiment_name} ({label}): {sample[:100]}...")