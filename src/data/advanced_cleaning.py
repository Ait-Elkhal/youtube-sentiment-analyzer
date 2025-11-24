# src/data/advanced_cleaning.py
import pandas as pd
import re
import string
from pathlib import Path

def demonstrate_cleaning_pipeline():
    """
    Démontre un pipeline de nettoyage complet même sur des données déjà nettoyées
    """
    print("🧹 DÉMONSTRATION DU PIPELINE DE NETTOYAGE")
    
    # Charger les données originales standardisées
    df = pd.read_csv("data/raw/reddit_standardized.csv")
    
    print(f"Données initiales: {len(df)} commentaires")
    print("Exemple avant nettoyage:")
    print(df['text'].iloc[0][:200] + "...")
    
    def advanced_text_cleaning(text):
        """
        Nettoyage avancé même sur du texte déjà nettoyé
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # 1. Supprimer les URLs résiduelles
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
        
        # 2. Supprimer les mentions @
        text = re.sub(r'@\w+', '[USER]', text)
        
        # 3. Supprimer les caractères spéciaux non désirés
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # 4. Gestion des emojis - les supprimer
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # 5. Normaliser la ponctuation
        text = re.sub(r'\.{2,}', '...', text)  # Points de suspension
        text = re.sub(r'\!{2,}', '!', text)    # Points d'exclamation multiples
        text = re.sub(r'\?{2,}', '?', text)    # Points d'interrogation multiples
        
        # 6. Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 7. Mettre en minuscules (optionnel - dépend du cas d'usage)
        text = text.lower()
        
        return text
    
    # Appliquer le nettoyage avancé
    df['advanced_cleaned_text'] = df['text'].apply(advanced_text_cleaning)
    
    # Calculer les statistiques de nettoyage
    original_lengths = df['text'].str.len()
    cleaned_lengths = df['advanced_cleaned_text'].str.len()
    length_reduction = ((original_lengths - cleaned_lengths) / original_lengths * 100).mean()
    
    print(f"\n📊 Impact du nettoyage avancé:")
    print(f"  Réduction moyenne de longueur: {length_reduction:.1f}%")
    print(f"  Longueur moyenne originale: {original_lengths.mean():.1f} caractères")
    print(f"  Longueur moyenne après nettoyage: {cleaned_lengths.mean():.1f} caractères")
    
    print("\nExemple après nettoyage avancé:")
    print(df['advanced_cleaned_text'].iloc[0][:200] + "...")
    
    # Comparaison avant/après pour quelques exemples
    print("\n🔍 COMPARAISON AVANT/APRÈS:")
    sample_indices = [0, 10, 100]
    for idx in sample_indices:
        print(f"\n--- Exemple {idx} ---")
        print(f"AVANT: {df['text'].iloc[idx][:150]}...")
        print(f"APRÈS: {df['advanced_cleaned_text'].iloc[idx][:150]}...")
    
    # Sauvegarder la version avec nettoyage avancé
    output_path = Path("data/processed/train_advanced_cleaned.csv")
    df[['advanced_cleaned_text', 'label']].to_csv(output_path, index=False)
    
    print(f"\n💾 Version avec nettoyage avancé sauvegardée: {output_path}")
    print("✅ DÉMONSTRATION DE NETTOYAGE TERMINÉE")

def analyze_cleaning_impact():
    """
    Analyse l'impact des différentes étapes de nettoyage
    """
    print("\n📈 ANALYSE DE L'IMPACT DU NETTOYAGE")
    
    df_original = pd.read_csv("data/raw/reddit_standardized.csv")
    df_cleaned = pd.read_csv("data/processed/train_advanced_cleaned.csv")
    
    # Statistiques comparatives
    stats = {
        'Original': {
            'count': len(df_original),
            'avg_length': df_original['text'].str.len().mean(),
            'min_length': df_original['text'].str.len().min(),
            'max_length': df_original['text'].str.len().max()
        },
        'Nettoyé': {
            'count': len(df_cleaned),
            'avg_length': df_cleaned['advanced_cleaned_text'].str.len().mean(),
            'min_length': df_cleaned['advanced_cleaned_text'].str.len().min(),
            'max_length': df_cleaned['advanced_cleaned_text'].str.len().max()
        }
    }
    
    print("Comparaison des statistiques:")
    print(f"{'Metric':<15} {'Original':<10} {'Nettoyé':<10} {'Différence':<12}")
    print("-" * 50)
    
    for metric in ['avg_length', 'min_length', 'max_length']:
        orig = stats['Original'][metric]
        clean = stats['Nettoyé'][metric]
        diff = clean - orig
        diff_pct = (diff / orig) * 100 if orig != 0 else 0
        
        print(f"{metric:<15} {orig:<10.1f} {clean:<10.1f} {diff:+.1f} ({diff_pct:+.1f}%)")

if __name__ == "__main__":
    demonstrate_cleaning_pipeline()
    analyze_cleaning_impact()