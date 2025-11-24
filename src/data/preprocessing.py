import pandas as pd
import re
import string
from pathlib import Path

def light_cleaning(text):
    """
    Nettoyage léger supplémentaire si nécessaire
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les espaces en début et fin
    text = text.strip()
    
    return text
def handle_class_imbalance(df):
    """
    Gère le déséquilibre des classes si nécessaire
    """
    from sklearn.utils import resample
    
    # Analyser le déséquilibre
    label_counts = df['label'].value_counts()
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"📈 Ratio de déséquilibre: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 1.5:  # Seuil arbitraire
        print("⚖️  Application de l'oversampling...")
        
        # Oversampling de la classe minoritaire
        dfs = []
        for label in df['label'].unique():
            df_label = df[df['label'] == label]
            if len(df_label) < max_count:
                df_upsampled = resample(df_label, 
                                      replace=True, 
                                      n_samples=max_count, 
                                      random_state=42)
                dfs.append(df_upsampled)
            else:
                dfs.append(df_label)
        
        balanced_df = pd.concat(dfs)
        print(f"✅ Dataset équilibré: {len(balanced_df)} échantillons")
        return balanced_df
    else:
        print("✅ Déséquilibre acceptable, pas de traitement nécessaire")
        return df

def prepare_train_test_split():
    """
    Prépare le split train/test reproductible
    """
    # Charger les données standardisées
    input_path = Path("data/raw/reddit_standardized.csv")
    df = pd.read_csv(input_path)
    
    print("🔧 Préparation des données pour l'entraînement...")
    
    # Appliquer un nettoyage léger supplémentaire
    df['cleaned_text'] = df['text'].apply(light_cleaning)
    
    # Supprimer les textes vides après nettoyage
    initial_count = len(df)
    df = df[df['cleaned_text'].str.len() > 0]
    final_count = len(df)
    
    print(f"📝 Textes après nettoyage: {final_count}/{initial_count}")
    
    # Analyser la longueur des textes
    df['text_length'] = df['cleaned_text'].str.len()
    
    # Split train/test (80/20)
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,  # Pour la reproductibilité
        stratify=df['label']  # Conserver la distribution des classes
    )
    
    print(f"📊 Split train/test:")
    print(f"   - Train: {len(train_df)} échantillons ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   - Test: {len(test_df)} échantillons ({len(test_df)/len(df)*100:.1f}%)")
    
    # Vérifier la distribution des classes dans chaque split
    print("📈 Distribution des classes:")
    for split_name, split_df in [('Train', train_df), ('Test', test_df)]:
        print(f"   {split_name}:")
        for label in [-1, 0, 1]:
            count = len(split_df[split_df['label'] == label])
            percentage = count / len(split_df) * 100
            sentiment = {1: 'Positif', 0: 'Neutre', -1: 'Négatif'}[label]
            print(f"     {sentiment}: {count} ({percentage:.1f}%)")
    
    # Sauvegarder les datasets
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(processed_path / "train.csv", index=False)
    test_df.to_csv(processed_path / "test.csv", index=False)
    
    print(f"💾 Datasets sauvegardés:")
    print(f"   - {processed_path / 'train.csv'}")
    print(f"   - {processed_path / 'test.csv'}")
    
    return train_df, test_df

if __name__ == "__main__":
    # Préparer les données
    train_df, test_df = prepare_train_test_split()
    
    # Aperçu
    print("\n👀 Aperçu du dataset d'entraînement:")
    print(train_df[['cleaned_text', 'label', 'text_length']].head())
    
    print("\n📏 Statistiques de longueur:")
    print(f"Train - Longueur moyenne: {train_df['text_length'].mean():.1f}")
    print(f"Test - Longueur moyenne: {test_df['text_length'].mean():.1f}")