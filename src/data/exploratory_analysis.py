# src/data/exploratory_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration pour les graphiques
plt.style.use('default')
sns.set_palette("husl")

def comprehensive_analysis():
    """
    Analyse exploratoire complète du dataset
    """
    print("🔍 DÉBUT DE L'ANALYSE EXPLORATOIRE")
    
    # Charger les données d'entraînement
    train_path = Path("data/processed/train.csv")
    df = pd.read_csv(train_path)
    
    print(f"📊 Dataset analysé: {len(df)} commentaires d'entraînement")
    
    # 1. DISTRIBUTION DES CLASSES
    print("\n🎯 1. DISTRIBUTION DES CLASSES DE SENTIMENT")
    
    plt.figure(figsize=(12, 6))
    
    # Données pour le graphique
    label_counts = df['label'].value_counts().sort_index()
    labels_map = {-1: 'Négatif', 0: 'Neutre', 1: 'Positif'}
    label_names = [labels_map[x] for x in label_counts.index]
    
    # Couleurs pour chaque sentiment
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
    
    # Graphique à barres
    plt.subplot(1, 2, 1)
    bars = plt.bar(label_names, label_counts.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, label_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                f'{count}\n({count/len(df)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.title('Distribution des Sentiments\nDataset d\'Entraînement', fontsize=14, fontweight='bold')
    plt.ylabel('Nombre de Commentaires')
    plt.grid(axis='y', alpha=0.3)
    
    # Camembert
    plt.subplot(1, 2, 2)
    plt.pie(label_counts.values, labels=label_names, colors=colors, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 12})
    plt.title('Répartition des Sentiments', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/processed/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. ANALYSE DES LONGUEURS DE TEXTE
    print("\n📏 2. ANALYSE DES LONGUEURS DE TEXTE")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogramme général
    axes[0, 0].hist(df['text_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(df['text_length'].mean(), color='red', linestyle='--', 
                      label=f'Moyenne: {df["text_length"].mean():.1f}')
    axes[0, 0].axvline(df['text_length'].median(), color='green', linestyle='--', 
                      label=f'Médiane: {df["text_length"].median():.1f}')
    axes[0, 0].set_xlabel('Longueur (caractères)')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].set_title('Distribution des Longueurs de Texte')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Boxplot par sentiment
    df['sentiment'] = df['label'].map(labels_map)
    sns.boxplot(data=df, x='sentiment', y='text_length', ax=axes[0, 1], palette=colors)
    axes[0, 1].set_xlabel('Sentiment')
    axes[0, 1].set_ylabel('Longueur (caractères)')
    axes[0, 1].set_title('Longueur des Textes par Sentiment')
    axes[0, 1].grid(alpha=0.3)
    
    # Densité par sentiment
    for sentiment, color in zip(['Négatif', 'Neutre', 'Positif'], colors):
        subset = df[df['sentiment'] == sentiment]
        axes[1, 0].hist(subset['text_length'], bins=30, alpha=0.6, label=sentiment, 
                       color=color, density=True)
    axes[1, 0].set_xlabel('Longueur (caractères)')
    axes[1, 0].set_ylabel('Densité')
    axes[1, 0].set_title('Densité des Longueurs par Sentiment')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Top 20 mots les plus fréquents (analyse basique)
    from collections import Counter
    all_text = ' '.join(df['cleaned_text'].astype(str)).lower()
    words = all_text.split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(20)
    
    words, counts = zip(*common_words)
    axes[1, 1].barh(range(len(words)), counts, color='lightcoral', alpha=0.7)
    axes[1, 1].set_yticks(range(len(words)))
    axes[1, 1].set_yticklabels(words, fontsize=9)
    axes[1, 1].set_xlabel('Fréquence')
    axes[1, 1].set_title('20 Mots les Plus Fréquents')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/text_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. STATISTIQUES DÉTAILLÉES
    print("\n📈 3. STATISTIQUES DÉTAILLÉES")
    
    print(f"Dataset: {len(df)} commentaires")
    
    print("\n📊 Distribution des classes:")
    for label, count in label_counts.items():
        sentiment_name = labels_map[label]
        percentage = count / len(df) * 100
        print(f"  {sentiment_name} ({label}): {count:>5} échantillons ({percentage:5.1f}%)")
    
    print(f"\n📏 Analyse des longueurs de texte:")
    print(f"  Moyenne: {df['text_length'].mean():.1f} caractères")
    print(f"  Médiane: {df['text_length'].median():.1f} caractères")
    print(f"  Écart-type: {df['text_length'].std():.1f} caractères")
    print(f"  Minimum: {df['text_length'].min()} caractères")
    print(f"  Maximum: {df['text_length'].max()} caractères")
    print(f"  95e percentile: {df['text_length'].quantile(0.95):.1f} caractères")
    
    print(f"\n📏 Longueur moyenne par sentiment:")
    for sentiment in ['Négatif', 'Neutre', 'Positif']:
        avg_length = df[df['sentiment'] == sentiment]['text_length'].mean()
        print(f"  {sentiment}: {avg_length:.1f} caractères")
    
    # 4. ANALYSE DU DÉSÉQUILIBRE
    print("\n⚖️ 4. ANALYSE DU DÉSÉQUILIBRE DES CLASSES")
    
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"  Classe majoritaire: {labels_map[label_counts.idxmax()]} ({max_count} échantillons)")
    print(f"  Classe minoritaire: {labels_map[label_counts.idxmin()]} ({min_count} échantillons)")
    print(f"  Ratio de déséquilibre: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 1.5:
        print("  ⚠️  Déséquilibre détecté - à considérer pendant l'entraînement")
        print("  Recommandation: Utiliser class_weight='balanced' ou oversampling")
    else:
        print("  ✅ Déséquilibre acceptable")
    
    # 5. QUALITÉ DES DONNÉES
    print("\n🔍 5. QUALITÉ DES DONNÉES")
    
    # Textes très courts
    short_texts = df[df['text_length'] < 10]
    print(f"  Textes très courts (<10 caractères): {len(short_texts)} ({len(short_texts)/len(df)*100:.1f}%)")
    
    # Textes très longs
    long_texts = df[df['text_length'] > 1000]
    print(f"  Textes très longs (>1000 caractères): {len(long_texts)} ({len(long_texts)/len(df)*100:.1f}%)")
    
    # Textes dupliqués
    duplicates = df.duplicated(subset=['cleaned_text']).sum()
    print(f"  Textes dupliqués: {duplicates} ({duplicates/len(df)*100:.1f}%)")
    
    print(f"\n💾 Graphiques sauvegardés dans data/processed/")
    print("✅ ANALYSE EXPLORATOIRE TERMINÉE")

if __name__ == "__main__":
    comprehensive_analysis()