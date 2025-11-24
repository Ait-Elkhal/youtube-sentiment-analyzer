# src/data/inspect_dataset.py
import pandas as pd

def inspect_dataset():
    # Charger le dataset
    df = pd.read_csv("data/raw/reddit.csv")
    
    print("🔍 Inspection du dataset:")
    print(f"Shape: {df.shape}")
    print(f"Colonnes: {df.columns.tolist()}")
    print(f"\nAperçu des premières lignes:")
    print(df.head())
    print(f"\nInformations sur les colonnes:")
    print(df.info())
    print(f"\nValeurs manquantes:")
    print(df.isnull().sum())

if __name__ == "__main__":
    inspect_dataset()