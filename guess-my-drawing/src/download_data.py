#!/usr/bin/env python3
"""
Téléchargeur de données Quick Draw pour Guess My Drawing
Télécharge les 5 classes de base : cat, dog, house, car, tree
"""

import os
import requests
import numpy as np
from pathlib import Path

# Classes à télécharger (MVP)
CLASSES = ['cat', 'dog', 'house', 'car', 'tree']

# URL de base Quick Draw
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

def download_class_data(class_name, data_dir):
    """Télécharge les données pour une classe donnée"""
    url = f"{BASE_URL}/{class_name}.npy"
    filepath = data_dir / f"{class_name}.npy"
    
    print(f"📥 Téléchargement de {class_name}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Téléchargement avec barre de progression
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ {class_name}.npy téléchargé ({downloaded / 1024 / 1024:.1f} MB)")
        
        # Vérification des données
        data = np.load(filepath)
        print(f"   📊 Shape: {data.shape} (images: {len(data)})")
        
        return True
        
    except requests.RequestException as e:
        print(f"❌ Erreur lors du téléchargement de {class_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue pour {class_name}: {e}")
        return False

def main():
    """Fonction principale"""
    print("🎨 TÉLÉCHARGEMENT DES DONNÉES QUICK DRAW")
    print("=" * 50)
    
    # Créer le dossier data s'il n'existe pas
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"📁 Dossier de destination: {data_dir}")
    print(f"🎯 Classes à télécharger: {', '.join(CLASSES)}")
    print()
    
    # Téléchargement de chaque classe
    success_count = 0
    for class_name in CLASSES:
        if download_class_data(class_name, data_dir):
            success_count += 1
        print()
    
    # Résumé
    print("📋 RÉSUMÉ DU TÉLÉCHARGEMENT")
    print("-" * 30)
    print(f"✅ Réussis: {success_count}/{len(CLASSES)}")
    print(f"❌ Échecs: {len(CLASSES) - success_count}/{len(CLASSES)}")
    
    if success_count == len(CLASSES):
        print("\n🎉 Tous les datasets sont prêts !")
        print("🚀 Prochaine étape: exploration des données")
    else:
        print("\n⚠️ Certains téléchargements ont échoué")
        print("🔄 Relance le script pour réessayer")

if __name__ == "__main__":
    main() 