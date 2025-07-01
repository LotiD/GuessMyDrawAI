#!/usr/bin/env python3
"""
Export du modèle PyTorch vers ONNX
Convertit le modèle .pth en .onnx pour déploiement léger
"""

import torch
from model import DrawingCNN
from pathlib import Path

def export_model():
    """Exporte le modèle PyTorch vers ONNX"""
    
    print("🔄 EXPORT PYTORCH → ONNX")
    print("=" * 40)
    
    # Vérifier que le modèle .pth existe
    model_path = Path("../models/best_model_20250701_170421.pth")
    if not model_path.exists():
        print("❌ Modèle PyTorch non trouvé!")
        print(f"   Cherché: {model_path}")
        return False
    
    try:
        # Charger le modèle PyTorch
        print(f"📦 Chargement: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Créer le modèle
        model = DrawingCNN(num_classes=4)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Modèle chargé: 4 classes")
        
        # Input factice (même format que l'entraînement)
        dummy_input = torch.randn(1, 1, 28, 28)
        
        # Export ONNX
        onnx_path = "../models/guess_my_drawing.onnx"
        print(f"🔄 Export vers: {onnx_path}")
        
        torch.onnx.export(
            model,                           # Le modèle
            dummy_input,                     # Input exemple
            onnx_path,                       # Fichier de sortie
            input_names=['input'],           # Nom input
            output_names=['output'],         # Nom output
            dynamic_axes={                   # Batch size dynamique
                'input': {0: 'batch_size'}, 
                'output': {0: 'batch_size'}
            },
            opset_version=11                 # Version ONNX
        )
        
        # Vérifier que le fichier a été créé
        onnx_file = Path(onnx_path)
        if onnx_file.exists():
            size_mb = onnx_file.stat().st_size / (1024 * 1024)
            print(f"✅ Export réussi!")
            print(f"   Fichier: {onnx_path}")
            print(f"   Taille: {size_mb:.1f} MB")
            print(f"   Format: ONNX v11")
            return True
        else:
            print("❌ Export échoué - fichier non créé")
            return False
            
    except Exception as e:
        print(f"❌ Erreur export: {e}")
        return False

if __name__ == "__main__":
    success = export_model()
    if success:
        print("\n🎉 Prêt pour déploiement ONNX Runtime!")
    else:
        print("\n💥 Export échoué")