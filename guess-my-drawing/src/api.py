#!/usr/bin/env python3
"""
API Flask pour Guess My Drawing
Sert le modèle CNN entraîné avec endpoints optimisés
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import io
from PIL import Image
import logging
from datetime import datetime
import os
import glob
from pathlib import Path

# Configuration Flask
app = Flask(__name__)
CORS(app)  # Pour permettre les requêtes depuis le frontend

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Classes et emojis (sans dog pour éviter confusion avec cat)
CLASSES = ['cat', 'house', 'car', 'tree']
CLASS_EMOJIS = ['🐱', '🏠', '🚗', '🌳']
CLASS_COLORS = ['#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7']

# Variables globales
model = None
device = None
model_info = {}

class DrawingCNN(nn.Module):
    """Architecture CNN identique à l'entraînement"""
    
    def __init__(self, num_classes=5):
        super(DrawingCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model():
    """Charge le modèle CNN en mémoire"""
    global model, device, model_info
    
    try:
        device = torch.device('cpu')  # Force CPU pour simplicité
        logger.info(f"🔥 Device: {device}")
        
        # Trouver le modèle
        model_files = glob.glob("../models/best_model_*.pth")
        if not model_files:
            raise FileNotFoundError("Aucun modèle trouvé!")
        
        latest_model = max(model_files, key=lambda x: x.split('_')[-1])
        logger.info(f"📦 Chargement modèle: {latest_model}")
        
        # Charger le checkpoint pour détecter le nombre de classes
        checkpoint = torch.load(latest_model, map_location=device)
        
        # Détecter automatiquement le nombre de classes
        fc2_shape = checkpoint['model_state_dict']['fc2.weight'].shape
        num_classes = fc2_shape[0]
        logger.info(f"🎯 Détection: {num_classes} classes")
        
        # Créer le modèle avec le bon nombre de classes
        model = DrawingCNN(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Stocker les infos
        model_info = {
            'model_path': latest_model,
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_accuracy': checkpoint.get('val_acc', 'N/A'),
            'train_accuracy': checkpoint.get('train_acc', 'N/A'),
            'parameters': sum(p.numel() for p in model.parameters()),
            'size_mb': sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024,
            'device': str(device),
            'loaded_at': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Modèle chargé!")
        logger.info(f"   Val Accuracy: {model_info['val_accuracy']:.2f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    """Documentation API"""
    docs = {
        "service": "Guess My Drawing API",
        "version": "1.0",
        "description": "API de reconnaissance de dessins avec CNN PyTorch",
        "model": {
            "accuracy": f"{model_info.get('val_accuracy', 'N/A'):.2f}%" if isinstance(model_info.get('val_accuracy'), (int, float)) else "N/A",
            "classes": len(CLASSES),
            "parameters": f"{model_info.get('parameters', 0):,}"
        },
        "endpoints": {
            "GET /health": "Status de l'API",
            "GET /classes": "Liste des classes", 
            "GET /test": "Test rapide",
            "POST /predict": "Prédiction de dessin"
        }
    }
    return jsonify(docs)

@app.route('/health', methods=['GET'])
def health():
    """Status de l'API"""
    status = {
        "status": "healthy" if model is not None else "error",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }
    
    if model is not None:
        status.update(model_info)
    
    return jsonify(status)

@app.route('/classes', methods=['GET'])
def get_classes():
    """Liste des classes"""
    classes_info = []
    for i, (cls, emoji, color) in enumerate(zip(CLASSES, CLASS_EMOJIS, CLASS_COLORS)):
        classes_info.append({
            "id": i,
            "name": cls,
            "emoji": emoji,
            "color": color,
            "display_name": f"{emoji} {cls.capitalize()}"
        })
    
    return jsonify({
        "classes": classes_info,
        "total": len(CLASSES)
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test avec image simulée"""
    try:
        if model is None:
            return jsonify({"error": "Modèle non chargé"}), 500
        
        # Image de test aléatoire
        test_image = torch.randn(1, 1, 28, 28).to(device)
        
        with torch.no_grad():
            outputs = model(test_image)
            probabilities = F.softmax(outputs, dim=1)
            pred_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0][pred_class].item()
        
        return jsonify({
            "test": "success",
            "prediction": CLASSES[pred_class],
            "emoji": CLASS_EMOJIS[pred_class],
            "confidence": float(confidence),
            "percentage": float(confidence) * 100,
            "message": "API fonctionnelle!"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_drawing():
    """Prédiction d'un dessin"""
    start_time = datetime.now()
    
    try:
        if model is None:
            return jsonify({"error": "Modèle non chargé"}), 500
        
        # Récupérer les données
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Données image manquantes"}), 400
        
        image_data = data['image']
        
        # Validation des données
        if not isinstance(image_data, list) or len(image_data) != 784:  # 28*28 = 784
            return jsonify({"error": "Format image invalide (doit être 784 pixels)"}), 400
        
        # Convertir en tensor PyTorch
        try:
            # Normaliser les données [0,255] vers [0,1] puis vers [-1,1]
            image_array = np.array(image_data, dtype=np.float32).reshape(1, 1, 28, 28)
            image_array = image_array / 255.0  # [0,1]
            image_array = (image_array - 0.5) / 0.5  # [-1,1] comme pendant l'entraînement
            
            image_tensor = torch.from_numpy(image_array).to(device)
            
        except Exception as e:
            return jsonify({"error": f"Erreur conversion image: {str(e)}"}), 400
        
        # Prédiction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze()
            
            # Le nouveau modèle a directement 4 classes dans l'ordre correct
            predictions = []
            
            # Log pour debug
            logger.info(f"🔍 Probabilités brutes: {[f'{p:.3f}' for p in probabilities]}")
            
            for i, (class_name, emoji, color) in enumerate(zip(CLASSES, CLASS_EMOJIS, CLASS_COLORS)):
                confidence = float(probabilities[i])
                
                predictions.append({
                    "class": class_name,
                    "confidence": confidence,
                    "percentage": confidence * 100,
                    "emoji": emoji,
                    "color": color
                })
                
            # Log des prédictions détaillées
            for pred in predictions:
                logger.info(f"   {pred['emoji']} {pred['class']:8s}: {pred['confidence']*100:5.1f}%")
            
            # Trier par confiance décroissante
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Stats
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            max_confidence = predictions[0]['confidence']
            predicted_class = predictions[0]['class']
            
        # Logs
        logger.info(f"🤖 Prédiction: {predicted_class} ({max_confidence:.1%}) en {processing_time:.0f}ms")
        
        return jsonify({
            "success": True,
            "predictions": predictions,
            "predicted_class": predicted_class,
            "max_confidence": max_confidence,
            "processing_time_ms": processing_time,
            "timestamp": start_time.isoformat(),
            "model_info": {
                "accuracy": model_info.get('val_accuracy', 'N/A'),
                "parameters": model_info.get('parameters', 0)
            }
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Erreur prédiction: {error_msg}")
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "predictions": [],
            "timestamp": start_time.isoformat()
        }), 500

if __name__ == '__main__':
    print("🚀 DÉMARRAGE API GUESS MY DRAWING")
    print("=" * 40)
    
    success = load_model()
    
    if success:
        print(f"✅ API prête!")
        print(f"🌐 Endpoints disponibles:")
        print(f"   GET  / → Documentation")
        print(f"   GET  /health → Status")
        print(f"   GET  /classes → Classes")
        print(f"   GET  /test → Test rapide")
        print(f"   POST /predict → Prédiction de dessin")
        print(f"\n🔥 Lancement serveur...")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Impossible de charger le modèle!")
