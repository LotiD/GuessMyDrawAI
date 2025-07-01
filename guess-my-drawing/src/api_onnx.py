#!/usr/bin/env python3
"""
API Flask ultra-légère avec ONNX Runtime
Version optimisée pour déploiement web (Vercel, Railway, etc.)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# Configuration Flask
app = Flask(__name__)
CORS(app)  # Pour le frontend

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Classes et emojis (identique au modèle)
CLASSES = ['cat', 'house', 'car', 'tree']
CLASS_EMOJIS = ['🐱', '🏠', '🚗', '🌳']
CLASS_COLORS = ['#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7']

# Variables globales
session = None
model_info = {}

def load_onnx_model():
    """Charge le modèle ONNX"""
    global session, model_info
    
    try:
        model_path = "../models/guess_my_drawing.onnx"
        
        if not Path(model_path).exists():
            logger.error(f"❌ Modèle ONNX non trouvé: {model_path}")
            return False
        
        # Charger le modèle ONNX
        logger.info(f"📦 Chargement modèle ONNX: {model_path}")
        session = ort.InferenceSession(model_path)
        
        # Informations du modèle
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        model_info = {
            'model_path': model_path,
            'input_name': input_info.name,
            'input_shape': input_info.shape,
            'output_name': output_info.name,
            'output_shape': output_info.shape,
            'classes': len(CLASSES),
            'runtime': 'ONNX Runtime',
            'loaded_at': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Modèle ONNX chargé!")
        logger.info(f"   Input: {input_info.name} {input_info.shape}")
        logger.info(f"   Output: {output_info.name} {output_info.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur chargement ONNX: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    """Documentation API"""
    docs = {
        "service": "Guess My Drawing API (ONNX)",
        "version": "2.0",
        "description": "API de reconnaissance de dessins ultra-légère avec ONNX Runtime",
        "model": {
            "runtime": "ONNX Runtime",
            "classes": len(CLASSES),
            "input_shape": model_info.get('input_shape', 'N/A')
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
        "status": "healthy" if session is not None else "error",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": session is not None,
        "runtime": "ONNX Runtime"
    }
    
    if session is not None:
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
        if session is None:
            return jsonify({"error": "Modèle ONNX non chargé"}), 500
        
        # Image de test aléatoire
        test_image = np.random.rand(1, 1, 28, 28).astype(np.float32)
        
        # Prédiction ONNX
        outputs = session.run(None, {model_info['input_name']: test_image})
        logits = outputs[0][0]
        
        # Appliquer softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        pred_class = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_class])
        
        return jsonify({
            "test": "success",
            "prediction": CLASSES[pred_class],
            "emoji": CLASS_EMOJIS[pred_class],
            "confidence": confidence,
            "percentage": confidence * 100,
            "runtime": "ONNX Runtime",
            "message": "API ONNX fonctionnelle!"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_drawing():
    """Prédiction d'un dessin avec ONNX Runtime"""
    start_time = datetime.now()
    
    try:
        if session is None:
            return jsonify({"error": "Modèle ONNX non chargé"}), 500
        
        # Récupérer les données
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Données image manquantes"}), 400
        
        image_data = data['image']
        
        # Validation
        if not isinstance(image_data, list) or len(image_data) != 784:
            return jsonify({"error": "Format image invalide (784 pixels requis)"}), 400
        
        # Préparation pour ONNX
        try:
            # Conversion et normalisation (identique à l'entraînement)
            image_array = np.array(image_data, dtype=np.float32).reshape(1, 1, 28, 28)
            image_array = image_array / 255.0  # [0,1]
            image_array = (image_array - 0.5) / 0.5  # [-1,1]
            
        except Exception as e:
            return jsonify({"error": f"Erreur traitement image: {str(e)}"}), 400
        
        # Prédiction ONNX (ultra-rapide!)
        outputs = session.run(None, {model_info['input_name']: image_array})
        logits = outputs[0][0]
        
        # IMPORTANT: Appliquer softmax aux logits pour obtenir des probabilités
        exp_logits = np.exp(logits - np.max(logits))  # Stabilité numérique
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Créer les résultats
        predictions = []
        for i, (class_name, emoji, color) in enumerate(zip(CLASSES, CLASS_EMOJIS, CLASS_COLORS)):
            confidence = float(probabilities[i])
            
            predictions.append({
                "class": class_name,
                "confidence": confidence,
                "percentage": confidence * 100,
                "emoji": emoji,
                "color": color
            })
        
        # Trier par confiance
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Stats
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        predicted_class = predictions[0]['class']
        max_confidence = predictions[0]['confidence']
        
        # Log
        logger.info(f"🤖 ONNX Prédiction: {predicted_class} ({max_confidence:.1%}) en {processing_time:.0f}ms")
        
        return jsonify({
            "success": True,
            "predictions": predictions,
            "predicted_class": predicted_class,
            "max_confidence": max_confidence,
            "processing_time_ms": processing_time,
            "runtime": "ONNX Runtime",
            "timestamp": start_time.isoformat()
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Erreur prédiction ONNX: {error_msg}")
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "predictions": [],
            "runtime": "ONNX Runtime",
            "timestamp": start_time.isoformat()
        }), 500

if __name__ == '__main__':
    print("🚀 DÉMARRAGE API ONNX - GUESS MY DRAWING")
    print("=" * 50)
    
    success = load_onnx_model()
    
    if success:
        print(f"✅ API ONNX prête!")
        print(f"🌐 Endpoints disponibles:")
        print(f"   GET  / → Documentation")
        print(f"   GET  /health → Status")
        print(f"   GET  /classes → Classes")
        print(f"   GET  /test → Test rapide")
        print(f"   POST /predict → Prédiction ONNX")
        print(f"")
        print(f"⚡ Runtime: ONNX (ultra-léger)")
        print(f"🔥 Prêt pour Vercel/Railway/Render!")
        print(f"\n🔥 Lancement serveur...")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Impossible de charger le modèle ONNX!")
        print("💡 Exécutez d'abord: python export_to_onnx.py")