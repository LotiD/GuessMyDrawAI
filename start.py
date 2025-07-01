#!/usr/bin/env python3
"""
Script de démarrage pour Guess My Drawing
Lance automatiquement l'API et le frontend
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def start_application():
    """Lance l'application complète"""
    
    print("🚀 DÉMARRAGE GUESS MY DRAWING")
    print("=" * 40)
    
    # Vérifier si le modèle existe
    model_dir = Path("guess-my-drawing/models")
    if not model_dir.exists() or not list(model_dir.glob("*.pth")):
        print("⚠️ Aucun modèle trouvé!")
        print("Veuillez d'abord:")
        print("1. cd guess-my-drawing/src")
        print("2. python download_data.py")
        print("3. python train_cnn.py")
        return
    
    try:
        print("🔥 Lancement de l'API Flask...")
        # Lancer l'API en arrière-plan
        api_process = subprocess.Popen([
            sys.executable, "guess-my-drawing/src/api.py"
        ])
        
        # Attendre que l'API démarre
        time.sleep(3)
        
        print("🌐 Lancement du frontend...")
        # Lancer le serveur web
        web_process = subprocess.Popen([
            sys.executable, "-m", "http.server", "8080"
        ], cwd="guess-my-drawing/web")
        
        # Attendre que le serveur démarre
        time.sleep(2)
        
        print("✅ Application démarrée!")
        print("📱 Frontend: http://localhost:8080")
        print("🔌 API: http://localhost:5000")
        print("\n🎨 Ouvrir dans le navigateur...")
        
        # Ouvrir automatiquement le navigateur
        webbrowser.open("http://localhost:8080")
        
        print("\n⏹️ Appuyez sur Ctrl+C pour arrêter")
        
        # Attendre la fin
        try:
            api_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Arrêt de l'application...")
            api_process.terminate()
            web_process.terminate()
            print("✅ Application arrêtée!")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    start_application() 