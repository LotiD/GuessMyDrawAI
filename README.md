# 🎨 Guess My Drawing - IA de Reconnaissance de Dessins

Une application web interactive utilisant l'intelligence artificielle pour deviner vos dessins en temps réel !

![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange) 
![Flask](https://img.shields.io/badge/Flask-API-blue) 
![Accuracy](https://img.shields.io/badge/Accuracy-98.89%25-brightgreen)
![Classes](https://img.shields.io/badge/Classes-4-purple)

## 🌟 **Fonctionnalités**

- **🎯 IA Ultra-Précise** : 98.89% d'accuracy sur 4 classes
- **⚡ Prédictions Temps Réel** : Reconnaissance instantanée pendant que vous dessinez  
- **🖼️ Interface Moderne** : Canvas HTML5 avec design responsive
- **🧠 CNN Avancé** : Architecture PyTorch avec 3.3M paramètres
- **📊 Feedback Visuel** : Barres de progression avec scores en temps réel

## 🎭 **Classes Reconnues**

| Emoji | Classe | Description |
|-------|--------|-------------|
| 🐱 | Cat | Chats et félins |
| 🏠 | House | Maisons et bâtiments |
| 🚗 | Car | Voitures et véhicules |
| 🌳 | Tree | Arbres et végétation |

## 🚀 **Démarrage Rapide**

### 1. **Installation**
```bash
git clone https://github.com/VOTRE-USERNAME/guess-my-drawing
cd guess-my-drawing
pip install -r requirements.txt
```

### 2. **Télécharger les données**
```bash
cd guess-my-drawing/src
python download_data.py
```

### 3. **Entraîner le modèle**
```bash
python train_cnn.py
```

### 4. **Lancer l'application**
```bash
# Terminal 1 : API Flask
python api.py

# Terminal 2 : Frontend Web  
cd ../web
python -m http.server 8080
```

### 5. **Utiliser**
Ouvrez `http://localhost:8080` et commencez à dessiner ! 🎨

## 🏗️ **Architecture**

```
guess-my-drawing/
├── 📁 src/           # Code source Python
│   ├── api.py        # API Flask
│   ├── train_cnn.py  # Entraînement CNN
│   ├── model.py      # Architecture PyTorch
│   └── download_data.py
├── 📁 web/           # Interface frontend
│   ├── index.html    # Application web
│   ├── style.css     # Design moderne
│   └── script.js     # Logique canvas
├── 📁 models/        # Modèles entraînés (.pth)
├── 📁 data/          # Datasets Quick Draw
└── 📁 notebooks/     # Analyses Jupyter
```

## 🧠 **Modèle CNN**

- **Architecture** : Conv2D → BatchNorm → MaxPool (x3) → Dense
- **Paramètres** : 3,306,948 
- **Taille** : 12.6 MB
- **Dataset** : Quick Draw (Google) - 120k images
- **Accuracy** : 98.89% validation, 98.80% test
- **Optimisations** : Data augmentation, AdamW, Cosine Annealing

## 📡 **API Endpoints**

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Documentation API |
| `/health` | GET | Status et métriques |
| `/classes` | GET | Liste des classes |
| `/test` | GET | Test rapide |
| `/predict` | POST | Prédiction de dessin |

## 🛠️ **Technologies**

- **Backend** : Python, PyTorch, Flask
- **Frontend** : HTML5 Canvas, CSS3, JavaScript Vanilla
- **IA** : CNN, Data Augmentation, Transfer Learning
- **Données** : Quick Draw Dataset (Google)

## 📊 **Performances**

- **Accuracy** : 98.89% (validation), 98.80% (test)
- **Vitesse** : ~5ms par prédiction
- **Temps d'entraînement** : 18 minutes
- **Robustesse** : Élimine confusion chien/chat

## 🔬 **Entraînement Avancé**

Le modèle utilise des techniques d'optimisation modernes :
- **Data Augmentation** : Translation, rotation
- **Regularization** : Dropout 0.5, Weight Decay
- **Optimiseur** : AdamW avec Cosine Annealing
- **Architecture** : BatchNorm pour stabilité

## 📈 **Résultats**

| Métrique | Valeur |
|----------|--------|
| Validation Accuracy | 98.89% |
| Test Accuracy | 98.80% |
| Temps d'entraînement | 18.3 min |
| Amélioration vs baseline | +4.31% |

## 🚀 **Déploiement**

Compatible avec :
- **Vercel** (serverless functions)
- **GitHub Pages** (frontend seulement)
- **Railway/Render** (application complète)

## 👨‍💻 **Développement**

Créé pour apprendre PyTorch et le deep learning en pratique.

## 📄 **Licence**

MIT License - Libre d'utilisation et modification.

---

⭐ **N'hésitez pas à star le projet si vous l'aimez !** 