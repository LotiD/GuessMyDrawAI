#!/usr/bin/env python3
"""
Script d'entraînement CNN - Guess My Drawing
Entraînement optimisé avec monitoring temps réel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import time
import random

try:
    from scipy.ndimage import rotate
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️ Scipy non disponible, rotation désactivée dans data augmentation")
    SCIPY_AVAILABLE = False

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Device: {device}")

# Paramètres d'entraînement RAPIDES ⚡
BATCH_SIZE = 256  # GROS batch pour moins d'itérations
LEARNING_RATE = 0.002  # LR plus élevé pour convergence rapide  
EPOCHS = 8  # MOINS d'epochs pour aller vite
NUM_CLASSES = 4  # Sans dog
WEIGHT_DECAY = 1e-4  # Régularisation L2

# Classes et emojis (optimisées sans dog)
CLASSES = ['cat', 'house', 'car', 'tree']
CLASS_EMOJIS = ['🐱', '🏠', '🚗', '🌳']

print(f"⚡ Configuration RAPIDE:")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Classes: {CLASSES}")
print(f"")
print(f"⏱️ ESTIMATION TEMPS TOTAL: ~25-35 minutes")

# Architecture CNN
class DrawingCNN(nn.Module):
    """CNN optimisé pour dessins Quick Draw 28x28"""
    
    def __init__(self, num_classes=4):
        super(DrawingCNN, self).__init__()
        
        # Bloc convolutionnel 1: 28x28 -> 14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloc convolutionnel 2: 14x14 -> 7x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bloc convolutionnel 3: 7x7 -> 7x7
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Couches fully connected
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Bloc 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Bloc 2  
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Bloc 3
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Dataset personnalisé
class QuickDrawDataset(Dataset):
    """Dataset pour les données Quick Draw avec Data Augmentation"""
    
    def __init__(self, data_dir, classes, max_samples_per_class=60000, train=True, augment=True):
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.train = train
        self.augment = augment and train
        
        # Charger les données
        print("📊 Chargement des données...")
        self.images = []
        self.labels = []
        
        for class_idx, class_name in enumerate(classes):
            print(f"   {CLASS_EMOJIS[class_idx]} Chargement {class_name}...")
            
            # Charger le fichier .npy
            filepath = self.data_dir / f"{class_name}.npy"
            try:
                data = np.load(filepath)
                
                # Plus d'échantillons pour compenser l'augmentation
                if len(data) > max_samples_per_class:
                    indices = np.random.permutation(len(data))[:max_samples_per_class]
                    data = data[indices]
                
                # Ajouter aux listes
                self.images.extend(data)
                self.labels.extend([class_idx] * len(data))
                
                print(f"     -> {len(data):,} images ajoutées")
            except FileNotFoundError:
                print(f"     ⚠️ Fichier {class_name}.npy non trouvé, ignoré")
        
        # Convertir en numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"✅ Dataset créé: {len(self.images):,} images total")
        print(f"   Distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Récupérer l'image et le label
        image = self.images[idx].reshape(28, 28).astype(np.float32)
        label = self.labels[idx]
        
        # Normalisation: Quick Draw a noir=255, blanc=0
        # On inverse pour avoir noir=0, blanc=255
        image = 255 - image
        
        # Data Augmentation RAPIDE si en mode train
        if self.augment and random.random() > 0.5:  # 50% chance seulement
            # SEULEMENT translation (plus rapide que rotation)
            shift_x = random.randint(-2, 2)
            shift_y = random.randint(-2, 2)
            image = np.roll(image, shift_x, axis=1)
            image = np.roll(image, shift_y, axis=0)
        
        # Normalisation finale [0,1] puis [-1,1]
        image = image / 255.0
        image = (image - 0.5) / 0.5
        
        # Ajouter dimension channel
        image = image[np.newaxis, ...]  # (1, 28, 28)
        
        return torch.FloatTensor(image), torch.LongTensor([label])[0]

def train_epoch(model, loader, optimizer, criterion, epoch, writer):
    """Entraîne le modèle pour une époque"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Progress bar
        if batch_idx % 50 == 0:
            progress = 100. * batch_idx / len(loader)
            acc = 100. * correct / total
            print(f'\r   Progress: {progress:5.1f}% | Loss: {loss.item():.4f} | Acc: {acc:.2f}%', end='', flush=True)
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    # Log TensorBoard
    if writer:
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
    
    print()  # Nouvelle ligne
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, epoch, writer):
    """Valide le modèle"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    val_loss /= len(loader)
    val_acc = 100. * correct / total
    
    # Log validation
    if writer:
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)
    
    return val_loss, val_acc

def main():
    """Fonction principale d'entraînement"""
    print("🔥 DÉBUT DE L'ENTRAÎNEMENT CNN")
    print("=" * 50)
    
    # Créer le dataset
    data_dir = Path("../data")
    if not data_dir.exists():
        data_dir = Path("./data")  # Si on lance depuis la racine
    
    # Datasets AMÉLIORÉS avec augmentation
    print("🔄 Création des datasets...")
    train_dataset = QuickDrawDataset(
        data_dir=data_dir,
        classes=CLASSES,
        max_samples_per_class=20000,  # ⚡ RÉDUIT pour rapidité
        train=True,
        augment=True  # Data augmentation pour train
    )
    
    val_dataset = QuickDrawDataset(
        data_dir=data_dir,
        classes=CLASSES,
        max_samples_per_class=5000,  # ⚡ RÉDUIT pour rapidité
        train=False,
        augment=False  # Pas d'augmentation pour validation
    )
    
    test_dataset = QuickDrawDataset(
        data_dir=data_dir,
        classes=CLASSES,
        max_samples_per_class=5000,  # ⚡ RÉDUIT pour rapidité  
        train=False,
        augment=False  # Pas d'augmentation pour test
    )
    
    print(f"📊 Splits:")
    print(f"   Train: {len(train_dataset):,} images")
    print(f"   Validation: {len(val_dataset):,} images") 
    print(f"   Test: {len(test_dataset):,} images")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Modèle et optimiseur AMÉLIORÉS
    model = DrawingCNN(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    
    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'../runs/drawing_cnn_{timestamp}')
    
    # Afficher infos modèle
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Architecture CNN:")
    print(f"   Paramètres: {params:,}")
    print(f"   Taille: {params * 4 / 1024 / 1024:.1f} MB")
    
    # Historique des métriques
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    
    start_time = time.time()
    
    # Boucle d'entraînement
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        print(f"\n📅 ÉPOQUE {epoch+1}/{EPOCHS}")
        print("-" * 30)
        
        # Entraînement
        print("🏋️ Entraînement...")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch, writer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        print("🔍 Validation...")
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, epoch, writer)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, f'../models/best_model_{timestamp}.pth')
            print(f"   💾 Nouveau meilleur modèle sauvé! (Val Acc: {val_acc:.2f}%)")
        
        epoch_time = time.time() - epoch_start
        print(f"   ⏱️ Temps époque: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"   📊 Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")
    
    total_time = time.time() - start_time
    
    # Évaluation finale sur test set
    print(f"\n🎯 ÉVALUATION FINALE...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)
    
    test_accuracy = 100. * test_correct / test_total
    
    # Résultats finaux
    print(f"\n🎉 ENTRAÎNEMENT TERMINÉ!")
    print("=" * 50)
    print(f"⏱️ Temps total: {total_time/60:.1f} minutes")
    print(f"🏆 Meilleure Val Accuracy: {best_val_acc:.2f}%")
    print(f"📊 Test Accuracy: {test_accuracy:.2f}%")
    print(f"💾 Modèle sauvé: models/best_model_{timestamp}.pth")
    
    # Évaluation de l'objectif
    if test_accuracy >= 85.0:
        print(f"\n🎉 OBJECTIF ATTEINT! Test accuracy {test_accuracy:.1f}% >= 85%")
        status = "SUCCESS ✅"
    else:
        print(f"\n⚠️ Objectif manqué. Test accuracy {test_accuracy:.1f}% < 85%")
        status = "NEEDS_IMPROVEMENT ⚠️"
    
    print(f"\n🏁 STATUS: {status}")
    
    # Fermer TensorBoard
    writer.close()
    print(f"\n🌐 TensorBoard: tensorboard --logdir=../runs")
    print(f"📁 Run: drawing_cnn_{timestamp}")
    print("✅ Entraînement terminé!")

if __name__ == "__main__":
    main() 