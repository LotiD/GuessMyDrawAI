#!/usr/bin/env python3
"""
Modèle CNN pour Guess My Drawing
Architecture DrawingCNN optimisée pour dessins 28x28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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