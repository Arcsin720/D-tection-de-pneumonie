"""
Architecture du modèle ResNet50 pour la détection de pneumonie
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from .config import IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


class PneumoniaResNet50(nn.Module):
    """
    ResNet50 pré-entraîné modifié pour la classification pneumonia/normal
    
    Architecture:
    - ResNet50 (pré-entraîné sur ImageNet)
    - Remplace la couche finale pour 2 classes
    - Fine-tuning des derniers blocs
    """
    
    def __init__(self, num_classes=2, freeze_backbone=True):
        """
        Args:
            num_classes: Nombre de classes (2 pour normal/pneumonia)
            freeze_backbone: Si True, gèle les couches pré-entraînées
        """
        super(PneumoniaResNet50, self).__init__()
        
        # Charger ResNet50 pré-entraîné
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Geler les poids pré-entraînés (transfer learning)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Remplacer la couche de classification finale
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Dégeler les derniers blocs pour fine-tuning
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)


class PneumoniaDataPreprocessor:
    """
    Prétraitement des images pour ResNet50
    - Redimensionnement à 224x224
    - Normalisation
    """
    
    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size
        
        # Transforms pour entraînement (avec augmentation)
        self.train_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
        
        # Transforms pour validation/test (sans augmentation)
        self.val_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    
    def preprocess_image(self, image, is_training=False):
        """
        Prétraite une image PIL
        
        Args:
            image: Image PIL
            is_training: Si True, applique l'augmentation
        
        Returns:
            Tensor d'image prétraité
        """
        transforms_to_use = self.train_transforms if is_training else self.val_transforms
        
        # Convertir en RGB si en niveaux de gris
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return transforms_to_use(image)
    
    @staticmethod
    def preprocess_batch(batch, is_training=False):
        """
        Prétraite un batch d'images du dataset HuggingFace
        
        Args:
            batch: Batch du dataset
            is_training: Mode entraînement ou validation
        
        Returns:
            Batch de tensors preprocessés
        """
        preprocessor = PneumoniaDataPreprocessor()
        
        images = []
        labels = []
        
        for item in batch:
            img = item['image']
            label = item['label']
            
            processed_img = preprocessor.preprocess_image(img, is_training=is_training)
            images.append(processed_img)
            labels.append(label)
        
        return {
            'images': torch.stack(images),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def create_model(num_classes=2, device='cuda'):
    """
    Factory function pour créer le modèle
    
    Args:
        num_classes: Nombre de classes
        device: Device ('cuda' ou 'cpu')
    
    Returns:
        Modèle ResNet50 sur le device spécifié
    """
    model = PneumoniaResNet50(num_classes=num_classes, freeze_backbone=True)
    model = model.to(device)
    return model


def get_model_summary(model):
    """Affiche un résumé du modèle"""
    return f"""
    ResNet50 Pneumonia Detector
    ===========================
    Total parameters: {sum(p.numel() for p in model.parameters()):,}
    Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}
    """
