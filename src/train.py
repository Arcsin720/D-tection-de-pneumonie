"""
Script d'entraînement du modèle ResNet50 pour la détection de pneumonie
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
import kagglehub
import os
from tqdm import tqdm
import json
from pathlib import Path

from .config import (
    DATASET_NAME, IMG_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    CLASS_WEIGHTS, MODEL_SAVE_PATH, RESULTS_SAVE_PATH, DEVICE,
    VAL_SPLIT, EARLY_STOPPING_PATIENCE, RANDOM_SEED
)
from .model import create_model, PneumoniaDataPreprocessor
from .data_loader import load_raw_pneumonia_dataset
from .evaluate import evaluate_model, compute_metrics


class TrainingMetrics:
    """Classe pour tracker les métriques d'entraînement"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def update_train(self, loss):
        self.train_losses.append(loss)
    
    def update_val(self, loss, accuracy):
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
        
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_epoch = len(self.val_losses) - 1
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def should_stop(self):
        return self.patience_counter >= EARLY_STOPPING_PATIENCE
    
    def to_dict(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }


def collate_fn(batch):
    """Custom collate function pour traiter les images du dataset"""
    preprocessor = PneumoniaDataPreprocessor(img_size=IMG_SIZE)
    
    images = []
    labels = []
    
    for item in batch:
        img = item['image']
        label = item['label']
        
        # Prétraiter l'image
        processed_img = preprocessor.preprocess_image(img, is_training=True)
        images.append(processed_img)
        labels.append(label)
    
    return {
        'images': torch.stack(images),
        'labels': torch.tensor(labels, dtype=torch.long)
    }


def create_dataloaders(train_dataset, batch_size=BATCH_SIZE):
    """
    Crée les DataLoaders pour train et validation
    
    Args:
        train_dataset: Dataset complet
        batch_size: Taille des batches
    
    Returns:
        train_loader, val_loader
    """
    # Split train/val
    val_size = int(len(train_dataset) * VAL_SPLIT)
    train_size = len(train_dataset) - val_size
    
    train_split, val_split = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Entraîne le modèle pour une époque
    
    Returns:
        loss moyenne de l'époque
    """
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """
    Évalue le modèle sur la validation set
    
    Returns:
        loss moyenne, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    device=DEVICE
):
    """
    Entraîne le modèle complet
    
    Args:
        model: Modèle ResNet50
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        num_epochs: Nombre d'épochs
        learning_rate: Taux d'apprentissage
        device: Device (cuda/cpu)
    
    Returns:
        Modèle entraîné, métriques
    """
    
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}\n")
    
    # Criterion avec class weights
    class_weights_tensor = torch.tensor(
        [CLASS_WEIGHTS[0], CLASS_WEIGHTS[1]],
        dtype=torch.float32,
        device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Tracking
    metrics = TrainingMetrics()
    
    # Entraînement
    print("Starting training...\n")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics.update_train(train_loss)
        
        # Validate
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)
        metrics.update_val(val_loss, val_accuracy)
        
        # Print stats
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if metrics.should_stop():
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    return model, metrics


def main():
    """Pipeline complet d'entraînement"""
    
    # Créer les répertoires
    Path(RESULTS_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(MODEL_SAVE_PATH)).mkdir(parents=True, exist_ok=True)
    
    # Charger le dataset
    print("Loading dataset...")
    train_dataset = load_raw_pneumonia_dataset("train")
    print(f"Dataset loaded: {len(train_dataset)} images\n")
    
    # Créer les dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(train_dataset, batch_size=BATCH_SIZE)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")
    
    # Créer le modèle
    print("Creating model...")
    model = create_model(num_classes=2, device=DEVICE)
    print(model)
    
    # Entraînement
    model, metrics = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )
    
    # Sauvegarder le modèle
    print(f"\nSaving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Sauvegarder les métriques
    metrics_path = os.path.join(RESULTS_SAVE_PATH, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
