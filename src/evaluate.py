"""
Évaluation du modèle et calcul des métriques
"""

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_model(model, dataloader, device='cuda'):
    """
    Évalue le modèle sur un dataloader et retourne les prédictions
    
    Args:
        model: Modèle entraîné
        dataloader: DataLoader
        device: Device (cuda/cpu)
    
    Returns:
        predictions, true_labels, probabilities
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)


def compute_metrics(predictions, true_labels, probabilities=None):
    """
    Calcule toutes les métriques d'évaluation
    
    Args:
        predictions: Prédictions du modèle
        true_labels: Labels vrais
        probabilities: Probabilités (optionnel, pour AUC)
    
    Returns:
        Dict avec toutes les métriques
    """
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    # AUC si les probabilités sont disponibles
    auc = None
    if probabilities is not None and probabilities.shape[1] == 2:
        try:
            auc = roc_auc_score(true_labels, probabilities[:, 1])
        except:
            auc = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }


def print_metrics(metrics, prefix=""):
    """Affiche les métriques formatées"""
    print(f"\n{prefix}")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    if metrics['auc'] is not None:
        print(f"AUC:       {metrics['auc']:.4f}")
    print("=" * 50)


def plot_confusion_matrix(predictions, true_labels, save_path=None):
    """Génère et affiche la matrice de confusion"""
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['NORMAL', 'PNEUMONIA'],
        yticklabels=['NORMAL', 'PNEUMONIA'],
        cbar_kws={'label': 'Count'}
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Pneumonia Detection')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return cm


def plot_roc_curve(probabilities, true_labels, save_path=None):
    """Génère et affiche la courbe ROC"""
    
    if probabilities.shape[1] != 2:
        print("ROC curve requires binary classification probabilities")
        return
    
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities[:, 1])
    auc = roc_auc_score(true_labels, probabilities[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Pneumonia Detection')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_classification_report(predictions, true_labels, save_path=None):
    """Génère un rapport de classification détaillé"""
    
    report = classification_report(
        true_labels,
        predictions,
        target_names=['NORMAL', 'PNEUMONIA'],
        output_dict=True
    )
    
    print("\nDetailed Classification Report:")
    print("=" * 60)
    print(classification_report(
        true_labels,
        predictions,
        target_names=['NORMAL', 'PNEUMONIA']
    ))
    
    # Visualiser les métriques
    metrics_names = ['precision', 'recall', 'f1-score']
    normal_metrics = [report['NORMAL'][m] for m in metrics_names]
    pneumonia_metrics = [report['PNEUMONIA'][m] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, normal_metrics, width, label='NORMAL', alpha=0.8)
    ax.bar(x + width/2, pneumonia_metrics, width, label='PNEUMONIA', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Classification report saved to {save_path}")
    
    plt.show()


def generate_test_report(
    model,
    test_loader,
    output_dir="results/",
    device='cuda'
):
    """
    Génère un rapport complet d'évaluation sur le test set
    
    Args:
        model: Modèle entraîné
        test_loader: DataLoader du test set
        output_dir: Répertoire pour sauvegarder les résultats
        device: Device (cuda/cpu)
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Évaluer
    predictions, true_labels, probabilities = evaluate_model(model, test_loader, device)
    
    # Calculer les métriques
    metrics = compute_metrics(predictions, true_labels, probabilities)
    
    # Afficher les métriques
    print_metrics(metrics, prefix="TEST SET RESULTS")
    
    # Sauvegarder les métriques
    import json
    with open(f"{output_dir}/test_metrics.json", 'w') as f:
        metrics_to_save = {k: float(v) if v is not None else None for k, v in metrics.items()}
        json.dump(metrics_to_save, f, indent=4)
    
    # Générer visualisations
    plot_confusion_matrix(
        predictions,
        true_labels,
        save_path=f"{output_dir}/confusion_matrix.png"
    )
    
    if probabilities is not None:
        plot_roc_curve(
            probabilities,
            true_labels,
            save_path=f"{output_dir}/roc_curve.png"
        )
    
    plot_classification_report(
        predictions,
        true_labels,
        save_path=f"{output_dir}/classification_report.png"
    )
    
    return metrics
