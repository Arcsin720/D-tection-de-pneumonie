"""
Configuration des hyperparamètres pour l'entraînement du modèle
"""

# Dataset
DATASET_NAME = "iamtanmayshukla/pneumonia-radiography-dataset"
IMG_SIZE = 224  # ResNet50 standard
RANDOM_SEED = 42

# Prétraitement
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet statistics
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Entraînement
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4

# Classes
CLASS_LABELS = {
    0: "NORMAL",
    1: "PNEUMONIA"
}

# Class weights (pour compenser le déséquilibre)
# Calculé depuis l'EDA: pneumonia=4273, normal=1583
CLASS_WEIGHTS = {
    0: 2.7,  # NORMAL (minoritaire)
    1: 1.0   # PNEUMONIA (majoritaire)
}

# Enregistrement du modèle
MODEL_SAVE_PATH = "models/resnet50_pneumonia.pth"
RESULTS_SAVE_PATH = "results/"

# Device
# Utilise GPU si disponible, sinon CPU
DEVICE = "cuda"  # Change to "cpu" si pas de GPU disponible

# Validation
VAL_SPLIT = 0.2  # 20% des données train pour validation
EARLY_STOPPING_PATIENCE = 5  # Arrêter après 5 épochs sans amélioration
