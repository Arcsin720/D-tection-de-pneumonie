# Pneumonia Detection Model

Modèle de machine learning pour la détection de pneumonie à partir de radiographies des poumons.

##  Dataset

- **Source** : [Hugging Face - raw_pneumonia_x_ray](https://huggingface.co/datasets/mmenendezg/raw_pneumonia_x_ray)
- **Taille** : 5,856 images
- **Classes** : Sain / Pneumonie

##  Architecture

```
pneumonia-detection/
├── src/               # Code Python modulaire
├── notebooks/         # Notebooks Jupyter
├── models/           # Modèles entraînés
├── results/          # Métriques et visualisations
└── tests/            # Tests unitaires
```

##  Installation

```bash
# Cloner le repo
git clone https://github.com/user/pneumonia-detection.git
cd pneumonia-detection

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

##  Utilisation

1. **Exploration des données** : `notebooks/01_eda.ipynb`
2. **Entraînement** : `notebooks/02_training.ipynb`
3. **Évaluation** : `notebooks/03_evaluation.ipynb`

##  Contribution

Créez une branche feature pour votre travail :
```bash
git checkout develop
git checkout -b feature/votre-nom-feature
```

Après terminer, créez une Pull Request vers `develop`.

##  Liste des Commandes Git Utiles

```bash
git status              # Voir les changements
git add fichier.py      # Ajouter un fichier
git commit -m "message" # Sauvegarder
git push origin branche # Envoyer sur GitHub
```
