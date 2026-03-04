from datasets import load_dataset
import kagglehub


def load_raw_pneumonia_dataset(split: str = "train"):
    """Charge le dataset de pneumonie depuis Kaggle.

    Args:
        split: split à charger (par défaut: train)

    Returns:
        Un objet Dataset.
    """
    # Download latest version
    path = kagglehub.dataset_download("iamtanmayshukla/pneumonia-radiography-dataset")
    print("Path to dataset files:", path)
    
    return load_dataset("imagefolder", data_dir=path, split=split)


def infer_label_column(dataset):
    """Trouve la colonne label la plus probable dans le dataset."""
    candidates = ["label", "labels", "class", "target", "diagnosis"]
    for col in candidates:
        if col in dataset.column_names:
            return col
    return None

