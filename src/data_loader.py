from datasets import load_dataset


def load_raw_pneumonia_dataset(split: str = "train"):
    """Charge le dataset raw_pneumonia_x_ray depuis Hugging Face.

    Args:
        split: split à charger (par défaut: train)

    Returns:
        Un objet Dataset Hugging Face.
    """
    return load_dataset("mmenendezg/raw_pneumonia_x_ray", split=split)


def infer_label_column(dataset):
    """Trouve la colonne label la plus probable dans le dataset."""
    candidates = ["label", "labels", "class", "target", "diagnosis"]
    for col in candidates:
        if col in dataset.column_names:
            return col
    return None
