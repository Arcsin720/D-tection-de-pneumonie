from collections import Counter

from src.data_loader import load_raw_pneumonia_dataset, infer_label_column


def run_basic_eda():
    dataset = load_raw_pneumonia_dataset("train")

    print("=== EDA RAPIDE ===")
    print(f"Nombre total d'images: {len(dataset)}")
    print(f"Colonnes disponibles: {dataset.column_names}")

    label_col = infer_label_column(dataset)
    if label_col is None:
        print("Colonne label non détectée automatiquement.")
        return

    counts = Counter(dataset[label_col])
    print(f"Colonne label utilisée: {label_col}")
    print("Distribution des classes:")
    for cls, count in counts.items():
        print(f"- Classe {cls}: {count}")


if __name__ == "__main__":
    run_basic_eda()
