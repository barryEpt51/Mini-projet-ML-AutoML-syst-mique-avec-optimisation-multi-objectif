"""
MODULE 1 : Chargement des 10 datasets depuis OpenML
======================================================
Ce module charge 10 datasets classiques pour évaluer notre AutoML.
OpenML est une plateforme publique avec des milliers de datasets standardisés.
"""

import openml
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------
# Liste des 10 datasets OpenML choisis
# Variés : petits/grands, régression/classification, domaines différents
# -------------------------------------------------------
DATASETS = {
    # (id_openml, nom_lisible, type_tache)
    "diabetes":         (37,    "Diabetes",              "classification"),
    "wine":             (40691, "Wine Quality",           "classification"),
    "breast_cancer":    (15,    "Breast Cancer",          "classification"),
    "iris":             (61,    "Iris",                   "classification"),
    "credit":           (31,    "Credit Approval",        "classification"),
    "heart":            (1462,  "Heart Disease",          "classification"),
    "titanic":          (40945, "Titanic",                "classification"),
    "vehicle":          (54,    "Vehicle Silhouettes",    "classification"),
    "bank":             (1461,  "Bank Marketing",         "classification"),
    "phoneme":          (1489,  "Phoneme",                "classification"),
}


def load_dataset(name: str) -> dict:
    """
    Charge un dataset par son nom et retourne un dictionnaire structuré.

    Paramètres
    ----------
    name : str
        Nom du dataset (clé dans DATASETS)

    Retourne
    --------
    dict avec les clés :
        - X_train, X_test : features
        - y_train, y_test : labels
        - n_samples        : nombre total d'échantillons
        - n_features       : nombre de features
        - n_classes        : nombre de classes
        - task_type        : 'classification' ou 'regression'
        - name             : nom lisible
    """
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' inconnu. Choisissez parmi : {list(DATASETS.keys())}")

    dataset_id, readable_name, task_type = DATASETS[name]

    print(f"  Chargement de '{readable_name}' (OpenML id={dataset_id})...")

    # Téléchargement depuis OpenML
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Conversion en DataFrame numpy pour uniformité
    X = pd.DataFrame(X).fillna(0)  # remplacer NaN par 0 (simple pour débutants)
    y = pd.Series(y)

    # Encodage des labels si catégoriels (ex: "yes"/"no" → 0/1)
    if y.dtype == object or str(y.dtype) == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    else:
        y = y.values

    # Encodage des features catégorielles
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.values.astype(np.float32)

    # Split train/test (80% / 20%) avec seed fixe pour reproductibilité
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task_type == "classification" else None
    )

    print(f"    → {X.shape[0]} échantillons, {X.shape[1]} features, {len(np.unique(y))} classes")

    return {
        "name":       readable_name,
        "task_type":  task_type,
        "X_train":    X_train,
        "X_test":     X_test,
        "y_train":    y_train,
        "y_test":     y_test,
        "n_samples":  X.shape[0],
        "n_features": X.shape[1],
        "n_classes":  len(np.unique(y)),
    }


def load_all_datasets() -> dict:
    """
    Charge tous les 10 datasets d'un coup.

    Retourne
    --------
    dict : {nom_dataset: dict_données}
    """
    print("=" * 50)
    print("Chargement des 10 datasets...")
    print("=" * 50)

    all_data = {}
    failed = []

    for name in DATASETS.keys():
        try:
            all_data[name] = load_dataset(name)
        except Exception as e:
            print(f"  ⚠️  Échec pour '{name}': {e}")
            failed.append(name)

    print(f"\n✅ {len(all_data)} datasets chargés avec succès.")
    if failed:
        print(f"❌ Échecs : {failed}")

    return all_data


def get_dataset_summary(all_data: dict) -> pd.DataFrame:
    """
    Crée un tableau résumé des caractéristiques de chaque dataset.
    Utile pour le rapport et l'interface Streamlit.
    """
    rows = []
    for name, data in all_data.items():
        rows.append({
            "Dataset":    data["name"],
            "Samples":    data["n_samples"],
            "Features":   data["n_features"],
            "Classes":    data["n_classes"],
            "Train size": len(data["X_train"]),
            "Test size":  len(data["X_test"]),
            "Task":       data["task_type"],
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------
# Test rapide si on exécute directement ce fichier
# -------------------------------------------------------
if __name__ == "__main__":
    # Charger un seul dataset pour tester
    data = load_dataset("diabetes")
    print(f"\nDataset chargé : {data['name']}")
    print(f"X_train shape  : {data['X_train'].shape}")
    print(f"y_train shape  : {data['y_train'].shape}")
