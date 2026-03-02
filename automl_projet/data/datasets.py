"""
data/datasets.py
================
Chargement des 10 datasets pour l'évaluation du framework AutoML.
Combine les datasets sklearn (disponibles sans connexion) et OpenML.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_diabetes, load_breast_cancer, load_iris,
    load_wine, load_digits
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ── Registre des 10 datasets ──────────────────────────
DATASETS = {
    # sklearn natifs
    "diabetes":      {"source": "sklearn", "task": "regression"},
    "breast_cancer": {"source": "sklearn", "task": "classification"},
    "iris":          {"source": "sklearn", "task": "classification"},
    "wine":          {"source": "sklearn", "task": "classification"},
    "digits":        {"source": "sklearn", "task": "classification"},
    # OpenML
    "credit":        {"source": "openml",  "task": "classification", "id": 31},
    "heart":         {"source": "openml",  "task": "classification", "id": 1461},
    "vehicle":       {"source": "openml",  "task": "classification", "id": 54},
    "bank":          {"source": "openml",  "task": "classification", "id": 1480},
    "phoneme":       {"source": "openml",  "task": "classification", "id": 1489},
}


def _preprocess(X, y, task: str, test_size: float = 0.2, seed: int = 42):
    """Preprocessing générique : imputation + normalisation + split."""
    # Encodage target si classification
    if task == "classification":
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = np.array(y, dtype=float)

    # Encodage features catégorielles
    if hasattr(X, "select_dtypes"):
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.values

    X = np.array(X, dtype=float)

    # Imputation + normalisation
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    return X_train, X_test, y_train, y_test


def load_dataset(name: str, seed: int = 42) -> dict:
    """
    Charge un dataset par son nom et retourne un dict standardisé.

    Retourne
    --------
    dict : X_train, X_test, y_train, y_test, n_samples, n_features, n_classes, task
    """
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' inconnu. Disponibles : {list(DATASETS.keys())}")

    info = DATASETS[name]
    task = info["task"]

    # ── Datasets sklearn ──────────────────────────────
    if info["source"] == "sklearn":
        loaders = {
            "diabetes":      load_diabetes,
            "breast_cancer": load_breast_cancer,
            "iris":          load_iris,
            "wine":          load_wine,
            "digits":        load_digits,
        }
        data = loaders[name]()
        X = pd.DataFrame(data.data,
                         columns=getattr(data, "feature_names",
                                         [f"f{i}" for i in range(data.data.shape[1])]))
        y = pd.Series(data.target, name="target")

    # ── Datasets OpenML ───────────────────────────────
    else:
        try:
            import openml
            dataset = openml.datasets.get_dataset(info["id"])
            X_raw, y_raw, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            X = X_raw
            y = y_raw
        except Exception as e:
            print(f"⚠️  OpenML non disponible pour '{name}': {e}")
            print("    Utilisation d'un dataset sklearn de substitution.")
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = _preprocess(X, y, task, seed=seed)

    return {
        "name":       name,
        "task":       task,
        "X_train":    X_train,
        "X_test":     X_test,
        "y_train":    y_train,
        "y_test":     y_test,
        "n_samples":  len(X_train) + len(X_test),
        "n_features": X_train.shape[1],
        "n_classes":  len(np.unique(y_train)) if task == "classification" else 0,
    }


def load_custom_dataset(df: pd.DataFrame, target_col: str = None,
                        task: str = "auto", seed: int = 42) -> dict:
    """
    Charge un dataset personnalisé (upload utilisateur).

    Paramètres
    ----------
    df         : DataFrame chargé depuis CSV ou Excel
    target_col : colonne cible (None = dernière colonne)
    task       : 'auto', 'classification' ou 'regression'
    """
    df = df.copy()

    # Séparation X / y
    if target_col and target_col in df.columns:
        y_raw = df[target_col]
        X_raw = df.drop(columns=[target_col])
    else:
        y_raw = df.iloc[:, -1]
        X_raw = df.iloc[:, :-1]

    # Auto-détection
    if task == "auto":
        task = ("regression"
                if y_raw.dtype in [np.float64, np.float32] and y_raw.nunique() > 20
                else "classification")

    # Suppression colonnes constantes
    X_raw = X_raw.loc[:, X_raw.nunique() > 1]

    X_train, X_test, y_train, y_test = _preprocess(X_raw, y_raw, task, seed=seed)

    return {
        "name":       "custom",
        "task":       task,
        "X_train":    X_train,
        "X_test":     X_test,
        "y_train":    y_train,
        "y_test":     y_test,
        "n_samples":  len(X_train) + len(X_test),
        "n_features": X_train.shape[1],
        "n_classes":  len(np.unique(y_train)) if task == "classification" else 0,
    }


def get_dataset_summary() -> pd.DataFrame:
    """Tableau résumé de tous les datasets disponibles."""
    rows = []
    for name, info in DATASETS.items():
        rows.append({
            "Nom":    name,
            "Source": info["source"],
            "Tâche":  info["task"],
        })
    return pd.DataFrame(rows)
