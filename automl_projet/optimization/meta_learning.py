"""
optimization/meta_learning.py
==============================
Méta-apprentissage : apprendre à apprendre.
Mémorise les meilleures configs par dataset pour warm-starter SMAC.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")


def extract_meta_features(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Extrait des méta-features statistiques décrivant un dataset.
    Utilisées pour mesurer la similarité entre datasets.
    """
    n_samples, n_features = X.shape
    classes, counts = np.unique(y, return_counts=True)

    class_ratios = counts / n_samples
    meta = {
        "n_samples":           float(n_samples),
        "n_features":          float(n_features),
        "n_classes":           float(len(classes)),
        "log_samples":         float(np.log1p(n_samples)),
        "log_features":        float(np.log1p(n_features)),
        "samples_per_feature": float(n_samples / max(n_features, 1)),
        "features_per_sample": float(n_features / max(n_samples, 1)),
        "class_imbalance":     float(np.max(class_ratios) / np.min(class_ratios)),
        "class_entropy":       float(-np.sum(class_ratios * np.log2(class_ratios + 1e-10))),
        "mean_feature_mean":   float(np.mean(np.mean(X, axis=0))),
        "mean_feature_std":    float(np.mean(np.std(X, axis=0))),
        "missing_ratio":       float(np.isnan(X).sum() / max(X.size, 1)),
        "sparsity":            float((X == 0).mean()),
        "relative_n_features": float(n_features / max(n_samples, 1)),
    }

    try:
        meta["mean_skewness"] = float(np.mean([
            np.mean((X[:, i] - np.mean(X[:, i])) ** 3) / (np.std(X[:, i]) + 1e-10) ** 3
            for i in range(min(n_features, 20))
        ]))
    except Exception:
        meta["mean_skewness"] = 0.0

    return meta


class MetaLearningBase:
    """
    Base de méta-données persistante.
    Stocke les meilleures configs et permet le warm-start de SMAC.
    """

    def __init__(self, storage_path: str = "./meta_learning_base.pkl"):
        self.storage_path = Path(storage_path)
        self.entries: list = []
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "rb") as f:
                    self.entries = pickle.load(f)
            except Exception:
                self.entries = []

    def save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "wb") as f:
            pickle.dump(self.entries, f)

    def add_entry(self, dataset_name: str, meta_features: dict,
                  best_config: dict, best_score: float, best_accuracy: float):
        """Ajoute ou met à jour une entrée dans la base méta."""
        for i, entry in enumerate(self.entries):
            if entry["dataset_name"] == dataset_name:
                if best_score < entry["best_score"]:
                    self.entries[i] = {
                        "dataset_name": dataset_name,
                        "meta_features": meta_features,
                        "best_config": best_config,
                        "best_score": best_score,
                        "best_accuracy": best_accuracy,
                    }
                    self.save()
                return
        self.entries.append({
            "dataset_name": dataset_name,
            "meta_features": meta_features,
            "best_config": best_config,
            "best_score": best_score,
            "best_accuracy": best_accuracy,
        })
        self.save()

    def find_similar_datasets(self, meta_features: dict, top_k: int = 3) -> list:
        """Trouve les top_k datasets les plus similaires via distance euclidienne."""
        if not self.entries:
            return []

        keys = list(meta_features.keys())
        new_vec = np.array([meta_features[k] for k in keys]).reshape(1, -1)
        stored  = np.array([[e["meta_features"].get(k, 0) for k in keys]
                             for e in self.entries])

        # Normalisation par écart-type
        scale = np.std(stored, axis=0) + 1e-10
        dists = cdist(new_vec / scale, stored / scale, metric="euclidean")[0]

        top_idx = np.argsort(dists)[:min(top_k, len(self.entries))]
        result  = []
        for i in top_idx:
            entry = dict(self.entries[i])
            entry["similarity_distance"] = float(dists[i])
            result.append(entry)
        return result

    def get_warm_start_configs(self, meta_features: dict, top_k: int = 3) -> list:
        """Retourne les configs recommandées pour le warm-start SMAC."""
        similar = self.find_similar_datasets(meta_features, top_k)
        return [s["best_config"] for s in similar]

    def get_summary(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame()
        return pd.DataFrame([{
            "Dataset":      e["dataset_name"],
            "Best algo":    e["best_config"].get("algorithm", "?"),
            "Best score":   round(e["best_score"], 4),
            "Accuracy":     round(e["best_accuracy"], 4),
            "n_samples":    int(e["meta_features"].get("n_samples", 0)),
            "n_features":   int(e["meta_features"].get("n_features", 0)),
        } for e in self.entries])

    def clear(self):
        self.entries = []
        self.save()
