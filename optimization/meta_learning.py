"""
MODULE 5 : Méta-apprentissage (Meta-Learning)
===============================================
Idée principale : "Apprendre à apprendre"

Problème : SMAC part de zéro sur chaque nouveau dataset → lent
Solution  : Mémoriser les meilleures configs des datasets précédents
            → "warm-start" SMAC avec ces configs comme point de départ

Comment ça marche ?
  1. On extrait des méta-features du dataset (taille, nb features, etc.)
  2. On cherche les datasets les plus similaires dans notre historique
  3. On récupère leurs meilleures configurations
  4. On initialise SMAC avec ces configs → il part d'un bon point de départ
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


# -------------------------------------------------------
# MÉTA-FEATURES : caractéristiques d'un dataset
# -------------------------------------------------------

def extract_meta_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """
    Extrait des méta-features statistiques d'un dataset.
    
    Ces features décrivent le dataset de façon numérique,
    permettant de comparer des datasets entre eux.

    Paramètres
    ----------
    X_train : features d'entraînement
    y_train : labels d'entraînement

    Retourne
    --------
    dict : méta-features du dataset
    """
    n_samples, n_features = X_train.shape
    classes, counts = np.unique(y_train, return_counts=True)
    n_classes = len(classes)

    # --- Features de base ---
    meta = {
        "n_samples":    float(n_samples),
        "n_features":   float(n_features),
        "n_classes":    float(n_classes),
        "log_samples":  float(np.log1p(n_samples)),
        "log_features": float(np.log1p(n_features)),
    }

    # --- Ratio samples/features ---
    meta["samples_per_feature"] = float(n_samples / n_features)
    meta["features_per_sample"] = float(n_features / n_samples)

    # --- Déséquilibre des classes ---
    class_ratios = counts / n_samples
    meta["class_imbalance"]    = float(np.max(class_ratios) / np.min(class_ratios))
    meta["class_entropy"]      = float(-np.sum(class_ratios * np.log2(class_ratios + 1e-10)))

    # --- Statistiques sur les features ---
    meta["mean_feature_mean"]  = float(np.mean(np.mean(X_train, axis=0)))
    meta["mean_feature_std"]   = float(np.mean(np.std(X_train, axis=0)))
    meta["mean_feature_min"]   = float(np.mean(np.min(X_train, axis=0)))
    meta["mean_feature_max"]   = float(np.mean(np.max(X_train, axis=0)))

    # --- Dispersion ---
    try:
        meta["mean_skewness"] = float(np.mean([
            np.mean((X_train[:, i] - np.mean(X_train[:, i]))**3) /
            (np.std(X_train[:, i]) + 1e-10)**3
            for i in range(min(n_features, 20))  # limiter pour la rapidité
        ]))
    except:
        meta["mean_skewness"] = 0.0

    # --- Valeurs manquantes ---
    meta["missing_ratio"] = float(np.isnan(X_train).sum() / (n_samples * n_features))

    # --- Cardinalité relative ---
    meta["relative_n_features"] = float(n_features / max(n_samples, 1))

    return meta


# -------------------------------------------------------
# BASE DE MÉTA-DONNÉES
# -------------------------------------------------------

class MetaLearningBase:
    """
    Base de données qui mémorise les performances de chaque dataset.
    
    Permet de :
      1. Stocker les résultats d'optimisation
      2. Retrouver les datasets similaires
      3. Recommander des configurations de départ (warm-start)
    """

    def __init__(self, storage_path: str = "./meta_learning_base.pkl"):
        self.storage_path = Path(storage_path)
        self.entries      = []   # liste des entrées mémorisées
        self._load()

    def _load(self):
        """Charge la base depuis le disque si elle existe."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "rb") as f:
                    self.entries = pickle.load(f)
                print(f"📚 Base méta chargée : {len(self.entries)} entrées")
            except:
                self.entries = []
        else:
            self.entries = []
            print("📚 Nouvelle base méta créée")

    def save(self):
        """Sauvegarde la base sur le disque."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "wb") as f:
            pickle.dump(self.entries, f)

    def add_entry(
        self,
        dataset_name:  str,
        meta_features: dict,
        best_config:   dict,
        best_score:    float,
        best_accuracy: float,
        all_results:   list = None,
    ):
        """
        Ajoute une nouvelle entrée dans la base méta.

        Paramètres
        ----------
        dataset_name  : nom du dataset
        meta_features : méta-features extraites (sortie de extract_meta_features)
        best_config   : meilleure config trouvée par SMAC
        best_score    : score combiné de la meilleure config
        best_accuracy : accuracy de la meilleure config
        all_results   : tous les résultats (optionnel)
        """
        # Vérifier si ce dataset existe déjà → mettre à jour si meilleur score
        for i, entry in enumerate(self.entries):
            if entry["dataset_name"] == dataset_name:
                if best_score < entry["best_score"]:
                    self.entries[i] = {
                        "dataset_name":  dataset_name,
                        "meta_features": meta_features,
                        "best_config":   best_config,
                        "best_score":    best_score,
                        "best_accuracy": best_accuracy,
                    }
                    print(f"  📝 Mise à jour de '{dataset_name}' (score amélioré)")
                    self.save()
                    return

        # Nouveau dataset
        self.entries.append({
            "dataset_name":  dataset_name,
            "meta_features": meta_features,
            "best_config":   best_config,
            "best_score":    best_score,
            "best_accuracy": best_accuracy,
        })
        print(f"  📝 Ajout de '{dataset_name}' dans la base méta")
        self.save()

    def find_similar_datasets(
        self,
        meta_features: dict,
        top_k:         int = 3,
    ) -> list:
        """
        Trouve les top_k datasets les plus similaires dans la base.
        
        Utilise la distance euclidienne sur les méta-features normalisées.

        Paramètres
        ----------
        meta_features : méta-features du nouveau dataset
        top_k         : nombre de voisins à retourner

        Retourne
        --------
        liste des top_k entrées les plus similaires (triées par similarité)
        """
        if not self.entries:
            return []

        # Extraire les clés communes
        all_keys = list(meta_features.keys())

        # Vecteur du nouveau dataset
        new_vec = np.array([meta_features[k] for k in all_keys]).reshape(1, -1)

        # Vecteurs des datasets mémorisés
        stored_vecs = []
        for entry in self.entries:
            vec = [entry["meta_features"].get(k, 0) for k in all_keys]
            stored_vecs.append(vec)

        stored_vecs = np.array(stored_vecs)

        # Normalisation (éviter que les grandes valeurs dominent)
        scale = np.std(stored_vecs, axis=0) + 1e-10
        new_vec_norm    = new_vec / scale
        stored_vecs_norm = stored_vecs / scale

        # Distance euclidienne
        distances = cdist(new_vec_norm, stored_vecs_norm, metric="euclidean")[0]

        # Trier par distance croissante
        sorted_indices = np.argsort(distances)
        top_indices    = sorted_indices[:min(top_k, len(self.entries))]

        similar = []
        for idx in top_indices:
            entry = dict(self.entries[idx])
            entry["similarity_distance"] = float(distances[idx])
            similar.append(entry)

        return similar

    def get_warm_start_configs(
        self,
        meta_features: dict,
        top_k:         int = 3,
    ) -> list:
        """
        Retourne les configurations recommandées pour le warm-start SMAC.

        Paramètres
        ----------
        meta_features : méta-features du nouveau dataset
        top_k         : nombre de configs à recommander

        Retourne
        --------
        liste de configs (dicts) → prêtes pour SMAC initial_design
        """
        similar = self.find_similar_datasets(meta_features, top_k)

        if not similar:
            print("  ⚠️  Base méta vide → SMAC démarre sans warm-start")
            return []

        configs = []
        for entry in similar:
            configs.append(entry["best_config"])
            print(f"  🎯 Warm-start depuis '{entry['dataset_name']}' "
                  f"(distance={entry['similarity_distance']:.3f}, "
                  f"algo={entry['best_config'].get('algorithm', '?')})")

        return configs

    def get_summary(self) -> pd.DataFrame:
        """Tableau résumé de la base méta."""
        if not self.entries:
            return pd.DataFrame()

        rows = []
        for entry in self.entries:
            rows.append({
                "Dataset":    entry["dataset_name"],
                "Best algo":  entry["best_config"].get("algorithm", "?"),
                "Best score": round(entry["best_score"], 4),
                "Accuracy":   round(entry["best_accuracy"], 4),
                "n_samples":  int(entry["meta_features"]["n_samples"]),
                "n_features": int(entry["meta_features"]["n_features"]),
            })

        return pd.DataFrame(rows)


# -------------------------------------------------------
# Test rapide
# -------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.model_selection import train_test_split

    print("Test du méta-apprentissage...")

    # Simuler 2 datasets
    for name, loader in [("breast_cancer", load_breast_cancer), ("iris", load_iris)]:
        data     = loader()
        X_train, _, y_train, _ = train_test_split(data.data, data.target, test_size=0.2)

        meta_features = extract_meta_features(X_train, y_train)
        print(f"\n📊 Méta-features de '{name}':")
        for k, v in list(meta_features.items())[:5]:
            print(f"  {k}: {v:.4f}")

    # Tester la base méta
    base = MetaLearningBase(storage_path="/tmp/test_meta.pkl")

    # Ajouter une entrée fictive
    base.add_entry(
        dataset_name  = "test_dataset",
        meta_features = meta_features,
        best_config   = {"algorithm": "random_forest", "rf_n_estimators": 100},
        best_score    = 0.15,
        best_accuracy = 0.95,
    )

    print("\n📚 Résumé de la base méta :")
    print(base.get_summary())
