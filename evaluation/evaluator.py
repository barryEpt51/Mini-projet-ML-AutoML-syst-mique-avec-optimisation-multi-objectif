"""
MODULE 3 : Évaluation multi-objectif
======================================
Ce module mesure les 3 objectifs simultanément pour chaque configuration :
  1. Accuracy  → performance prédictive (à MAXIMISER → on minimise 1-accuracy)
  2. Latence   → temps d'inférence en ms (à MINIMISER)
  3. Mémoire   → pic de RAM utilisé en MB (à MINIMISER)

C'est le cœur de notre AutoML systémique !
"""

import time
import tracemalloc
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


# -------------------------------------------------------
# CONTRAINTES RESSOURCES
# Définir des limites "dures" : si dépassées → configuration rejetée
# -------------------------------------------------------
RESOURCE_CONSTRAINTS = {
    "max_latency_ms":  500.0,   # max 500ms pour prédire le test set
    "max_memory_mb":   512.0,   # max 512 MB de RAM
    "max_train_time_s": 120.0,  # max 2 minutes d'entraînement
}


def measure_latency(model, X_test: np.ndarray, n_repeats: int = 5) -> float:
    """
    Mesure le temps d'inférence moyen en millisecondes.

    On répète n_repeats fois pour avoir une mesure stable.
    On prend la médiane (plus robuste que la moyenne).

    Paramètres
    ----------
    model    : modèle sklearn déjà entraîné
    X_test   : données de test
    n_repeats: nombre de répétitions

    Retourne
    --------
    float : latence médiane en ms
    """
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        model.predict(X_test)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # convertir en ms

    return float(np.median(times))


def measure_memory(model, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Mesure le pic de mémoire RAM utilisé pendant l'entraînement en MB.

    Utilise tracemalloc (bibliothèque standard Python).

    Paramètres
    ----------
    model  : modèle sklearn NON entraîné (sera entraîné ici)
    X_train: données d'entraînement
    y_train: labels d'entraînement

    Retourne
    --------
    float : pic de mémoire en MB
    """
    tracemalloc.start()

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        tracemalloc.stop()
        raise e

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak / (1024 * 1024)  # convertir bytes → MB


def evaluate_configuration(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    budget:  float = 1.0,
    use_cv:  bool  = False,
) -> dict:
    """
    Évalue une configuration sur les 3 objectifs avec gestion des contraintes.

    Paramètres
    ----------
    model    : modèle sklearn à évaluer
    X_train  : features d'entraînement
    y_train  : labels d'entraînement
    X_test   : features de test
    y_test   : labels de test
    budget   : fraction du dataset à utiliser (0.1 à 1.0) → pour Hyperband
               budget=0.1 signifie "utilise seulement 10% des données"
    use_cv   : utiliser la cross-validation au lieu du simple split

    Retourne
    --------
    dict avec :
        - error        : 1 - accuracy (SMAC minimise, donc on inverse)
        - latency_ms   : temps d'inférence en ms
        - memory_mb    : pic RAM en MB
        - accuracy     : accuracy brute (pour affichage)
        - train_time_s : temps d'entraînement en secondes
        - constraint_violated : True si contraintes dépassées
        - status       : "SUCCESS" ou "FAILED"
    """

    # --- Appliquer le budget (sous-échantillonnage pour Hyperband) ---
    if budget < 1.0:
        n_samples = max(int(len(X_train) * budget), 10)  # minimum 10 samples
        indices   = np.random.choice(len(X_train), n_samples, replace=False)
        X_tr      = X_train[indices]
        y_tr      = y_train[indices]
    else:
        X_tr = X_train
        y_tr = y_train

    # --- Entraînement avec mesure du temps et de la mémoire ---
    try:
        tracemalloc.start()
        train_start = time.perf_counter()

        model.fit(X_tr, y_tr)

        train_time = time.perf_counter() - train_start
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_mb = peak_memory / (1024 * 1024)

    except Exception as e:
        tracemalloc.stop()
        print(f"  ⚠️  Erreur d'entraînement: {e}")
        return {
            "error":              1.0,
            "latency_ms":         9999.0,
            "memory_mb":          9999.0,
            "accuracy":           0.0,
            "train_time_s":       9999.0,
            "constraint_violated": True,
            "status":             "FAILED",
        }

    # --- Vérifier contrainte temps d'entraînement ---
    if train_time > RESOURCE_CONSTRAINTS["max_train_time_s"]:
        return {
            "error":              1.0,
            "latency_ms":         9999.0,
            "memory_mb":          memory_mb,
            "accuracy":           0.0,
            "train_time_s":       train_time,
            "constraint_violated": True,
            "status":             "TIMEOUT",
        }

    # --- Accuracy ---
    if use_cv:
        scores  = cross_val_score(model, X_tr, y_tr, cv=3, scoring="accuracy")
        accuracy = float(np.mean(scores))
    else:
        y_pred   = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

    # --- Latence ---
    latency_ms = measure_latency(model, X_test)

    # --- Vérifier contraintes ressources ---
    constraint_violated = (
        latency_ms > RESOURCE_CONSTRAINTS["max_latency_ms"] or
        memory_mb  > RESOURCE_CONSTRAINTS["max_memory_mb"]
    )

    return {
        "error":               1.0 - accuracy,   # SMAC minimise → erreur
        "latency_ms":          latency_ms,
        "memory_mb":           memory_mb,
        "accuracy":            accuracy,
        "train_time_s":        train_time,
        "constraint_violated": constraint_violated,
        "status":              "SUCCESS",
    }


def compute_pareto_front(results: list) -> list:
    """
    Calcule le front de Pareto à partir d'une liste de résultats.
    
    Une solution est Pareto-optimale si aucune autre solution n'est
    meilleure sur TOUS les objectifs simultanément.

    Les 3 objectifs (tous à minimiser) :
      - error      (1 - accuracy)
      - latency_ms
      - memory_mb

    Paramètres
    ----------
    results : liste de dicts (sortie de evaluate_configuration)

    Retourne
    --------
    liste des indices des solutions Pareto-optimales
    """
    n = len(results)
    is_pareto = np.ones(n, dtype=bool)

    objectives = np.array([
        [r["error"], r["latency_ms"], r["memory_mb"]]
        for r in results
    ])

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j domine i si j est meilleur ou égal sur tous les objectifs
            # et strictement meilleur sur au moins un
            if (np.all(objectives[j] <= objectives[i]) and
                np.any(objectives[j] <  objectives[i])):
                is_pareto[i] = False
                break

    return list(np.where(is_pareto)[0])


# -------------------------------------------------------
# Test rapide
# -------------------------------------------------------
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_breast_cancer

    print("Test du module d'évaluation...")
    data = load_breast_cancer()
    X, y = data.data, data.target

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    result = evaluate_configuration(model, X_train, y_train, X_test, y_test)

    print(f"\n📊 Résultats de l'évaluation :")
    print(f"  Accuracy     : {result['accuracy']:.4f}")
    print(f"  Erreur       : {result['error']:.4f}")
    print(f"  Latence      : {result['latency_ms']:.2f} ms")
    print(f"  Mémoire      : {result['memory_mb']:.2f} MB")
    print(f"  Temps train  : {result['train_time_s']:.2f} s")
    print(f"  Contraintes  : {'⚠️ VIOLÉES' if result['constraint_violated'] else '✅ OK'}")
    print(f"  Statut       : {result['status']}")
