"""
evaluation/evaluator.py
=======================
Mesure des 3 objectifs simultanément pour chaque configuration :
  1. Accuracy  → performance prédictive (classification: accuracy_score, régression: R²)
  2. Latence   → temps d'inférence médian en ms sur n_repeats répétitions
  3. Mémoire   → pic de RAM réel pendant l'entraînement via tracemalloc (MB)
"""

import time
import tracemalloc
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Contraintes ressources par défaut ─────────────────
RESOURCE_CONSTRAINTS = {
    "max_latency_ms":   500.0,
    "max_memory_mb":    512.0,
    "max_train_time_s": 120.0,
}


def measure_latency(model, X_test: np.ndarray, n_repeats: int = 5) -> float:
    """
    Mesure la latence médiane d'inférence en ms.
    On prend la médiane sur n_repeats répétitions pour réduire le bruit.
    """
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        model.predict(X_test)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def evaluate_configuration(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    task:    str   = "classification",
    budget:  float = 1.0,
) -> dict:
    """
    Évalue une configuration sur les 3 objectifs avec gestion des contraintes.

    budget : fraction du dataset à utiliser (0.1 à 1.0) pour Hyperband.
             budget=0.1 → seulement 10% des données d'entraînement.

    Retourne
    --------
    dict : error, latency_ms, memory_mb, accuracy, train_time_s,
           constraint_violated, status
    """
    # ── Appliquer le budget Hyperband ─────────────────
    if budget < 1.0:
        n = max(int(len(X_train) * budget), 10)
        idx = np.random.choice(len(X_train), n, replace=False)
        X_tr, y_tr = X_train[idx], y_train[idx]
    else:
        X_tr, y_tr = X_train, y_train

    # ── Entraînement + mesure mémoire (tracemalloc) ───
    try:
        tracemalloc.start()
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / (1024 * 1024)
    except Exception as e:
        tracemalloc.stop()
        return {
            "error": 1.0, "latency_ms": 9999.0, "memory_mb": 9999.0,
            "accuracy": 0.0, "train_time_s": 9999.0,
            "constraint_violated": True, "status": "FAILED"
        }

    # ── Contrainte temps ──────────────────────────────
    if train_time > RESOURCE_CONSTRAINTS["max_train_time_s"]:
        return {
            "error": 1.0, "latency_ms": 9999.0, "memory_mb": memory_mb,
            "accuracy": 0.0, "train_time_s": train_time,
            "constraint_violated": True, "status": "TIMEOUT"
        }

    # ── Score de performance ───────────────────────────
    preds = model.predict(X_test)
    if task == "classification":
        accuracy = float(accuracy_score(y_test, preds))
    else:
        accuracy = float(max(r2_score(y_test, preds), 0.0))  # R² borné à 0

    error = 1.0 - accuracy  # SMAC minimise → on inverse

    # ── Latence (médiane sur 5 répétitions) ───────────
    latency_ms = measure_latency(model, X_test)

    # ── Vérification contraintes ──────────────────────
    constraint_violated = (
        latency_ms > RESOURCE_CONSTRAINTS["max_latency_ms"] or
        memory_mb  > RESOURCE_CONSTRAINTS["max_memory_mb"]
    )

    return {
        "error":               error,
        "latency_ms":          latency_ms,
        "memory_mb":           memory_mb,
        "accuracy":            accuracy,
        "train_time_s":        train_time,
        "constraint_violated": constraint_violated,
        "status":              "SUCCESS",
    }


def compute_pareto_front(results: list) -> list:
    """
    Calcule les indices des solutions Pareto-optimales.
    Une solution est Pareto-optimale si aucune autre ne la domine
    sur tous les objectifs simultanément.
    Objectifs (tous à minimiser) : error, latency_ms, memory_mb
    """
    n = len(results)
    is_pareto = np.ones(n, dtype=bool)
    objectives = np.array([
        [r.get("error", 1.0), r.get("latency_ms", 9999.0), r.get("memory_mb", 9999.0)]
        for r in results
    ])
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if (np.all(objectives[j] <= objectives[i]) and
                    np.any(objectives[j] < objectives[i])):
                is_pareto[i] = False
                break
    return list(np.where(is_pareto)[0])
