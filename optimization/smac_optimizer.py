"""
MODULE 4 : Optimisation SMAC3 + Hyperband (cœur du projet)
============================================================
C'est ici que tout se passe :
  - SMAC3 : optimisation bayésienne (apprend des évaluations passées)
  - Hyperband : allocation budgétaire adaptative (arrête tôt les mauvaises configs)
  - Multi-objectif : on optimise accuracy + latence + mémoire

SMAC3 vs GridSearch :
  - GridSearch : teste TOUTES les combinaisons → très lent
  - SMAC3 : apprend quelles configurations sont prometteuses → beaucoup plus rapide
  
Hyperband :
  - Commence avec beaucoup de configs sur peu de données (budget faible)
  - Garde seulement les meilleures et augmente le budget
  - Résultat : trouve de bonnes configs sans tout évaluer
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from smac import MultiFidelityFacade, Scenario
from smac.intensifier import Hyperband
from ConfigSpace import Configuration

# Nos modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.search_space import get_configspace, build_model
from evaluation.evaluator import evaluate_configuration, compute_pareto_front, RESOURCE_CONSTRAINTS


# -------------------------------------------------------
# POIDS pour combiner les 3 objectifs en un seul score
# (nécessaire car SMAC optimise 1 seul objectif à la fois)
# On peut jouer sur ces poids selon la priorité
# -------------------------------------------------------
OBJECTIVE_WEIGHTS = {
    "error":      0.6,   # accuracy = priorité principale
    "latency":    0.2,   # latence = secondaire
    "memory":     0.2,   # mémoire = secondaire
}

# Valeurs de normalisation (pour que les 3 objectifs soient comparables)
NORMALIZATION = {
    "error_max":     1.0,      # max possible : 100% d'erreur
    "latency_max":   RESOURCE_CONSTRAINTS["max_latency_ms"],
    "memory_max":    RESOURCE_CONSTRAINTS["max_memory_mb"],
}


def weighted_objective(error: float, latency_ms: float, memory_mb: float) -> float:
    """
    Combine les 3 objectifs en un score scalaire normalisé.
    
    SMAC minimise ce score → on veut : faible erreur, faible latence, faible mémoire.
    
    Chaque objectif est normalisé entre 0 et 1 avant la combinaison.
    """
    # Normalisation
    norm_error   = error      / NORMALIZATION["error_max"]
    norm_latency = min(latency_ms / NORMALIZATION["latency_max"], 1.0)
    norm_memory  = min(memory_mb  / NORMALIZATION["memory_max"],  1.0)

    # Score pondéré
    score = (
        OBJECTIVE_WEIGHTS["error"]   * norm_error   +
        OBJECTIVE_WEIGHTS["latency"] * norm_latency +
        OBJECTIVE_WEIGHTS["memory"]  * norm_memory
    )

    return float(score)


class AutoMLOptimizer:
    """
    Classe principale qui orchestre SMAC3 + Hyperband.

    Utilisation typique :
        optimizer = AutoMLOptimizer(dataset_name="diabetes")
        optimizer.run(X_train, y_train, X_test, y_test)
        best = optimizer.get_best_config()
        results = optimizer.get_all_results()
    """

    def __init__(
        self,
        dataset_name:   str  = "dataset",
        n_trials:       int  = 50,     # nombre d'évaluations total
        min_budget:     float = 0.1,   # budget min Hyperband (10% des données)
        max_budget:     float = 1.0,   # budget max (100% des données)
        output_dir:     str  = "./smac_output",
        seed:           int  = 42,
    ):
        self.dataset_name = dataset_name
        self.n_trials     = n_trials
        self.min_budget   = min_budget
        self.max_budget   = max_budget
        self.output_dir   = Path(output_dir) / dataset_name
        self.seed         = seed

        # Stockage des résultats
        self.all_results  = []
        self.best_config  = None
        self.best_score   = float("inf")
        self.smac         = None

        # Données (seront définies dans run())
        self.X_train = None
        self.y_train = None
        self.X_test  = None
        self.y_test  = None

    def _target_function(self, config: Configuration, seed: int, budget: float) -> float:
        """
        Fonction objectif appelée par SMAC à chaque évaluation.
        
        SMAC appelle cette fonction avec :
          - config : une configuration à évaluer
          - seed   : seed aléatoire
          - budget : fraction du dataset à utiliser (Hyperband)
        
        Elle retourne le score combiné (à minimiser).
        """
        try:
            # Construire le modèle
            model = build_model(dict(config), random_state=seed)

            # Évaluer sur les 3 objectifs
            result = evaluate_configuration(
                model    = model,
                X_train  = self.X_train,
                y_train  = self.y_train,
                X_test   = self.X_test,
                y_test   = self.y_test,
                budget   = budget,
            )

            # Calculer le score combiné
            if result["status"] == "FAILED" or result["constraint_violated"]:
                score = 1.0  # pénalité maximale
            else:
                score = weighted_objective(
                    error      = result["error"],
                    latency_ms = result["latency_ms"],
                    memory_mb  = result["memory_mb"],
                )

            # Stocker le résultat complet pour analyse ultérieure
            result["config"]      = dict(config)
            result["score"]       = score
            result["budget"]      = budget
            result["dataset"]     = self.dataset_name
            result["timestamp"]   = time.time()
            self.all_results.append(result)

            # Mettre à jour le meilleur résultat
            if score < self.best_score and result["status"] == "SUCCESS":
                self.best_score  = score
                self.best_config = dict(config)

            return score

        except Exception as e:
            print(f"  ⚠️  Erreur dans target_function: {e}")
            return 1.0  # pénalité maximale

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
    ) -> dict:
        """
        Lance l'optimisation SMAC + Hyperband.
        
        Paramètres
        ----------
        X_train, y_train : données d'entraînement
        X_test, y_test   : données de test

        Retourne
        --------
        dict : meilleure configuration trouvée avec ses scores
        """
        print(f"\n{'='*60}")
        print(f"🚀 Optimisation SMAC + Hyperband : {self.dataset_name}")
        print(f"   n_trials  : {self.n_trials}")
        print(f"   budget    : {self.min_budget:.0%} → {self.max_budget:.0%} des données")
        print(f"{'='*60}")

        # Stocker les données
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test

        # Obtenir l'espace de configuration
        cs = get_configspace()

        # --- Configuration du Scenario SMAC ---
        scenario = Scenario(
            configspace        = cs,
            name               = f"automl_{self.dataset_name}",
            output_directory   = self.output_dir,
            deterministic      = False,
            n_trials           = self.n_trials,
            min_budget         = self.min_budget,
            max_budget         = self.max_budget,
            seed               = self.seed,
        )

        # --- Créer SMAC avec MultiFidelityFacade (SMAC + Hyperband) ---
        # MultiFidelityFacade = SMAC bayésien + Hyperband pour le budget
        self.smac = MultiFidelityFacade(
            scenario          = scenario,
            target_function   = self._target_function,
            intensifier       = Hyperband(scenario, eta=3),
            # eta=3 : facteur de réduction Hyperband (standard)
            overwrite         = True,
        )

        # --- Lancer l'optimisation ---
        start_time = time.time()
        incumbent  = self.smac.optimize()
        total_time = time.time() - start_time

        # --- Rapport final ---
        best_result = self._get_final_evaluation(incumbent)
        best_result["total_optimization_time_s"] = total_time
        best_result["n_evaluations"] = len(self.all_results)

        print(f"\n✅ Optimisation terminée en {total_time:.1f}s")
        print(f"   {len(self.all_results)} configurations évaluées")
        print(f"\n🏆 Meilleure configuration :")
        print(f"   Algorithme : {self.best_config.get('algorithm', 'N/A')}")
        print(f"   Accuracy   : {best_result.get('accuracy', 0):.4f}")
        print(f"   Latence    : {best_result.get('latency_ms', 0):.2f} ms")
        print(f"   Mémoire    : {best_result.get('memory_mb', 0):.2f} MB")
        print(f"   Score      : {self.best_score:.4f}")

        return best_result

    def _get_final_evaluation(self, config: Configuration) -> dict:
        """Évaluation finale de la meilleure config sur budget=1.0 (données complètes)."""
        model  = build_model(dict(config), random_state=self.seed)
        result = evaluate_configuration(
            model   = model,
            X_train = self.X_train,
            y_train = self.y_train,
            X_test  = self.X_test,
            y_test  = self.y_test,
            budget  = 1.0,
        )
        result["config"] = dict(config)
        result["score"]  = weighted_objective(
            result["error"], result["latency_ms"], result["memory_mb"]
        )
        return result

    def get_best_config(self) -> dict:
        """Retourne la meilleure configuration trouvée."""
        return self.best_config

    def get_all_results(self) -> pd.DataFrame:
        """Retourne tous les résultats sous forme de DataFrame."""
        if not self.all_results:
            return pd.DataFrame()

        rows = []
        for r in self.all_results:
            row = {
                "dataset":    r.get("dataset", ""),
                "algorithm":  r.get("config", {}).get("algorithm", ""),
                "accuracy":   r.get("accuracy", 0),
                "error":      r.get("error", 1),
                "latency_ms": r.get("latency_ms", 0),
                "memory_mb":  r.get("memory_mb", 0),
                "score":      r.get("score", 1),
                "budget":     r.get("budget", 0),
                "status":     r.get("status", ""),
                "train_time_s": r.get("train_time_s", 0),
                "constraint_violated": r.get("constraint_violated", False),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def get_pareto_front(self) -> pd.DataFrame:
        """Retourne uniquement les solutions Pareto-optimales."""
        df = self.get_all_results()
        if df.empty:
            return df

        successful = df[df["status"] == "SUCCESS"].reset_index(drop=True)
        if successful.empty:
            return successful

        results_list = successful.to_dict("records")
        pareto_indices = compute_pareto_front(results_list)

        return successful.iloc[pareto_indices].reset_index(drop=True)

    def save_results(self, filepath: str):
        """Sauvegarde tous les résultats en CSV."""
        df = self.get_all_results()
        df.to_csv(filepath, index=False)
        print(f"💾 Résultats sauvegardés : {filepath}")


# -------------------------------------------------------
# Test rapide
# -------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    print("Test de l'optimiseur SMAC + Hyperband...")

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    optimizer = AutoMLOptimizer(
        dataset_name = "breast_cancer_test",
        n_trials     = 15,   # peu de trials pour le test
        output_dir   = "/tmp/smac_test"
    )

    best = optimizer.run(X_train, y_train, X_test, y_test)

    print("\n📊 Toutes les évaluations :")
    df = optimizer.get_all_results()
    print(df[["algorithm", "accuracy", "latency_ms", "memory_mb", "score"]].head(10))

    print("\n🎯 Front de Pareto :")
    pareto = optimizer.get_pareto_front()
    print(pareto[["algorithm", "accuracy", "latency_ms", "memory_mb"]])
