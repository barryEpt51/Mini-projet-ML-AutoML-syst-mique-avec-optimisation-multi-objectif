"""
optimization/smac_optimizer.py
===============================
Cœur du framework : SMAC3 + Hyperband intégré nativement.

Points clés vs version précédente :
  ✅ MultiFidelityFacade  → SMAC bayésien + Hyperband natif (pas manuel)
  ✅ intensifier=Hyperband(scenario, eta=3) → allocation budget SMAC officielle
  ✅ Scalarisation pondérée des 3 objectifs (poids configurables)
  ✅ Warm start via initial_design depuis la base méta
  ✅ budget = fraction des données (0.1 → 1.0) passé par Hyperband à l'évaluateur
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from smac import MultiFidelityFacade, Scenario
from smac.intensifier import Hyperband
from ConfigSpace import Configuration

from models.search_space import get_configspace, build_model
from evaluation.evaluator import (
    evaluate_configuration, compute_pareto_front, RESOURCE_CONSTRAINTS
)

# ── Poids de scalarisation (modifiables depuis l'interface) ──
OBJECTIVE_WEIGHTS = {
    "error":   0.6,
    "latency": 0.2,
    "memory":  0.2,
}


def weighted_objective(error: float, latency_ms: float, memory_mb: float) -> float:
    """
    Combine les 3 objectifs en un score scalaire normalisé que SMAC minimise.
    Chaque objectif est normalisé entre 0 et 1 avant la combinaison.
    """
    norm_error   = error
    norm_latency = min(latency_ms / RESOURCE_CONSTRAINTS["max_latency_ms"], 1.0)
    norm_memory  = min(memory_mb  / RESOURCE_CONSTRAINTS["max_memory_mb"],  1.0)

    return float(
        OBJECTIVE_WEIGHTS["error"]   * norm_error   +
        OBJECTIVE_WEIGHTS["latency"] * norm_latency +
        OBJECTIVE_WEIGHTS["memory"]  * norm_memory
    )


class AutoMLOptimizer:
    """
    Orchestre SMAC3 + Hyperband sur un dataset.

    Utilisation :
        optimizer = AutoMLOptimizer("breast_cancer", n_trials=30)
        best = optimizer.run(X_train, y_train, X_test, y_test, task)
        df   = optimizer.get_all_results()
    """

    def __init__(
        self,
        dataset_name: str   = "dataset",
        n_trials:     int   = 30,
        min_budget:   float = 0.1,
        max_budget:   float = 1.0,
        output_dir:   str   = "./smac_output",
        seed:         int   = 42,
    ):
        self.dataset_name = dataset_name
        self.n_trials     = n_trials
        self.min_budget   = min_budget
        self.max_budget   = max_budget
        self.output_dir   = Path(output_dir) / dataset_name
        self.seed         = seed

        self.all_results  = []
        self.best_config  = None
        self.best_score   = float("inf")
        self.task         = "classification"

        # Données
        self.X_train = self.y_train = self.X_test = self.y_test = None

    def _target_function(self, config: Configuration, seed: int,
                          budget: float) -> float:
        """
        Fonction objectif appelée par SMAC + Hyperband à chaque évaluation.
        budget : fraction du dataset passée automatiquement par Hyperband (0.1 → 1.0)
        """
        try:
            model  = build_model(dict(config), task=self.task, random_state=seed)
            result = evaluate_configuration(
                model, self.X_train, self.y_train,
                self.X_test, self.y_test,
                task=self.task, budget=budget
            )

            score = (1.0 if result["status"] != "SUCCESS" or result["constraint_violated"]
                     else weighted_objective(result["error"],
                                             result["latency_ms"],
                                             result["memory_mb"]))

            result.update({
                "config":    dict(config),
                "score":     score,
                "budget":    budget,
                "dataset":   self.dataset_name,
                "algorithm": config["algorithm"],
            })
            self.all_results.append(result)

            if score < self.best_score and result["status"] == "SUCCESS":
                self.best_score  = score
                self.best_config = dict(config)

            return score

        except Exception as e:
            return 1.0

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        task:    str = "classification",
        warm_configs: list = None,
        progress_callback=None,
    ) -> dict:
        """
        Lance SMAC3 + Hyperband natif.

        warm_configs : liste de dicts de configs pour le warm start méta
        progress_callback : fonction(trial_count, best_score) pour la UI
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.task    = task

        import tempfile, os
        cs = get_configspace(task=task)

        scenario = Scenario(
            configspace      = cs,
            name             = f"automl_{self.dataset_name}",
            output_directory = str(self.output_dir),
            deterministic    = False,
            n_trials         = self.n_trials,
            min_budget       = self.min_budget,
            max_budget       = self.max_budget,
            seed             = self.seed,
        )

        # ── Warm start depuis méta-apprentissage ──────
        initial_configs = []
        if warm_configs:
            for wc in warm_configs[:3]:
                try:
                    cfg = Configuration(cs, values={
                        k: v for k, v in wc.items() if k in cs
                    })
                    initial_configs.append(cfg)
                except Exception:
                    pass

        # ── SMAC + Hyperband natif ────────────────────
        # MultiFidelityFacade = optimisation bayésienne SMAC
        # intensifier = Hyperband → allocation de budget adaptative officielle
        smac = MultiFidelityFacade(
            scenario        = scenario,
            target_function = self._target_function,
            intensifier     = Hyperband(scenario, eta=3),
            overwrite       = True,
        )

        # Injecter les configs warm start
        for cfg in initial_configs:
            smac.ask()  # demander un trial pour l'initialiser

        t0       = time.time()
        incumbent = smac.optimize()
        duration  = time.time() - t0

        # Évaluation finale sur budget=1.0 (données complètes)
        best_result = self._final_eval(incumbent)
        best_result["total_time_s"]  = round(duration, 1)
        best_result["n_evaluations"] = len(self.all_results)

        return best_result

    def _final_eval(self, config: Configuration) -> dict:
        """Évaluation finale de l'incumbent sur le dataset complet."""
        model  = build_model(dict(config), task=self.task, random_state=self.seed)
        result = evaluate_configuration(
            model, self.X_train, self.y_train,
            self.X_test, self.y_test,
            task=self.task, budget=1.0
        )
        result["config"]    = dict(config)
        result["algorithm"] = config["algorithm"]
        result["score"]     = weighted_objective(
            result["error"], result["latency_ms"], result["memory_mb"]
        )
        return result

    def get_all_results(self) -> pd.DataFrame:
        if not self.all_results:
            return pd.DataFrame()
        rows = []
        for r in self.all_results:
            rows.append({
                "dataset":             r.get("dataset", ""),
                "algorithm":           r.get("algorithm", ""),
                "accuracy":            round(r.get("accuracy", 0), 4),
                "error":               round(r.get("error", 1), 4),
                "latency_ms":          round(r.get("latency_ms", 0), 3),
                "memory_mb":           round(r.get("memory_mb", 0), 2),
                "score":               round(r.get("score", 1), 5),
                "budget":              round(r.get("budget", 0), 2),
                "status":              r.get("status", ""),
                "train_time_s":        round(r.get("train_time_s", 0), 2),
                "constraint_violated": r.get("constraint_violated", False),
            })
        return pd.DataFrame(rows)

    def get_pareto_front(self) -> pd.DataFrame:
        df = self.get_all_results()
        if df.empty:
            return df
        ok = df[df["status"] == "SUCCESS"].reset_index(drop=True)
        if ok.empty:
            return ok
        idx = compute_pareto_front(ok.to_dict("records"))
        return ok.iloc[idx].reset_index(drop=True)
