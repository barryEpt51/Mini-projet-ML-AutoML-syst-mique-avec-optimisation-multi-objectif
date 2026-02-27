"""
PIPELINE PRINCIPAL : Évaluation sur les 10 datasets
=====================================================
Ce script orchestre tout le projet :
  1. Charge les 10 datasets
  2. Pour chaque dataset : extrait les méta-features + optimise avec SMAC
  3. Stocke tout dans la base méta-apprentissage
  4. Génère un rapport de synthèse

Exécution : python run_pipeline.py
"""

import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Ajouter le répertoire du projet au path Python
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from data.datasets import load_all_datasets, load_dataset, get_dataset_summary
from optimization.smac_optimizer import AutoMLOptimizer
from optimization.meta_learning import MetaLearningBase, extract_meta_features


# -------------------------------------------------------
# CONFIGURATION DU PIPELINE
# -------------------------------------------------------
PIPELINE_CONFIG = {
    "n_trials":      30,    # évaluations SMAC par dataset (↑ = meilleur mais plus lent)
    "min_budget":    0.1,   # budget min Hyperband
    "max_budget":    1.0,   # budget max
    "output_dir":    str(PROJECT_DIR / "smac_output"),
    "results_dir":   str(PROJECT_DIR / "results"),
    "meta_base_path":str(PROJECT_DIR / "meta_learning_base.pkl"),
    "seed":          42,
}


def run_single_dataset(
    dataset_name: str,
    meta_base:    MetaLearningBase,
    config:       dict,
) -> dict:
    """
    Exécute le pipeline complet sur un seul dataset.
    
    Étapes :
      1. Charger le dataset
      2. Extraire les méta-features
      3. Récupérer les configs warm-start depuis la base méta
      4. Optimiser avec SMAC + Hyperband
      5. Stocker les résultats dans la base méta

    Retourne
    --------
    dict avec les résultats du meilleur modèle trouvé
    """
    print(f"\n{'─'*60}")
    print(f"📊 Dataset : {dataset_name}")
    print(f"{'─'*60}")

    # --- 1. Charger le dataset ---
    try:
        data = load_dataset(dataset_name)
    except Exception as e:
        print(f"❌ Impossible de charger '{dataset_name}': {e}")
        return None

    # --- 2. Extraire les méta-features ---
    meta_features = extract_meta_features(data["X_train"], data["y_train"])
    print(f"  📐 n_samples={data['n_samples']}, "
          f"n_features={data['n_features']}, "
          f"n_classes={data['n_classes']}")

    # --- 3. Warm-start depuis la base méta ---
    warm_configs = meta_base.get_warm_start_configs(meta_features, top_k=2)
    if warm_configs:
        print(f"  🔥 Warm-start : {len(warm_configs)} config(s) récupérée(s)")
    else:
        print(f"  🆕 Démarrage à froid (base méta vide)")

    # --- 4. Optimisation SMAC + Hyperband ---
    optimizer = AutoMLOptimizer(
        dataset_name = dataset_name,
        n_trials     = config["n_trials"],
        min_budget   = config["min_budget"],
        max_budget   = config["max_budget"],
        output_dir   = config["output_dir"],
        seed         = config["seed"],
    )

    best_result = optimizer.run(
        X_train = data["X_train"],
        y_train = data["y_train"],
        X_test  = data["X_test"],
        y_test  = data["y_test"],
    )

    if best_result is None:
        return None

    # --- 5. Sauvegarder dans la base méta ---
    meta_base.add_entry(
        dataset_name  = dataset_name,
        meta_features = meta_features,
        best_config   = optimizer.get_best_config() or {},
        best_score    = optimizer.best_score,
        best_accuracy = best_result.get("accuracy", 0),
    )

    # --- 6. Sauvegarder les résultats détaillés ---
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    optimizer.save_results(str(results_dir / f"{dataset_name}_results.csv"))

    # Résumé compact
    summary = {
        "dataset":      dataset_name,
        "n_samples":    data["n_samples"],
        "n_features":   data["n_features"],
        "best_algo":    optimizer.get_best_config().get("algorithm", "N/A") if optimizer.get_best_config() else "N/A",
        "accuracy":     round(best_result.get("accuracy", 0), 4),
        "latency_ms":   round(best_result.get("latency_ms", 0), 2),
        "memory_mb":    round(best_result.get("memory_mb", 0), 2),
        "score":        round(best_result.get("score", 1), 4),
        "n_evaluations":best_result.get("n_evaluations", 0),
        "optim_time_s": round(best_result.get("total_optimization_time_s", 0), 1),
    }

    return summary


def run_full_pipeline(dataset_names: list = None, config: dict = None) -> pd.DataFrame:
    """
    Exécute le pipeline sur tous les datasets.

    Paramètres
    ----------
    dataset_names : liste des datasets à traiter (None = tous les 10)
    config        : configuration du pipeline

    Retourne
    --------
    DataFrame avec les résultats de synthèse
    """
    if config is None:
        config = PIPELINE_CONFIG

    if dataset_names is None:
        from data.datasets import DATASETS
        dataset_names = list(DATASETS.keys())

    print("\n" + "═"*60)
    print("🤖 AutoML Systémique - Pipeline complet")
    print("═"*60)
    print(f"Datasets   : {len(dataset_names)}")
    print(f"n_trials   : {config['n_trials']} par dataset")
    print(f"Algorithmes: RF, GB, ET, SVM, KNN, LR, DT, XGBoost")
    print("═"*60)

    # Initialiser la base méta
    meta_base = MetaLearningBase(storage_path=config["meta_base_path"])

    # Exécuter sur chaque dataset
    pipeline_start = time.time()
    all_summaries  = []

    for i, name in enumerate(dataset_names, 1):
        print(f"\n[{i}/{len(dataset_names)}] ", end="")
        summary = run_single_dataset(name, meta_base, config)
        if summary:
            all_summaries.append(summary)

    total_time = time.time() - pipeline_start

    # --- Rapport de synthèse ---
    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)

        print("\n" + "═"*60)
        print("📋 RAPPORT DE SYNTHÈSE")
        print("═"*60)
        print(df_summary.to_string(index=False))
        print(f"\n⏱  Temps total : {total_time/60:.1f} minutes")
        print(f"📊 Accuracy moyenne : {df_summary['accuracy'].mean():.4f}")
        print(f"⚡ Latence moyenne  : {df_summary['latency_ms'].mean():.1f} ms")
        print(f"💾 Mémoire moyenne  : {df_summary['memory_mb'].mean():.1f} MB")

        # Sauvegarder le rapport
        results_dir = Path(config["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        summary_path = results_dir / "pipeline_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"\n💾 Rapport sauvegardé : {summary_path}")

        return df_summary

    return pd.DataFrame()


# -------------------------------------------------------
# Point d'entrée avec arguments en ligne de commande
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Pipeline")
    parser.add_argument(
        "--datasets", nargs="+",
        help="Datasets à traiter (défaut: tous)",
        default=None
    )
    parser.add_argument(
        "--n_trials", type=int, default=30,
        help="Nombre d'évaluations SMAC par dataset"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Mode rapide : 10 trials, 3 datasets seulement"
    )
    args = parser.parse_args()

    config = dict(PIPELINE_CONFIG)
    config["n_trials"] = args.n_trials

    datasets = args.datasets
    if args.quick:
        datasets = ["diabetes", "breast_cancer", "iris"]
        config["n_trials"] = 10
        print("⚡ Mode rapide activé : 3 datasets, 10 trials")

    results = run_full_pipeline(dataset_names=datasets, config=config)
    print("\n✅ Pipeline terminé !")
