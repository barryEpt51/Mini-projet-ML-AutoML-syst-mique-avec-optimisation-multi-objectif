"""
run_pipeline.py
===============
Pipeline CLI : évaluation sur les 10 datasets avec SMAC + Hyperband.
Exécution : python run_pipeline.py
             python run_pipeline.py --quick          (3 datasets, 10 trials)
             python run_pipeline.py --datasets breast_cancer iris --n_trials 20
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from data.datasets               import load_dataset, DATASETS
from optimization.smac_optimizer import AutoMLOptimizer, OBJECTIVE_WEIGHTS
from optimization.meta_learning  import MetaLearningBase, extract_meta_features

PIPELINE_CONFIG = {
    "n_trials":    30,
    "min_budget":  0.1,
    "max_budget":  1.0,
    "output_dir":  str(PROJECT_DIR / "smac_output"),
    "results_dir": str(PROJECT_DIR / "results"),
    "meta_path":   str(PROJECT_DIR / "meta_learning_base.pkl"),
    "seed":        42,
}


def run_single(name: str, meta_base: MetaLearningBase, cfg: dict) -> dict:
    print(f"\n{'─'*55}\n📊 {name}\n{'─'*55}")
    try:
        data = load_dataset(name, seed=cfg["seed"])
    except Exception as e:
        print(f"❌ Impossible de charger '{name}': {e}")
        return None

    meta_feats   = extract_meta_features(data["X_train"], data["y_train"])
    warm_configs = meta_base.get_warm_start_configs(meta_feats, top_k=2)
    if warm_configs:
        print(f"  🔥 Warm start → algo : {warm_configs[0].get('algorithm','?')}")
    else:
        print("  🆕 Démarrage à froid")

    optimizer = AutoMLOptimizer(
        dataset_name = name,
        n_trials     = cfg["n_trials"],
        min_budget   = cfg["min_budget"],
        max_budget   = cfg["max_budget"],
        output_dir   = cfg["output_dir"],
        seed         = cfg["seed"],
    )

    best = optimizer.run(
        data["X_train"], data["y_train"],
        data["X_test"],  data["y_test"],
        task         = data["task"],
        warm_configs = warm_configs,
    )
    if best is None:
        return None

    meta_base.add_entry(
        dataset_name  = name,
        meta_features = meta_feats,
        best_config   = optimizer.best_config or {},
        best_score    = optimizer.best_score,
        best_accuracy = best.get("accuracy", 0),
    )

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    optimizer.get_all_results().to_csv(results_dir / f"{name}_results.csv", index=False)

    return {
        "Dataset":     name,
        "Tâche":       data["task"],
        "Best algo":   best.get("algorithm", "N/A"),
        "Accuracy":    round(best.get("accuracy", 0), 4),
        "Latence_ms":  round(best.get("latency_ms", 0), 2),
        "Mémoire_MB":  round(best.get("memory_mb", 0), 2),
        "Score SMAC":  round(best.get("score", 1), 4),
        "Temps (s)":   round(best.get("total_time_s", 0), 1),
        "N évals":     best.get("n_evaluations", 0),
    }


def run_pipeline(dataset_names=None, cfg=None):
    if cfg is None:
        cfg = PIPELINE_CONFIG
    if dataset_names is None:
        dataset_names = list(DATASETS.keys())

    print("\n" + "═"*55)
    print("🤖  AutoML Systémique — Pipeline complet")
    print(f"    Datasets : {len(dataset_names)} | n_trials : {cfg['n_trials']}")
    print("═"*55)

    meta_base  = MetaLearningBase(storage_path=cfg["meta_path"])
    summaries  = []
    t0         = time.time()

    for i, name in enumerate(dataset_names, 1):
        print(f"\n[{i}/{len(dataset_names)}] ", end="")
        s = run_single(name, meta_base, cfg)
        if s:
            summaries.append(s)

    total = time.time() - t0
    if summaries:
        df = pd.DataFrame(summaries)
        print("\n" + "═"*55)
        print("📋  RAPPORT DE SYNTHÈSE")
        print("═"*55)
        print(df.to_string(index=False))
        print(f"\n⏱  Durée totale : {total/60:.1f} min")
        print(f"🎯  Accuracy moy. : {df['Accuracy'].mean():.4f}")
        print(f"⚡  Latence moy.  : {df['Latence_ms'].mean():.1f} ms")
        print(f"💾  Mémoire moy.  : {df['Mémoire_MB'].mean():.1f} MB")

        results_dir = Path(cfg["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / "pipeline_summary.csv"
        df.to_csv(path, index=False)
        print(f"\n💾  Rapport : {path}")
        return df
    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Pipeline — Sujet 3")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets à traiter (défaut: tous les 10)")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--quick", action="store_true",
                        help="Mode rapide : iris + breast_cancer + wine, 10 trials")
    args = parser.parse_args()

    cfg = dict(PIPELINE_CONFIG)
    cfg["n_trials"] = args.n_trials
    datasets = args.datasets

    if args.quick:
        datasets = ["iris", "breast_cancer", "wine"]
        cfg["n_trials"] = 10
        print("⚡ Mode rapide : 3 datasets, 10 trials")

    run_pipeline(dataset_names=datasets, cfg=cfg)
    print("\n✅ Pipeline terminé !")
