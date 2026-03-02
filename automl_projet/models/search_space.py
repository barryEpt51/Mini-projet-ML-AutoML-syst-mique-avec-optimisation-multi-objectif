"""
models/search_space.py
======================
Espace de recherche des algorithmes classiques pour SMAC3.
8 algorithmes avec leurs hyperparamètres et plages de valeurs.
"""

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace import Float, Integer, Categorical
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_configspace(task: str = "classification") -> ConfigurationSpace:
    """
    Définit l'espace de configuration complet pour SMAC3.
    Utilise la syntaxe ConfigSpace moderne (Float, Integer, Categorical).

    Paramètres
    ----------
    task : 'classification' ou 'regression'

    Retourne
    --------
    ConfigurationSpace prêt pour SMAC MultiFidelityFacade
    """
    cs = ConfigurationSpace(seed=42)

    # ── Choix de l'algorithme ──────────────────────────
    algos = [
        "random_forest", "gradient_boosting", "extra_trees",
        "svm", "knn", "linear_model", "decision_tree"
    ]
    if XGBOOST_AVAILABLE:
        algos.append("xgboost")

    algo = Categorical("algorithm", algos, default="random_forest")
    cs.add(algo)

    # ── Random Forest ──────────────────────────────────
    cs.add(Integer("rf_n_estimators",      (10, 500),  default=100, log=True))
    cs.add(Integer("rf_max_depth",         (2,  30),   default=10))
    cs.add(Integer("rf_min_samples_split", (2,  20),   default=2))
    cs.add(Integer("rf_min_samples_leaf",  (1,  10),   default=1))
    cs.add(Categorical("rf_max_features",  ["sqrt", "log2"], default="sqrt"))

    # ── Gradient Boosting ──────────────────────────────
    cs.add(Integer("gb_n_estimators",  (10, 300),   default=100, log=True))
    cs.add(Float("gb_learning_rate",   (0.01, 0.5), default=0.1, log=True))
    cs.add(Integer("gb_max_depth",     (1,  10),    default=3))
    cs.add(Float("gb_subsample",       (0.5, 1.0),  default=1.0))

    # ── Extra Trees ────────────────────────────────────
    cs.add(Integer("et_n_estimators",      (10, 500), default=100, log=True))
    cs.add(Integer("et_max_depth",         (2,  30),  default=10))
    cs.add(Integer("et_min_samples_split", (2,  20),  default=2))

    # ── SVM ────────────────────────────────────────────
    cs.add(Float("svm_C",      (0.01, 100.0), default=1.0, log=True))
    cs.add(Categorical("svm_kernel", ["rbf", "linear", "poly"], default="rbf"))
    cs.add(Categorical("svm_gamma",  ["scale", "auto"],         default="scale"))

    # ── KNN ────────────────────────────────────────────
    cs.add(Integer("knn_n_neighbors", (1, 50),                    default=5))
    cs.add(Categorical("knn_weights", ["uniform", "distance"],    default="uniform"))
    cs.add(Categorical("knn_metric",  ["euclidean", "manhattan"], default="euclidean"))

    # ── Modèle Linéaire ────────────────────────────────
    cs.add(Float("linear_C",        (0.001, 100.0), default=1.0, log=True))
    cs.add(Float("linear_alpha",    (0.001, 100.0), default=1.0, log=True))

    # ── Decision Tree ──────────────────────────────────
    cs.add(Integer("dt_max_depth",         (1, 30), default=5))
    cs.add(Integer("dt_min_samples_split", (2, 20), default=2))
    cs.add(Categorical("dt_criterion",     ["gini", "entropy"] if task == "classification"
                                           else ["squared_error", "absolute_error"],
                       default="gini" if task == "classification" else "squared_error"))

    # ── XGBoost ────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        cs.add(Integer("xgb_n_estimators",  (10, 500),   default=100, log=True))
        cs.add(Float("xgb_learning_rate",   (0.01, 0.5), default=0.1, log=True))
        cs.add(Integer("xgb_max_depth",     (1, 10),     default=3))
        cs.add(Float("xgb_subsample",       (0.5, 1.0),  default=0.8))

    return cs


def build_model(config: dict, task: str = "classification",
                random_state: int = 42):
    """
    Instancie un modèle sklearn depuis une configuration SMAC.

    Paramètres
    ----------
    config       : dict de la configuration SMAC
    task         : 'classification' ou 'regression'
    random_state : seed

    Retourne
    --------
    Estimateur sklearn non entraîné
    """
    algo = config["algorithm"]
    is_clf = (task == "classification")

    if algo == "random_forest":
        params = dict(
            n_estimators      = int(config["rf_n_estimators"]),
            max_depth         = int(config["rf_max_depth"]),
            min_samples_split = int(config["rf_min_samples_split"]),
            min_samples_leaf  = int(config["rf_min_samples_leaf"]),
            max_features      = config["rf_max_features"],
            random_state      = random_state, n_jobs=-1
        )
        return RandomForestClassifier(**params) if is_clf else RandomForestRegressor(**params)

    elif algo == "gradient_boosting":
        params = dict(
            n_estimators  = int(config["gb_n_estimators"]),
            learning_rate = float(config["gb_learning_rate"]),
            max_depth     = int(config["gb_max_depth"]),
            subsample     = float(config["gb_subsample"]),
            random_state  = random_state
        )
        return GradientBoostingClassifier(**params) if is_clf else GradientBoostingRegressor(**params)

    elif algo == "extra_trees":
        params = dict(
            n_estimators      = int(config["et_n_estimators"]),
            max_depth         = int(config["et_max_depth"]),
            min_samples_split = int(config["et_min_samples_split"]),
            random_state      = random_state, n_jobs=-1
        )
        return ExtraTreesClassifier(**params) if is_clf else ExtraTreesRegressor(**params)

    elif algo == "svm":
        params = dict(C=float(config["svm_C"]), kernel=config["svm_kernel"],
                      gamma=config["svm_gamma"])
        return SVC(**params, random_state=random_state if config["svm_kernel"] != "linear" else None,
                   probability=True) if is_clf else SVR(**params)

    elif algo == "knn":
        params = dict(n_neighbors=int(config["knn_n_neighbors"]),
                      weights=config["knn_weights"],
                      metric=config["knn_metric"], n_jobs=-1)
        return KNeighborsClassifier(**params) if is_clf else KNeighborsRegressor(**params)

    elif algo == "linear_model":
        if is_clf:
            return LogisticRegression(C=float(config["linear_C"]),
                                      max_iter=500, random_state=random_state, n_jobs=-1)
        else:
            return Ridge(alpha=float(config["linear_alpha"]))

    elif algo == "decision_tree":
        params = dict(max_depth=int(config["dt_max_depth"]),
                      min_samples_split=int(config["dt_min_samples_split"]),
                      criterion=config["dt_criterion"],
                      random_state=random_state)
        return DecisionTreeClassifier(**params) if is_clf else DecisionTreeRegressor(**params)

    elif algo == "xgboost" and XGBOOST_AVAILABLE:
        params = dict(
            n_estimators     = int(config["xgb_n_estimators"]),
            learning_rate    = float(config["xgb_learning_rate"]),
            max_depth        = int(config["xgb_max_depth"]),
            subsample        = float(config["xgb_subsample"]),
            random_state     = random_state, n_jobs=-1,
            verbosity=0, eval_metric="logloss" if is_clf else "rmse"
        )
        return XGBClassifier(**params) if is_clf else XGBRegressor(**params)

    raise ValueError(f"Algorithme inconnu : {algo}")
