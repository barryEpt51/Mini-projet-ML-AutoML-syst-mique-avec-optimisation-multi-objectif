"""
MODULE 2 : Espace de recherche des algorithmes classiques
==========================================================
Ce module définit :
  1. Les algorithmes disponibles (RF, SVM, XGBoost, KNN, Ridge...)
  2. Leurs hyperparamètres et plages de valeurs pour SMAC3
  3. Une fonction factory pour instancier n'importe quel modèle
"""

import numpy as np
from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical, Constant
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost non installé. Utilisation des autres algorithmes.")


# -------------------------------------------------------
# DÉFINITION DE L'ESPACE DE RECHERCHE GLOBAL
# -------------------------------------------------------
# SMAC3 utilise ConfigurationSpace pour définir les hyperparamètres
# On crée UN seul espace qui inclut :
#   - Le choix de l'algorithme
#   - Les hyperparamètres de chaque algorithme
# -------------------------------------------------------

def get_configspace() -> ConfigurationSpace:
    """
    Définit l'espace de configuration complet pour SMAC3.
    Inclut le choix de l'algorithme + ses hyperparamètres.

    Retourne
    --------
    ConfigurationSpace : l'espace de recherche complet
    """
    cs = ConfigurationSpace(seed=42)

    # --- Choix de l'algorithme ---
    algorithms = ["random_forest", "gradient_boosting", "extra_trees",
                  "svm", "knn", "logistic_regression", "decision_tree"]
    if XGBOOST_AVAILABLE:
        algorithms.append("xgboost")

    algorithm = Categorical("algorithm", algorithms, default="random_forest")
    cs.add(algorithm)

    # -------------------------------------------------------
    # RANDOM FOREST
    # -------------------------------------------------------
    rf_n_estimators     = Integer("rf_n_estimators",     (10, 500),   default=100, log=True)
    rf_max_depth        = Integer("rf_max_depth",        (1, 50),     default=10)
    rf_min_samples_split= Integer("rf_min_samples_split",(2, 20),     default=2)
    rf_min_samples_leaf = Integer("rf_min_samples_leaf", (1, 10),     default=1)
    rf_max_features     = Categorical("rf_max_features", ["sqrt", "log2", "None"], default="sqrt")
    cs.add([rf_n_estimators, rf_max_depth, rf_min_samples_split,
            rf_min_samples_leaf, rf_max_features])

    # -------------------------------------------------------
    # GRADIENT BOOSTING
    # -------------------------------------------------------
    gb_n_estimators  = Integer("gb_n_estimators",  (10, 300),    default=100, log=True)
    gb_learning_rate = Float("gb_learning_rate",   (0.01, 0.5),  default=0.1, log=True)
    gb_max_depth     = Integer("gb_max_depth",     (1, 10),      default=3)
    gb_subsample     = Float("gb_subsample",       (0.5, 1.0),   default=1.0)
    cs.add([gb_n_estimators, gb_learning_rate, gb_max_depth, gb_subsample])

    # -------------------------------------------------------
    # EXTRA TREES
    # -------------------------------------------------------
    et_n_estimators      = Integer("et_n_estimators",      (10, 500), default=100, log=True)
    et_max_depth         = Integer("et_max_depth",         (1, 50),   default=10)
    et_min_samples_split = Integer("et_min_samples_split", (2, 20),   default=2)
    cs.add([et_n_estimators, et_max_depth, et_min_samples_split])

    # -------------------------------------------------------
    # SVM
    # -------------------------------------------------------
    svm_C       = Float("svm_C",      (0.01, 100.0), default=1.0, log=True)
    svm_kernel  = Categorical("svm_kernel", ["rbf", "linear", "poly"], default="rbf")
    svm_gamma   = Categorical("svm_gamma",  ["scale", "auto"],         default="scale")
    cs.add([svm_C, svm_kernel, svm_gamma])

    # -------------------------------------------------------
    # KNN
    # -------------------------------------------------------
    knn_n_neighbors = Integer("knn_n_neighbors", (1, 50),              default=5)
    knn_weights     = Categorical("knn_weights", ["uniform", "distance"], default="uniform")
    knn_metric      = Categorical("knn_metric",  ["euclidean", "manhattan", "minkowski"], default="euclidean")
    cs.add([knn_n_neighbors, knn_weights, knn_metric])

    # -------------------------------------------------------
    # LOGISTIC REGRESSION
    # -------------------------------------------------------
    lr_C        = Float("lr_C",       (0.001, 100.0), default=1.0, log=True)
    lr_max_iter = Integer("lr_max_iter", (100, 1000), default=200)
    lr_solver   = Categorical("lr_solver", ["lbfgs", "liblinear", "saga"], default="lbfgs")
    cs.add([lr_C, lr_max_iter, lr_solver])

    # -------------------------------------------------------
    # DECISION TREE
    # -------------------------------------------------------
    dt_max_depth         = Integer("dt_max_depth",         (1, 50), default=5)
    dt_min_samples_split = Integer("dt_min_samples_split", (2, 20), default=2)
    dt_min_samples_leaf  = Integer("dt_min_samples_leaf",  (1, 10), default=1)
    dt_criterion         = Categorical("dt_criterion", ["gini", "entropy"], default="gini")
    cs.add([dt_max_depth, dt_min_samples_split, dt_min_samples_leaf, dt_criterion])

    # -------------------------------------------------------
    # XGBOOST (si disponible)
    # -------------------------------------------------------
    if XGBOOST_AVAILABLE:
        xgb_n_estimators  = Integer("xgb_n_estimators",  (10, 500),   default=100, log=True)
        xgb_learning_rate = Float("xgb_learning_rate",   (0.01, 0.5), default=0.1, log=True)
        xgb_max_depth     = Integer("xgb_max_depth",     (1, 10),     default=3)
        xgb_subsample     = Float("xgb_subsample",       (0.5, 1.0),  default=0.8)
        xgb_colsample     = Float("xgb_colsample_bytree",(0.5, 1.0),  default=0.8)
        cs.add([xgb_n_estimators, xgb_learning_rate, xgb_max_depth,
                xgb_subsample, xgb_colsample])

    return cs


# -------------------------------------------------------
# FACTORY : créer un modèle depuis une configuration SMAC
# -------------------------------------------------------

def build_model(config: dict, n_features: int = None, random_state: int = 42):
    """
    Instancie un modèle sklearn à partir d'une configuration SMAC.

    Paramètres
    ----------
    config       : dict-like, configuration SMAC (config[param] = valeur)
    n_features   : int, nombre de features (pour adapter max_features RF)
    random_state : int, seed pour reproductibilité

    Retourne
    --------
    Un estimateur sklearn non entraîné
    """
    algo = config["algorithm"]

    if algo == "random_forest":
        max_feat = config["rf_max_features"]
        if max_feat == "None":
            max_feat = None
        return RandomForestClassifier(
            n_estimators      = int(config["rf_n_estimators"]),
            max_depth         = int(config["rf_max_depth"]),
            min_samples_split = int(config["rf_min_samples_split"]),
            min_samples_leaf  = int(config["rf_min_samples_leaf"]),
            max_features      = max_feat,
            random_state      = random_state,
            n_jobs            = -1,
        )

    elif algo == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators  = int(config["gb_n_estimators"]),
            learning_rate = float(config["gb_learning_rate"]),
            max_depth     = int(config["gb_max_depth"]),
            subsample     = float(config["gb_subsample"]),
            random_state  = random_state,
        )

    elif algo == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators      = int(config["et_n_estimators"]),
            max_depth         = int(config["et_max_depth"]),
            min_samples_split = int(config["et_min_samples_split"]),
            random_state      = random_state,
            n_jobs            = -1,
        )

    elif algo == "svm":
        return SVC(
            C            = float(config["svm_C"]),
            kernel       = config["svm_kernel"],
            gamma        = config["svm_gamma"],
            random_state = random_state,
            probability  = True,  # nécessaire pour predict_proba
        )

    elif algo == "knn":
        return KNeighborsClassifier(
            n_neighbors = int(config["knn_n_neighbors"]),
            weights     = config["knn_weights"],
            metric      = config["knn_metric"],
            n_jobs      = -1,
        )

    elif algo == "logistic_regression":
        return LogisticRegression(
            C            = float(config["lr_C"]),
            max_iter     = int(config["lr_max_iter"]),
            solver       = config["lr_solver"],
            random_state = random_state,
            n_jobs       = -1,
        )

    elif algo == "decision_tree":
        return DecisionTreeClassifier(
            max_depth         = int(config["dt_max_depth"]),
            min_samples_split = int(config["dt_min_samples_split"]),
            min_samples_leaf  = int(config["dt_min_samples_leaf"]),
            criterion         = config["dt_criterion"],
            random_state      = random_state,
        )

    elif algo == "xgboost" and XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators        = int(config["xgb_n_estimators"]),
            learning_rate       = float(config["xgb_learning_rate"]),
            max_depth           = int(config["xgb_max_depth"]),
            subsample           = float(config["xgb_subsample"]),
            colsample_bytree    = float(config["xgb_colsample_bytree"]),
            random_state        = random_state,
            n_jobs              = -1,
            eval_metric         = "logloss",
            verbosity           = 0,
        )

    else:
        raise ValueError(f"Algorithme inconnu : {algo}")


# -------------------------------------------------------
# Test rapide
# -------------------------------------------------------
if __name__ == "__main__":
    cs = get_configspace()
    print("Espace de configuration créé !")
    print(f"Nombre de hyperparamètres : {len(cs.get_hyperparameters())}")
    print("\nHyperparamètres disponibles :")
    for hp in cs.get_hyperparameters():
        print(f"  - {hp.name}")

    # Tester une configuration par défaut
    default_config = cs.get_default_configuration()
    model = build_model(default_config)
    print(f"\nModèle par défaut : {model}")
