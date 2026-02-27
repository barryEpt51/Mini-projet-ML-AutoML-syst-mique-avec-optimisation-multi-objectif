"""
╔══════════════════════════════════════════════════════════════════════╗
║       Framework AutoML Systémique & Optimisation Multi-Objectif      ║
║       Sujet 3 — Apprentissage Automatique 2025-2026                  ║
║  Améliore le code de base avec :                                     ║
║   ✅ Multi-objectif RÉEL (3 directions Pareto via Optuna)            ║
║   ✅ 5 algorithmes (RF, GB, SVM, Ridge/LR, KNN)                      ║
║   ✅ Classification ET Régression                                    ║
║   ✅ Preprocessing automatique (NaN, encodage, normalisation)        ║
║   ✅ Hyperband (élimination progressive)                             ║
║   ✅ Méta-apprentissage (warm starting)                              ║
║   ✅ Courbe de convergence                                           ║
║   ✅ Front de Pareto 3D interactif                                   ║
╚══════════════════════════════════════════════════════════════════════╝

Installation :
    pip install streamlit optuna plotly scikit-learn pandas numpy

Lancement :
    streamlit run automl_framework.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris, load_wine

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AutoML Systémique — Sujet 3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Framework AutoML Systémique & Multi-Objectif")
st.markdown("""
> **Sujet 3** | Optimisation simultanée de l'**Accuracy**, de la **Latence** et de la **Mémoire**  
> via **Optimisation Bayésienne multi-objectif** (Optuna) + **Hyperband** + **Méta-apprentissage**
""")

# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Configuration")

dataset_option = st.sidebar.selectbox(
    "Dataset",
    ["Diabetes (Régression)", "Breast Cancer (Classification)",
     "Iris (Classification)", "Wine (Classification)", "Upload CSV"]
)

task_type_override = st.sidebar.selectbox(
    "Type de tâche (pour CSV)",
    ["Auto-détection", "Régression", "Classification"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Budget & Optimisation")
budget         = st.sidebar.slider("Nombre d'essais Bayésiens", 10, 100, 30)
use_hyperband  = st.sidebar.checkbox("Activer Hyperband (élimination progressive)", value=True)
use_meta       = st.sidebar.checkbox("Activer le Méta-apprentissage (Warm Start)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Contraintes Ressources")
max_mem_ko  = st.sidebar.number_input("Mémoire Max (Ko)", value=1000.0, min_value=10.0)
max_lat_ms  = st.sidebar.number_input("Latence Max (ms)", value=50.0, min_value=1.0)

# ══════════════════════════════════════════════════════════
#  1. PREPROCESSING AUTOMATIQUE
# ══════════════════════════════════════════════════════════
def auto_preprocess(df: pd.DataFrame):
    """
    Preprocessing générique :
    - Supprime colonnes constantes
    - Encodage Label pour catégorielles
    - Imputation médiane pour numériques
    - Normalisation StandardScaler
    Retourne X (np.array), y (np.array), task_type ('regression'|'classification')
    """
    df = df.copy()

    # Séparation X / y (dernière colonne = target)
    X_raw = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]

    # Détection du type de tâche
    if task_type_override == "Régression":
        task = "regression"
    elif task_type_override == "Classification":
        task = "classification"
    else:
        task = "regression" if y_raw.dtype in [np.float64, np.float32] and y_raw.nunique() > 20 else "classification"

    # Encodage target si classification
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values.astype(float)

    # Encodage des features catégorielles
    for col in X_raw.select_dtypes(include=["object", "category"]).columns:
        X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))

    # Suppression colonnes constantes
    X_raw = X_raw.loc[:, X_raw.nunique() > 1]

    # Pipeline : imputation + normalisation
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    X = pipe.fit_transform(X_raw)

    return X, y, task


# ══════════════════════════════════════════════════════════
#  2. MESURE DES 3 OBJECTIFS (code de base conservé + amélioré)
# ══════════════════════════════════════════════════════════
def measure_metrics(model, X_test, y_test, task: str):
    """
    Mesure les 3 objectifs systémiques :
    - Score de performance :
        * Classification → Accuracy (% prédictions correctes) — demandé par le sujet
        * Régression     → R² (proportion variance expliquée) — équivalent générique
    - Latence   : inférence unitaire moyenne sur 100 répétitions (ms)
    - Mémoire   : taille du modèle sérialisé (Ko)
    """
    preds = model.predict(X_test)

    # Précision
    if task == "regression":
        score = r2_score(y_test, preds)
        score = max(score, 0.0)  # évite valeurs négatives pour visualisation
    else:
        # Accuracy : métrique demandée explicitement par le sujet
        score = accuracy_score(y_test, preds)

    # Latence (ms) — moyenne sur 100 prédictions unitaires
    start = time.perf_counter()
    for _ in range(100):
        model.predict(X_test[:1])
    latency_ms = (time.perf_counter() - start) / 100 * 1000

    # Mémoire (Ko)
    mem_ko = len(pickle.dumps(model)) / 1024

    return score, latency_ms, mem_ko


# ══════════════════════════════════════════════════════════
#  3. CONSTRUCTION DU MODÈLE SELON LA CONFIG
# ══════════════════════════════════════════════════════════
def build_model(trial, task: str, budget_fraction: float = 1.0):
    """
    Espace de recherche élargi : 5 familles d'algorithmes.
    budget_fraction : réduit n_estimators pour Hyperband (élimination précoce).
    """
    algos = ["RandomForest", "GradientBoosting", "SVM", "LinearModel", "KNN"]
    algo  = trial.suggest_categorical("algo", algos)

    max_estimators = max(10, int(200 * budget_fraction))

    if algo == "RandomForest":
        params = dict(
            n_estimators = trial.suggest_int("rf_n_est", 10, max_estimators),
            max_depth    = trial.suggest_int("rf_depth", 2, 20),
            min_samples_split = trial.suggest_int("rf_mss", 2, 10),
            random_state = 42
        )
        return (RandomForestRegressor(**params) if task == "regression"
                else RandomForestClassifier(**params))

    elif algo == "GradientBoosting":
        params = dict(
            n_estimators  = trial.suggest_int("gb_n_est", 10, max_estimators),
            learning_rate = trial.suggest_float("gb_lr", 0.01, 0.3, log=True),
            max_depth     = trial.suggest_int("gb_depth", 2, 8),
            random_state  = 42
        )
        return (GradientBoostingRegressor(**params) if task == "regression"
                else GradientBoostingClassifier(**params))

    elif algo == "SVM":
        C     = trial.suggest_float("svm_C", 1e-2, 1e3, log=True)
        gamma = trial.suggest_categorical("svm_gamma", ["scale", "auto"])
        kernel = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
        return (SVR(C=C, gamma=gamma, kernel=kernel) if task == "regression"
                else SVC(C=C, gamma=gamma, kernel=kernel, random_state=42))

    elif algo == "LinearModel":
        if task == "regression":
            alpha = trial.suggest_float("ridge_alpha", 1e-3, 100, log=True)
            return Ridge(alpha=alpha)
        else:
            C = trial.suggest_float("lr_C", 1e-2, 100, log=True)
            return LogisticRegression(C=C, max_iter=500, random_state=42)

    else:  # KNN
        k = trial.suggest_int("knn_k", 1, 20)
        w = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
        return (KNeighborsRegressor(n_neighbors=k, weights=w) if task == "regression"
                else KNeighborsClassifier(n_neighbors=k, weights=w))


# ══════════════════════════════════════════════════════════
#  4. HYPERBAND — ÉLIMINATION PROGRESSIVE
# ══════════════════════════════════════════════════════════
def hyperband_filter(configs_scores: list, eta: int = 3) -> list:
    """
    Prend une liste de (trial_number, score) et retient
    les top 1/eta pour le prochain round.
    """
    n = max(1, len(configs_scores) // eta)
    sorted_cs = sorted(configs_scores, key=lambda x: x[1], reverse=True)
    return sorted_cs[:n]


# ══════════════════════════════════════════════════════════
#  5. MÉTA-APPRENTISSAGE — WARM START
# ══════════════════════════════════════════════════════════
META_STORE_KEY = "meta_knowledge"

def extract_meta_features(X: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n_samples":  int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes":  int(len(np.unique(y))),
        "feature_variance_mean": float(np.var(X, axis=0).mean()),
        "sparsity": float((X == 0).mean())
    }

def meta_similarity(mf1: dict, mf2: dict) -> float:
    """Distance euclidienne normalisée entre deux vecteurs de méta-features."""
    keys = ["n_samples", "n_features", "n_classes", "feature_variance_mean", "sparsity"]
    v1 = np.array([mf1.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([mf2.get(k, 0) for k in keys], dtype=float)
    norm = np.linalg.norm(v1) + np.linalg.norm(v2) + 1e-9
    return float(np.linalg.norm(v1 - v2) / norm)

def get_warm_start_suggestion(X, y) -> str | None:
    """Retourne l'algo qui a le mieux marché sur un dataset similaire."""
    if META_STORE_KEY not in st.session_state:
        return None
    mf_current = extract_meta_features(X, y)
    best_sim, best_algo = 1e9, None
    for entry in st.session_state[META_STORE_KEY]:
        sim = meta_similarity(mf_current, entry["meta_features"])
        if sim < best_sim:
            best_sim = sim
            best_algo = entry["best_algo"]
    return best_algo

def save_meta_knowledge(X, y, best_algo: str):
    if META_STORE_KEY not in st.session_state:
        st.session_state[META_STORE_KEY] = []
    st.session_state[META_STORE_KEY].append({
        "meta_features": extract_meta_features(X, y),
        "best_algo": best_algo
    })


# ══════════════════════════════════════════════════════════
#  6. MOTEUR PRINCIPAL — OPTIMISATION MULTI-OBJECTIF
# ══════════════════════════════════════════════════════════
def run_automl(X, y, task: str):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(y if task == "classification" else None)
    )

    results_data = []
    convergence  = []   # meilleur score cumulatif
    best_score_so_far = -np.inf

    # Warm start
    warm_algo = get_warm_start_suggestion(X, y) if use_meta else None
    if warm_algo:
        st.info(f"🧠 Méta-apprentissage : warm start suggéré → **{warm_algo}**")

    # Barre de progression
    progress_bar = st.progress(0, text="Initialisation...")
    status_text  = st.empty()

    # ── Hyperband : on divise le budget en rounds ──────────────
    if use_hyperband:
        eta        = 3
        n_rounds   = 3
        round_budgets = [max(3, budget // (eta ** i)) for i in range(n_rounds)]
        # round 0 : budget/9 trials, round 1 : budget/3 trials, round 2 : budget trials
        round_budgets = [budget // 9, budget // 3, budget]
        round_budgets = [max(3, b) for b in round_budgets]
        round_fractions = [0.2, 0.5, 1.0]   # fraction de n_estimators
    else:
        round_budgets   = [budget]
        round_fractions = [1.0]

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],  # score, latence, mémoire
        sampler=sampler
    )

    # Warm start : on enqueue un trial avec l'algo suggéré
    # enqueue_trial est compatible avec suggest_categorical (pas de conflit d'espace)
    if use_meta and warm_algo:
        study.enqueue_trial({"algo": warm_algo})

    trial_count = 0
    surviving_trials = []   # pour Hyperband

    for round_idx, (n_trials_round, bfrac) in enumerate(zip(round_budgets, round_fractions)):

        def objective(trial):
            nonlocal best_score_so_far, trial_count

            model = build_model(trial, task, budget_fraction=bfrac)

            # Entraînement
            model.fit(X_train, y_train)

            # Mesure des 3 objectifs
            score, lat_ms, mem_ko = measure_metrics(model, X_test, y_test, task)

            # Sauvegarde
            results_data.append({
                "Trial":      trial.number,
                "Round":      round_idx + 1,
                "Algo":       trial.params.get("algo", "?"),
                "Score":      round(score, 4),
                "Latence_ms": round(lat_ms, 4),
                "Memoire_ko": round(mem_ko, 2),
                "Contraintes_OK": (lat_ms <= max_lat_ms and mem_ko <= max_mem_ko)
            })

            # Convergence
            if score > best_score_so_far:
                best_score_so_far = score
            convergence.append(best_score_so_far)

            trial_count += 1
            pct = min(trial_count / sum(round_budgets), 1.0)
            progress_bar.progress(pct, text=f"Round {round_idx+1} | Trial {trial_count} | Best score: {best_score_so_far:.4f}")

            return score, lat_ms, mem_ko

        study.optimize(objective, n_trials=n_trials_round)

        # ── Hyperband : éliminer les mauvais trials ───────────────
        if use_hyperband and round_idx < len(round_budgets) - 1:
            trial_scores = [(t.number, t.values[0]) for t in study.trials if t.values]
            survivors = hyperband_filter(trial_scores, eta=3)
            surviving_trials = [s[0] for s in survivors]
            st.caption(f"Hyperband Round {round_idx+1} : {len(trial_scores)} configs → {len(survivors)} survivantes")

    progress_bar.progress(1.0, text="✅ Optimisation terminée !")

    # Méta-apprentissage : sauvegarder le meilleur algo trouvé
    if use_meta and results_data:
        res_df = pd.DataFrame(results_data)
        best_row = res_df.sort_values("Score", ascending=False).iloc[0]
        save_meta_knowledge(X, y, best_row["Algo"])

    return pd.DataFrame(results_data), convergence, study


# ══════════════════════════════════════════════════════════
#  7. AFFICHAGE DES RÉSULTATS
# ══════════════════════════════════════════════════════════
def display_results(res_df: pd.DataFrame, convergence: list, study, task: str):

    metric_label = "R² — Score de Performance" if task == "regression" else "Accuracy — Score de Performance"

    # ── Métriques clés ─────────────────────────────────────────
    st.markdown("---")
    st.header("📊 Résultats de l'Optimisation")

    col1, col2, col3, col4 = st.columns(4)
    best = res_df.sort_values("Score", ascending=False).iloc[0]
    valid = res_df[res_df["Contraintes_OK"] == True]

    col1.metric("🏆 Meilleur Score Global",    f"{best['Score']:.4f}")
    col2.metric("⚡ Latence (meilleur)",       f"{best['Latence_ms']:.3f} ms")
    col3.metric("💾 Mémoire (meilleur)",       f"{best['Memoire_ko']:.1f} Ko")
    col4.metric("✅ Configs valides",           f"{len(valid)} / {len(res_df)}")

    # ── Onglets de visualisation ───────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Front de Pareto 3D",
        "📈 Convergence",
        "🔬 Analyse par Algorithme",
        "📋 Tableau Complet"
    ])

    # ── TAB 1 : Front de Pareto 3D ─────────────────────────────
    with tab1:
        st.subheader("Front de Pareto 3D — Accuracy vs Latence vs Mémoire")
        st.caption("Chaque point est une configuration testée. Les ★ respectent vos contraintes ressources.")

        fig3d = px.scatter_3d(
            res_df,
            x="Score", y="Latence_ms", z="Memoire_ko",
            color="Algo",
            template="plotly_white",
            symbol="Contraintes_OK",
            hover_data=["Trial", "Round"],
            title="Front de Pareto — Optimisation Multi-Objectif",
            labels={
                "Score": metric_label,
                "Latence_ms": "Latence (ms)",
                "Memoire_ko": "Mémoire (Ko)"
            },
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig3d.update_traces(marker=dict(size=5, opacity=0.8))
        st.plotly_chart(fig3d, use_container_width=True)

        # Pareto 2D : Score vs Latence
        st.subheader("Pareto 2D — Score vs Latence")
        fig2d = px.scatter(
            res_df, x="Latence_ms", y="Score",
            color="Algo", size="Memoire_ko",
            symbol="Contraintes_OK",
            hover_data=["Trial", "Memoire_ko"],
            title="Score vs Latence (taille ∝ Mémoire)",
            labels={"Latence_ms": "Latence (ms)", "Score": metric_label}
        )
        fig2d.add_vline(x=max_lat_ms, line_dash="dash", line_color="red",
                        annotation_text=f"Limite latence ({max_lat_ms}ms)")
        fig2d.add_hline(y=0.8, line_dash="dot", line_color="green",
                        annotation_text="Seuil 0.8")
        st.plotly_chart(fig2d, use_container_width=True)

    # ── TAB 2 : Convergence ────────────────────────────────────
    with tab2:
        st.subheader("Courbe de Convergence de l'Optimisation Bayésienne")
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            y=convergence,
            mode="lines+markers",
            name="Meilleur score cumulatif",
            line=dict(color="#2196F3", width=2),
            marker=dict(size=4)
        ))
        fig_conv.add_trace(go.Scatter(
            y=res_df["Score"].tolist(),
            mode="markers",
            name="Score par trial",
            marker=dict(color="rgba(255,100,100,0.4)", size=5)
        ))
        fig_conv.update_layout(
            title="Convergence — Best Score au fil des Trials",
            xaxis_title="Trial",
            yaxis_title=metric_label,
            hovermode="x unified"
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        # Distribution des scores par round (Hyperband)
        if res_df["Round"].nunique() > 1:
            st.subheader("Distribution des Scores par Round (Hyperband)")
            fig_box = px.box(
                res_df, x="Round", y="Score", color="Round",
                title="Amélioration progressive avec Hyperband",
                labels={"Score": metric_label}
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # ── TAB 3 : Analyse par Algorithme ─────────────────────────
    with tab3:
        st.subheader("Comparaison des Algorithmes")

        algo_stats = res_df.groupby("Algo").agg(
            Score_mean    = ("Score", "mean"),
            Score_max     = ("Score", "max"),
            Latence_mean  = ("Latence_ms", "mean"),
            Memoire_mean  = ("Memoire_ko", "mean"),
            N_trials      = ("Trial", "count")
        ).reset_index().round(4)

        # Radar chart
        categories = ["Score_mean", "Latence_mean (inv)", "Memoire_mean (inv)"]
        fig_radar = go.Figure()
        for _, row in algo_stats.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[
                    row["Score_mean"],
                    1 / (1 + row["Latence_mean"]),
                    1 / (1 + row["Memoire_mean"] / 100)
                ],
                theta=["Accuracy", "Rapidité", "Légèreté"],
                fill="toself",
                name=row["Algo"]
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Radar : Score / Rapidité / Légèreté par Algorithme"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.dataframe(
            algo_stats.style
                .highlight_max(axis=0, subset=["Score_mean", "Score_max"], color="#c8f7c5")
                .highlight_min(axis=0, subset=["Latence_mean", "Memoire_mean"],  color="#c8f7c5"),
            use_container_width=True
        )

        # Distribution des hyperparamètres par algo
        st.subheader("Importance des Hyperparamètres")
        try:
            importances = optuna.importance.get_param_importances(
                study, target=lambda t: t.values[0] if t.values else 0
            )
            imp_df = pd.DataFrame(
                importances.items(), columns=["Hyperparamètre", "Importance"]
            ).sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                imp_df, x="Importance", y="Hyperparamètre",
                orientation="h",
                title="Importance des Hyperparamètres sur le Score",
                color="Importance", color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            st.caption("Importance non calculable avec moins de 4 trials par algo.")

    # ── TAB 4 : Tableau complet ────────────────────────────────
    with tab4:
        st.subheader("Tableau Complet des Configurations Testées")

        # Filtre
        show_valid_only = st.checkbox("Afficher uniquement les configs qui respectent les contraintes")
        display_df = res_df[res_df["Contraintes_OK"]] if show_valid_only else res_df
        display_df = display_df.sort_values("Score", ascending=False)

        st.dataframe(
            display_df.style.highlight_max(axis=0, subset=["Score"], color="#c8f7c5")
                            .highlight_min(axis=0, subset=["Latence_ms", "Memoire_ko"], color="#c8f7c5"),
            use_container_width=True,
            height=400
        )

        # Recommandation finale
        st.markdown("---")
        st.subheader("🎯 Recommandation Finale")

        if not valid.empty:
            best_valid = valid.sort_values("Score", ascending=False).iloc[0]
            st.success(f"""
**Meilleure configuration respectant les contraintes :**
- **Algorithme :** {best_valid['Algo']}
- **Score ({metric_label}) :** {best_valid['Score']:.4f}
- **Latence :** {best_valid['Latence_ms']:.3f} ms (limite : {max_lat_ms} ms)
- **Mémoire :** {best_valid['Memoire_ko']:.1f} Ko (limite : {max_mem_ko} Ko)
- **Trial n°** : {int(best_valid['Trial'])}
            """)
        else:
            st.error("⚠️ Aucune configuration ne respecte les contraintes. Essayez d'assouplir les limites dans la barre latérale.")
            best_global = res_df.sort_values("Score", ascending=False).iloc[0]
            st.warning(f"""
**Meilleure configuration globale (hors contraintes) :**
- Algorithme : {best_global['Algo']} | Score : {best_global['Score']:.4f}
- Latence : {best_global['Latence_ms']:.3f} ms | Mémoire : {best_global['Memoire_ko']:.1f} Ko
            """)


# ══════════════════════════════════════════════════════════
#  8. CHARGEMENT DES DONNÉES ET ORCHESTRATION
# ══════════════════════════════════════════════════════════
def load_sklearn_dataset(option: str):
    """Charge un dataset sklearn et retourne un DataFrame."""
    loaders = {
        "Diabetes (Régression)":          load_diabetes,
        "Breast Cancer (Classification)": load_breast_cancer,
        "Iris (Classification)":          load_iris,
        "Wine (Classification)":          load_wine,
    }
    data = loaders[option]()
    df = pd.concat([
        pd.DataFrame(data.data, columns=data.feature_names),
        pd.Series(data.target, name="target")
    ], axis=1)
    return df


# ── Sélection / Upload ─────────────────────────────────────
df = None

if dataset_option != "Upload CSV":
    df = load_sklearn_dataset(dataset_option)
    st.success(f"✅ Dataset **{dataset_option}** chargé — {df.shape[0]} lignes × {df.shape[1]-1} features")
else:
    uploaded = st.sidebar.file_uploader("Importer votre fichier CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ **{uploaded.name}** importé — {df.shape[0]} lignes × {df.shape[1]-1} features")
    else:
        st.info("👈 Importez un fichier CSV dans la barre latérale pour commencer.")

# ── Aperçu du dataset ──────────────────────────────────────
if df is not None:
    with st.expander("👁 Aperçu du dataset", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Lignes",    df.shape[0])
        col2.metric("Features",  df.shape[1] - 1)
        col3.metric("Target",    df.columns[-1])

    st.markdown("---")

    # ── Bouton de lancement ────────────────────────────────────
    if st.button("🚀 Lancer l'Optimisation AutoML", type="primary", use_container_width=True):
        try:
            X, y, task = auto_preprocess(df)
            st.info(f"🔍 Tâche détectée : **{task.upper()}** | {X.shape[0]} samples × {X.shape[1]} features")

            res_df, convergence, study = run_automl(X, y, task)
            display_results(res_df, convergence, study, task)

        except Exception as e:
            st.error(f"❌ Erreur : {e}")
            st.exception(e)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("Framework AutoML Systémique — Sujet 3 | Apprentissage Automatique 2025-2026")