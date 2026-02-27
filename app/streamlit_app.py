"""
APPLICATION STREAMLIT : Interface AutoML Systémique
=====================================================
Interface complète pour :
  1. Configurer et lancer l'optimisation
  2. Visualiser les résultats en temps réel
  3. Explorer le front de Pareto
  4. Comparer les algorithmes
  5. Analyser le méta-apprentissage
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# Ajouter le répertoire du projet au path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# -------------------------------------------------------
# CONFIGURATION DE LA PAGE
# -------------------------------------------------------
st.set_page_config(
    page_title = "AutoML Systémique",
    page_icon  = "🤖",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# -------------------------------------------------------
# STYLE CSS PERSONNALISÉ
# -------------------------------------------------------
st.markdown("""
<style>
    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: white; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #B3C7E6; font-size: 1rem; margin: 0.5rem 0 0 0; }

    /* Cartes de métriques */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid;
    }
    .metric-card.accuracy  { border-color: #4CAF50; }
    .metric-card.latency   { border-color: #2196F3; }
    .metric-card.memory    { border-color: #FF9800; }
    .metric-card .value    { font-size: 2rem; font-weight: 700; }
    .metric-card .label    { font-size: 0.85rem; color: #666; }

    /* Badges algorithme */
    .algo-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
    }

    /* Sections */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e3c72;
        border-bottom: 2px solid #E0E8F0;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Tableau résultats */
    .results-table { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# ÉTAT DE LA SESSION (persistance entre interactions)
# -------------------------------------------------------
if "all_results"     not in st.session_state: st.session_state.all_results     = {}
if "pipeline_summary"not in st.session_state: st.session_state.pipeline_summary= None
if "is_running"      not in st.session_state: st.session_state.is_running       = False
if "meta_base_loaded"not in st.session_state: st.session_state.meta_base_loaded = False


# -------------------------------------------------------
# EN-TÊTE
# -------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🤖 AutoML Systémique</h1>
    <p>Optimisation multi-objectif : Accuracy · Latence · Mémoire | SMAC3 + Hyperband + Méta-apprentissage</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# SIDEBAR : CONFIGURATION
# -------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/robot.png", width=60)
    st.title("⚙️ Configuration")

    st.markdown("### 📊 Datasets")
    try:
        from data.datasets import DATASETS
        all_dataset_names = list(DATASETS.keys())
    except:
        all_dataset_names = ["diabetes", "breast_cancer", "iris", "wine",
                              "credit", "heart", "titanic", "vehicle", "bank", "phoneme"]

    selected_datasets = st.multiselect(
        "Datasets à évaluer",
        options  = all_dataset_names,
        default  = ["diabetes", "breast_cancer", "iris"],
        help     = "Sélectionnez les datasets à inclure dans l'évaluation"
    )

    st.markdown("### 🔧 SMAC + Hyperband")
    n_trials = st.slider(
        "Nombre d'évaluations par dataset",
        min_value = 5,
        max_value = 100,
        value     = 20,
        step      = 5,
        help      = "Plus = meilleure qualité, mais plus lent"
    )

    min_budget = st.slider("Budget minimum Hyperband", 0.05, 0.5, 0.1, 0.05,
                            help="Fraction des données au premier niveau")
    max_budget = 1.0

    st.markdown("### ⚖️ Poids des objectifs")
    w_accuracy = st.slider("Poids Accuracy",  0.0, 1.0, 0.6, 0.1)
    w_latency  = st.slider("Poids Latence",   0.0, 1.0, 0.2, 0.1)
    w_memory   = st.slider("Poids Mémoire",   0.0, 1.0, 0.2, 0.1)

    # Normalisation automatique
    total_w = w_accuracy + w_latency + w_memory
    if total_w > 0:
        w_accuracy /= total_w
        w_latency  /= total_w
        w_memory   /= total_w
        st.caption(f"Normalisés → A:{w_accuracy:.2f}, L:{w_latency:.2f}, M:{w_memory:.2f}")

    st.markdown("### 🔒 Contraintes ressources")
    max_latency = st.number_input("Latence max (ms)", 10, 5000, 500, 50)
    max_memory  = st.number_input("Mémoire max (MB)", 10, 2048, 512, 64)

    st.divider()

    # Bouton de lancement
    run_button = st.button(
        "🚀 Lancer l'optimisation",
        type      = "primary",
        use_container_width = True,
        disabled  = len(selected_datasets) == 0,
    )

    # Charger résultats existants
    st.divider()
    load_button = st.button("📂 Charger résultats existants", use_container_width=True)


# -------------------------------------------------------
# ONGLETS PRINCIPAUX
# -------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Accueil",
    "📊 Résultats",
    "🎯 Front de Pareto",
    "📈 Convergence",
    "🧠 Méta-apprentissage"
])


# ========================================================
# ONGLET 1 : ACCUEIL + LANCEMENT
# ========================================================
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">🎯 Objectifs du projet</div>', unsafe_allow_html=True)
        st.markdown("""
        Ce framework AutoML optimise **simultanément 3 objectifs** :

        | Objectif | Description | Direction |
        |----------|-------------|-----------|
        | 🎯 **Accuracy** | Performance prédictive | Maximiser ↑ |
        | ⚡ **Latence** | Temps d'inférence (ms) | Minimiser ↓ |
        | 💾 **Mémoire** | RAM utilisée (MB) | Minimiser ↓ |

        **Algorithmes explorés :** Random Forest, Gradient Boosting, Extra Trees, SVM, KNN, Logistic Regression, Decision Tree, XGBoost
        """)

    with col2:
        st.markdown('<div class="section-header">🏗️ Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        ```
        📥 10 Datasets OpenML
               ↓
        🔍 Méta-features extraction
               ↓
        🧠 Warm-start méta-apprentissage
               ↓
        🤖 SMAC3 (optimisation bayésienne)
          + Hyperband (budget adaptatif)
               ↓
        📊 Évaluation multi-objectif
               ↓
        🎯 Front de Pareto 3D
        ```
        """)

    # Lancement du pipeline
    if run_button and selected_datasets:
        st.markdown('<div class="section-header">⏳ Optimisation en cours...</div>', unsafe_allow_html=True)

        progress_bar    = st.progress(0)
        status_text     = st.empty()
        results_preview = st.empty()

        # Importer les modules nécessaires
        try:
            from optimization.smac_optimizer import AutoMLOptimizer, OBJECTIVE_WEIGHTS
            from optimization.meta_learning  import MetaLearningBase, extract_meta_features
            from data.datasets               import load_dataset

            # Mise à jour des poids
            OBJECTIVE_WEIGHTS["error"]   = w_accuracy
            OBJECTIVE_WEIGHTS["latency"] = w_latency
            OBJECTIVE_WEIGHTS["memory"]  = w_memory

            meta_base = MetaLearningBase(
                storage_path=str(PROJECT_DIR / "meta_learning_base.pkl")
            )

            summaries = []

            for i, dataset_name in enumerate(selected_datasets):
                progress = i / len(selected_datasets)
                progress_bar.progress(progress)
                status_text.info(f"🔄 Traitement de **{dataset_name}** ({i+1}/{len(selected_datasets)})...")

                try:
                    data = load_dataset(dataset_name)
                    meta_feats = extract_meta_features(data["X_train"], data["y_train"])

                    optimizer = AutoMLOptimizer(
                        dataset_name = dataset_name,
                        n_trials     = n_trials,
                        min_budget   = min_budget,
                        max_budget   = max_budget,
                        output_dir   = str(PROJECT_DIR / "smac_output"),
                        seed         = 42,
                    )

                    best = optimizer.run(
                        data["X_train"], data["y_train"],
                        data["X_test"],  data["y_test"],
                    )

                    if best:
                        meta_base.add_entry(
                            dataset_name  = dataset_name,
                            meta_features = meta_feats,
                            best_config   = optimizer.get_best_config() or {},
                            best_score    = optimizer.best_score,
                            best_accuracy = best.get("accuracy", 0),
                        )
                        st.session_state.all_results[dataset_name] = optimizer.get_all_results()

                        summaries.append({
                            "dataset":    dataset_name,
                            "best_algo":  optimizer.get_best_config().get("algorithm", "N/A") if optimizer.get_best_config() else "N/A",
                            "accuracy":   round(best.get("accuracy", 0), 4),
                            "latency_ms": round(best.get("latency_ms", 0), 2),
                            "memory_mb":  round(best.get("memory_mb", 0), 2),
                        })

                        # Aperçu en temps réel
                        results_preview.dataframe(
                            pd.DataFrame(summaries),
                            use_container_width=True,
                        )

                except Exception as e:
                    st.warning(f"⚠️ Erreur sur '{dataset_name}': {e}")

            progress_bar.progress(1.0)

            if summaries:
                st.session_state.pipeline_summary = pd.DataFrame(summaries)
                status_text.success(f"✅ Optimisation terminée ! {len(summaries)} datasets traités.")
            else:
                status_text.error("❌ Aucun résultat obtenu.")

        except ImportError as e:
            st.error(f"❌ Module manquant: {e}")
            st.info("Installez les dépendances : `pip install smac openml xgboost`")

    # Afficher le résumé si disponible
    if st.session_state.pipeline_summary is not None:
        st.markdown('<div class="section-header">🏆 Meilleurs résultats</div>', unsafe_allow_html=True)
        df_summary = st.session_state.pipeline_summary

        # Métriques globales
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("🎯 Accuracy moyenne",  f"{df_summary['accuracy'].mean():.4f}")
        with c2:
            st.metric("⚡ Latence moyenne",   f"{df_summary['latency_ms'].mean():.1f} ms")
        with c3:
            st.metric("💾 Mémoire moyenne",   f"{df_summary['memory_mb'].mean():.1f} MB")
        with c4:
            st.metric("📊 Datasets évalués",  len(df_summary))

        st.dataframe(df_summary, use_container_width=True, hide_index=True)


# ========================================================
# ONGLET 2 : RÉSULTATS DÉTAILLÉS
# ========================================================
with tab2:
    st.markdown('<div class="section-header">📊 Résultats détaillés par dataset</div>',
                unsafe_allow_html=True)

    if not st.session_state.all_results:
        st.info("👆 Lancez d'abord l'optimisation dans l'onglet **Accueil**.")
    else:
        try:
            from analysis.visualizations import plot_algorithm_comparison, plot_datasets_heatmap

            # Sélecteur de dataset
            available = list(st.session_state.all_results.keys())
            selected  = st.selectbox("Sélectionner un dataset", available)

            df_sel = st.session_state.all_results[selected]

            if not df_sel.empty:
                # KPIs du dataset sélectionné
                best_row = df_sel.loc[df_sel["accuracy"].idxmax()]
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("🎯 Meilleure accuracy",   f"{best_row['accuracy']:.4f}")
                with c2: st.metric("⚡ Latence",              f"{best_row['latency_ms']:.1f} ms")
                with c3: st.metric("💾 Mémoire",              f"{best_row['memory_mb']:.1f} MB")
                with c4: st.metric("🤖 Meilleur algorithme",  best_row.get('algorithm', 'N/A'))

                # Graphique comparaison algorithmes
                fig = plot_algorithm_comparison(df_sel, f"Comparaison algorithmes - {selected}")
                st.plotly_chart(fig, use_container_width=True)

                # Tableau complet
                with st.expander("📋 Voir toutes les évaluations"):
                    cols_to_show = ["algorithm", "accuracy", "latency_ms", "memory_mb",
                                    "score", "budget", "status", "train_time_s"]
                    cols_exist   = [c for c in cols_to_show if c in df_sel.columns]
                    st.dataframe(df_sel[cols_exist].round(4), use_container_width=True)

            # Heatmap tous datasets
            st.markdown("### 🗺️ Vue globale : performance par dataset")
            fig_heatmap = plot_datasets_heatmap(st.session_state.all_results)
            st.plotly_chart(fig_heatmap, use_container_width=True)

        except ImportError as e:
            st.error(f"Module manquant: {e}")


# ========================================================
# ONGLET 3 : FRONT DE PARETO
# ========================================================
with tab3:
    st.markdown('<div class="section-header">🎯 Analyse du Front de Pareto</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Le **Front de Pareto** contient les solutions pour lesquelles il est impossible
    d'améliorer un objectif sans en dégrader un autre.
    Ces solutions représentent les **meilleurs compromis** possibles.
    """)

    if not st.session_state.all_results:
        st.info("👆 Lancez d'abord l'optimisation dans l'onglet **Accueil**.")
    else:
        try:
            from analysis.visualizations import plot_pareto_front_3d, plot_tradeoff_scatter

            # Combiner tous les résultats
            all_dfs   = list(st.session_state.all_results.values())
            df_all    = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

            dataset_filter = st.selectbox(
                "Filtrer par dataset",
                ["Tous"] + list(st.session_state.all_results.keys())
            )

            if dataset_filter != "Tous" and "dataset" in df_all.columns:
                df_filtered = df_all[df_all["dataset"] == dataset_filter]
            else:
                df_filtered = df_all

            col1, col2 = st.columns([3, 2])

            with col1:
                fig_3d = plot_pareto_front_3d(
                    df_filtered,
                    title=f"Front de Pareto 3D – {dataset_filter}"
                )
                st.plotly_chart(fig_3d, use_container_width=True)

            with col2:
                fig_2d = plot_tradeoff_scatter(
                    df_filtered,
                    title="Compromis Accuracy vs Latence"
                )
                st.plotly_chart(fig_2d, use_container_width=True)

        except ImportError as e:
            st.error(f"Module manquant: {e}")


# ========================================================
# ONGLET 4 : CONVERGENCE SMAC
# ========================================================
with tab4:
    st.markdown('<div class="section-header">📈 Convergence de l\'optimisation SMAC</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Ces graphiques montrent comment **SMAC apprend** au fil des évaluations.
    Une bonne convergence = le score s'améliore rapidement puis se stabilise.
    """)

    if not st.session_state.all_results:
        st.info("👆 Lancez d'abord l'optimisation dans l'onglet **Accueil**.")
    else:
        try:
            from analysis.visualizations import plot_convergence

            dataset_sel = st.selectbox(
                "Dataset à analyser",
                list(st.session_state.all_results.keys()),
                key="conv_dataset"
            )

            df_conv = st.session_state.all_results[dataset_sel]
            if not df_conv.empty:
                fig = plot_convergence(df_conv, f"Convergence SMAC – {dataset_sel}")
                st.plotly_chart(fig, use_container_width=True)

                # Stats convergence
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Évaluations totales", len(df_conv))
                with col2:
                    df_ok = df_conv[df_conv["status"] == "SUCCESS"] if "status" in df_conv.columns else df_conv
                    st.metric("Évaluations réussies", len(df_ok))
                with col3:
                    if len(df_ok) > 1:
                        first_half = df_ok.head(len(df_ok)//2)["accuracy"].max()
                        second_half= df_ok.tail(len(df_ok)//2)["accuracy"].max()
                        improvement = second_half - first_half
                        st.metric("Amélioration (2ème moitié)", f"+{improvement:.4f}")

        except ImportError as e:
            st.error(f"Module manquant: {e}")


# ========================================================
# ONGLET 5 : MÉTA-APPRENTISSAGE
# ========================================================
with tab5:
    st.markdown('<div class="section-header">🧠 Base de Méta-apprentissage</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Le **méta-apprentissage** mémorise les meilleures configurations de chaque dataset.
    Quand on arrive sur un nouveau dataset, on cherche les datasets similaires
    et on utilise leurs configurations comme **point de départ** pour SMAC.
    """)

    try:
        from optimization.meta_learning import MetaLearningBase
        meta_base = MetaLearningBase(
            storage_path=str(PROJECT_DIR / "meta_learning_base.pkl")
        )

        df_meta = meta_base.get_summary()

        if df_meta.empty:
            st.info("🆕 La base méta est vide. Lancez le pipeline pour la remplir.")
        else:
            st.success(f"✅ {len(df_meta)} dataset(s) en mémoire")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(df_meta, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("**Distribution des meilleurs algorithmes**")
                algo_counts = df_meta["Best algo"].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels = algo_counts.index,
                    values = algo_counts.values,
                    hole   = 0.4,
                )])
                fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True)

    except Exception as e:
        st.warning(f"Impossible de charger la base méta: {e}")

    # Tester la similarité
    st.markdown("### 🔍 Tester la similarité de dataset")
    st.markdown("Entrez les caractéristiques d'un nouveau dataset pour voir quels datasets sont similaires :")

    col1, col2, col3 = st.columns(3)
    with col1: n_samples_test  = st.number_input("n_samples",  10, 100000, 500)
    with col2: n_features_test = st.number_input("n_features", 1,  1000,   20)
    with col3: n_classes_test  = st.number_input("n_classes",  2,  100,    2)

    if st.button("🔍 Trouver datasets similaires"):
        try:
            from optimization.meta_learning import MetaLearningBase, extract_meta_features
            import numpy as np

            meta_base = MetaLearningBase(
                storage_path=str(PROJECT_DIR / "meta_learning_base.pkl")
            )

            # Simuler des méta-features simples
            fake_meta = {
                "n_samples":    float(n_samples_test),
                "n_features":   float(n_features_test),
                "n_classes":    float(n_classes_test),
                "log_samples":  float(np.log1p(n_samples_test)),
                "log_features": float(np.log1p(n_features_test)),
                "samples_per_feature": float(n_samples_test / n_features_test),
                "features_per_sample": float(n_features_test / n_samples_test),
                "class_imbalance": 1.0,
                "class_entropy": float(np.log2(n_classes_test)),
                "mean_feature_mean": 0.0,
                "mean_feature_std":  1.0,
                "mean_feature_min":  -1.0,
                "mean_feature_max":  1.0,
                "mean_skewness":     0.0,
                "missing_ratio":     0.0,
                "relative_n_features": float(n_features_test / n_samples_test),
            }

            similar = meta_base.find_similar_datasets(fake_meta, top_k=3)

            if similar:
                for s in similar:
                    st.info(
                        f"📌 **{s['dataset_name']}** "
                        f"(distance={s['similarity_distance']:.3f}) → "
                        f"Meilleur algo: **{s['best_config'].get('algorithm', '?')}**, "
                        f"Accuracy: {s['best_accuracy']:.4f}"
                    )
            else:
                st.warning("Base méta vide. Lancez d'abord le pipeline !")
        except Exception as e:
            st.error(f"Erreur: {e}")
