"""
app/streamlit_app.py
====================
Interface Streamlit fusionnée :
  ✅ Architecture modulaire (binôme)
  ✅ SMAC + MultiFidelityFacade + Hyperband natif (binôme)
  ✅ 8 algorithmes + 10 datasets (binôme)
  ✅ tracemalloc pour la mémoire (binôme)
  ✅ Upload CSV + Excel avec choix de la colonne target (nous)
  ✅ Aperçu + statistiques descriptives avant lancement (nous)
  ✅ Poids de scalarisation réglables (nous)
  ✅ Fond blanc sur tous les graphiques (nous)

Lancement :
    streamlit run app/streamlit_app.py
"""

import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# ── Path ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="AutoML Systémique — Sujet 3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1A3C6E 0%, #2E6DB4 100%);
    padding: 1.8rem 2rem; border-radius: 12px; color: white;
    text-align: center; margin-bottom: 1.5rem;
}
.main-header h1 { color: white; font-size: 2rem; margin: 0; }
.main-header p  { color: #B3C7E6; font-size: 0.95rem; margin: 0.4rem 0 0 0; }
.section-hdr {
    font-size: 1.2rem; font-weight: 700; color: #1A3C6E;
    border-bottom: 2px solid #D6E4F7;
    padding-bottom: 0.4rem; margin: 1.2rem 0 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────
for key, val in [
    ("all_results", {}),
    ("pipeline_summary", None),
    ("meta_entries", []),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Header ────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 AutoML Systémique &amp; Multi-Objectif</h1>
    <p>Sujet 3 | SMAC3 + Hyperband natif + Méta-apprentissage
       | Accuracy · Latence · Mémoire</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Configuration")

    # ── Dataset ───────────────────────────────────────
    st.markdown("### 📊 Source de données")
    SKLEARN_DATASETS = [
        "diabetes", "breast_cancer", "iris", "wine", "digits"
    ]
    OPENML_DATASETS  = ["credit", "heart", "vehicle", "bank", "phoneme"]

    dataset_source = st.selectbox(
        "Source", ["sklearn (intégré)", "OpenML", "Upload CSV", "Upload Excel (.xlsx)"]
    )

    df_loaded   = None
    target_col  = None
    dataset_name = "custom"

    if dataset_source == "sklearn (intégré)":
        dataset_name = st.selectbox("Dataset sklearn", SKLEARN_DATASETS)

    elif dataset_source == "OpenML":
        dataset_name = st.selectbox("Dataset OpenML", OPENML_DATASETS)

    elif dataset_source == "Upload CSV":
        upl = st.file_uploader("Fichier CSV", type=["csv"])
        if upl:
            try:
                df_loaded = pd.read_csv(upl)
                st.success(f"✅ {upl.name} — {df_loaded.shape[0]} × {df_loaded.shape[1]}")
            except Exception as e:
                st.error(f"Erreur CSV : {e}")

    elif dataset_source == "Upload Excel (.xlsx)":
        upl = st.file_uploader("Fichier Excel", type=["xlsx", "xls"])
        if upl:
            try:
                xls    = pd.ExcelFile(upl)
                sheets = xls.sheet_names
                sheet  = st.selectbox("Feuille", sheets) if len(sheets) > 1 else sheets[0]
                df_loaded = pd.read_excel(upl, sheet_name=sheet)
                st.success(f"✅ {upl.name} ({sheet}) — {df_loaded.shape[0]} × {df_loaded.shape[1]}")
            except Exception as e:
                st.error(f"Erreur Excel : {e}")

    # ── Choix colonne target (upload uniquement) ──────
    if df_loaded is not None:
        st.markdown("---")
        st.markdown("### 🎯 Colonne target")
        target_col = st.selectbox(
            "Colonne cible",
            options=df_loaded.columns.tolist(),
            index=len(df_loaded.columns) - 1,
            help="Colonne que le modèle doit apprendre à prédire"
        )
        task_override = st.selectbox(
            "Type de tâche", ["Auto-détection", "Classification", "Régression"]
        )
    else:
        task_override = "Auto-détection"

    # ── SMAC + Hyperband ──────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 SMAC + Hyperband")
    n_trials   = st.slider("Évaluations par dataset", 5, 100, 20, 5)
    min_budget = st.slider("Budget min Hyperband", 0.05, 0.5, 0.1, 0.05,
                            help="Fraction des données au 1er niveau Hyperband")

    # Datasets multiples (seulement pour sklearn/OpenML)
    if dataset_source in ["sklearn (intégré)", "OpenML"]:
        all_ds = SKLEARN_DATASETS if dataset_source == "sklearn (intégré)" else OPENML_DATASETS
        selected_datasets = st.multiselect(
            "Datasets à évaluer",
            options=all_ds,
            default=[dataset_name],
        )
    else:
        selected_datasets = ["custom"]

    use_meta = st.checkbox("Méta-apprentissage (warm start)", value=True)

    # ── Contraintes ressources ────────────────────────
    st.markdown("---")
    st.markdown("### 🔒 Contraintes ressources")
    max_latency = st.number_input("Latence max (ms)", 10, 5000, 500, 50)
    max_memory  = st.number_input("Mémoire max (MB)", 10, 2048, 512, 64)

    # ── Poids de scalarisation ────────────────────────
    st.markdown("---")
    st.markdown("### ⚖️ Poids scalarisation SMAC")
    st.caption("SMAC optimise un coût scalaire = combinaison pondérée des 3 objectifs.")
    w_acc = st.slider("Accuracy",  0.0, 1.0, 0.6, 0.1)
    w_lat = st.slider("Latence",   0.0, 1.0, 0.2, 0.1)
    w_mem = st.slider("Mémoire",   0.0, 1.0, 0.2, 0.1)
    _tot  = w_acc + w_lat + w_mem + 1e-9
    w_acc, w_lat, w_mem = w_acc/_tot, w_lat/_tot, w_mem/_tot
    st.caption(f"Normalisés → A:{w_acc:.2f} L:{w_lat:.2f} M:{w_mem:.2f}")

    st.markdown("---")
    run_btn = st.button("🚀 Lancer l'optimisation", type="primary",
                         use_container_width=True)

# ══════════════════════════════════════════════════════
#  ONGLETS PRINCIPAUX
# ══════════════════════════════════════════════════════
tab_home, tab_data, tab_results, tab_pareto, tab_conv, tab_meta = st.tabs([
    "🏠 Accueil",
    "📂 Données",
    "📊 Résultats",
    "🎯 Pareto",
    "📈 Convergence",
    "🧠 Méta-apprentissage",
])

# ══════════════════════════════════════════════════════
#  ONGLET ACCUEIL
# ══════════════════════════════════════════════════════
with tab_home:

    st.markdown('<div class="section-hdr">🎯 Objectifs du projet</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Ce framework AutoML optimise **simultanément 3 objectifs** :

    | Objectif | Mesure | Direction |
    |---|---|---|
    | 🎯 **Accuracy** | accuracy_score / R² | Maximiser ↑ |
    | ⚡ **Latence** | Médiane inférence (ms) | Minimiser ↓ |
    | 💾 **Mémoire** | Pic RAM tracemalloc (MB) | Minimiser ↓ |

    **8 algorithmes :** Random Forest, Gradient Boosting, Extra Trees,
    SVM, KNN, Modèle Linéaire, Decision Tree, XGBoost
    """)

    # ── Résumé si pipeline déjà lancé ─────────────────
    if st.session_state.pipeline_summary is not None:
        st.markdown('<div class="section-hdr">🏆 Derniers résultats</div>',
                    unsafe_allow_html=True)
        df_s = st.session_state.pipeline_summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Accuracy moy.",  f"{df_s['accuracy'].mean():.4f}")
        c2.metric("⚡ Latence moy.",   f"{df_s['latency_ms'].mean():.1f} ms")
        c3.metric("💾 Mémoire moy.",   f"{df_s['memory_mb'].mean():.1f} MB")
        c4.metric("📊 Datasets",        len(df_s))
        st.dataframe(df_s, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════
#  ONGLET DONNÉES — aperçu + stats avant lancement
# ══════════════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="section-hdr">📂 Aperçu & Statistiques du dataset</div>',
                unsafe_allow_html=True)

    # Charger pour affichage
    preview_df = None
    if df_loaded is not None:
        preview_df = df_loaded
    else:
        try:
            from data.datasets import load_dataset
            d = load_dataset(dataset_name)
            full = np.vstack([d["X_train"], d["X_test"]])
            preview_df = pd.DataFrame(full, columns=[f"f{i}" for i in range(full.shape[1])])
            preview_df["target"] = np.concatenate([d["y_train"], d["y_test"]])
        except Exception:
            pass

    if preview_df is not None:
        t_prev, t_stats, t_missing = st.tabs([
            "👁 Aperçu", "📊 Statistiques", "🔍 Valeurs manquantes"
        ])

        with t_prev:
            eff_target = target_col if target_col else preview_df.columns[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lignes",    preview_df.shape[0])
            c2.metric("Colonnes",  preview_df.shape[1])
            c3.metric("Target",    eff_target)
            c4.metric("Features",  preview_df.shape[1] - 1)
            st.dataframe(preview_df.head(10), use_container_width=True)

        with t_stats:
            num_cols = preview_df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = preview_df.select_dtypes(include=["object","category"]).columns.tolist()

            if num_cols:
                st.markdown("**Colonnes numériques**")
                st.dataframe(preview_df[num_cols].describe().round(3),
                             use_container_width=True)
            if cat_cols:
                st.markdown("**Colonnes catégorielles**")
                cat_stats = pd.DataFrame({
                    col: {
                        "Valeurs uniques":       preview_df[col].nunique(),
                        "Valeur la + fréquente": preview_df[col].mode()[0]
                                                 if not preview_df[col].mode().empty else "N/A",
                        "Fréquence (%)":         f"{preview_df[col].value_counts(normalize=True).iloc[0]*100:.1f}%"
                    }
                    for col in cat_cols
                }).T
                st.dataframe(cat_stats, use_container_width=True)

            # Distribution target
            eff_target = target_col if target_col else preview_df.columns[-1]
            st.markdown(f"**Distribution de la target : `{eff_target}`**")
            import plotly.express as px
            if preview_df[eff_target].nunique() <= 20:
                vc = preview_df[eff_target].value_counts().reset_index()
                vc.columns = [eff_target, "count"]
                fig_dist = px.bar(vc, x=eff_target, y="count",
                                  template="plotly_white", color="count",
                                  color_continuous_scale="Blues")
            else:
                fig_dist = px.histogram(preview_df, x=eff_target,
                                        template="plotly_white",
                                        color_discrete_sequence=["#2E6DB4"])
            st.plotly_chart(fig_dist, use_container_width=True)

        with t_missing:
            miss_info = pd.DataFrame({
                "Type":              preview_df.dtypes.astype(str),
                "Valeurs manquantes":preview_df.isnull().sum(),
                "% manquant":        (preview_df.isnull().sum() / len(preview_df) * 100).round(2),
                "Valeurs uniques":   preview_df.nunique(),
            })

            def _color_miss(val):
                if isinstance(val, float):
                    if val > 30: return "background-color:#FFCCCC"
                    if val > 0:  return "background-color:#FFF3CC"
                return ""

            st.dataframe(
                miss_info.style.applymap(_color_miss, subset=["% manquant"]),
                use_container_width=True
            )
            total_miss = preview_df.isnull().sum().sum()
            if total_miss == 0:
                st.success("✅ Aucune valeur manquante.")
            else:
                st.warning(f"⚠️ {total_miss} valeurs manquantes → remplacées "
                           f"automatiquement par la médiane.")
    else:
        st.info("👈 Sélectionnez ou importez un dataset dans la sidebar.")

# ══════════════════════════════════════════════════════
#  LANCEMENT DU PIPELINE
# ══════════════════════════════════════════════════════
if run_btn:
    try:
        from optimization.smac_optimizer import AutoMLOptimizer, OBJECTIVE_WEIGHTS
        from optimization.meta_learning  import MetaLearningBase, extract_meta_features
        from evaluation.evaluator        import RESOURCE_CONSTRAINTS
        from data.datasets               import load_dataset, load_custom_dataset

        # Mettre à jour les poids et contraintes globaux
        OBJECTIVE_WEIGHTS["error"]   = w_acc
        OBJECTIVE_WEIGHTS["latency"] = w_lat
        OBJECTIVE_WEIGHTS["memory"]  = w_mem
        RESOURCE_CONSTRAINTS["max_latency_ms"] = max_latency
        RESOURCE_CONSTRAINTS["max_memory_mb"]  = max_memory

        meta_base = MetaLearningBase(
            storage_path=str(PROJECT_DIR / "meta_learning_base.pkl")
        )

        # Définir la liste finale des datasets
        if df_loaded is not None:
            task_map = {"Classification": "classification",
                        "Régression":     "regression",
                        "Auto-détection": "auto"}
            datasets_to_run = [{
                "name": "custom",
                "data": load_custom_dataset(
                    df_loaded, target_col=target_col,
                    task=task_map.get(task_override, "auto")
                )
            }]
        else:
            datasets_to_run = [{"name": n, "data": None} for n in selected_datasets]

        progress_bar = st.progress(0, text="Initialisation…")
        status_txt   = st.empty()
        preview_tbl  = st.empty()
        summaries    = []

        for i, ds_info in enumerate(datasets_to_run):
            dname = ds_info["name"]
            progress_bar.progress(i / len(datasets_to_run),
                                  text=f"🔄 {dname} ({i+1}/{len(datasets_to_run)})…")

            try:
                data = ds_info["data"] if ds_info["data"] else load_dataset(dname)

                meta_feats   = extract_meta_features(data["X_train"], data["y_train"])
                warm_configs = meta_base.get_warm_start_configs(meta_feats, top_k=2) \
                               if use_meta else []

                if warm_configs:
                    status_txt.info(f"🧠 Warm start : algo suggéré → "
                                    f"**{warm_configs[0].get('algorithm','?')}**")

                optimizer = AutoMLOptimizer(
                    dataset_name = dname,
                    n_trials     = n_trials,
                    min_budget   = min_budget,
                    max_budget   = 1.0,
                    output_dir   = str(PROJECT_DIR / "smac_output"),
                    seed         = 42,
                )

                best = optimizer.run(
                    data["X_train"], data["y_train"],
                    data["X_test"],  data["y_test"],
                    task         = data["task"],
                    warm_configs = warm_configs,
                )

                if best and best.get("status") != "FAILED":
                    meta_base.add_entry(
                        dataset_name  = dname,
                        meta_features = meta_feats,
                        best_config   = optimizer.best_config or {},
                        best_score    = optimizer.best_score,
                        best_accuracy = best.get("accuracy", 0),
                    )
                    st.session_state.all_results[dname] = optimizer.get_all_results()
                    summaries.append({
                        "Dataset":     dname,
                        "Meilleur algo": best.get("algorithm", "N/A"),
                        "Accuracy":    round(best.get("accuracy", 0), 4),
                        "Latence_ms":  round(best.get("latency_ms", 0), 2),
                        "Mémoire_MB":  round(best.get("memory_mb", 0), 2),
                        "Score SMAC":  round(best.get("score", 1), 4),
                        "Temps (s)":   round(best.get("total_time_s", 0), 1),
                    })
                    preview_tbl.dataframe(pd.DataFrame(summaries),
                                          use_container_width=True)

            except Exception as e:
                st.warning(f"⚠️ Erreur sur '{dname}': {e}")

        progress_bar.progress(1.0, text="✅ Optimisation terminée !")

        if summaries:
            st.session_state.pipeline_summary = pd.DataFrame(summaries)
            status_txt.success(f"✅ {len(summaries)} dataset(s) traité(s) avec succès !")
        else:
            status_txt.error("❌ Aucun résultat obtenu.")

    except ImportError as e:
        st.error(f"Module manquant : {e}")
        st.code("pip install smac openml xgboost", language="bash")

# ══════════════════════════════════════════════════════
#  ONGLET RÉSULTATS
# ══════════════════════════════════════════════════════
with tab_results:
    st.markdown('<div class="section-hdr">📊 Résultats détaillés</div>',
                unsafe_allow_html=True)

    if not st.session_state.all_results:
        st.info("👆 Lancez l'optimisation pour voir les résultats.")
    else:
        try:
            from analysis.visualizations import plot_algorithm_comparison, plot_datasets_heatmap

            available = list(st.session_state.all_results.keys())
            sel = st.selectbox("Dataset", available, key="res_ds")
            df_sel = st.session_state.all_results[sel]

            if not df_sel.empty:
                best_row = df_sel.loc[df_sel["accuracy"].idxmax()]
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("🎯 Meilleure accuracy",  f"{best_row['accuracy']:.4f}")
                c2.metric("⚡ Latence",             f"{best_row['latency_ms']:.1f} ms")
                c3.metric("💾 Mémoire",             f"{best_row['memory_mb']:.1f} MB")
                c4.metric("🤖 Meilleur algo",       best_row.get("algorithm","N/A"))

                metric_label = "R²" if st.session_state.all_results.get(
                    sel, pd.DataFrame()).get("task", [""])[0:1] == ["regression"] else "Accuracy"

                fig_box = plot_algorithm_comparison(df_sel, f"Comparaison — {sel}")
                st.plotly_chart(fig_box, use_container_width=True)

                with st.expander("📋 Toutes les évaluations"):
                    cols_show = [c for c in ["algorithm","accuracy","latency_ms",
                                              "memory_mb","score","budget","status",
                                              "train_time_s"] if c in df_sel.columns]
                    df_show = df_sel[cols_show].sort_values("accuracy", ascending=False)
                    c_ok    = df_show["accuracy"] == df_show["accuracy"].max()
                    st.dataframe(
                        df_show.style.highlight_max(axis=0, subset=["accuracy"],
                                                     color="#C8F7C5")
                                     .highlight_min(axis=0, subset=["latency_ms","memory_mb"],
                                                     color="#C8F7C5"),
                        use_container_width=True, height=380
                    )

            if len(st.session_state.all_results) > 1:
                st.markdown("### 🗺️ Vue globale — tous les datasets")
                fig_heat = plot_datasets_heatmap(st.session_state.all_results)
                st.plotly_chart(fig_heat, use_container_width=True)

        except ImportError as e:
            st.error(f"Module manquant: {e}")

# ══════════════════════════════════════════════════════
#  ONGLET PARETO
# ══════════════════════════════════════════════════════
with tab_pareto:
    st.markdown('<div class="section-hdr">🎯 Front de Pareto</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Le **front de Pareto** contient les solutions pour lesquelles il est "
        "impossible d'améliorer un objectif sans en dégrader un autre."
    )

    if not st.session_state.all_results:
        st.info("👆 Lancez l'optimisation pour voir le front de Pareto.")
    else:
        try:
            from analysis.visualizations import plot_pareto_front_3d, plot_tradeoff_scatter

            all_dfs = list(st.session_state.all_results.values())
            df_all  = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

            ds_filter = st.selectbox(
                "Filtrer par dataset",
                ["Tous"] + list(st.session_state.all_results.keys()),
                key="par_filter"
            )
            df_filt = (df_all[df_all["dataset"] == ds_filter]
                       if ds_filter != "Tous" and "dataset" in df_all.columns
                       else df_all)

            col1, col2 = st.columns([3, 2])
            with col1:
                st.plotly_chart(
                    plot_pareto_front_3d(df_filt, f"Pareto 3D — {ds_filter}"),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    plot_tradeoff_scatter(df_filt, max_latency=max_latency),
                    use_container_width=True
                )

            # Tableau des solutions Pareto
            from evaluation.evaluator import compute_pareto_front
            df_ok = df_filt[df_filt["status"] == "SUCCESS"].reset_index(drop=True) \
                    if "status" in df_filt.columns else df_filt.reset_index(drop=True)
            if not df_ok.empty:
                pidx    = compute_pareto_front(df_ok.to_dict("records"))
                df_pare = df_ok.iloc[pidx].sort_values("accuracy", ascending=False)
                st.markdown(f"**{len(df_pare)} solutions Pareto-optimales**")
                cols_p  = [c for c in ["algorithm","accuracy","latency_ms",
                                        "memory_mb","score"] if c in df_pare.columns]
                st.dataframe(df_pare[cols_p].round(4),
                             use_container_width=True, hide_index=True)

        except ImportError as e:
            st.error(f"Module manquant: {e}")

# ══════════════════════════════════════════════════════
#  ONGLET CONVERGENCE
# ══════════════════════════════════════════════════════
with tab_conv:
    st.markdown('<div class="section-hdr">📈 Convergence SMAC + Hyperband</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Ces graphiques montrent comment SMAC apprend au fil des évaluations. "
        "Hyperband alloue plus de budget aux configurations prometteuses."
    )

    if not st.session_state.all_results:
        st.info("👆 Lancez l'optimisation pour voir la convergence.")
    else:
        try:
            from analysis.visualizations import plot_convergence, plot_radar

            ds_conv = st.selectbox("Dataset", list(st.session_state.all_results.keys()),
                                    key="conv_ds")
            df_conv = st.session_state.all_results[ds_conv]

            if not df_conv.empty:
                st.plotly_chart(
                    plot_convergence(df_conv, f"Convergence SMAC — {ds_conv}"),
                    use_container_width=True
                )

                c1, c2, c3 = st.columns(3)
                df_ok_conv = df_conv[df_conv["status"] == "SUCCESS"] \
                             if "status" in df_conv.columns else df_conv
                c1.metric("Évaluations totales",   len(df_conv))
                c2.metric("Évaluations réussies",   len(df_ok_conv))
                if len(df_ok_conv) > 4:
                    h1 = df_ok_conv.head(len(df_ok_conv)//2)["accuracy"].max()
                    h2 = df_ok_conv.tail(len(df_ok_conv)//2)["accuracy"].max()
                    c3.metric("Gain (2ème moitié)",    f"+{h2-h1:.4f}")

                # Radar des algorithmes
                st.markdown("### 🕸️ Profil des algorithmes")
                st.plotly_chart(plot_radar(df_conv, f"Profil — {ds_conv}"),
                                use_container_width=True)

                # Distribution par budget (impact Hyperband)
                if "budget" in df_conv.columns and df_conv["budget"].nunique() > 1:
                    import plotly.express as px
                    st.markdown("### 📦 Distribution accuracy par budget Hyperband")
                    fig_bud = px.box(df_ok_conv, x="budget", y="accuracy",
                                     color="budget", template="plotly_white",
                                     title="Impact du budget Hyperband sur l'accuracy")
                    st.plotly_chart(fig_bud, use_container_width=True)

        except ImportError as e:
            st.error(f"Module manquant: {e}")

# ══════════════════════════════════════════════════════
#  ONGLET MÉTA-APPRENTISSAGE
# ══════════════════════════════════════════════════════
with tab_meta:
    st.markdown('<div class="section-hdr">🧠 Base de Méta-apprentissage</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Le méta-apprentissage mémorise les meilleures configurations de chaque dataset. "
        "Sur un nouveau dataset, il cherche les datasets similaires et utilise leurs "
        "configurations comme **warm start** pour SMAC."
    )

    try:
        from optimization.meta_learning import MetaLearningBase, extract_meta_features

        meta_base = MetaLearningBase(
            storage_path=str(PROJECT_DIR / "meta_learning_base.pkl")
        )
        df_meta = meta_base.get_summary()

        if df_meta.empty:
            st.info("🆕 Base méta vide. Lancez l'optimisation pour la remplir.")
        else:
            st.success(f"✅ {len(df_meta)} dataset(s) en mémoire")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(df_meta, use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**Algos les plus souvent gagnants**")
                ac = df_meta["Best algo"].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=ac.index, values=ac.values, hole=0.4,
                )])
                fig_pie.update_layout(height=300, template="plotly_white",
                                      margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            if st.button("🗑️ Vider la base méta"):
                meta_base.clear()
                st.success("Base méta vidée.")
                st.rerun()

        # ── Test similarité ────────────────────────────
        st.markdown("### 🔍 Tester la similarité d'un nouveau dataset")
        c1, c2, c3 = st.columns(3)
        with c1: ns = st.number_input("n_samples",  10, 100000, 500)
        with c2: nf = st.number_input("n_features", 1,  1000,   20)
        with c3: nc = st.number_input("n_classes",  2,  100,    2)

        if st.button("🔍 Trouver datasets similaires"):
            fake_meta = {
                "n_samples":           float(ns),
                "n_features":          float(nf),
                "n_classes":           float(nc),
                "log_samples":         float(np.log1p(ns)),
                "log_features":        float(np.log1p(nf)),
                "samples_per_feature": float(ns / max(nf, 1)),
                "features_per_sample": float(nf / max(ns, 1)),
                "class_imbalance":     1.0,
                "class_entropy":       float(np.log2(nc)),
                "mean_feature_mean":   0.0,
                "mean_feature_std":    1.0,
                "missing_ratio":       0.0,
                "sparsity":            0.0,
                "relative_n_features": float(nf / max(ns, 1)),
                "mean_skewness":       0.0,
            }
            similar = meta_base.find_similar_datasets(fake_meta, top_k=3)
            if similar:
                for s in similar:
                    st.info(
                        f"📌 **{s['dataset_name']}** "
                        f"(dist={s['similarity_distance']:.3f}) → "
                        f"Meilleur algo : **{s['best_config'].get('algorithm','?')}** "
                        f"| Accuracy : {s['best_accuracy']:.4f}"
                    )
            else:
                st.warning("Base méta vide. Lancez d'abord le pipeline.")

    except Exception as e:
        st.warning(f"Impossible de charger la base méta : {e}")

# ── Footer ────────────────────────────────────────────
st.markdown("---")
st.caption("AutoML Systémique — Sujet 3 | Apprentissage Automatique 2025-2026 | "
           "SMAC3 + MultiFidelityFacade + Hyperband natif + Méta-apprentissage")
