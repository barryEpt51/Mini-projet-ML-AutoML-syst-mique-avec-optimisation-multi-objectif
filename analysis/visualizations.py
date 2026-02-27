"""
MODULE 6 : Analyse des compromis et visualisations
====================================================
Ce module génère toutes les visualisations pour analyser les résultats :
  1. Front de Pareto 3D interactif (accuracy vs latence vs mémoire)
  2. Comparaison des algorithmes par dataset
  3. Évolution du score au fil des évaluations (convergence SMAC)
  4. Heatmap des performances sur les 10 datasets
  5. Distribution des hyperparamètres des meilleures configs
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# -------------------------------------------------------
# COULEURS PAR ALGORITHME (cohérence visuelle)
# -------------------------------------------------------
ALGO_COLORS = {
    "random_forest":       "#2196F3",  # bleu
    "gradient_boosting":   "#4CAF50",  # vert
    "extra_trees":         "#00BCD4",  # cyan
    "svm":                 "#FF9800",  # orange
    "knn":                 "#9C27B0",  # violet
    "logistic_regression": "#F44336",  # rouge
    "decision_tree":       "#795548",  # marron
    "xgboost":             "#FF5722",  # rouge-orange
}

LAYOUT_THEME = dict(
    template   = "plotly_white",
    font       = dict(family="Inter, Arial", size=13),
    paper_bgcolor = "#FAFAFA",
    plot_bgcolor  = "#FFFFFF",
)


def plot_pareto_front_3d(df: pd.DataFrame, title: str = "Front de Pareto 3D") -> go.Figure:
    """
    Graphique 3D interactif montrant le front de Pareto.
    
    Les 3 axes :
      - X : Accuracy (à maximiser → à droite = meilleur)
      - Y : Latence en ms (à minimiser → en bas = meilleur)  
      - Z : Mémoire en MB (à minimiser → en bas = meilleur)
    
    Les points Pareto-optimaux sont mis en évidence.

    Paramètres
    ----------
    df    : DataFrame avec colonnes accuracy, latency_ms, memory_mb, algorithm
    title : titre du graphique
    """
    if df.empty:
        return go.Figure().update_layout(title="Pas de données")

    df_plot = df[df["status"] == "SUCCESS"].copy() if "status" in df.columns else df.copy()

    if df_plot.empty:
        return go.Figure().update_layout(title="Pas de données valides")

    # Calculer le front de Pareto
    from evaluation.evaluator import compute_pareto_front
    results_list    = df_plot.to_dict("records")
    pareto_indices  = compute_pareto_front(results_list)
    df_plot["pareto"] = False
    df_plot.iloc[pareto_indices, df_plot.columns.get_loc("pareto")] = True

    fig = go.Figure()

    # --- Points non-Pareto (gris, petits) ---
    non_pareto = df_plot[~df_plot["pareto"]]
    if not non_pareto.empty:
        fig.add_trace(go.Scatter3d(
            x    = non_pareto["accuracy"],
            y    = non_pareto["latency_ms"],
            z    = non_pareto["memory_mb"],
            mode = "markers",
            name = "Autres configurations",
            marker = dict(
                size   = 3,
                color  = "lightgray",
                opacity= 0.5,
            ),
            hovertemplate = (
                "Algorithme: %{customdata[0]}<br>"
                "Accuracy: %{x:.3f}<br>"
                "Latence: %{y:.1f}ms<br>"
                "Mémoire: %{z:.1f}MB<extra></extra>"
            ),
            customdata = non_pareto[["algorithm"]].values if "algorithm" in non_pareto.columns else None,
        ))

    # --- Points Pareto (colorés par algorithme, grands) ---
    pareto_df = df_plot[df_plot["pareto"]]
    if not pareto_df.empty:
        algos = pareto_df["algorithm"].unique() if "algorithm" in pareto_df.columns else ["unknown"]
        for algo in algos:
            subset = pareto_df[pareto_df["algorithm"] == algo] if "algorithm" in pareto_df.columns else pareto_df
            color  = ALGO_COLORS.get(algo, "#333333")
            fig.add_trace(go.Scatter3d(
                x    = subset["accuracy"],
                y    = subset["latency_ms"],
                z    = subset["memory_mb"],
                mode = "markers",
                name = f"Pareto - {algo}",
                marker = dict(
                    size   = 8,
                    color  = color,
                    opacity= 0.9,
                    symbol = "diamond",
                    line   = dict(color="white", width=1),
                ),
                hovertemplate = (
                    f"<b>{algo}</b><br>"
                    "Accuracy: %{x:.4f}<br>"
                    "Latence: %{y:.1f}ms<br>"
                    "Mémoire: %{z:.1f}MB<br>"
                    "<extra>✅ Pareto-optimal</extra>"
                ),
            ))

    fig.update_layout(
        title  = dict(text=title, x=0.5, font=dict(size=16)),
        scene  = dict(
            xaxis_title = "Accuracy ↑",
            yaxis_title = "Latence (ms) ↓",
            zaxis_title = "Mémoire (MB) ↓",
            xaxis = dict(backgroundcolor="#EEF"),
            yaxis = dict(backgroundcolor="#EFE"),
            zaxis = dict(backgroundcolor="#FEE"),
        ),
        legend = dict(x=0, y=1),
        height = 700,
        **LAYOUT_THEME,
    )

    return fig


def plot_convergence(df: pd.DataFrame, title: str = "Convergence SMAC") -> go.Figure:
    """
    Montre comment le meilleur score évolue au fil des évaluations.
    Permet de visualiser la convergence de l'optimisation bayésienne.
    """
    if df.empty or "score" not in df.columns:
        return go.Figure().update_layout(title="Pas de données")

    df_sorted = df.sort_index().reset_index(drop=True)
    df_sorted["best_score_so_far"] = df_sorted["score"].cummin()
    df_sorted["best_accuracy_so_far"] = df_sorted["accuracy"].cummax()
    df_sorted["evaluation"] = range(1, len(df_sorted) + 1)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Meilleur score combiné (↓)", "Meilleure accuracy (↑)"),
        vertical_spacing=0.15,
    )

    # Score
    fig.add_trace(go.Scatter(
        x    = df_sorted["evaluation"],
        y    = df_sorted["score"],
        mode = "markers",
        name = "Score par évaluation",
        marker = dict(size=4, color="lightblue", opacity=0.6),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x    = df_sorted["evaluation"],
        y    = df_sorted["best_score_so_far"],
        mode = "lines",
        name = "Meilleur score",
        line = dict(color="#2196F3", width=2.5),
    ), row=1, col=1)

    # Accuracy
    fig.add_trace(go.Scatter(
        x    = df_sorted["evaluation"],
        y    = df_sorted["accuracy"],
        mode = "markers",
        name = "Accuracy par évaluation",
        marker = dict(size=4, color="lightgreen", opacity=0.6),
        showlegend=True,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x    = df_sorted["evaluation"],
        y    = df_sorted["best_accuracy_so_far"],
        mode = "lines",
        name = "Meilleure accuracy",
        line = dict(color="#4CAF50", width=2.5),
    ), row=2, col=1)

    fig.update_layout(
        title  = dict(text=title, x=0.5),
        height = 600,
        **LAYOUT_THEME,
    )
    fig.update_xaxes(title_text="Nombre d'évaluations")

    return fig


def plot_algorithm_comparison(df: pd.DataFrame, title: str = "Comparaison des algorithmes") -> go.Figure:
    """
    Boxplot comparant la distribution des performances par algorithme.
    """
    if df.empty or "algorithm" not in df.columns:
        return go.Figure().update_layout(title="Pas de données")

    df_success = df[df["status"] == "SUCCESS"].copy() if "status" in df.columns else df.copy()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Accuracy ↑", "Latence (ms) ↓", "Mémoire (MB) ↓"),
    )

    algos = sorted(df_success["algorithm"].unique())

    for i, (col_name, ylabel, row, col) in enumerate([
        ("accuracy",   "Accuracy",     1, 1),
        ("latency_ms", "Latence (ms)", 1, 2),
        ("memory_mb",  "Mémoire (MB)", 1, 3),
    ]):
        for algo in algos:
            subset = df_success[df_success["algorithm"] == algo]
            color  = ALGO_COLORS.get(algo, "#333")
            fig.add_trace(go.Box(
                y    = subset[col_name],
                name = algo,
                marker_color = color,
                showlegend   = (i == 0),
                boxmean      = True,
            ), row=row, col=col)

    fig.update_layout(
        title  = dict(text=title, x=0.5),
        height = 500,
        **LAYOUT_THEME,
        boxmode = "group",
    )

    return fig


def plot_datasets_heatmap(results_by_dataset: dict, title: str = "Performance sur 10 datasets") -> go.Figure:
    """
    Heatmap montrant les performances de chaque algorithme sur chaque dataset.
    
    Paramètres
    ----------
    results_by_dataset : dict {dataset_name: DataFrame_résultats}
    """
    if not results_by_dataset:
        return go.Figure().update_layout(title="Pas de données")

    # Construire la matrice : dataset x algorithme → meilleure accuracy
    datasets   = []
    algorithms_set = set()

    for name, df in results_by_dataset.items():
        if df.empty:
            continue
        datasets.append(name)
        if "algorithm" in df.columns:
            algorithms_set.update(df["algorithm"].unique())

    algorithms = sorted(algorithms_set)
    if not datasets or not algorithms:
        return go.Figure().update_layout(title="Pas assez de données")

    matrix = np.full((len(datasets), len(algorithms)), np.nan)

    for i, dataset_name in enumerate(datasets):
        df = results_by_dataset.get(dataset_name, pd.DataFrame())
        if df.empty or "algorithm" not in df.columns:
            continue
        df_ok = df[df["status"] == "SUCCESS"] if "status" in df.columns else df
        for j, algo in enumerate(algorithms):
            subset = df_ok[df_ok["algorithm"] == algo]
            if not subset.empty:
                matrix[i, j] = subset["accuracy"].max()

    fig = go.Figure(data=go.Heatmap(
        z           = matrix,
        x           = algorithms,
        y           = datasets,
        colorscale  = "RdYlGn",
        zmin        = 0.5,
        zmax        = 1.0,
        text        = np.where(np.isnan(matrix), "", np.round(matrix, 3).astype(str)),
        texttemplate= "%{text}",
        hovertemplate = "Dataset: %{y}<br>Algo: %{x}<br>Accuracy: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title  = dict(text=title, x=0.5),
        height = max(400, len(datasets) * 50),
        **LAYOUT_THEME,
    )

    return fig


def plot_tradeoff_scatter(df: pd.DataFrame, title: str = "Compromis Accuracy vs Latence") -> go.Figure:
    """
    Scatter 2D montrant le compromis entre accuracy et latence,
    avec la taille des points proportionnelle à la mémoire.
    """
    if df.empty:
        return go.Figure().update_layout(title="Pas de données")

    df_ok = df[df["status"] == "SUCCESS"].copy() if "status" in df.columns else df.copy()
    if df_ok.empty:
        return go.Figure().update_layout(title="Pas de données")

    fig = go.Figure()

    algos = df_ok["algorithm"].unique() if "algorithm" in df_ok.columns else ["unknown"]
    for algo in algos:
        subset = df_ok[df_ok["algorithm"] == algo] if "algorithm" in df_ok.columns else df_ok
        color  = ALGO_COLORS.get(algo, "#333")

        fig.add_trace(go.Scatter(
            x    = subset["accuracy"],
            y    = subset["latency_ms"],
            mode = "markers",
            name = algo,
            marker = dict(
                size   = np.clip(subset["memory_mb"] / 10, 4, 20),
                color  = color,
                opacity= 0.7,
                line   = dict(color="white", width=0.5),
            ),
            hovertemplate = (
                f"<b>{algo}</b><br>"
                "Accuracy: %{x:.4f}<br>"
                "Latence: %{y:.1f}ms<br>"
                "Mémoire: %{marker.size:.0f}×10 MB<extra></extra>"
            ),
        ))

    fig.update_layout(
        title      = dict(text=title + "<br><sup>Taille des points ∝ Mémoire utilisée</sup>", x=0.5),
        xaxis_title = "Accuracy ↑",
        yaxis_title = "Latence (ms) ↓",
        height     = 550,
        **LAYOUT_THEME,
    )

    return fig


# -------------------------------------------------------
# Test rapide
# -------------------------------------------------------
if __name__ == "__main__":
    # Créer des données fictives pour tester
    np.random.seed(42)
    n = 50
    algos = ["random_forest", "svm", "knn", "gradient_boosting"]

    df_test = pd.DataFrame({
        "algorithm":  np.random.choice(algos, n),
        "accuracy":   np.random.uniform(0.7, 0.99, n),
        "latency_ms": np.random.uniform(1, 200, n),
        "memory_mb":  np.random.uniform(10, 200, n),
        "error":      np.random.uniform(0.01, 0.3, n),
        "score":      np.random.uniform(0.05, 0.5, n),
        "status":     ["SUCCESS"] * n,
    })

    fig1 = plot_pareto_front_3d(df_test)
    fig1.show()

    fig2 = plot_convergence(df_test)
    fig2.show()

    print("✅ Visualisations générées avec succès !")
