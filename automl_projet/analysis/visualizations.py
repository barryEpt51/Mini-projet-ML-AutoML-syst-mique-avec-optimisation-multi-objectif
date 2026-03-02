"""
analysis/visualizations.py
===========================
Toutes les visualisations du framework AutoML.
Fond blanc (template='plotly_white') sur tous les graphiques.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from evaluation.evaluator import compute_pareto_front

# ── Palette cohérente par algorithme ──────────────────
ALGO_COLORS = {
    "random_forest":    "#2196F3",
    "gradient_boosting":"#4CAF50",
    "extra_trees":      "#00BCD4",
    "svm":              "#FF9800",
    "knn":              "#9C27B0",
    "linear_model":     "#F44336",
    "decision_tree":    "#795548",
    "xgboost":          "#FF5722",
}

LAYOUT = dict(template="plotly_white",
              font=dict(family="Arial", size=13),
              paper_bgcolor="#FAFAFA",
              plot_bgcolor="#FFFFFF")


# ── 1. Front de Pareto 3D ─────────────────────────────
def plot_pareto_front_3d(df: pd.DataFrame, title: str = "Front de Pareto 3D",
                          metric_label: str = "Accuracy") -> go.Figure:
    if df.empty:
        return go.Figure().update_layout(title="Pas de données", **LAYOUT)

    df_ok = df[df["status"] == "SUCCESS"].copy() if "status" in df.columns else df.copy()
    if df_ok.empty:
        return go.Figure().update_layout(title="Pas de données valides", **LAYOUT)

    # Calcul front de Pareto
    records      = df_ok.to_dict("records")
    pareto_idx   = compute_pareto_front(records)
    df_ok        = df_ok.reset_index(drop=True)
    df_ok["pareto"] = False
    df_ok.loc[pareto_idx, "pareto"] = True

    fig = go.Figure()

    # Points non-Pareto
    non_p = df_ok[~df_ok["pareto"]]
    if not non_p.empty:
        fig.add_trace(go.Scatter3d(
            x=non_p["accuracy"], y=non_p["latency_ms"], z=non_p["memory_mb"],
            mode="markers", name="Autres",
            marker=dict(size=3, color="lightgray", opacity=0.5),
            hovertemplate="Algo: %{customdata}<br>Accuracy: %{x:.3f}<br>"
                          "Latence: %{y:.1f}ms<br>Mémoire: %{z:.1f}MB<extra></extra>",
            customdata=non_p["algorithm"].values if "algorithm" in non_p.columns else None
        ))

    # Points Pareto par algorithme
    pareto_df = df_ok[df_ok["pareto"]]
    for algo in (pareto_df["algorithm"].unique() if "algorithm" in pareto_df.columns else []):
        sub   = pareto_df[pareto_df["algorithm"] == algo]
        color = ALGO_COLORS.get(algo, "#333")
        fig.add_trace(go.Scatter3d(
            x=sub["accuracy"], y=sub["latency_ms"], z=sub["memory_mb"],
            mode="markers", name=f"★ {algo}",
            marker=dict(size=8, color=color, opacity=0.9, symbol="diamond",
                        line=dict(color="white", width=1)),
            hovertemplate=f"<b>{algo}</b><br>Accuracy: %{{x:.4f}}<br>"
                          f"Latence: %{{y:.1f}}ms<br>Mémoire: %{{z:.1f}}MB"
                          f"<extra>✅ Pareto-optimal</extra>"
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title=f"{metric_label} ↑",
            yaxis_title="Latence (ms) ↓",
            zaxis_title="Mémoire (MB) ↓",
            xaxis=dict(backgroundcolor="#EEF4FF"),
            yaxis=dict(backgroundcolor="#F0FFF0"),
            zaxis=dict(backgroundcolor="#FFF8EE"),
        ),
        height=650, legend=dict(x=0, y=1), **LAYOUT
    )
    return fig


# ── 2. Scatter 2D : compromis accuracy vs latence ─────
def plot_tradeoff_scatter(df: pd.DataFrame, title: str = "Compromis Accuracy vs Latence",
                           max_latency: float = 500.0,
                           metric_label: str = "Accuracy") -> go.Figure:
    if df.empty:
        return go.Figure().update_layout(title="Pas de données", **LAYOUT)

    df_ok = df[df["status"] == "SUCCESS"].copy() if "status" in df.columns else df.copy()
    fig   = go.Figure()

    for algo in (df_ok["algorithm"].unique() if "algorithm" in df_ok.columns else []):
        sub   = df_ok[df_ok["algorithm"] == algo]
        color = ALGO_COLORS.get(algo, "#333")
        sizes = np.clip(sub["memory_mb"] / 10, 4, 25) if "memory_mb" in sub.columns else 8
        fig.add_trace(go.Scatter(
            x=sub["accuracy"], y=sub["latency_ms"],
            mode="markers", name=algo,
            marker=dict(size=sizes, color=color, opacity=0.75,
                        line=dict(color="white", width=0.5)),
            hovertemplate=f"<b>{algo}</b><br>Accuracy: %{{x:.4f}}<br>"
                          f"Latence: %{{y:.1f}}ms<extra></extra>"
        ))

    fig.add_hline(y=max_latency, line_dash="dash", line_color="red",
                  annotation_text=f"Limite latence ({max_latency}ms)")

    fig.update_layout(
        title=dict(text=title + "<br><sup>Taille ∝ Mémoire utilisée</sup>", x=0.5),
        xaxis_title=f"{metric_label} ↑",
        yaxis_title="Latence (ms) ↓",
        height=520, **LAYOUT
    )
    return fig


# ── 3. Courbe de convergence SMAC ─────────────────────
def plot_convergence(df: pd.DataFrame, title: str = "Convergence SMAC",
                      metric_label: str = "Accuracy") -> go.Figure:
    if df.empty or "accuracy" not in df.columns:
        return go.Figure().update_layout(title="Pas de données", **LAYOUT)

    df_s = df.reset_index(drop=True)
    df_s["best_acc"]  = df_s["accuracy"].cummax()
    df_s["best_score"] = df_s["score"].cummin() if "score" in df_s.columns else None
    df_s["eval"]      = range(1, len(df_s) + 1)

    fig = make_subplots(rows=2, cols=1,
                         subplot_titles=(f"Meilleure {metric_label} cumulée (↑)",
                                         "Meilleur score SMAC cumulé (↓)"),
                         vertical_spacing=0.15)

    # Accuracy
    fig.add_trace(go.Scatter(x=df_s["eval"], y=df_s["accuracy"], mode="markers",
                              name="Accuracy / éval", showlegend=True,
                              marker=dict(size=4, color="lightblue", opacity=0.6)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_s["eval"], y=df_s["best_acc"], mode="lines",
                              name=f"Meilleure {metric_label}",
                              line=dict(color="#2196F3", width=2.5)),
                  row=1, col=1)

    # Score SMAC
    if "score" in df_s.columns:
        fig.add_trace(go.Scatter(x=df_s["eval"], y=df_s["score"], mode="markers",
                                  name="Score SMAC / éval",
                                  marker=dict(size=4, color="#FFCC80", opacity=0.6)),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=df_s["eval"], y=df_s["best_score"], mode="lines",
                                  name="Meilleur score SMAC",
                                  line=dict(color="#FF9800", width=2.5)),
                      row=2, col=1)

    fig.update_layout(title=dict(text=title, x=0.5), height=600, **LAYOUT)
    fig.update_xaxes(title_text="Évaluations")
    return fig


# ── 4. Comparaison algorithmes (boxplot) ──────────────
def plot_algorithm_comparison(df: pd.DataFrame,
                               title: str = "Comparaison des algorithmes",
                               metric_label: str = "Accuracy") -> go.Figure:
    if df.empty or "algorithm" not in df.columns:
        return go.Figure().update_layout(title="Pas de données", **LAYOUT)

    df_ok  = df[df["status"] == "SUCCESS"].copy() if "status" in df.columns else df.copy()
    algos  = sorted(df_ok["algorithm"].unique())
    fig    = make_subplots(rows=1, cols=3,
                            subplot_titles=(f"{metric_label} ↑",
                                            "Latence (ms) ↓",
                                            "Mémoire (MB) ↓"))

    for i, (col_name, row, col) in enumerate([
        ("accuracy",   1, 1),
        ("latency_ms", 1, 2),
        ("memory_mb",  1, 3),
    ]):
        for algo in algos:
            sub = df_ok[df_ok["algorithm"] == algo]
            fig.add_trace(go.Box(
                y=sub[col_name], name=algo,
                marker_color=ALGO_COLORS.get(algo, "#333"),
                showlegend=(i == 0), boxmean=True
            ), row=row, col=col)

    fig.update_layout(title=dict(text=title, x=0.5),
                       height=480, boxmode="group", **LAYOUT)
    return fig


# ── 5. Heatmap performances sur N datasets ────────────
def plot_datasets_heatmap(results_by_dataset: dict,
                           title: str = "Performance sur les datasets") -> go.Figure:
    if not results_by_dataset:
        return go.Figure().update_layout(title="Pas de données", **LAYOUT)

    datasets, algos_set = [], set()
    for name, df in results_by_dataset.items():
        if df is not None and not df.empty:
            datasets.append(name)
            if "algorithm" in df.columns:
                algos_set.update(df["algorithm"].unique())

    algos = sorted(algos_set)
    if not datasets or not algos:
        return go.Figure().update_layout(title="Données insuffisantes", **LAYOUT)

    matrix = np.full((len(datasets), len(algos)), np.nan)
    for i, dname in enumerate(datasets):
        df = results_by_dataset.get(dname)
        if df is None or df.empty:
            continue
        df_ok = df[df["status"] == "SUCCESS"] if "status" in df.columns else df
        for j, algo in enumerate(algos):
            sub = df_ok[df_ok["algorithm"] == algo] if "algorithm" in df_ok.columns else pd.DataFrame()
            if not sub.empty:
                matrix[i, j] = sub["accuracy"].max()

    text = np.where(np.isnan(matrix), "", np.round(matrix, 3).astype(str))
    fig  = go.Figure(data=go.Heatmap(
        z=matrix, x=algos, y=datasets,
        colorscale="RdYlGn", zmin=0.5, zmax=1.0,
        text=text, texttemplate="%{text}",
        hovertemplate="Dataset: %{y}<br>Algo: %{x}<br>Accuracy: %{z:.4f}<extra></extra>"
    ))
    fig.update_layout(title=dict(text=title, x=0.5),
                       height=max(400, len(datasets) * 55), **LAYOUT)
    return fig


# ── 6. Radar chart par algorithme ─────────────────────
def plot_radar(df: pd.DataFrame, title: str = "Profil des algorithmes") -> go.Figure:
    if df.empty or "algorithm" not in df.columns:
        return go.Figure().update_layout(title="Pas de données", **LAYOUT)

    df_ok = df[df["status"] == "SUCCESS"].copy() if "status" in df.columns else df.copy()
    stats = df_ok.groupby("algorithm").agg(
        acc_mean=("accuracy",   "mean"),
        lat_mean=("latency_ms", "mean"),
        mem_mean=("memory_mb",  "mean"),
    ).reset_index()

    fig = go.Figure()
    for _, row in stats.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[
                row["acc_mean"],
                1 / (1 + row["lat_mean"] / 100),
                1 / (1 + row["mem_mean"] / 100),
            ],
            theta=["Accuracy", "Rapidité", "Légèreté"],
            fill="toself", name=row["algorithm"],
            line=dict(color=ALGO_COLORS.get(row["algorithm"], "#333"))
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=480, **LAYOUT
    )
    return fig
