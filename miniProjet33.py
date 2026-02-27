import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import pickle
import optuna
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_diabetes, load_breast_cancer

# ---------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------------------------------------
st.set_page_config(page_title="Framework AutoML Systémique", layout="wide")

st.title("🚀 Framework AutoML Systémique & Multi-Objectif")
st.markdown("""
**Sujet 3 :** Ce framework optimise simultanément l'**Accuracy (R²)**, la **Latence** et la **Mémoire**.
Il utilise l'**Optimisation Bayésienne** pour explorer l'espace de recherche et identifier le front de Pareto.
""")

# ---------------------------------------------------------
# SIDEBAR - PARAMÈTRES ET BUDGET
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuration du Framework")
dataset_option = st.sidebar.selectbox("Choisir un Dataset", ["Diabetes (Régression)", "Upload CSV"])
budget = st.sidebar.slider("Budget (Nombre d'essais Bayésiens)", 5, 50, 20)

st.sidebar.subheader("🎯 Contraintes Ressources")
max_mem_allowed = st.sidebar.number_input("Mémoire Max autorisée (Ko)", value=500.0)
max_lat_allowed = st.sidebar.number_input("Latence Max autorisée (ms)", value=10.0) / 1000

# ---------------------------------------------------------
# FONCTIONS TECHNIQUES (MÉMOIRE & LATENCE)
# ---------------------------------------------------------
def measure_metrics(model, X_test, y_test):
    # 1. Mesure de la Précision
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    
    # 2. Mesure de la Latence (Inférence unitaire moyenne)
    start_time = time.time()
    for _ in range(100): # Moyenne sur 100 prédictions pour la stabilité
        model.predict(X_test[:1])
    latency = (time.time() - start_time) / 100
    
    # 3. Mesure de la Mémoire (Taille du modèle sérialisé)
    mem_size = len(pickle.dumps(model)) / 1024  # En Ko
    
    return r2, latency, mem_size

# ---------------------------------------------------------
# MOTEUR D'OPTIMISATION BAYÉSIENNE (MULTI-OBJECTIF)
# ---------------------------------------------------------
def run_automl_process(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results_data = []

    # Fonction objectif pour Optuna
    def objective(trial):
        # Espace de recherche
        algo = trial.suggest_categorical("algo", ["RandomForest", "GradientBoosting"])
        
        if algo == "RandomForest":
            n_estimators = trial.suggest_int("rf_n_estimators", 10, 200)
            max_depth = trial.suggest_int("rf_max_depth", 2, 20)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            n_estimators = trial.suggest_int("gb_n_estimators", 10, 200)
            learning_rate = trial.suggest_float("gb_lr", 0.01, 0.3)
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

        # Entraînement
        model.fit(X_train, y_train)
        
        # Évaluation systémique
        r2, lat, mem = measure_metrics(model, X_test, y_test)
        
        # Sauvegarde des résultats
        results_data.append({
            "Trial": trial.number,
            "Algo": algo,
            "R2": r2,
            "Latence": lat,
            "Memoire": mem,
            "Respect_Contraintes": (lat <= max_lat_allowed and mem <= max_mem_allowed)
        })
        
        # On retourne le R2 (Optuna va chercher à le maximiser)
        return r2

    # Lancement de l'optimisation
    with st.spinner("Optimisation Bayésienne en cours..."):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=budget)

    # ---------------------------------------------------------
    # AFFICHAGE DES RÉSULTATS ET ANALYSE DE COMPROMIS
    # ---------------------------------------------------------
    res_df = pd.DataFrame(results_data)
    
    st.header("📊 Analyse des Compromis (Front de Pareto)")
    
    # Graphique Multi-Objectif
    fig = px.scatter(
        res_df, x="Latence", y="R2", size="Memoire", color="Algo",
        symbol="Respect_Contraintes",
        title="Optimisation : Précision vs Latence (Taille du point = Mémoire)",
        labels={"Latence": "Latence (s)", "R2": "Précision (R²)"},
        hover_data=["Trial", "Memoire"]
    )
    # Ajout des lignes de contraintes
    fig.add_vline(x=max_lat_allowed, line_dash="dash", line_color="red", annotation_text="Limite Latence")
    st.plotly_chart(fig, use_container_width=True)

    # Tableau des Meilleurs Modèles (ceux qui respectent les contraintes)
    st.subheader("🏆 Meilleurs modèles respectant les contraintes")
    valid_models = res_df[res_df["Respect_Contraintes"] == True].sort_values(by="R2", ascending=False)
    
    if not valid_models.empty:
        st.dataframe(valid_models.style.highlight_max(axis=0, subset=['R2']))
    else:
        st.error("Aucun modèle ne respecte vos contraintes de ressources. Essayez d'augmenter les limites dans la barre latérale.")

# ---------------------------------------------------------
# CHARGEMENT DES DONNÉES
# ---------------------------------------------------------
if dataset_option == "Diabetes (Régression)":
    data = load_diabetes()
    df = pd.concat([pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target, name='target')], axis=1)
    st.write("Dataset chargé : Diabetes (SKLearn)")
    run_automl_process(df)
elif dataset_option == "Upload CSV":
    file = st.sidebar.file_uploader("Importer votre fichier CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        st.write("Dataset importé :", file.name)
        run_automl_process(df)
    else:
        st.info("Veuillez importer un fichier CSV pour commencer.")