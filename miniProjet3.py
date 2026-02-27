import streamlit as st # pour la creationd e l'interface utilisateur
import pandas as pd
import numpy as np
import base64 # pour le telechargment de fichiers
import plotly.graph_objects as go # pour la visualisation de données
import time # pour la gestion de temps d'execution
import sys 
import psutil # pour la getion de la memoire
import joblib # pour la sauvegarde et le chargment de modeles
from sklearn.metrics import mean_squared_error, r2_score # pour l'evaluation de models
from sklearn.model_selection import train_test_split # pour la division des données
from sklearn.datasets import load_diabetes # pour charger un dataset d'exemple
from sklearn.ensemble import RandomForestRegressor # pour le model de regression
import optuna # pour l'optimisation d'hyperparametres


#---------------------------------#
# En-tête du Projet
st.write("""
#  Framework AutoML Systémique
### Optimisation Multi-Objectif : Précision, Latence et Mémoire

Framework d'*AutoML* capable d'optimiser simultanément plusieurs critères. 
Ce framework utilise l'**Optimisation Bayésienne** pour trouver le meilleur équilibre entre la performance prédictive et l'efficacité technique

""")
#---------------------------------#

# --- SIDEBAR : CONFIGURATION DE L'ESPACE DE RECHERCHE ---
st.sidebar.header('Configuration de la Recherche') # pour la configuration de l'espace de recherche
split_size = st.sidebar.slider('Ratio Train/Test (%)', 10, 90, 80) # pour la division des données en train et test
n_trials = st.sidebar.number_input('Nombre d\'essais (Budget)', min_value=5, max_value=100, value=20) # pour Le nombre d'essais pour l'optimisation d'hyperparamètres

st.sidebar.subheader('Espace des Hyperparamètres') # pour la cinfiguration de l'espace des hyperparametres
n_estimators_max = st.sidebar.slider('Max n_estimators', 10, 500, 100) # pour le nombre d'arbres dans la foret aleatoire
max_depth_max = st.sidebar.slider('Max Depth', 1, 50, 20) # pour la profondeur maximale des arbres