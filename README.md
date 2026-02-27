# 🤖 AutoML Systémique avec Optimisation Multi-Objectif

## Description
Framework AutoML qui optimise **simultanément** :
- 🎯 **Accuracy** (performance prédictive)
- ⚡ **Latence** (temps d'inférence en ms)
- 💾 **Mémoire** (RAM utilisée en MB)

**Méthodes utilisées :**
- **SMAC3** : optimisation bayésienne
- **Hyperband** : allocation budgétaire adaptative
- **Méta-apprentissage** : warm-start depuis les datasets précédents
- **Front de Pareto** : analyse des compromis multi-objectif

---

## 📁 Structure du projet

```
automl_project/
├── data/
│   └── datasets.py          → Chargement des 10 datasets OpenML
├── models/
│   └── search_space.py      → Espace de recherche (RF, SVM, XGBoost...)
├── evaluation/
│   └── evaluator.py         → Mesure accuracy + latence + mémoire
├── optimization/
│   ├── smac_optimizer.py    → SMAC3 + Hyperband
│   └── meta_learning.py     → Méta-apprentissage (warm-start)
├── analysis/
│   └── visualizations.py    → Pareto front, graphiques
├── app/
│   └── streamlit_app.py     → Interface web complète
├── run_pipeline.py          → Pipeline principal (tous les datasets)
├── notebook.ipynb           → Notebook pédagogique
└── requirements.txt         → Dépendances Python
```

---

## 🚀 Installation et lancement

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Tester rapidement (3 datasets, 10 trials)
```bash
python run_pipeline.py --quick
```

### 3. Pipeline complet (10 datasets)
```bash
python run_pipeline.py --n_trials 30
```

### 4. Lancer l'application Streamlit
```bash
streamlit run app/streamlit_app.py
```

### 5. Explorer le notebook pédagogique
```bash
jupyter notebook notebook.ipynb
```

---

## 📊 Datasets utilisés (OpenML)

| Dataset | Samples | Features | Classes |
|---------|---------|----------|---------|
| Diabetes | 768 | 8 | 2 |
| Wine Quality | 6497 | 11 | 7 |
| Breast Cancer | 699 | 9 | 2 |
| Iris | 150 | 4 | 3 |
| Credit Approval | 690 | 15 | 2 |
| Heart Disease | 303 | 13 | 2 |
| Titanic | 1309 | 13 | 2 |
| Vehicle Silhouettes | 946 | 18 | 4 |
| Bank Marketing | 45211 | 16 | 2 |
| Phoneme | 5404 | 5 | 2 |

---

## 🤖 Algorithmes explorés

- Random Forest
- Gradient Boosting
- Extra Trees
- SVM (Support Vector Machine)
- KNN (K-Nearest Neighbors)
- Logistic Regression
- Decision Tree
- XGBoost

---

## 🎯 Concepts clés

### SMAC3 (Sequential Model-Based Algorithm Configuration)
SMAC apprend un modèle probabiliste des performances en fonction des hyperparamètres. 
Il utilise ce modèle pour choisir intelligemment la prochaine configuration à évaluer 
(au lieu de tout tester comme GridSearch).

### Hyperband
Hyperband utilise un budget progressif :
1. Évalue beaucoup de configs avec peu de données (budget=10%)
2. Garde les meilleures et augmente le budget
3. La meilleure config reçoit tout le budget (100%)

### Méta-apprentissage
Mémorise les meilleures configs de chaque dataset. Sur un nouveau dataset :
1. Extrait des méta-features (taille, nb features, etc.)
2. Trouve les datasets similaires dans la base
3. Initialise SMAC avec leurs meilleures configs (warm-start)

### Front de Pareto
Une solution est Pareto-optimale si aucune autre solution n'est meilleure 
sur TOUS les objectifs simultanément. Le front de Pareto montre l'ensemble 
des meilleurs compromis possibles.

---

## ⚙️ Configuration

Modifier `run_pipeline.py` → `PIPELINE_CONFIG` :

```python
PIPELINE_CONFIG = {
    "n_trials":   30,    # ↑ = meilleur mais plus lent
    "min_budget": 0.1,   # fraction min des données (Hyperband)
    "max_budget": 1.0,   # fraction max des données
}
```

Modifier les poids des objectifs dans `optimization/smac_optimizer.py` :

```python
OBJECTIVE_WEIGHTS = {
    "error":   0.6,  # priorité à l'accuracy
    "latency": 0.2,  # importance de la vitesse
    "memory":  0.2,  # importance de la mémoire
}
```

---

## 📝 Auteurs
Projet AutoML - Cours de Machine Learning Avancé
