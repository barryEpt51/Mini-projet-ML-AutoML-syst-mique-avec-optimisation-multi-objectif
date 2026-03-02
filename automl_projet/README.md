# AutoML Systémique & Optimisation Multi-Objectif
### Sujet 3 — Apprentissage Automatique 2025-2026

---

## Objectif

Framework AutoML optimisant **simultanément 3 objectifs** :

| Objectif | Mesure | Direction |
|---|---|---|
| Accuracy | `accuracy_score` (classif) / R² (régression) | Maximiser ↑ |
| Latence | Médiane sur 5 inférences (ms) | Minimiser ↓ |
| Mémoire | Pic RAM via `tracemalloc` (MB) | Minimiser ↓ |

---

## Architecture

```
automl_projet/
├── app/
│   └── streamlit_app.py      # Interface web (Streamlit)
├── models/
│   └── search_space.py       # Espace de recherche ConfigSpace (8 algos)
├── evaluation/
│   └── evaluator.py          # Mesure accuracy / latence / mémoire
├── optimization/
│   ├── smac_optimizer.py     # SMAC + MultiFidelityFacade + Hyperband natif
│   └── meta_learning.py      # Méta-apprentissage & warm start
├── analysis/
│   └── visualizations.py     # Pareto 3D, convergence, heatmap, radar
├── data/
│   └── datasets.py           # 10 datasets (sklearn + OpenML + upload)
├── run_pipeline.py            # Pipeline CLI complet
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Lancement

### Interface web (recommandée)
```bash
streamlit run app/streamlit_app.py
```

### Pipeline CLI
```bash
# Tous les 10 datasets
python run_pipeline.py

# Mode rapide (3 datasets, 10 trials)
python run_pipeline.py --quick

# Datasets spécifiques
python run_pipeline.py --datasets breast_cancer iris wine --n_trials 20
```

---

## Points clés techniques

### SMAC + Hyperband natif
```python
from smac import MultiFidelityFacade, Scenario
from smac.intensifier import Hyperband

smac = MultiFidelityFacade(
    scenario        = scenario,
    target_function = self._target_function,
    intensifier     = Hyperband(scenario, eta=3),  # natif, pas manuel
)
```

### Scalarisation multi-objectif
SMAC optimise un seul objectif scalaire (combinaison pondérée des 3) :
```
coût = w_acc*(1-accuracy) + w_lat*(latence/500) + w_mem*(mémoire/512)
```
Les poids sont réglables dans la sidebar de l'interface.

### Méta-apprentissage (warm start)
- Extraction de méta-features du dataset (taille, n_features, entropie…)
- Distance euclidienne normalisée entre datasets
- Injection des meilleures configs des datasets similaires comme point de départ SMAC

### 8 algorithmes explorés
Random Forest, Gradient Boosting, Extra Trees, SVM, KNN,
Modèle Linéaire (Ridge/LogReg), Decision Tree, XGBoost

### 10 datasets
- **sklearn** : diabetes, breast_cancer, iris, wine, digits
- **OpenML** : credit, heart, vehicle, bank, phoneme
