# Revised Cost Prediction — ML Pipeline for Building Permits

Pipeline de **Machine Learning** pour la prédiction du **Revised Cost** (coût révisé) à partir de données de permis de construction : régression XGBoost, PCA, clustering KMeans et visualisations.

---

## Sommaire

1. [Vue d'ensemble](#1-vue-densemble)
2. [Structure du projet](#2-structure-du-projet)
3. [Données](#3-données)
4. [Installation et dépendances](#4-installation-et-dépendances)
5. [Pipeline (résumé)](#5-pipeline-résumé)
6. [Résultats et captures](#6-résultats-et-captures)
7. [Exécution](#7-exécution)
8. [Pousser sur Git](#8-pousser-sur-git)

---

## 1. Vue d'ensemble

| Élément | Détail |
|--------|--------|
| **Objectif** | Prédire **Revised Cost (log)** à partir des caractéristiques des permis |
| **Modèle** | XGBoost (régression) sur 5 composantes PCA |
| **Prétraitement** | Normalisation colonnes (DataSF), Lat/Lon, encodage, outliers IQR, log des coûts, StandardScaler, PCA |
| **Clustering** | KMeans sur X_pca ; Silhouette score (échantillon 10k pour gros volumes) |
| **Environnement** | Python 3.x ; compatible Cursor / VS Code et Google Colab |

---

## 2. Structure du projet

```
ML/
├── ml.ipynb                    # Notebook principal (analyse + modélisation)
├── README.md                   # Ce fichier
├── requirements.txt            # Dépendances pip
├── install_deps.py             # Script d'installation des paquets
├── data_avec_material_type.csv # Données d'entrée (à placer ici)
├── assets/                     # Graphiques exportés (générés à l'exécution)
│   ├── 01_distributions.png   # Distributions des variables numériques
│   ├── 02_pca_variance.png     # Variance expliquée cumulée (PCA)
│   ├── 03_y_test_vs_y_pred.png # Prédictions vs réelles
│   ├── 04_elbow_method.png     # Méthode du coude (KMeans)
│   └── 05_clustering_pca.png  # Clusters sur les 2 premières composantes PCA
├── xgb_model.pkl               # Modèle XGBoost (généré)
├── scaler.pkl                  # StandardScaler (généré)
└── pca.pkl                     # PCA (généré)
```

---

## 3. Données

- **Fichier** : `data_avec_material_type.csv`
- **Colonnes utilisées** : Permit Type Definition, Estimated Cost, Revised Cost, Materials Cost, Labor Cost, Neighborhoods - Analysis Boundaries, Location (ou Latitude/Longitude), Material_Type, etc.
- Le notebook inclut une **cellule de normalisation** pour adapter les données DataSF (renommage colonnes, parsing `Location` type POINT, création de Materials/Labor Cost si absents, coercion numérique, `dropna` ciblé).

---

## 4. Installation et dépendances

**Option A — Script d’installation (recommandé, Cursor / VS Code)**

```bash
python install_deps.py
```

**Option B — pip**

```bash
pip install -r requirements.txt
```

**Paquets principaux** : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `joblib`.

---

## 5. Pipeline (résumé)

1. **Chargement** : `pd.read_csv("data_avec_material_type.csv")`
2. **Normalisation** : renommage colonnes, Lat/Lon depuis `Location`, coûts numériques, colonnes optionnelles, `dropna` sur colonnes clés
3. **Nettoyage** : suppression colonnes inutiles (Permit Number, Description, etc.)
4. **Encodage** : LabelEncoder sur Permit Type Definition et Neighborhoods → Neighborhood
5. **Outliers** : IQR sur Estimated Cost, Revised Cost, Materials Cost, Labor Cost
6. **Feature engineering** : `Estimated Cost_log`, `Revised Cost_log` (log1p)
7. **Variables numériques** : Materials Cost, Labor Cost, Estimated Cost_log, Permit Type Definition, Neighborhood, Latitude, Longitude
8. **StandardScaler** puis **PCA (5 composantes)** → `X_pca`
9. **Régression** : train_test_split, XGBRegressor, MSE / R²
10. **Clustering** : KMeans (3 clusters), Silhouette score (sample_size=10000 pour gros jeux)
11. **Sauvegarde** : joblib (model, scaler, pca)
12. **Export des figures** : les graphiques sont enregistrés dans `assets/` à l’exécution

---

## 6. Résultats et captures

Après exécution du notebook, les graphiques suivants sont générés dans `assets/` :

| Figure | Fichier | Description |
|--------|---------|-------------|
| Distributions | `assets/01_distributions.png` | Histogrammes des variables numériques |
| PCA | `assets/02_pca_variance.png` | Variance expliquée cumulée (choix du nombre de composantes) |
| Régression | `assets/03_y_test_vs_y_pred.png` | Valeurs réelles vs prédites (droite idéale en rouge) |
| Elbow | `assets/04_elbow_method.png` | Méthode du coude pour le choix de k (KMeans) |
| Clustering | `assets/05_clustering_pca.png` | Nuage de points PCA (composantes 1 et 2) coloré par cluster |

Exemple d’intégration en local :

```markdown
![Distributions](assets/01_distributions.png)
![PCA](assets/02_pca_variance.png)
![Régression](assets/03_y_test_vs_y_pred.png)
![Elbow](assets/04_elbow_method.png)
![Clustering](assets/05_clustering_pca.png)
```

*(Les images s’affichent une fois le notebook exécuté et les fichiers présents dans `assets/`.)*

---

**Captures (après exécution du notebook)** :

| | |
|---|---|
| ![Distributions](assets/01_distributions.png) | ![PCA](assets/02_pca_variance.png) |
| *Distributions des variables* | *Variance expliquée PCA* |
| ![Régression](assets/03_y_test_vs_y_pred.png) | ![Elbow](assets/04_elbow_method.png) |
| *y_test vs y_pred* | *Méthode du coude (KMeans)* |
| ![Clustering](assets/05_clustering_pca.png) | |
| *Clusters sur PCA (2 composantes)* | |

---

## 7. Exécution

1. Placer **data_avec_material_type.csv** à la racine du dossier `ML` (même niveau que `ml.ipynb`).
2. Installer les dépendances : `python install_deps.py` ou `pip install -r requirements.txt`.
3. Ouvrir **ml.ipynb** dans Cursor, VS Code ou Jupyter.
4. Choisir le noyau Python où les paquets sont installés.
5. Lancer **Run All** (ou exécuter les cellules dans l’ordre).

Les artefacts (`.pkl`) et les figures dans `assets/` sont créés à la fin des cellules concernées. La cellule Google Colab (Drive) est ignorée en local grâce à un `try/except ModuleNotFoundError`.

---

## 8. Pousser sur Git

- **Nom de dépôt suggéré** : `revised-cost-prediction-ml` ou `building-permits-ml-pipeline`
- **.gitignore** : exclure `*.csv`, `*.pkl`, `.venv/`, `venv/`, `.ipynb_checkpoints/` (adapter si vous versionnez les CSV ou modèles).

**Résumé du commit (Summary)** — à coller dans le champ *Summary (required)* :

```
Revised Cost Prediction: XGBoost, PCA, KMeans pipeline for building permits
```

**Description du commit (Description)** — à coller dans le champ *Description* :

```
- ML pipeline to predict Revised Cost (log) from building permit data
- Preprocessing: column normalization (DataSF), Lat/Lon parsing, encoding, IQR outliers, log transform, StandardScaler, PCA (5 components)
- Model: XGBRegressor on PCA features; metrics: MSE, R²
- Clustering: KMeans on PCA space; silhouette score with sample_size for large datasets
- Exported figures in assets/ (distributions, PCA variance, y_test vs y_pred, elbow method, clustering scatter)
- Compatible Cursor/VS Code and Colab; install_deps.py and requirements.txt for setup
- README with structure, run instructions, and Git push steps
```

---

## Auteur et licence

Projet portfolio ML. Adapter la licence selon l’usage (ex. MIT).
