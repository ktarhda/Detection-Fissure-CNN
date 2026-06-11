# 🏗️ Détection de Fissures sur Béton par Deep Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![EILCO](https://img.shields.io/badge/EILCO-Calais-blue?style=for-the-badge)

## 📌 Présentation du Projet
La maintenance des infrastructures en génie civil est un enjeu économique et sécuritaire majeur. Ce projet, réalisé dans le cadre du Master M2 à l'**EILCO (Calais)**, propose une approche automatisée basée sur le **Deep Learning** pour détecter l'apparition de fissures sur des surfaces en béton à partir d'analyses d'images.

L'objectif est de remplacer les inspections visuelles humaines (longues et subjectives) par un système d'intelligence artificielle hautement fiable, rapide et déployable à grande échelle.

---

## 📊 Base de Données (Dataset)
* **Source :** Surface Crack Detection Dataset (Kaggle).
* **Volume :** **40 000 images** haute résolution (20 000 avec fissures / 20 000 saines).
* **Traitement :** Prétraitement des images, normalisation et division des données en ensembles d'entraînement, de validation et de test.

---

## 🔬 Approches Méthodologiques & Architectures

Deux approches distinctes ont été implémentées et rigoureusement comparées :

### 1. Modèle CNN "From Scratch" (Propriétaire)
* Conception d'une architecture personnalisée de réseaux de neurones convolutifs.
* Alignement séquentiel de couches de **Convolution (Conv2D)**, de **Max-Pooling** pour la réduction de dimension, et de **Dropout** pour éviter le surapprentissage (*overfitting*).

### 2. Optimisation par Transfer Learning (ResNet50)
* Exploitation du modèle pré-entraîné **ResNet50** (sur ImageNet) pour bénéficier de l'extraction de caractéristiques complexes profondes.
* Ajustement fin (*Fine-Tuning*) des dernières couches denses pour l'adapter spécifiquement à la classification des fissures du béton.

---

## 🏆 Résultats & Performances du Modèle

L'utilisation du Transfer Learning a apporté des améliorations critiques tant sur le plan de la précision globale que sur l'efficacité informatique :

| Métrique / Indicateur | 🧪 CNN Propriétaire | 🚀 Optimisation ResNet50 |
| :--- | :---: | :---: |
| **Précision (Accuracy)** | **99.60 %** | **99.85 %** |
| **Temps d'Entraînement** | Baseline ($T$) | **Divisé par 3** ($\frac{T}{3}$) |
| **Convergence** | Standard | Ultra-rapide |

* **Optimiseur utilisé :** Adam (*Adaptive Moment Estimation*) pour une descente de gradient stable et optimisée.

---

## 🛠️ Technologies & Outils

* **Langage de programmation :** Python
* **Librairies de Deep Learning :** TensorFlow, Keras
* **Traitement de données & Visualisation :** NumPy, Pandas, Matplotlib, Seaborn
* **Environnement de développement :** Jupyter Notebook / VS Code

---

## 📂 Structure des Fichiers (Recommandée)
```text
├── datasets/                 # Images d'entraînement et de test (exclues de Git)
├── notebooks/                # Notebooks Jupyter contenant le code de modélisation
├── src/                      # Scripts Python (.py) pour l'entraînement du modèle
├── .gitignore                # Fichier pour ignorer les gros fichiers (ZIP, modèles .keras)
└── README.md                 # Présentation du projet