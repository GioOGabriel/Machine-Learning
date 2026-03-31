# Alzheimer Detection with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Machine Learning project for predicting Alzheimer's disease using clinical and demographic data. This project implements and compares 5 different classification algorithms with detailed analysis and visualizations.

## Table of Contents

- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results & Visualizations](#results--visualizations)
- [Technologies](#technologies)
- [Author](#author)

## Overview

This project aims to predict Alzheimer's disease probability using various Machine Learning techniques. Each algorithm is thoroughly analyzed in dedicated Jupyter notebooks with:

- Data preprocessing and exploratory analysis
- Hyperparameter tuning with GridSearchCV
- Cross-validation and performance metrics
- Visualizations (confusion matrices, ROC curves, feature importance)
- Model comparison and optimization

## Algorithms Implemented

| Algorithm | Description | Notebook |
|-----------|-------------|----------|
| **Decision Tree** | Rule-based classification model | `Alzheimer_DecisionTree.ipynb` |
| **K-Nearest Neighbors (KNN)** | Instance-based learning classifier | `Alzheimer_KNN.ipynb` |
| **Logistic Regression** | Linear model for binary classification | `Alzheimer_LogisticRegression.ipynb` |
| **Random Forest** | Ensemble of decision trees | `Alzheimer_RandomForest.ipynb` |
| **Support Vector Machine (SVM)** | Hyperplane-based classification | `Alzheimer_SVM.ipynb` |

Additional notebooks:
- `Alzheimer_Otimizado.ipynb` - Optimized model with best parameters
- `Comparacao_Algoritmos.ipynb` - Comparative analysis of all algorithms

## Project Structure

```
Machine-Learning/
├── notebooks/                    # Jupyter Notebooks with detailed analysis
│   ├── Alzheimer_DecisionTree.ipynb
│   ├── Alzheimer_KNN.ipynb
│   ├── Alzheimer_LogisticRegression.ipynb
│   ├── Alzheimer_RandomForest.ipynb
│   ├── Alzheimer_SVM.ipynb
│   ├── Alzheimer_Otimizado.ipynb
│   ├── Alzheimerquedeucerto.ipynb
│   └── Comparacao_Algoritmos.ipynb
│
├── models/                       # Trained models
│   ├── alzheimer_model.pkl       # Best trained model
│   └── alzheimer_scaler.pkl      # Feature scaler
│
├── data/                         # Dataset
│   └── alzheimers_disease_data.csv
│
├── visualizations/               # Generated plots and charts
│   ├── *_matriz_confusao.png     # Confusion matrices
│   ├── *_curvas_roc_pr.png       # ROC and PR curves
│   ├── *_importancia_features.png # Feature importance
│   └── comparacao_*.png          # Algorithm comparisons
│
├── alzheimer_predictor.py        # Prediction script
├── alzheimer_interface_terminal.py # Terminal interface
├── comparacao_algoritmos.py      # Algorithm comparison script
├── requirements.txt              # Project dependencies
└── README.md
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/GioOVander/Machine-Learning.git
cd Machine-Learning
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Terminal Interface
Run the interactive terminal interface for predictions:
```bash
python alzheimer_interface_terminal.py
```

### Algorithm Comparison
Compare all implemented algorithms:
```bash
python comparacao_algoritmos.py
```

### Jupyter Notebooks
Explore detailed analysis in the notebooks:
```bash
jupyter notebook notebooks/
```

## Dataset

The dataset contains clinical and demographic data for Alzheimer's disease prediction, including:

- Patient demographics (age, gender, education)
- Clinical measurements
- Cognitive assessments
- Lifestyle factors

## Results & Visualizations

The project generates comprehensive visualizations stored in the `visualizations/` folder:

### Confusion Matrices
Shows model prediction accuracy for each class.

### ROC and PR Curves
Evaluates model performance across different thresholds.

### Feature Importance
Identifies the most relevant features for prediction.

### Algorithm Comparison
Side-by-side comparison of all implemented algorithms.

| Visualization Type | Files |
|-------------------|-------|
| Confusion Matrices | `*_matriz_confusao.png` |
| ROC/PR Curves | `*_curvas_roc_pr.png` |
| Feature Importance | `*_importancia_features.png` |
| Class Distribution | `*_distribuicao_classes.png` |
| GridSearch Results | `*_comparacao_gridsearch.png` |

## Technologies

- **Python 3.x** - Programming language
- **Scikit-learn** - Machine Learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization
- **Jupyter Notebook** - Interactive development
- **Imbalanced-learn** - Handling imbalanced datasets (SMOTE, NearMiss)

## Author

**GioOVander**

- GitHub: [@GioOVander](https://github.com/GioOVander)

---

If you find this project useful, please consider giving it a star!
