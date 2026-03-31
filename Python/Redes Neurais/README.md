# Deteccao de Alzheimer com Machine Learning

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

Projeto de Machine Learning para prever a probabilidade de Alzheimer com base em dados clinicos e demograficos. Implementa e compara 5 algoritmos de classificacao diferentes com analises detalhadas e visualizacoes.

## Indice

- [Visao Geral](#visao-geral)
- [Algoritmos Implementados](#algoritmos-implementados)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalacao](#instalacao)
- [Como Usar](#como-usar)
- [Dataset](#dataset)
- [Resultados e Visualizacoes](#resultados-e-visualizacoes)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Autor](#autor)

## Visao Geral

Este projeto tem como objetivo prever a probabilidade de Alzheimer utilizando diversas tecnicas de Machine Learning. Cada algoritmo e analisado em detalhes em Jupyter Notebooks dedicados com:

- Pre-processamento e analise exploratoria de dados
- Ajuste de hiperparametros com GridSearchCV
- Validacao cruzada e metricas de desempenho
- Visualizacoes (matrizes de confusao, curvas ROC, importancia de features)
- Comparacao e otimizacao de modelos

## Algoritmos Implementados

| Algoritmo | Descricao | Notebook |
|-----------|-----------|----------|
| **Decision Tree** | Modelo de classificacao baseado em regras | `Alzheimer_DecisionTree.ipynb` |
| **KNN (K-Nearest Neighbors)** | Classificacao baseada em vizinhos proximos | `Alzheimer_KNN.ipynb` |
| **Logistic Regression** | Modelo linear para classificacao binaria | `Alzheimer_LogisticRegression.ipynb` |
| **Random Forest** | Ensemble de arvores de decisao | `Alzheimer_RandomForest.ipynb` |
| **SVM (Support Vector Machine)** | Classificacao com hiperplanos | `Alzheimer_SVM.ipynb` |

Notebooks adicionais:
- `Alzheimer_Otimizado.ipynb` - Modelo otimizado com melhores parametros
- `Comparacao_Algoritmos.ipynb` - Analise comparativa de todos os algoritmos

## Estrutura do Projeto

```
Machine-Learning/
├── notebooks/                    # Jupyter Notebooks com analises
│   ├── Alzheimer_DecisionTree.ipynb
│   ├── Alzheimer_KNN.ipynb
│   ├── Alzheimer_LogisticRegression.ipynb
│   ├── Alzheimer_RandomForest.ipynb
│   ├── Alzheimer_SVM.ipynb
│   ├── Alzheimer_Otimizado.ipynb
│   ├── Alzheimerquedeucerto.ipynb
│   └── Comparacao_Algoritmos.ipynb
│
├── models/                       # Modelos treinados
│   ├── alzheimer_model.pkl       # Melhor modelo treinado
│   └── alzheimer_scaler.pkl      # Scaler das features
│
├── data/                         # Dataset
│   └── alzheimers_disease_data.csv
│
├── visualizations/               # Graficos gerados
│   ├── *_matriz_confusao.png     # Matrizes de confusao
│   ├── *_curvas_roc_pr.png       # Curvas ROC e PR
│   ├── *_importancia_features.png # Importancia das features
│   └── comparacao_*.png          # Comparacoes entre algoritmos
│
├── alzheimer_predictor.py        # Script de predicao
├── alzheimer_interface_terminal.py # Interface de terminal
├── comparacao_algoritmos.py      # Script de comparacao
├── requirements.txt              # Dependencias do projeto
└── README.md
```

## Instalacao

1. **Clone o repositorio**
```bash
git clone https://github.com/GioOGabriel/Machine-Learning.git
cd Machine-Learning
```

2. **Crie um ambiente virtual (recomendado)**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. **Instale as dependencias**
```bash
pip install -r requirements.txt
```

## Como Usar

### Interface de Terminal
Execute a interface interativa para predicoes:
```bash
python alzheimer_interface_terminal.py
```

### Comparacao de Algoritmos
Compare todos os algoritmos implementados:
```bash
python comparacao_algoritmos.py
```

### Jupyter Notebooks
Explore as analises detalhadas nos notebooks:
```bash
jupyter notebook notebooks/
```

## Dataset

O dataset contem dados clinicos e demograficos para predicao de Alzheimer, incluindo:

- Dados demograficos dos pacientes (idade, genero, educacao)
- Medicoes clinicas
- Avaliacoes cognitivas
- Fatores de estilo de vida

## Resultados e Visualizacoes

O projeto gera visualizacoes completas armazenadas na pasta `visualizations/`:

### Matrizes de Confusao
Mostra a precisao das predicoes do modelo para cada classe.

### Curvas ROC e PR
Avalia o desempenho do modelo em diferentes limiares.

### Importancia de Features
Identifica as features mais relevantes para a predicao.

### Comparacao de Algoritmos
Comparacao lado a lado de todos os algoritmos implementados.

| Tipo de Visualizacao | Arquivos |
|---------------------|----------|
| Matrizes de Confusao | `*_matriz_confusao.png` |
| Curvas ROC/PR | `*_curvas_roc_pr.png` |
| Importancia de Features | `*_importancia_features.png` |
| Distribuicao de Classes | `*_distribuicao_classes.png` |
| Resultados GridSearch | `*_comparacao_gridsearch.png` |

## Tecnologias Utilizadas

- **Python 3.x** - Linguagem de programacao
- **Scikit-learn** - Biblioteca de Machine Learning
- **Pandas** - Manipulacao de dados
- **NumPy** - Computacao numerica
- **Matplotlib** - Visualizacao de dados
- **Seaborn** - Visualizacao estatistica
- **Jupyter Notebook** - Desenvolvimento interativo
- **Imbalanced-learn** - Tratamento de datasets desbalanceados (SMOTE, NearMiss)

## Autor

**GioOGabriel**

- GitHub: [@GioOGabriel](https://github.com/GioOGabriel)
