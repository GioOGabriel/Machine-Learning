# Deteccao de Alzheimer com Machine Learning

Este projeto utiliza diferentes algoritmos de Machine Learning para prever a probabilidade de Alzheimer com base em dados clinicos e demograficos.

## Estrutura do Projeto

```
Redes Neurais/
├── notebooks/           # Jupyter Notebooks com analises
│   ├── Alzheimer_DecisionTree.ipynb
│   ├── Alzheimer_KNN.ipynb
│   ├── Alzheimer_LogisticRegression.ipynb
│   ├── Alzheimer_RandomForest.ipynb
│   ├── Alzheimer_SVM.ipynb
│   ├── Alzheimer_Otimizado.ipynb
│   ├── Alzheimerquedeucerto.ipynb
│   └── Comparacao_Algoritmos.ipynb
├── models/              # Modelos treinados salvos
│   ├── alzheimer_model.pkl
│   └── alzheimer_scaler.pkl
├── data/                # Dados utilizados
│   └── alzheimers_disease_data.csv
├── visualizations/      # Graficos e visualizacoes geradas
├── alzheimer_predictor.py        # Script de predicao
├── alzheimer_interface_terminal.py  # Interface de terminal
├── comparacao_algoritmos.py      # Comparacao entre algoritmos
└── requirements.txt     # Dependencias do projeto
```

## Algoritmos Implementados

- **Decision Tree (Arvore de Decisao)** - Modelo de classificacao baseado em regras
- **KNN (K-Nearest Neighbors)** - Classificacao baseada em vizinhos proximos
- **Logistic Regression (Regressao Logistica)** - Modelo linear para classificacao
- **Random Forest (Floresta Aleatoria)** - Ensemble de arvores de decisao
- **SVM (Support Vector Machine)** - Classificacao com hiperplanos

## Como Usar

### Instalacao

```bash
pip install -r requirements.txt
```

### Executar Interface de Terminal

```bash
python alzheimer_interface_terminal.py
```

### Executar Comparacao de Algoritmos

```bash
python comparacao_algoritmos.py
```

## Dataset

O dataset utilizado contem dados clinicos e demograficos de pacientes para predicao de Alzheimer.

## Visualizacoes

Os graficos gerados pelos notebooks estao na pasta `visualizations/`, incluindo:
- Matrizes de confusao
- Curvas ROC e PR
- Importancia de features
- Comparacao entre algoritmos

## Tecnologias Utilizadas

- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Autor

GioOVander
