# Detecção de Alzheimer com Machine Learning

Projeto de análise comparativa de algoritmos de machine learning para detecção de doença de Alzheimer.

## 🚀 Quick Start - Aplicação Streamlit

### Forma Mais Fácil (Windows)
1. Duplo clique em `executar_streamlit.bat`
2. A aplicação abrirá automaticamente em `http://localhost:8501`

### Outras Opções
```bash
# Opção 1: Usar launcher Python (todos os SOs)
python executar_streamlit.py

# Opção 2: Manual
streamlit run app_streamlit.py
```

**Nota:** Na primeira execução, o modelo será treinado automaticamente (~1 minuto)

---

## 📁 Estrutura do Projeto

\\\
Redes Neurais/
├── data/                    # Dados e CSVs
│   ├── alzheimers_disease_data.csv
│   ├── resultados_base.csv
│   ├── resultados_otimizados.csv
│   └── resultados_completos.csv
│
├── models/                  # Modelos treinados
│   ├── alzheimer_model.pkl
│   └── alzheimer_scaler.pkl
│
├── notebooks/               # Jupyter Notebooks
│   ├── Alzheimerquedeucerto.ipynb
│   ├── Alzheimer_DecisionTree.ipynb
│   ├── Alzheimer_KNN.ipynb
│   ├── Alzheimer_LogisticRegression.ipynb
│   ├── Alzheimer_Otimizado.ipynb
│   ├── Alzheimer_RandomForest.ipynb
│   ├── Alzheimer_SVM.ipynb
│   └── Comparacao_Algoritmos.ipynb
│
├── visualizations/          # Gráficos e imagens
│   ├── comparacao_metricas.png
│   ├── ranking_f1score.png
│   ├── curvas_roc_comparacao.png
│   ├── heatmap_melhoria.png
│   ├── tempo_treinamento.png
│   ├── radar_desempenho.png
│   ├── scatter_acuracia_f1.png
│   ├── boxplot_metricas.png
│   ├── summary_performance.png
│   ├── matrizes_confusao_modelos_base.png
│   ├── matrizes_confusao_modelos_otimizados.png
│   └── balanceamento_smote.png
│
├── relatorios/              # Relatórios HTML
│   └── comparacao_algoritmos.html
│
├── scripts/                 # Scripts Python auxiliares
│   ├── comparacao_algoritmos.py
│   ├── gerar_relatorio_html.py
│   ├── visualizador_interativo.py
│   ├── sumario_final.py
│   └── treinar_e_exportar_modelo.py  # NOVO: Exporta modelo para produção
│
├── docs/                    # Documentação
│   ├── README.md
│   ├── COMPARACAO_COMPLETA.md
│   ├── ANALISE_DETALHADA.txt
│   ├── GUIA_RAPIDO.txt
│   └── CORRECAO_SMOTE.md
│
├── .gitignore
├── requirements.txt
├── app_streamlit.py                 # NOVO: Aplicação web com Streamlit
├── executar_streamlit.bat           # NOVO: Launcher para Windows
├── executar_streamlit.py            # NOVO: Launcher multiplataforma
├── README_PROJETO.md                # Este arquivo
├── README_STREAMLIT.md              # Documentação da aplicação
├── alzheimer_interface_terminal.py
├── alzheimer_predictor.py
└── venv/                    # Ambiente virtual Python

\\\

## 🚀 Como Usar

### ⭐ NOVO: Aplicação Streamlit (Recomendado)

**Opção 1 (Windows - Mais Fácil):**
- Duplo clique em `executar_streamlit.bat`

**Opção 2 (Todos os SOs):**
```bash
python executar_streamlit.py
```

**Opção 3 (Manual):**
```bash
streamlit run app_streamlit.py
```

Acesse em: `http://localhost:8501`

---

### Alternativas Legadas

#### 1. Visualizar o Relatório HTML

Abra o arquivo com clique duplo ou arraste para o navegador:
```
relatorios/comparacao_algoritmos.html
```

#### 2. Executar os Scripts de Análise

```bash
python scripts/comparacao_algoritmos.py
python scripts/gerar_relatorio_html.py
```

#### 3. Usar a Interface Terminal

Execute a interface interativa:
```bash
python alzheimer_predictor.py
python alzheimer_interface_terminal.py
```
relatorios/comparacao_algoritmos.html
\\\

### 2. Executar os Scripts

Certifique-se de ter o ambiente virtual ativado:
\\\ash
source venv/Scripts/activate  # Windows
\\\

Execute os scripts:
\\\ash
python scripts/comparacao_algoritmos.py
python scripts/gerar_relatorio_html.py
\\\

### 3. Usar a Interface Terminal

Execute a interface interativa:
\\\ash
python alzheimer_predictor.py
python alzheimer_interface_terminal.py
\\\

## 📊 Algoritmos Comparados

- **MLP** (Multi-Layer Perceptron) - Rede Neural
- **Decision Tree** - Árvore de Decisão
- **KNN** - K-Nearest Neighbors
- **Logistic Regression** - Regressão Logística
- **Random Forest** - Floresta Aleatória
- **SVM** - Support Vector Machine

## 📈 Principais Resultados

| Algoritmo | Acurácia | F1-Score | Tempo (s) |
|-----------|----------|----------|-----------|
| Random Forest | 91.22% | 90.80% | 0.60 |
| SVM | 85.90% | 86.08% | 0.85 |
| Decision Tree | 83.02% | 83.14% | 0.04 |
| Logistic Regression | 82.45% | 83.06% | 0.01 |
| MLP | 82.30% | 82.60% | 4.46 |
| KNN | 71.37% | 75.46% | 0.00 |

## 🔧 Requisitos

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

Ver \equirements.txt\ para lista completa.

## 📝 Documentação

- **docs/README.md** - Informações gerais do projeto
- **docs/COMPARACAO_COMPLETA.md** - Análise detalhada dos algoritmos
- **docs/ANALISE_DETALHADA.txt** - Resultados técnicos completos
- **docs/GUIA_RAPIDO.txt** - Guia rápido para iniciantes

## 👨‍💻 Autor

Projeto de Machine Learning para Detecção de Alzheimer

## 📅 Data de Criação

27/05/2026
