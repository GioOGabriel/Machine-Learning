# 🧠 Comparação de Algoritmos - Detecção de Alzheimer

## 📌 Visão Geral

Este projeto implementa uma **análise comparativa completa** de **6 algoritmos de Machine Learning** para diagnóstico automático de **Alzheimer**, comparando versões **BASE** (padrão) e **OTIMIZADA** (com hiperparâmetros ajustados).

### ⚡ Resultado Principal
**Random Forest alcançou 91.22% de acurácia**, sendo o melhor escolha para produção.

---

## 🚀 Quick Start (3 passos)

### 1. Instale as dependências
```bash
pip install -r requirements.txt
```

### 2. Execute a análise
```bash
python comparacao_algoritmos.py
```

### 3. Visualize os resultados
```bash
# Opção A: Ver relatório HTML
# Abra "relatorios/comparacao_algoritmos.html" no navegador

# Opção B: Visualizador interativo
python visualizador_interativo.py

# Opção C: Sumário em texto
python sumario_final.py
```

---

## 📊 O Que Você Recebe

### 📁 Arquivos Gerados

#### Dados (CSV)
- `resultados_base.csv` - Métricas dos 6 modelos base
- `resultados_otimizados.csv` - Métricas dos 6 modelos otimizados
- `resultados_completos.csv` - Dataset consolidado

#### Gráficos (PNG)
1. **comparacao_metricas.png** - Comparação lado a lado das 4 métricas principais
2. **ranking_f1score.png** - Ranking final dos algoritmos
3. **curvas_roc_comparacao.png** - Curvas ROC para análise de discriminação
4. **heatmap_melhoria.png** - Visualização de melhoria percentual
5. **tempo_treinamento.png** - Comparação de velocidade
6. **radar_desempenho.png** - Gráfico radar multidimensional
7. **scatter_acuracia_f1.png** - Análise de dispersão
8. **boxplot_metricas.png** - Distribuição estatística
9. **summary_performance.png** - Sumário visual completo
10. **matrizes_confusao_modelos_base.png** - Matrizes de confusão (base)
11. **matrizes_confusao_modelos_otimizados.png** - Matrizes de confusão (otimizado)

#### Relatórios
- `relatorios/comparacao_algoritmos.html` - Relatório interativo visual

#### Documentação
- `COMPARACAO_COMPLETA.md` - Documentação técnica completa
- `ANALISE_DETALHADA.txt` - Análise profunda com recomendações
- `README.md` - Este arquivo

---

## 📈 Resultados Resumidos

### Ranking Final (F1-Score)

```
🥇 1º - Random Forest:     0.9080 (91.22% acurácia)
🥈 2º - SVM:               0.8608 (85.90% acurácia)
🥉 3º - Decision Tree:     0.8588 (86.04% acurácia - otimizado)
4º   - MLP:               0.8354 (83.17% acurácia - otimizado)
5º   - Logistic Reg:      0.8306 (82.45% acurácia)
6º   - KNN:               0.7687 (72.37% acurácia - otimizado)
```

### Performance por Critério

| Métrica | Melhor Algoritmo | Valor |
|---------|------------------|-------|
| **Acurácia** | Random Forest | 91.22% |
| **F1-Score** | Random Forest | 0.9080 |
| **Precisão** | Random Forest | 95.25% |
| **Recall** | KNN (otimizado) | 91.93% |
| **AUC-ROC** | Random Forest | 0.9605 |
| **Velocidade** | KNN | 0.0007s |

---

## 🎯 Recomendações por Caso de Uso

### 1️⃣ Produção Clínica (Melhor Acurácia)
```
Algoritmo: Random Forest (BASE)
Acurácia: 91.22%
F1-Score: 0.9080
Tempo: 0.60 segundos
Recomendação: USE ESTE!
```

### 2️⃣ Screening Massivo (Alta Sensibilidade)
```
Algoritmo: KNN (OTIMIZADO)
Recall: 91.93% (detecta 92% dos casos)
Tempo: 0.0007s (< 1ms) ⚡
Recomendação: Para aplicações de screening
```

### 3️⃣ Explicabilidade (Interpretável)
```
Algoritmo: Decision Tree (OTIMIZADO)
Acurácia: 86.04%
F1-Score: 0.8588
Tempo: 0.032s
Recomendação: Fácil de explicar ao paciente
```

### 4️⃣ Trade-off Ideal
```
Algoritmo: SVM (BASE)
Acurácia: 85.90%
AUC-ROC: 0.9362
Tempo: 0.85s
Recomendação: Bom em tudo
```

---

## 🔬 Algoritmos Comparados

### 1. Random Forest ⭐ VENCEDOR
- Floresta Aleatória (100-200 árvores)
- Melhor acurácia geral
- Bom em dados desbalanceados
- Menos interpretável

### 2. Support Vector Machine (SVM)
- Separador de margens máximas
- Bom para alta dimensionalidade
- AUC-ROC excelente
- Tempo computacional maior

### 3. Decision Tree 📚 INTERPRETÁVEL
- Árvore de decisão (profundidade 10)
- Melhor melhoria com otimização (+3.29%)
- Muito rápida (35ms)
- Altamente interpretável

### 4. Multi-Layer Perceptron (MLP)
- Rede neural com 3 camadas
- Bom desempenho
- Early stopping implementado
- Menos estável que outros

### 5. Logistic Regression
- Regressão logística simples
- Baseline de referência
- Extremamente rápido (5ms)
- Performance modesta

### 6. K-Nearest Neighbors (KNN) ⚡ RÁPIDO
- Classificação por vizinhos
- Mais rápido do mercado (< 1ms)
- Alto recall (91.93%)
- Menor acurácia geral

---

## 📋 Dataset

### Origem
Alzheimer's Disease Detection

### Características
- **Amostras**: 2.149 registros
- **Features**: 33 características médicas
- **Classes**: Binária (Saudável vs Alzheimer)
- **Distribuição Original**: 64.6% saudável, 35.4% Alzheimer
- **Balanceamento**: SMOTE (50.0% cada)

### Divisão
- Treino: 75% (2.083 amostras)
- Teste: 25% (695 amostras)
- Validação: 5-fold Stratified K-Fold

---

## 🛠️ Técnicas Aplicadas

### Pré-processamento
✅ **SMOTE** - Balanceamento de classes  
✅ **StandardScaler** - Normalização (média=0, std=1)  
✅ **Stratified K-Fold** - Validação cruzada respeitando distribuição  

### Otimização
✅ **GridSearchCV** - Busca exaustiva de hiperparâmetros  
✅ **5-Fold CV** - Validação cruzada robusta  
✅ **F1-Score** - Métrica de otimização principal  

---

## 🔍 Métricas Explicadas

### Acurácia
- % de previsões corretas (TP + TN) / Total
- **Uso**: Problemas balanceados
- **Este projeto**: Suplementar

### Precisão
- De quantos Alzheimer previstos, quantos estão corretos
- **Fórmula**: TP / (TP + FP)
- **Importa**: Evitar alarmes falsos

### Recall (Sensibilidade)
- De todos os Alzheimer reais, quantos foram detectados
- **Fórmula**: TP / (TP + FN)
- **CRÍTICO**: Não deixar passar casos reais

### F1-Score ⭐ PRINCIPAL
- Média harmônica entre Precisão e Recall
- **Fórmula**: 2 × (Precisão × Recall) / (Precisão + Recall)
- **Melhor para**: Este problema (balanceia ambos)

### AUC-ROC
- Área sob a curva ROC
- **Significa**: Capacidade de discriminação entre classes
- **Intervalo**: 0 (péssimo) a 1 (perfeito)

---

## 💻 Estrutura de Arquivos

```
.
├── comparacao_algoritmos.py          # ⭐ Script principal
├── gerar_relatorio_html.py           # Gera relatório HTML
├── sumario_final.py                  # Sumário em texto
├── visualizador_interativo.py        # Visualizador GUI
├── requirements.txt                  # Dependências
│
├── data/
│   └── alzheimers_disease_data.csv   # Dataset original
│
├── resultados_base.csv               # Métricas base
├── resultados_otimizados.csv         # Métricas otimizadas
├── resultados_completos.csv          # Dataset consolidado
│
├── *.png                             # 11 gráficos gerados
│
├── relatorios/
│   └── comparacao_algoritmos.html    # Relatório visual
│
└── DOCUMENTACAO/
    ├── COMPARACAO_COMPLETA.md        # Guia técnico completo
    ├── ANALISE_DETALHADA.txt         # Análise profunda
    └── README.md                     # Este arquivo
```

---

## ⚙️ Configuração

### Parâmetros Base
Editáveis em `comparacao_algoritmos.py`, função `get_algoritmos_base()`:

```python
'Random Forest': RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

### Parâmetros Otimizados
Editáveis em `comparacao_algoritmos.py`, função `get_algoritmos_otimizados()`:

```python
'Random Forest': RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
```

### Grid de Busca
Editável em `comparacao_algoritmos.py`, função `get_param_grids()`:

```python
'Random Forest': {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```

---

## 📚 Scripts Disponíveis

### 1. `comparacao_algoritmos.py` ⭐ PRINCIPAL
Executa a análise completa:
```bash
python comparacao_algoritmos.py
```

Duração: ~15 minutos  
Saída: 11 gráficos + 3 CSVs + 1 HTML

### 2. `gerar_relatorio_html.py`
Gera relatório HTML interativo:
```bash
python gerar_relatorio_html.py
```

Abre em navegador: `relatorios/comparacao_algoritmos.html`

### 3. `sumario_final.py`
Exibe sumário em texto:
```bash
python sumario_final.py
```

### 4. `visualizador_interativo.py`
Interface para visualizar gráficos:
```bash
python visualizador_interativo.py
```

Menu interativo para explorar resultados.

---

## 🐛 Resolução de Problemas

### Erro: "ModuleNotFoundError: matplotlib"
```bash
pip install matplotlib seaborn
```

### Erro: "Dataset não encontrado"
1. Coloque o arquivo CSV em `data/alzheimers_disease_data.csv`
2. Ou selecione via diálogo de arquivo

### Erro: "SMOTE falhou"
- Verifique distribuição de classes
- Instale: `pip install imbalanced-learn`

### Lentidão durante GridSearchCV
- Reduzir `cv` de 5 para 3
- Reduzir número de parâmetros no grid
- Usar `n_jobs=-1` (já está implementado)

---

## 📊 Interpretação dos Gráficos

### Comparação de Métricas
4 subgráficos mostrando Acurácia, Precisão, Recall e F1-Score

### Ranking F1-Score
Barras horizontais ordenadas - Random Forest está no topo

### Curvas ROC
Quanto mais próximo do canto superior esquerdo, melhor

### Heatmap de Melhoria
Verde = melhoria positiva  
Vermelho = piora  
Branco = sem mudança

### Summary Performance
Resumo visual completo com 8 subgráficos

---

## 🔬 Investigações Futuras

1. **Feature Importance**
   ```python
   # Adicionar após treinamento
   importances = rf_model.feature_importances_
   ```

2. **Ensemble Methods**
   - Combinar Random Forest + SVM
   - Potencial: +1-2% em acurácia

3. **Deep Learning**
   - Experimentar com TensorFlow/PyTorch
   - Se houver dados de imagem

4. **Validação Externa**
   - Testar em dataset completamente novo
   - Validar robustez em produção

5. **Explainability**
   - SHAP values
   - LIME para explicações

---

## 📞 Suporte & Contribuição

### Dúvidas sobre:
- **Algoritmos**: Consulte `ANALISE_DETALHADA.txt`
- **Técnicas**: Consulte `COMPARACAO_COMPLETA.md`
- **Código**: Consulte comentários nos scripts

### Para melhorias:
- Edite os scripts conforme necessário
- Teste com novos datasets
- Implemente novas visualizações

---

## ✅ Checklist de Conclusão

- ✅ Dados carregados e balanceados
- ✅ 6 algoritmos treinados (base)
- ✅ 6 algoritmos otimizados
- ✅ 11 gráficos gerados
- ✅ Métricas calculadas
- ✅ Relatório HTML criado
- ✅ Recomendações fornecidas
- ✅ Documentação completa

---

## 📝 Licença & Créditos

**Dataset**: Alzheimer's Disease Detection (Kaggle)  
**Autores**: Desenvolvimento automático de comparação  
**Data**: 27/05/2026  
**Status**: ✅ PRODUÇÃO  

---

## 🎯 Conclusão

Este projeto fornece uma **análise técnica rigorosa** de 6 algoritmos de Machine Learning para diagnóstico de Alzheimer. 

**Random Forest é a escolha recomendada** para produção com **91.22% de acurácia**.

Todos os dados, gráficos e análises estão disponíveis para decisões informadas sobre implementação.

---

**Obrigado por usar este sistema de análise! 🧠**
