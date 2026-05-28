# Comparação de Algoritmos de Redes Neurais - Detecção de Alzheimer

Uma análise completa e sistemática comparando 6 algoritmos de Machine Learning para diagnóstico automático de Alzheimer.

## 📊 Resumo Executivo

Foram comparados **6 algoritmos** em duas versões (BASE e OTIMIZADA):

| Ranking | Algoritmo | F1-Score (BASE) | F1-Score (OTM) | Acurácia | Recomendação |
|---------|-----------|-----------------|----------------|----------|--------------|
| 🥇 1º   | Random Forest | 0.9080 | 0.9074 | **91.22%** | **MELHOR PRODUÇÃO** |
| 🥈 2º   | SVM | 0.8608 | 0.8588 | 85.90% | Trade-off |
| 🥉 3º   | Decision Tree | 0.8314 | 0.8588 | 86.04% | Interpretável |
| 4º     | Logistic Regression | 0.8306 | 0.8306 | 82.45% | Baseline |
| 5º     | MLP | 0.8260 | 0.8354 | 83.17% | Simples |
| 6º     | KNN | 0.7546 | 0.7687 | 72.37% | Rápido (0.0007s) |

## 🎯 Dados

- **Dataset**: Alzheimer's Disease Detection
- **Amostras**: 2.149 registros
- **Features**: 33 características médicas
- **Classes**: Saudável (64.6%) vs Alzheimer (35.4%)
- **Balanceamento**: SMOTE aplicado
- **Normalização**: StandardScaler
- **Divisão**: 75% treino, 25% teste (estratificado)

## 🔬 Metodologia

### Técnicas Aplicadas:
- **SMOTE**: Balanceamento de classes desbalanceadas
- **StandardScaler**: Normalização de features
- **Validação Cruzada**: 5-fold Stratified K-Fold
- **GridSearchCV**: Otimização de hiperparâmetros

### Algoritmos Implementados:

1. **MLP (Multi-Layer Perceptron)** - Rede Neural
   - Base: 1 camada oculta, 500 iterações
   - Otimizado: 3 camadas (128-64-32), early stopping, lr adaptativa

2. **Decision Tree** - Árvore de Decisão
   - Base: Sem restrições
   - Otimizado: max_depth=10, min_samples_split=5

3. **KNN** - K-Nearest Neighbors
   - Base: k=5, uniformemente ponderado
   - Otimizado: k=7, pesos por distância

4. **Logistic Regression** - Regressão Logística
   - Base: parâmetros padrão
   - Otimizado: C=1.0, penalty L2, solver lbfgs

5. **Random Forest** - Floresta Aleatória
   - Base: 100 árvores
   - Otimizado: 200 árvores, max_depth=15

6. **SVM** - Support Vector Machine
   - Base: kernel linear
   - Otimizado: kernel RBF, C=10

## 📈 Resultados Principais

### Desempenho Geral (Médias):

**MODELOS BASE:**
- Acurácia: 82.71% ± 6.51%
- F1-Score: 83.52% ± 5.01%
- AUC-ROC: 89.80% ± 5.21%
- Tempo médio: 0.99s

**MODELOS OTIMIZADOS:**
- Acurácia: 83.53% ± 6.27%
- F1-Score: 84.33% ± 4.56%
- AUC-ROC: 91.04% ± 3.05%
- Tempo médio: 0.67s

### Melhores por Critério:

| Métrica | Vencedor BASE | Vencedor OTM |
|---------|---|---|
| **F1-Score** | Random Forest (0.9080) | Random Forest (0.9074) |
| **Acurácia** | Random Forest (91.22%) | Random Forest (91.22%) |
| **Precisão** | Random Forest (95.25%) | Random Forest (95.83%) |
| **Recall** | KNN (88.18%) | KNN (91.93%) |
| **AUC-ROC** | Random Forest (0.9605) | Random Forest (0.9578) |
| **Tempo** | KNN (0.0013s) | KNN (0.0007s) |

### Análise de Melhoria (Base → Otimizado):

```
MLP                  : F1: UP (+1.14%)  | Acc: UP (+1.05%)  | Tempo: DOWN (-69.83%)
Decision Tree        : F1: UP (+3.29%)  | Acc: UP (+3.64%)  | Tempo: DOWN (-8.87%)
KNN                  : F1: UP (+1.86%)  | Acc: UP (+1.41%)  | Tempo: DOWN (-47.24%)
Logistic Regression  : F1: MESMO        | Acc: MESMO        | Tempo: DOWN (-9.35%)
Random Forest        : F1: DOWN (-0.06%)| Acc: MESMO        | Tempo: UP (+88.86%)
SVM                  : F1: DOWN (-0.23%)| Acc: MESMO        | Tempo: UP (+77.30%)
```

## 📊 Visualizações Geradas

Foram gerados **10 gráficos** para análise visual:

1. **comparacao_metricas.png** - Comparação lado a lado de todas as métricas
2. **ranking_f1score.png** - Ranking final por F1-Score
3. **curvas_roc_comparacao.png** - Curvas ROC dos modelos
4. **heatmap_melhoria.png** - Mapa de calor com melhoria percentual
5. **tempo_treinamento.png** - Comparação de tempo de treino
6. **radar_desempenho.png** - Gráfico radar de desempenho
7. **scatter_acuracia_f1.png** - Scatter plot F1-Score vs Acurácia
8. **boxplot_metricas.png** - Distribuição de métricas
9. **summary_performance.png** - Sumário visual completo
10. **matrizes_confusao_modelos_base.png** - Matrizes de confusão (base)
11. **matrizes_confusao_modelos_otimizados.png** - Matrizes de confusão (otimizado)

## 💾 Arquivos de Saída

### Dados:
- `resultados_base.csv` - Métricas dos modelos base
- `resultados_otimizados.csv` - Métricas dos modelos otimizados
- `resultados_completos.csv` - Dataset completo para análise

### Relatórios:
- `relatorios/comparacao_algoritmos.html` - Relatório interativo HTML

## 🚀 Como Executar

### 1. Instalação de Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar Comparação Completa
```bash
python comparacao_algoritmos.py
```

Este script:
- Carrega e pré-processa os dados
- Treina 6 algoritmos em versão base
- Otimiza hiperparâmetros com GridSearchCV
- Treina 6 algoritmos em versão otimizada
- Gera todos os gráficos
- Salva relatórios e dados

### 3. Gerar Relatório HTML
```bash
python gerar_relatorio_html.py
```

### 4. Ver Sumário Final
```bash
python sumario_final.py
```

## 🏆 Recomendações Finais

### 1. PARA PRODUÇÃO (Melhor Acurácia)
```
Algoritmo: Random Forest
F1-Score: 0.9080 (Base) / 0.9074 (Otimizado)
Acurácia: 91.22%
Tempo: 0.60s
Recomendação: Use versão BASE por ser mais rápida
```

### 2. PARA INTERPRETABILIDADE
```
Algoritmo: Decision Tree (Otimizado)
F1-Score: 0.8588
Acurácia: 86.04%
Tempo: 0.032s
Parâmetros: max_depth=10, min_samples_split=5
Recomendação: Ideal quando explicabilidade é crítica
```

### 3. PARA VELOCIDADE CRÍTICA
```
Algoritmo: KNN (Otimizado)
Tempo: 0.0007s (< 1ms!)
F1-Score: 0.7687
Acurácia: 72.37%
Recomendação: Para sistemas em tempo real
```

### 4. PARA TRADE-OFF IDEAL
```
Algoritmo: SVM (Base)
F1-Score: 0.8608
Acurácia: 85.90%
AUC-ROC: 0.9362
Tempo: 0.85s
Recomendação: Bom equilíbrio entre todas as métricas
```

## 📋 Conclusões

1. **Random Forest é o melhor escolha geral** com acurácia de 91.22% e F1-Score de 0.9080
2. **Decision Tree oferece boa interpretabilidade** com melhoria de 3.29% em F1-Score após otimização
3. **KNN é extremamente rápido** (< 1ms) mas com menor acurácia - útil para restrições de latência
4. **A otimização melhorou a maioria dos modelos**, exceto Random Forest e SVM (já bem ajustados)
5. **Trade-off entre velocidade e acurácia**: Random Forest é 1000x mais lento que KNN, mas 20% mais preciso

## 📁 Estrutura de Pastas

```
.
├── comparacao_algoritmos.py       # Script principal
├── gerar_relatorio_html.py        # Gerador de HTML
├── sumario_final.py               # Sumário em texto
├── requirements.txt               # Dependências
├── data/
│   └── alzheimers_disease_data.csv
├── models/
├── notebooks/
├── visualizations/
├── relatorios/
│   └── comparacao_algoritmos.html
├── *.csv                          # Resultados
└── *.png                          # Gráficos
```

## 🔧 Modificações Possíveis

Para ajustar os parâmetros, edite as funções:
- `get_algoritmos_base()` - Alterar configurações base
- `get_algoritmos_otimizados()` - Alterar otimizações
- `get_param_grids()` - Alterar grid de busca

## 📞 Suporte

Para dúvidas ou melhorias, consulte a documentação ou execute os scripts com `-h` para ajuda.

---

**Versão**: 1.0  
**Data**: 27/05/2026  
**Autor**: Sistema de Comparação Automática
