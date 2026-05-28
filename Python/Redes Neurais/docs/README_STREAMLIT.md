# 🧠 Aplicação Streamlit - Detecção de Alzheimer

## 🚀 Quick Start

### Pré-requisitos
- Python 3.8+
- pip

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Treinar e Exportar o Modelo
Execute este comando uma única vez para treinar o Random Forest e exportar:

```bash
python scripts/treinar_e_exportar_modelo.py
```

Saída esperada:
```
[OK] Modelo salvo: models/alzheimer_rf_model.pkl
[OK] Scaler salvo: models/alzheimer_rf_scaler.pkl
[OK] Features salvo: models/alzheimer_rf_features.pkl

MODELO TREINADO E EXPORTADO COM SUCESSO!
```

### 3. Executar a Aplicação Streamlit
```bash
streamlit run app_streamlit.py
```

A aplicação abrirá em: `http://localhost:8501`

---

## 📋 Como Usar a Aplicação

### Tab "Prognóstico" (Principal)
1. Preencha os dados do paciente no formulário
2. Clique no botão "Fazer Prognóstico"
3. Veja o resultado com a probabilidade

**Campos do formulário:**
- **Idade**: 18-100 anos
- **Fumante**: Sim/Não
- **Consome álcool**: Sim/Não
- **Atividade Física**: 0-40 horas/semana
- **Pressão Arterial**: 60-200 mmHg
- **Frequência Cardíaca**: 40-150 bpm
- **Colesterol**: 100-400 mg/dL
- **Qualidade do Sono**: 0-10 (escala)
- **Estresse Mental**: 0-10 (escala)

### Tab "Informações"
- Métricas de desempenho do modelo
- Informações técnicas

### Tab "Sobre"
- Disclaimer legal
- Metodologia utilizada
- Tecnologias usadas

---

## 📊 Desempenho do Modelo

| Métrica | Valor |
|---------|-------|
| **Acurácia** | 91.82% |
| **Precisão** | 92.94% |
| **Recall** | 83.16% |
| **F1-Score** | 87.78% |
| **AUC-ROC** | 93.81% |

---

## 🔍 Características Técnicas

- **Algoritmo**: Random Forest (200 estimadores)
- **Features**: 32 características clínicas
- **Dataset**: 2.149 amostras
- **Balanceamento**: SMOTE (apenas no treino)
- **Normalização**: StandardScaler
- **Validação**: 5-fold Cross-Validation

---

## 📁 Estrutura de Arquivos

```
Redes Neurais/
├── app_streamlit.py                    # Aplicação principal
├── scripts/
│   ├── treinar_e_exportar_modelo.py   # Script de treinamento
│   └── comparacao_algoritmos.py        # Comparação de algoritmos
├── models/
│   ├── alzheimer_rf_model.pkl         # Modelo treinado
│   ├── alzheimer_rf_scaler.pkl        # Scaler
│   └── alzheimer_rf_features.pkl      # Nomes de features
├── data/
│   └── alzheimers_disease_data.csv    # Dataset
└── requirements.txt                    # Dependências
```

---

## 🔄 Pipeline de Processamento

```
1. Carregar dados originais (2.149 amostras)
   ↓
2. Train/Test Split (75% / 25%)
   ↓
3. SMOTE aplicado APENAS no treino
   ↓
4. StandardScaler normalização
   ↓
5. Treinar Random Forest
   ↓
6. Avaliar no teste (dados reais)
```

---

## ⚠️ Aviso Legal

Esta é uma ferramenta **EDUCACIONAL** de detecção de Alzheimer. 

**NÃO substitui diagnóstico médico profissional.**

Sempre consulte um médico especialista para confirmação de diagnóstico.

---

## 🛠️ Troubleshooting

### Erro: "Modelo não encontrado"
**Solução:** Execute `python scripts/treinar_e_exportar_modelo.py`

### Erro: "ModuleNotFoundError: No module named 'streamlit'"
**Solução:** Execute `pip install -r requirements.txt`

### A aplicação não abre no navegador
**Solução:** Copie manualmente a URL exibida no terminal (geralmente `http://localhost:8501`)

---

## 📚 Referências

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-Learn Guide](https://scikit-learn.org/)
- [SMOTE Algorithm](https://arxiv.org/abs/1106.1813)

---

## 👨‍💻 Desenvolvido com

- Python 3.8+
- Streamlit 1.57.0+
- Scikit-Learn
- Pandas & NumPy
- Matplotlib

---

_Projeto de Machine Learning para Detecção de Alzheimer - 2026_
