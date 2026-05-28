# 🧠 Detecção de Alzheimer com Machine Learning

Projeto completo de análise e detecção de Doença de Alzheimer usando múltiplos algoritmos de Machine Learning com interface interativa.

---

## 📁 Estrutura do Projeto

```
Redes Neurais/
│
├── 📂 core/                          # Funcionalidades principais
│   ├── utils_kagglehub.py           # Download automático do Kaggle
│   ├── exemplo_kagglehub.py         # Exemplos de uso
│   └── __init__.py
│
├── 📂 config/                        # Configurações
│   ├── launcher_streamlit.py        # Script para iniciar Streamlit
│   ├── launcher_streamlit.bat       # Batch para Windows
│   └── __init__.py
│
├── 📂 tests/                         # Testes e diagnóstico
│   ├── teste_kaggle.py              # Teste de configuração Kaggle
│   ├── test_imports.py              # Testes de importação
│   ├── diagnostico.py               # Script de diagnóstico
│   └── __init__.py
│
├── 📂 docs/                          # Documentação
│   ├── GUIA_KAGGLEHUB.md            # Setup do Kaggle
│   ├── KAGGLE_CONFIGURADO.md        # Status da configuração
│   ├── SETUP_KAGGLEHUB_RESUMO.md    # Quick start
│   └── ... outros guias
│
├── 📂 scripts/                       # Scripts principais
│   ├── treinar_e_exportar_modelo.py
│   └── comparacao_algoritmos.py
│
├── 📂 data/                          # Dados (do Kaggle)
│   └── alzheimers_disease_data.csv
│
├── 📂 models/                        # Modelos treinados
│   ├── alzheimer_rf_model.pkl
│   └── alzheimer_rf_scaler.pkl
│
├── 📂 notebooks/                     # Jupyter Notebooks
│   └── ... análises
│
├── 📂 archive/                       # Arquivos antigos/backup
│   └── ... versões antigas
│
├── 🎯 Aplicações Principais
│   ├── app_streamlit.py             # Interface web principal
│   ├── alzheimer_predictor.py       # Interface Gradio
│   └── alzheimer_interface_terminal.py
│
├── 📋 Configuração
│   ├── requirements.txt
│   ├── .gitignore
│   └── README_PROJETO.md
│
└── 📝 Este arquivo
    └── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Primeira Execução (Setup)

```bash
# Clonar o repositório
git clone <seu-repo>
cd "Redes Neurais"

# Instalar dependências
pip install -r requirements.txt
```

### 2️⃣ Configurar Kaggle (Uma única vez)

```bash
# 1. Instalar Kaggle CLI
pip install kaggle

# 2. Obter token em: https://www.kaggle.com/settings/account
# 3. Criar arquivo em: ~/.kaggle/kaggle.json

# Ou testar se já está configurado:
python tests/teste_kaggle.py
```

### 3️⃣ Executar a Aplicação

**Opção A: Streamlit (Recomendado)**
```bash
python config/launcher_streamlit.py
# Ou no Windows:
config\launcher_streamlit.bat
```

**Opção B: Interface de Terminal**
```bash
python alzheimer_interface_terminal.py
```

**Opção C: Interface Gradio**
```bash
python alzheimer_predictor.py
```

**Opção D: Treinar Modelo**
```bash
python scripts/treinar_e_exportar_modelo.py
```

---

## 📊 Funcionalidades

### ✅ Modelos Suportados
- Random Forest (Melhor desempenho ⭐)
- MLP (Multi-Layer Perceptron)
- Decision Tree
- K-Nearest Neighbors
- Logistic Regression
- Support Vector Machine

### ✅ Features
- Download automático do dataset do Kaggle
- Treinamento e avaliação de múltiplos algoritmos
- Comparação de desempenho
- Interface web interativa (Streamlit)
- Interface terminal com cores
- Cache e otimizações
- Validação cruzada

### ✅ Dataset
- **Fonte**: Kaggle - [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)
- **Tamanho**: 2149 amostras
- **Features**: 31 atributos clínicos
- **Target**: Diagnosis (Saudável vs Alzheimer)

---

## 🔧 Configuração Detalhada

### Kaggle CLI

Veja `docs/GUIA_KAGGLEHUB.md` para instruções completas.

### Dependências

Veja `requirements.txt` para lista completa.

Principais:
- pandas, numpy
- scikit-learn, tensorflow
- streamlit, gradio
- matplotlib, seaborn

---

## 📊 Resultados Esperados

**Melhor Modelo: Random Forest**
- Acurácia: ~95%
- Precisão: ~94%
- Recall: ~96%
- F1-Score: ~95%
- AUC-ROC: ~0.98

---

## 🛠️ Desenvolvimento

### Adicionar Novo Modelo

1. Edite `scripts/comparacao_algoritmos.py`
2. Adicione import do modelo
3. Configure hiperparâmetros
4. Adicione à comparação

### Executar Testes

```bash
# Teste de Kaggle
python tests/teste_kaggle.py

# Teste de importações
python tests/test_imports.py

# Diagnóstico geral
python tests/diagnostico.py
```

---

## 📚 Documentação

| Arquivo | Descrição |
|---------|-----------|
| `docs/GUIA_KAGGLEHUB.md` | Setup completo do Kaggle |
| `docs/KAGGLE_CONFIGURADO.md` | Status da configuração |
| `docs/SETUP_KAGGLEHUB_RESUMO.md` | Quick start |
| `docs/GUIA_STREAMLIT.md` | Usando Streamlit |
| `docs/README_STREAMLIT.md` | Funcionalidades Streamlit |

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Arquivo CSV não encontrado"
```bash
python tests/teste_kaggle.py
# Configure o Kaggle se necessário
```

### "Conexão recusada (Streamlit)"
```bash
# Verifique se porta 8501 está disponível
# Ou force outra porta:
streamlit run app_streamlit.py --server.port 8502
```

---

## 🔒 Segurança

⚠️ **Nunca commitar**:
- `~/.kaggle/kaggle.json` (credenciais)
- Arquivos `*.csv` grandes
- Tokens ou senhas

Use `.gitignore`:
```
.kaggle/
data/*.csv
data/*.zip
*.pkl
```

---

## 📈 Performance

- **Treinamento**: ~2-5 minutos (primeira vez)
- **Predição**: <100ms por amostra
- **Interface Streamlit**: Responsiva

---

## 👥 Contribuições

Para reportar bugs ou sugerir melhorias:
1. Abra uma issue
2. Descreva o problema
3. Forneça exemplos

---

## 📄 Licença

MIT License - Veja LICENSE para detalhes

---

## 🎓 TCC

**Autor**: Giovani Gabriel  
**Projeto**: Detecção de Alzheimer com Machine Learning  
**Instituição**: [Sua instituição]  
**Ano**: 2026

---

## 📞 Suporte

- 📧 Email: giovanigabriel@example.com
- 🐛 Issues: GitHub Issues
- 📚 Documentação: `/docs/`

---

**Última atualização**: 28 de Maio de 2026
