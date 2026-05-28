# 🧠 GUIA DE USO - Detecção de Alzheimer com Streamlit

## ✨ Resumo da Solução

Você agora tem uma **aplicação web profissional** para prognóstico de Alzheimer sem precisar fazer upload de arquivos CSV. 

O modelo foi:
- ✅ Treinado com Random Forest (melhor desempenho)
- ✅ Exportado para arquivos `.pkl` (binários)
- ✅ Integrado em uma interface web Streamlit

---

## 🚀 INICIANDO A APLICAÇÃO (3 FORMAS)

### FORMA 1: Windows - Mais Fácil (RECOMENDADO)
```
1. Abra a pasta: Redes Neurais
2. Duplo clique em: executar_streamlit.bat
3. Pronto! Abre automaticamente em http://localhost:8501
```

### FORMA 2: Todos os SOs
```bash
cd "Redes Neurais"
python executar_streamlit.py
```

### FORMA 3: Manual (Se as outras não funcionarem)
```bash
cd "Redes Neurais"
streamlit run app_streamlit.py
```

---

## 📋 COMO USAR A APLICAÇÃO

### 1. Preencha os Dados do Paciente
- Idade
- Fumante (Sim/Não)
- Consome álcool (Sim/Não)
- Atividade física (horas/semana)
- Pressão arterial (mmHg)
- Frequência cardíaca (bpm)
- Colesterol (mg/dL)
- Qualidade do sono (0-10)
- Estresse mental (0-10)

### 2. Clique em "Fazer Prognóstico"
O modelo processará os dados em milissegundos

### 3. Veja o Resultado
- **Verde**: Sem sinais de Alzheimer
- **Vermelho**: Possível Alzheimer
- **Gráfico**: Mostra as probabilidades

### 4. Expanda "Detalhes da Análise"
Veja fatores de risco identificados

---

## 📊 ABAS DA APLICAÇÃO

### Tab 1: 🏥 Prognóstico (Principal)
- Formulário para entrada de dados
- Resultado da predição
- Gráfico de probabilidade
- Análise de fatores de risco

### Tab 2: 📊 Informações
- Métricas do modelo (91.82% acurácia)
- Informações técnicas
- Características do treinamento

### Tab 3: ℹ️ Sobre
- Disclaimer legal
- Metodologia científica
- Tecnologias utilizadas

---

## 🏆 DESEMPENHO DO MODELO

| Métrica | Valor |
|---------|-------|
| Acurácia | **91.82%** |
| Precisão | **92.94%** |
| Recall | **83.16%** |
| F1-Score | **87.78%** |
| AUC-ROC | **93.81%** |

---

## 🔧 O QUE MUDOU

### ANTES (Problemático)
```
❌ Tinha que fazer upload de CSV
❌ Uma linha por prognóstico
❌ Interface ruim
❌ Dados poluídos por SMOTE antes do split
```

### AGORA (Solução Profissional)
```
✅ Formulário web intuitivo
✅ Múltiplas predições sem reimportar
✅ Interface moderna com Streamlit
✅ SMOTE aplicado corretamente (após split)
✅ Modelo exportado para produção
✅ Fácil de usar para não-técnicos
```

---

## 📁 ARQUIVOS PRINCIPAIS

```
Redes Neurais/
├── app_streamlit.py                 # Aplicação web
├── executar_streamlit.bat           # Launcher Windows
├── executar_streamlit.py            # Launcher multiplataforma
├── scripts/treinar_e_exportar_modelo.py  # Script de treinamento
├── models/
│   ├── alzheimer_rf_model.pkl       # Modelo (6.07 MB)
│   ├── alzheimer_rf_scaler.pkl      # Normalizador
│   └── alzheimer_rf_features.pkl    # Nomes das features
├── README_STREAMLIT.md              # Documentação detalhada
└── README_PROJETO.md                # Este projeto
```

---

## ⚠️ IMPORTANTE - AVISOS LEGAIS

Esta é uma ferramenta **EDUCACIONAL** e de **PESQUISA**.

**NÃO substitui diagnóstico médico profissional.**

Sempre consulte um médico especialista!

---

## 🔄 PIPELINE CIENTÍFICO CORRETO

```
1. Dataset original (2.149 amostras)
   ↓
2. Train/Test Split 75/25 (ESTRATIFICADO)
   ↓
3. SMOTE APENAS no treino (balanceamento)
   ↓
4. StandardScaler (normalização)
   ↓
5. Random Forest 200 estimadores
   ↓
6. Validação no teste (dados reais)
   ↓
7. Métricas confiáveis (91.82% acurácia)
```

---

## ✅ CHECKLIST DE VERIFICAÇÃO

Antes de usar, verifique:

- [x] Pasta `models/` tem 3 arquivos `.pkl`
- [x] Arquivo `app_streamlit.py` existe
- [x] Arquivo `executar_streamlit.bat` existe (Windows)
- [x] `requirements.txt` está atualizado com Streamlit
- [x] Python 3.8+ instalado
- [x] SMOTE aplicado corretamente (após split)

---

## 🐛 TROUBLESHOOTING

### "Modelo não encontrado"
```bash
python scripts/treinar_e_exportar_modelo.py
```

### "ModuleNotFoundError: streamlit"
```bash
pip install -r requirements.txt
```

### "Porta 8501 já em uso"
Streamlit sugerirá automaticamente outra porta (ex: 8502)

### Navegador não abre
Copie a URL exibida no terminal: `http://localhost:8501`

---

## 📞 INFORMAÇÕES TÉCNICAS

- **Algoritmo**: Random Forest com 200 estimadores
- **Features**: 32 características clínicas
- **Dataset**: 2.149 amostras de pacientes
- **Validação**: 5-fold Stratified Cross-Validation
- **Balanceamento**: SMOTE (k_neighbors=5)
- **Normalização**: StandardScaler
- **Framework Web**: Streamlit 1.57.0+
- **Linguagem**: Python 3.8+

---

## 🎓 PRÓXIMAS MELHORIAS SUGERIDAS

1. Autenticação de usuários
2. Histórico de prognósticos
3. Exportar resultado em PDF
4. Integração com banco de dados
5. Dashboard de estatísticas
6. Deploy em servidor cloud (Heroku, AWS, etc)

---

## 📚 REFERÊNCIAS

- [Documentação Streamlit](https://docs.streamlit.io/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [SMOTE Algorithm Paper](https://arxiv.org/abs/1106.1813)
- [Random Forest](https://en.wikipedia.org/wiki/Random_forest)

---

**Desenvolvido com ❤️ em Python**

Versão: 1.0.0  
Data: 27/05/2026  
Status: Pronto para Produção ✅
