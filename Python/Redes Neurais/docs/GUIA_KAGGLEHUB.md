# Integração com KaggleHub - Guia de Uso

## 📥 Automatização do Dataset do Kaggle

Este projeto agora suporta **download automático do dataset do Kaggle** sem necessidade de fazer upload manual do CSV a cada execução!

## ✨ O que foi alterado?

Os seguintes arquivos foram atualizados para usar KaggleHub:

1. **utils_kagglehub.py** (NOVO)
   - Arquivo central com funções para carregar dados do Kaggle
   - Tenta carregar localmente primeiro, depois do Kaggle

2. **scripts/treinar_e_exportar_modelo.py**
   - Agora usa `carregar_dataset_kaggle()` automaticamente

3. **alzheimer_interface_terminal.py**
   - Opção de treinar modelo com dados do Kaggle

4. **alzheimer_predictor.py**
   - Suporta treinamento com dados do Kaggle

5. **scripts/comparacao_algoritmos.py**
   - Baixa dados do Kaggle se não encontrar localmente

## 🚀 Como usar?

### Opção 1: Arquivo CSV Local (Mais Rápido)

Se você já tem o `alzheimers_disease_data.csv` na pasta `data/`:

```bash
python scripts/treinar_e_exportar_modelo.py
# Carregará automaticamente do arquivo local
```

### Opção 2: Baixar do Kaggle Automaticamente

Se o arquivo local não existir:

```bash
python scripts/treinar_e_exportar_modelo.py
# Baixará automaticamente do Kaggle
```

## 🔧 Configuração Necessária (Primeira Vez)

Para usar o download automático do Kaggle, você precisa:

### 1. Instalar Kaggle CLI

```bash
pip install kaggle
```

### 2. Configurar Credenciais

1. Vá para: https://www.kaggle.com/settings/account
2. Clique em "Create New API Token"
3. Um arquivo `kaggle.json` será baixado
4. Coloque o arquivo em:
   - **Windows**: `C:\Users\seu_usuario\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
5. Dê permissão ao arquivo:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json  # Linux/Mac
   ```

### 3. Testando a Configuração

```bash
kaggle datasets list
# Se funcionar, as credenciais estão OK
```

## 📝 Exemplo de Código

Você pode usar as funções diretamente em seu próprio código:

```python
from utils_kagglehub import carregar_dataset_kaggle

# Download automático do Kaggle
path, df = carregar_dataset_kaggle("rabieelkharoua/alzheimers-disease-dataset")

print(f"Dataset baixado em: {path}")
print(f"Forma: {df.shape}")
```

Ou tente local primeiro, depois Kaggle:

```python
from utils_kagglehub import carregar_dataset_local_ou_kaggle

local_path = 'data/alzheimers_disease_data.csv'
path, df = carregar_dataset_local_ou_kaggle(local_path)
```

## 💾 No GitHub

Agora você pode:

1. **NÃO fazer upload do CSV** (arquivo é grande)
2. **Colocar na pasta `.gitignore`**:
   ```
   data/*.csv
   data/*.zip
   ```
3. **Fazer commit sem o CSV** do dataset
4. No repositório, adicione um README informando:
   - Como configurar Kaggle CLI
   - Como executar o projeto

## 📊 Benefícios

✅ **Sem upload manual** do CSV toda vez  
✅ **Automático** - download acontece quando necessário  
✅ **Reutilização** - usa cache do Kaggle se já baixou  
✅ **Menor repo** - não precisa armazenar CSV grande no Git  
✅ **Reprodutível** - qualquer um pode executar o projeto  

## 🐛 Troubleshooting

### Erro: "ImportError: cannot import name 'get_web_endpoint'"
- Atualize: `pip install --upgrade kagglehub kagglesdk`

### Erro: "Arquivo não encontrado"
- Verifique se `kaggle.json` está em `~/.kaggle/`
- Teste com: `kaggle datasets list`

### Erro: "Connection refused"
- Verifique conexão com internet
- Verifique credenciais do Kaggle

## 📚 Mais Informações

- Kaggle Hub: https://github.com/Kaggle/kagglehub
- Kaggle API: https://github.com/Kaggle/kaggle-api
- Dataset: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
