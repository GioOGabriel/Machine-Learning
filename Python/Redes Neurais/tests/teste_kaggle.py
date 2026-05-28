"""
TESTE DE DOWNLOAD DO KAGGLE

Este script testa se a configuração do Kaggle está funcionando
e faz download de uma pequena quantidade de dados para teste.
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("TESTE DE CONFIGURACAO DO KAGGLE")
print("=" * 70)
print()

# Verificar arquivo kaggle.json
kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
print(f"[1] Verificando kaggle.json...")
if kaggle_json.exists():
    print(f"    ✓ Encontrado: {kaggle_json}")
else:
    print(f"    ✗ Arquivo não encontrado: {kaggle_json}")
    print("    Configure primeiro!")
    sys.exit(1)

print()
print(f"[2] Testando utils_kagglehub...")
try:
    from utils_kagglehub import carregar_dataset_kaggle
    print("    ✓ utils_kagglehub importado com sucesso")
except ImportError as e:
    print(f"    ✗ Erro ao importar: {e}")
    sys.exit(1)

print()
print(f"[3] Tentando baixar dataset do Kaggle...")
print(f"    Dataset: rabieelkharoua/alzheimers-disease-dataset")
print(f"    Isso pode levar alguns minutos na primeira vez...")
print()

try:
    path, df = carregar_dataset_kaggle("rabieelkharoua/alzheimers-disease-dataset")
    
    print()
    print("=" * 70)
    print("✓ DOWNLOAD CONCLUIDO COM SUCESSO!")
    print("=" * 70)
    print()
    print(f"Localização: {path}")
    print(f"Tamanho do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print()
    print("Colunas:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print()
    print("Primeiras 5 linhas:")
    print(df.head())
    
    print()
    print("=" * 70)
    print("CONFIGURACAO VALIDADA COM SUCESSO!")
    print("=" * 70)
    print()
    print("Agora você pode executar:")
    print("  python scripts/treinar_e_exportar_modelo.py")
    print()
    
except Exception as e:
    print()
    print("=" * 70)
    print("✗ ERRO AO BAIXAR DATASET")
    print("=" * 70)
    print()
    print(f"Erro: {e}")
    print()
    print("Possíveis soluções:")
    print("1. Verifique sua conexão com a internet")
    print("2. Verifique se o token do Kaggle está correto")
    print("3. Verifique se o arquivo kaggle.json está em: ~/.kaggle/")
    print("4. Tente: kaggle datasets list (para testar CLI)")
    print()
    sys.exit(1)
