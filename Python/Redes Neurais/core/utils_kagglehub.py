"""
Utilitário para carregar dados do Kaggle usando KaggleHub
"""

import pandas as pd
import os
from pathlib import Path
import subprocess

def carregar_dataset_kaggle(dataset_name="rabieelkharoua/alzheimers-disease-dataset"):
    """
    Carrega o dataset do Kaggle usando a CLI do Kaggle.
    
    Args:
        dataset_name (str): Nome do dataset no Kaggle
    
    Returns:
        tuple: (path_do_dataset, dataframe_carregado)
    """
    try:
        print(f"[KaggleHub] Baixando dataset: {dataset_name}...")
        
        # Criar diretório para dados se não existir
        data_dir = Path('./data')
        data_dir.mkdir(exist_ok=True)
        
        # Tentar usar kaggle CLI
        try:
            import kagglehub
            path = kagglehub.dataset_download(dataset_name)
        except (ImportError, Exception):
            # Se kagglehub não funcionar, usar subprocess com kaggle CLI
            print("[KaggleHub] Tentando alternativa com CLI do Kaggle...")
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', str(data_dir)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Erro ao baixar com Kaggle CLI: {result.stderr}")
            
            # Descompactar
            import zipfile
            zip_path = list(data_dir.glob('*.zip'))
            if zip_path:
                with zipfile.ZipFile(zip_path[0], 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                zip_path[0].unlink()
            
            path = str(data_dir)
        
        print(f"[KaggleHub] Dataset baixado em: {path}")
        
        # Encontrar e carregar o CSV
        csv_files = []
        for root, dirs, files in os.walk(path):
            csv_files.extend([f for f in files if f.endswith('.csv')])
        
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {path}")
        
        # Pegar o primeiro CSV ou o 'alzheimers_disease_data.csv'
        csv_file = next(
            (f for f in csv_files if 'alzheimer' in f.lower()),
            csv_files[0]
        )
        
        # Encontrar o caminho completo do arquivo
        csv_path = None
        for root, dirs, files in os.walk(path):
            if csv_file in files:
                csv_path = os.path.join(root, csv_file)
                break
        
        if not csv_path:
            raise FileNotFoundError(f"Não foi possível localizar {csv_file}")
        
        print(f"[KaggleHub] Carregando: {csv_file}")
        
        df = pd.read_csv(csv_path)
        print(f"[KaggleHub] Dataset carregado com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        return path, df
    
    except Exception as e:
        print(f"[ERRO KaggleHub] {str(e)}")
        print("\n[INFO] Para usar KaggleHub, você precisa:")
        print("  1. Instalar Kaggle CLI: pip install kaggle")
        print("  2. Configurar credenciais: https://www.kaggle.com/settings/account")
        print("  3. Fazer download de 'kaggle.json' e colocar em ~/.kaggle/")
        raise

def carregar_dataset_local_ou_kaggle(local_path=None, dataset_name="rabieelkharoua/alzheimers-disease-dataset"):
    """
    Tenta carregar do caminho local primeiro, se não encontrar, baixa do Kaggle.
    
    Args:
        local_path (str): Caminho local do arquivo CSV
        dataset_name (str): Nome do dataset no Kaggle
    
    Returns:
        tuple: (caminho_do_arquivo, dataframe)
    """
    # Tentar carregar localmente primeiro
    if local_path and os.path.exists(local_path):
        print(f"[LOCAL] Carregando dataset de: {local_path}")
        df = pd.read_csv(local_path)
        print(f"[LOCAL] Dataset carregado com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas")
        return local_path, df
    
    # Se não encontrar localmente, baixar do Kaggle
    print(f"[LOCAL] Arquivo não encontrado em {local_path}, buscando no Kaggle...")
    return carregar_dataset_kaggle(dataset_name)

