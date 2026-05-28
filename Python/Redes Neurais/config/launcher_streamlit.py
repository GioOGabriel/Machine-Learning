#!/usr/bin/env python
"""
Launcher para a aplicação Streamlit - Detecção de Alzheimer

Uso: python executar_streamlit.py
"""

import subprocess
import sys
import os
from pathlib import Path

def executar_comando(cmd, descricao=""):
    """Executa um comando e retorna o código de saída."""
    if descricao:
        print(f"[INFO] {descricao}...")
    try:
        resultado = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if resultado.returncode != 0:
            print(f"[ERRO] {resultado.stderr}")
            return False
        return True
    except Exception as e:
        print(f"[ERRO] {str(e)}")
        return False

def verificar_python():
    """Verifica se Python está disponível."""
    print("[CHECK] Verificando Python...")
    if sys.version_info < (3, 8):
        print("[ERRO] Python 3.8+ é necessário!")
        return False
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor} encontrado")
    return True

def instalar_dependencias():
    """Instala as dependências se necessário."""
    print("[CHECK] Verificando dependencias...")
    
    # Verificar Streamlit
    try:
        import streamlit
        print("[OK] Streamlit encontrado")
        return True
    except ImportError:
        print("[INFO] Instalando dependencias...")
        req_path = Path(__file__).parent.parent / 'requirements.txt'
        return executar_comando(
            f"{sys.executable} -m pip install -r {req_path}",
            "Instalando pacotes"
        )

def treinar_modelo():
    """Treina o modelo se ele não existir."""
    modelo_path = Path(__file__).parent.parent / "models/alzheimer_rf_model.pkl"
    
    if modelo_path.exists():
        print(f"[OK] Modelo encontrado: {modelo_path}")
        return True
    
    print("[INFO] Modelo nao encontrado. Treinando...")
    scripts_path = Path(__file__).parent.parent / "scripts/treinar_e_exportar_modelo.py"
    return executar_comando(
        f"{sys.executable} {scripts_path}",
        "Treinando modelo Random Forest"
    )

def iniciar_streamlit():
    """Inicia a aplicação Streamlit."""
    print("\n" + "=" * 60)
    print("INICIANDO APLICACAO STREAMLIT")
    print("=" * 60)
    print("\nA aplicacao estara disponivel em: http://localhost:8501")
    print("Pressione Ctrl+C para parar\n")
    
    app_path = Path(__file__).parent.parent / "app_streamlit.py"
    os.system(f"{sys.executable} -m streamlit run {app_path}")

def main():
    print("\n" + "=" * 60)
    print("LAUNCHER - DETECCAO DE ALZHEIMER COM STREAMLIT")
    print("=" * 60 + "\n")
    
    # 1. Verificar Python
    if not verificar_python():
        sys.exit(1)
    
    # 2. Instalar dependências
    if not instalar_dependencias():
        sys.exit(1)
    
    # 3. Treinar/verificar modelo
    if not treinar_modelo():
        sys.exit(1)
    
    # 4. Iniciar Streamlit
    iniciar_streamlit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Aplicacao interrompida pelo usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERRO] {str(e)}")
        sys.exit(1)
