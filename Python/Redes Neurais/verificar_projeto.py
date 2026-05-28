#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CHECKLIST DE VERIFICAÇÃO - Projeto Reorganizado

Execute este script para verificar se tudo está correto após a reorganização.
"""

import os
import sys
from pathlib import Path

def check_folder(name, path):
    """Verifica se uma pasta existe."""
    if Path(path).exists():
        items = len(list(Path(path).iterdir()))
        print(f"  OK {name.ljust(20)} ({items} itens)")
        return True
    else:
        print(f"  XX {name.ljust(20)} FALTANDO!")
        return False

def check_file(name, path):
    """Verifica se um arquivo existe."""
    if Path(path).exists():
        size = Path(path).stat().st_size / 1024
        print(f"  OK {name.ljust(30)} ({size:.1f} KB)")
        return True
    else:
        print(f"  XX {name.ljust(30)} FALTANDO!")
        return False

def check_import(name, module):
    """Verifica se um módulo pode ser importado."""
    try:
        __import__(module)
        print(f"  OK {name.ljust(30)}")
        return True
    except ImportError:
        print(f"  XX {name.ljust(30)} (pip install {module})")
        return False

def main():
    base = Path(__file__).parent
    
    print("\n" + "=" * 70)
    print("CHECKLIST - VERIFICAÇÃO DO PROJETO")
    print("=" * 70)
    
    all_ok = True
    
    # 1. Pastas
    print("\n[1] ESTRUTURA DE PASTAS:")
    folders = {
        "core/": base / "core",
        "config/": base / "config",
        "tests/": base / "tests",
        "docs/": base / "docs",
        "scripts/": base / "scripts",
        "data/": base / "data",
        "models/": base / "models",
        "notebooks/": base / "notebooks",
    }
    
    for name, path in folders.items():
        if not check_folder(name, path):
            all_ok = False
    
    # 2. Arquivos Principais
    print("\n[2] ARQUIVOS PRINCIPAIS:")
    files = {
        "README.md": base / "README.md",
        "ESTRUTURA.md": base / "ESTRUTURA.md",
        "app_streamlit.py": base / "app_streamlit.py",
        "requirements.txt": base / "requirements.txt",
        ".gitignore": base / ".gitignore",
    }
    
    for name, path in files.items():
        if not check_file(name, path):
            all_ok = False
    
    # 3. Core Files
    print("\n[3] ARQUIVOS CORE:")
    core_files = {
        "utils_kagglehub.py": base / "core" / "utils_kagglehub.py",
        "exemplo_kagglehub.py": base / "core" / "exemplo_kagglehub.py",
    }
    
    for name, path in core_files.items():
        if not check_file(name, path):
            all_ok = False
    
    # 4. Config Files
    print("\n[4] ARQUIVOS CONFIG:")
    config_files = {
        "launcher_streamlit.py": base / "config" / "launcher_streamlit.py",
        "launcher_streamlit.bat": base / "config" / "launcher_streamlit.bat",
    }
    
    for name, path in config_files.items():
        if not check_file(name, path):
            all_ok = False
    
    # 5. Test Files
    print("\n[5] ARQUIVOS TESTES:")
    test_files = {
        "teste_kaggle.py": base / "tests" / "teste_kaggle.py",
        "test_imports.py": base / "tests" / "test_imports.py",
        "diagnostico.py": base / "tests" / "diagnostico.py",
    }
    
    for name, path in test_files.items():
        if not check_file(name, path):
            all_ok = False
    
    # 6. Documentation
    print("\n[6] DOCUMENTAÇÃO:")
    docs = {
        "GUIA_KAGGLEHUB.md": base / "docs" / "GUIA_KAGGLEHUB.md",
        "KAGGLE_CONFIGURADO.md": base / "docs" / "KAGGLE_CONFIGURADO.md",
        "GUIA_STREAMLIT.md": base / "docs" / "GUIA_STREAMLIT.md",
    }
    
    for name, path in docs.items():
        if not check_file(name, path):
            all_ok = False
    
    # 7. Python Imports
    print("\n[7] DEPENDÊNCIAS PYTHON:")
    imports = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "streamlit": "streamlit",
    }
    
    for name, module in imports.items():
        if not check_import(name, module):
            all_ok = False
    
    # 8. Kaggle Config
    print("\n[8] CONFIGURACAO KAGGLE:")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if check_file("~/.kaggle/kaggle.json", kaggle_json):
        print("  INFO - Configure Kaggle em: https://www.kaggle.com/settings/account")
    else:
        print("  INFO - Kaggle ainda nao configurado")
        all_ok = False
    
    # Final Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("OK - TUDO OK! Seu projeto esta pronto!")
        print("\nProximos passos:")
        print("  1. python tests/teste_kaggle.py    (testar Kaggle)")
        print("  2. python config/launcher_streamlit.py  (iniciar app)")
    else:
        print("XX - ALGUNS ITENS FALTANDO")
        print("\nResolva os problemas acima e tente novamente.")
    print("=" * 70 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
