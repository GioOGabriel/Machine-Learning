import sys
print("Testando importacoes...")

try:
    import streamlit as st
    print("[OK] Streamlit importado")
except Exception as e:
    print(f"[ERRO] Streamlit: {e}")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    print("[OK] Pandas e NumPy importados")
except Exception as e:
    print(f"[ERRO] Pandas/NumPy: {e}")
    sys.exit(1)

try:
    import joblib
    print("[OK] Joblib importado")
except Exception as e:
    print(f"[ERRO] Joblib: {e}")
    sys.exit(1)

try:
    from pathlib import Path
    print("[OK] Pathlib importado")
except Exception as e:
    print(f"[ERRO] Pathlib: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("[OK] Matplotlib importado")
except Exception as e:
    print(f"[ERRO] Matplotlib: {e}")
    sys.exit(1)

print("\nTodas as importacoes OK!")
print("Agora testando carregamento do modelo...")

try:
    BASE_PATH = Path('.')
    MODELS_PATH = BASE_PATH / 'models'
    MODEL_FILE = MODELS_PATH / 'alzheimer_rf_model.pkl'
    SCALER_FILE = MODELS_PATH / 'alzheimer_rf_scaler.pkl'
    FEATURES_FILE = MODELS_PATH / 'alzheimer_rf_features.pkl'
    
    print(f"Carregando modelo de: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    print("[OK] Modelo carregado")
    
    print(f"Carregando scaler de: {SCALER_FILE}")
    scaler = joblib.load(SCALER_FILE)
    print("[OK] Scaler carregado")
    
    print(f"Carregando features de: {FEATURES_FILE}")
    features = joblib.load(FEATURES_FILE)
    print(f"[OK] Features carregado ({len(features)} features)")
    
except Exception as e:
    print(f"[ERRO] ao carregar modelo: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCESSO] Tudo funcionando!")
