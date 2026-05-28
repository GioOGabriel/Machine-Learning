"""
Script de diagnóstico para a aplicação Streamlit
Ajuda a identificar e resolver problemas
"""

import sys
from pathlib import Path
import subprocess

print("=" * 70)
print("DIAGNOSTICO STREAMLIT - DETECCAO DE ALZHEIMER")
print("=" * 70)
print()

# 1. Verificar Python
print("[1/6] Verificando Python...")
print(f"  Versao: {sys.version}")
if sys.version_info < (3, 8):
    print("  [AVISO] Python 3.8+ e recomendado")
else:
    print("  [OK]")
print()

# 2. Verificar Streamlit
print("[2/6] Verificando Streamlit...")
try:
    import streamlit as st
    print(f"  [OK] Streamlit {st.__version__} instalado")
except ImportError:
    print("  [ERRO] Streamlit nao instalado")
    print("  Solucao: pip install streamlit")
    sys.exit(1)
print()

# 3. Verificar dependências
print("[3/6] Verificando dependencias...")
deps = {
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'sklearn': 'Scikit-Learn',
    'joblib': 'Joblib',
    'matplotlib': 'Matplotlib'
}

for module, name in deps.items():
    try:
        __import__(module)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [ERRO] {name} - pip install {module}")
print()

# 4. Verificar arquivos do projeto
print("[4/6] Verificando arquivos...")
files = [
    ('app_streamlit_v2.py', 'Aplicacao Streamlit'),
    ('models/alzheimer_rf_model.pkl', 'Modelo Random Forest'),
    ('models/alzheimer_rf_scaler.pkl', 'Scaler'),
    ('models/alzheimer_rf_features.pkl', 'Features'),
    ('.streamlit/config.toml', 'Configuracao Streamlit'),
]

base_path = Path(__file__).parent
all_ok = True

for filepath, name in files:
    full_path = base_path / filepath
    if full_path.exists():
        size = full_path.stat().st_size / 1024
        print(f"  [OK] {name} ({size:.1f} KB)")
    else:
        print(f"  [FALTA] {name} - {filepath}")
        all_ok = False

if not all_ok:
    print("\n  Solucao: Execute 'python scripts/treinar_e_exportar_modelo.py'")
print()

# 5. Testar carregamento do modelo
print("[5/6] Testando carregamento do modelo...")
try:
    import joblib
    model = joblib.load(base_path / 'models/alzheimer_rf_model.pkl')
    scaler = joblib.load(base_path / 'models/alzheimer_rf_scaler.pkl')
    features = joblib.load(base_path / 'models/alzheimer_rf_features.pkl')
    
    print(f"  [OK] Modelo carregado com sucesso")
    print(f"  [OK] Scaler carregado com sucesso")
    print(f"  [OK] Features carregado ({len(features)} features)")
except Exception as e:
    print(f"  [ERRO] {e}")
    sys.exit(1)
print()

# 6. Teste rápido
print("[6/6] Teste rapido de predicao...")
try:
    import pandas as pd
    import numpy as np
    
    # Dados de teste (com TODAS as features)
    test_data = {
        'Age': 65,
        'Gender': 0,
        'Ethnicity': 0,
        'EducationLevel': 2,
        'BMI': 25.0,
        'Smoking': 0,
        'AlcoholConsumption': 0,
        'PhysicalActivity': 3.0,
        'DietQuality': 5,
        'SleepQuality': 7,
        'FamilyHistoryAlzheimers': 0,
        'CardiovascularDisease': 0,
        'Diabetes': 0,
        'Depression': 0,
        'HeadInjury': 0,
        'Hypertension': 0,
        'SystolicBP': 120,
        'DiastolicBP': 80,
        'CholesterolTotal': 200,
        'CholesterolLDL': 100,
        'CholesterolHDL': 50,
        'CholesterolTriglycerides': 150,
        'MMSE': 28,
        'FunctionalAssessment': 8,
        'MemoryComplaints': 0,
        'BehavioralProblems': 0,
        'ADL': 0,
        'Confusion': 0,
        'Disorientation': 0,
        'PersonalityChanges': 0,
        'DifficultyCompletingTasks': 0,
        'Forgetfulness': 0,
    }
    
    df = pd.DataFrame([test_data])
    df_scaled = scaler.transform(df[features])
    
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    
    resultado = "Saudavel" if prediction == 0 else "Alzheimer"
    print(f"  [OK] Predicao: {resultado}")
    print(f"  [OK] Confianca: {probability[prediction]*100:.1f}%")
except Exception as e:
    print(f"  [ERRO] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("TODOS OS TESTES PASSARAM COM SUCESSO!")
print("=" * 70)
print()
print("Para iniciar a aplicação, execute:")
print("  streamlit run app_streamlit_v2.py")
print()
