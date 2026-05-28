"""
===============================================================================
TREINAR E EXPORTAR MODELO RANDOM FOREST PARA DETECÇÃO DE ALZHEIMER
===============================================================================

Este script treina o modelo Random Forest (melhor desempenho) e o exporta
para uso em produção com a aplicação Streamlit.

Autor: Machine Learning Pipeline
===============================================================================
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.utils_kagglehub import carregar_dataset_kaggle

# Configurações
RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent.parent / 'data' / 'alzheimers_disease_data.csv'
MODELS_PATH = Path(__file__).parent.parent / 'models'
MODELS_PATH.mkdir(exist_ok=True)

MODEL_FILE = MODELS_PATH / 'alzheimer_rf_model.pkl'
SCALER_FILE = MODELS_PATH / 'alzheimer_rf_scaler.pkl'
FEATURES_FILE = MODELS_PATH / 'alzheimer_rf_features.pkl'

def carregar_dados():
    """Carrega e prepara os dados do Kaggle usando KaggleHub."""
    print("[1] CARREGANDO DADOS")
    print("-" * 50)
    
    # Tentar carregar localmente primeiro, depois do Kaggle
    try:
        if DATA_PATH.exists():
            print(f"[LOCAL] Carregando de: {DATA_PATH}")
            df = pd.read_csv(DATA_PATH)
        else:
            print("[KAGGLE] Arquivo local não encontrado, baixando do Kaggle...")
            _, df = carregar_dataset_kaggle("rabieelkharoua/alzheimers-disease-dataset")
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar os dados: {e}")
        raise
    
    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} features")
    
    # Remover colunas desnecessárias
    columns_to_remove = ['PatientID', 'DoctorInCharge']
    df = df.drop(columns=[c for c in columns_to_remove if c in df.columns])
    
    # Separar features e target
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    
    print(f"Features: {X.shape[1]} | Target shape: {y.shape[0]}")
    print(f"Classes: {y.value_counts().to_dict()}")
    
    return X, y, X.columns.tolist()

def preparar_dados(X, y):
    """Divide em treino/teste, aplica SMOTE apenas no treino e normaliza."""
    print("\n[2] PREPARAÇÃO DOS DADOS")
    print("-" * 50)
    
    # Divisão treino/teste ANTES do balanceamento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # Balanceamento com SMOTE APENAS no conjunto de treino
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Após SMOTE (treino): {len(y_train_balanced)} amostras (balanceado)")
    print(f"Teste mantido intacto: {len(X_test)} amostras (não poluído)")
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    print("Normalização aplicada (StandardScaler)")
    
    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler

def treinar_modelo(X_train, X_test, y_train, y_test):
    """Treina o Random Forest com hiperparâmetros otimizados."""
    print("\n[3] TREINAMENTO DO RANDOM FOREST")
    print("-" * 50)
    
    # Hiperparâmetros otimizados
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    print("Treinando modelo...")
    model.fit(X_train, y_train)
    
    # Avaliação
    print("\n[4] AVALIAÇÃO DO MODELO")
    print("-" * 50)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Acurácia:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precisão: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    
    return model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc
    }

def salvar_modelo(model, scaler, feature_names):
    """Salva o modelo, scaler e nomes de features."""
    print("\n[5] EXPORTANDO MODELO")
    print("-" * 50)
    
    # Salvar modelo
    joblib.dump(model, MODEL_FILE)
    print(f"[OK] Modelo salvo: {MODEL_FILE}")
    
    # Salvar scaler
    joblib.dump(scaler, SCALER_FILE)
    print(f"[OK] Scaler salvo: {SCALER_FILE}")
    
    # Salvar nomes de features
    joblib.dump(feature_names, FEATURES_FILE)
    print(f"[OK] Features salvo: {FEATURES_FILE}")
    
    # Informações do modelo
    print(f"\nTamanho do modelo: {MODEL_FILE.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Pronto para usar na aplicacao Streamlit!")

def main():
    print("=" * 60)
    print("TREINAMENTO E EXPORTAÇÃO DO MODELO RANDOM FOREST")
    print("=" * 60)
    
    try:
        # Carregar dados
        X, y, feature_names = carregar_dados()
        
        # Preparar dados
        X_train, X_test, y_train, y_test, scaler = preparar_dados(X, y)
        
        # Treinar modelo
        model, metrics = treinar_modelo(X_train, X_test, y_train, y_test)
        
        # Salvar modelo
        salvar_modelo(model, scaler, feature_names)
        
        print("\n" + "=" * 60)
        print("MODELO TREINADO E EXPORTADO COM SUCESSO!")
        print("=" * 60)
        print("\nVoce pode agora executar a aplicacao Streamlit:")
        print("streamlit run app_streamlit.py")
        
    except Exception as e:
        print(f"\n[ERRO] {str(e)}")
        raise

if __name__ == "__main__":
    main()
