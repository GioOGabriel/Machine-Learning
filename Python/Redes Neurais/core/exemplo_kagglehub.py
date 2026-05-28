"""
EXEMPLO DE USO: Como usar KaggleHub nos seus scripts

Este arquivo demonstra como usar a integração com KaggleHub
"""

from utils_kagglehub import carregar_dataset_kaggle, carregar_dataset_local_ou_kaggle
import pandas as pd

# ============================================================================
# EXEMPLO 1: Download automático do Kaggle
# ============================================================================

def exemplo_1_download_automatico():
    """Exemplo simples: baixa dataset do Kaggle"""
    print("\n=== EXEMPLO 1: Download Automático ===\n")
    
    try:
        path, df = carregar_dataset_kaggle("rabieelkharoua/alzheimers-disease-dataset")
        print(f"✓ Dataset baixado com sucesso!")
        print(f"  Localização: {path}")
        print(f"  Tamanho: {df.shape[0]} linhas x {df.shape[1]} colunas")
        print(f"  Primeiras colunas: {list(df.columns[:5])}")
        
    except Exception as e:
        print(f"✗ Erro ao baixar: {e}")

# ============================================================================
# EXEMPLO 2: Tenta local primeiro, depois Kaggle
# ============================================================================

def exemplo_2_local_ou_kaggle():
    """Tenta carregar arquivo local, se não encontrar, baixa do Kaggle"""
    print("\n=== EXEMPLO 2: Local ou Kaggle ===\n")
    
    local_path = 'data/alzheimers_disease_data.csv'
    
    try:
        path, df = carregar_dataset_local_ou_kaggle(local_path)
        print(f"✓ Dataset carregado!")
        print(f"  Localização: {path}")
        print(f"  Tamanho: {df.shape[0]} linhas x {df.shape[1]} colunas")
        
    except Exception as e:
        print(f"✗ Erro ao carregar: {e}")

# ============================================================================
# EXEMPLO 3: Usar em pipeline de ML
# ============================================================================

def exemplo_3_pipeline_ml():
    """Exemplo: Pipeline completo com download e treinamento"""
    print("\n=== EXEMPLO 3: Pipeline ML Completo ===\n")
    
    try:
        # 1. Carregar dados
        print("[1] Carregando dados...")
        path, df = carregar_dataset_kaggle("rabieelkharoua/alzheimers-disease-dataset")
        
        # 2. Pré-processar
        print("[2] Pré-processando...")
        from sklearn.preprocessing import StandardScaler
        
        # Remover colunas desnecessárias
        df = df.drop(columns=[c for c in ['PatientID', 'DoctorInCharge'] if c in df.columns])
        
        X = df.drop(columns=['Diagnosis'])
        y = df['Diagnosis']
        
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"✓ Dados preparados: {X_scaled.shape}")
        
        # 3. Treinar modelo (exemplo)
        print("[3] Treinando modelo...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        print(f"✓ Modelo treinado com acurácia: {score:.2%}")
        
    except Exception as e:
        print(f"✗ Erro no pipeline: {e}")

# ============================================================================
# EXEMPLO 4: Integração com Streamlit
# ============================================================================

def exemplo_4_streamlit():
    """Código para usar em aplicação Streamlit"""
    
    codigo_streamlit = """
    import streamlit as st
    from utils_kagglehub import carregar_dataset_kaggle
    
    st.title("Detector de Alzheimer")
    
    @st.cache_data
    def carregar_dados():
        _, df = carregar_dataset_kaggle("rabieelkharoua/alzheimers-disease-dataset")
        return df
    
    df = carregar_dados()
    st.write(f"Dataset carregado: {df.shape[0]} amostras")
    st.dataframe(df.head())
    """
    
    print("\n=== EXEMPLO 4: Streamlit ===\n")
    print(codigo_streamlit)

# ============================================================================
# EXECUTAR EXEMPLOS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXEMPLOS DE USO DO KAGGLEHUB")
    print("=" * 60)
    
    # Descomente o exemplo que quer testar:
    
    # exemplo_1_download_automatico()
    # exemplo_2_local_ou_kaggle()
    # exemplo_3_pipeline_ml()
    exemplo_4_streamlit()
    
    print("\n" + "=" * 60)
    print("Para mais informações, veja GUIA_KAGGLEHUB.md")
    print("=" * 60)
