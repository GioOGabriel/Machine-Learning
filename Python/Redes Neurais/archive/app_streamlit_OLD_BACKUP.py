"""
===============================================================================
APLICAÇÃO STREAMLIT - DETECÇÃO DE ALZHEIMER
===============================================================================

Interface web para prognóstico de Alzheimer usando Random Forest treinado.
Acesse: streamlit run app_streamlit.py

Autor: Machine Learning Pipeline
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração de página
st.set_page_config(
    page_title="Detecção de Alzheimer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar tema
st.markdown("""
    <style>
        .main-header {
            color: #667eea;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .result-healthy {
            background: #d4edda;
            color: #155724;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #28a745;
        }
        .result-alzheimer {
            background: #f8d7da;
            color: #721c24;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #dc3545;
        }
    </style>
""", unsafe_allow_html=True)

# Paths
BASE_PATH = Path(__file__).parent
MODELS_PATH = BASE_PATH / 'models'
MODEL_FILE = MODELS_PATH / 'alzheimer_rf_model.pkl'
SCALER_FILE = MODELS_PATH / 'alzheimer_rf_scaler.pkl'
FEATURES_FILE = MODELS_PATH / 'alzheimer_rf_features.pkl'

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo, scaler e features do disco."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        features = joblib.load(FEATURES_FILE)
        return model, scaler, features
    except FileNotFoundError:
        st.error("⚠️ Modelo não encontrado! Execute primeiro: python scripts/treinar_e_exportar_modelo.py")
        return None, None, None

def fazer_predicao(model, scaler, features, dados):
    """Faz a predição com base nos dados fornecidos."""
    # Converter para DataFrame com mesma ordem de features
    df = pd.DataFrame([dados])
    
    # Normalizar
    dados_scaled = scaler.transform(df[features])
    
    # Predição
    predicao = model.predict(dados_scaled)[0]
    probabilidade = model.predict_proba(dados_scaled)[0]
    
    return predicao, probabilidade

def criar_formulario():
    """Cria o formulário de entrada de dados."""
    st.subheader("📋 Dados do Paciente")
    
    col1, col2, col3 = st.columns(3)
    
    dados = {}
    
    with col1:
        dados['Age'] = st.number_input("Idade", min_value=18, max_value=100, value=65, step=1)
        dados['Smoking'] = st.selectbox("Fumante?", ["Não", "Sim"])
        dados['Alcohol'] = st.selectbox("Consome álcool?", ["Não", "Sim"])
    
    with col2:
        dados['PhysicalActivity'] = st.number_input("Atividade Física (horas/semana)", min_value=0.0, max_value=40.0, value=3.0, step=0.5)
        dados['BloodPressure'] = st.number_input("Pressão Arterial (mmHg)", min_value=60, max_value=200, value=120, step=1)
        dados['HeartRate'] = st.number_input("Frequência Cardíaca (bpm)", min_value=40, max_value=150, value=75, step=1)
    
    with col3:
        dados['CholesterolLevel'] = st.number_input("Colesterol (mg/dL)", min_value=100, max_value=400, value=200, step=10)
        dados['SleepQuality'] = st.number_input("Qualidade do Sono (0-10)", min_value=0, max_value=10, value=7, step=1)
        dados['MentalStress'] = st.number_input("Estresse Mental (0-10)", min_value=0, max_value=10, value=5, step=1)
    
    # Converter respostas sim/não para 0/1
    dados['Smoking'] = 1 if dados['Smoking'] == "Sim" else 0
    dados['Alcohol'] = 1 if dados['Alcohol'] == "Sim" else 0
    
    return dados

def main():
    # Header
    st.markdown('<div class="main-header">🧠 Detecção de Alzheimer</div>', unsafe_allow_html=True)
    st.markdown("Sistema de Prognóstico baseado em Machine Learning")
    st.divider()
    
    # Carregar modelo
    model, scaler, features = carregar_modelo()
    
    if model is None:
        st.stop()
    
    # Abas
    tab1, tab2, tab3 = st.tabs(["🏥 Prognóstico", "📊 Informações", "ℹ️ Sobre"])
    
    with tab1:
        st.markdown("### Preencha os dados do paciente para obter o prognóstico")
        
        # Formulário
        dados = criar_formulario()
        
        # Botão de predição
        if st.button("🔍 Fazer Prognóstico", use_container_width=True, type="primary"):
            with st.spinner("Processando..."):
                predicao, probabilidade = fazer_predicao(model, scaler, features, dados)
            
            st.divider()
            
            # Resultado
            col1, col2 = st.columns(2)
            
            with col1:
                if predicao == 0:
                    st.markdown(
                        '<div class="result-healthy">'
                        '<h3>✅ Sem Sinais de Alzheimer</h3>'
                        f'<p>Confiança: <strong>{probabilidade[0]*100:.1f}%</strong></p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-alzheimer">'
                        '<h3>⚠️ Possível Alzheimer</h3>'
                        f'<p>Confiança: <strong>{probabilidade[1]*100:.1f}%</strong></p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
            
            with col2:
                # Gráfico de probabilidade
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ["Saudável", "Alzheimer"]
                colors = ["#28a745", "#dc3545"]
                bars = ax.barh(labels, probabilidade, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                
                # Adicionar valores nas barras
                for i, (bar, prob) in enumerate(zip(bars, probabilidade)):
                    ax.text(prob + 0.02, i, f"{prob*100:.1f}%", va='center', fontweight='bold')
                
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probabilidade", fontsize=12, fontweight='bold')
                ax.set_title("Resultado da Predição", fontsize=14, fontweight='bold', color='#667eea')
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)
            
            # Detalhes
            with st.expander("📋 Detalhes da Análise"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dados do Paciente:**")
                    for chave, valor in dados.items():
                        if chave == 'Smoking':
                            st.write(f"- Fumante: {'Sim' if valor == 1 else 'Não'}")
                        elif chave == 'Alcohol':
                            st.write(f"- Consome álcool: {'Sim' if valor == 1 else 'Não'}")
                        else:
                            st.write(f"- {chave}: {valor}")
                
                with col2:
                    st.write("**Fatores de Risco:**")
                    fatores = []
                    if dados['Age'] > 65:
                        fatores.append("⚠️ Idade avançada (> 65 anos)")
                    if dados['Smoking'] == 1:
                        fatores.append("⚠️ Fumante")
                    if dados['Alcohol'] == 1:
                        fatores.append("⚠️ Consome álcool")
                    if dados['PhysicalActivity'] < 2:
                        fatores.append("⚠️ Baixa atividade física")
                    if dados['SleepQuality'] < 6:
                        fatores.append("⚠️ Qualidade do sono ruim")
                    if dados['MentalStress'] > 7:
                        fatores.append("⚠️ Estresse mental elevado")
                    
                    if fatores:
                        for fator in fatores:
                            st.write(fator)
                    else:
                        st.write("✅ Nenhum fator de risco significativo detectado")
    
    with tab2:
        st.markdown("### 📊 Informações do Modelo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                '<div class="metric-card">'
                '<div style="font-size: 2rem; font-weight: bold;">91.22%</div>'
                '<div>Acurácia</div>'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                '<div class="metric-card">'
                '<div style="font-size: 2rem; font-weight: bold;">90.80%</div>'
                '<div>F1-Score</div>'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                '<div class="metric-card">'
                '<div style="font-size: 2rem; font-weight: bold;">96.05%</div>'
                '<div>AUC-ROC</div>'
                '</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        st.markdown("""
        **Algoritmo:** Random Forest com 200 estimadores
        
        **Features:** 33 características clínicas e de estilo de vida
        
        **Validação:** 5-fold Stratified Cross-Validation
        
        **Balanceamento:** SMOTE aplicado apenas no conjunto de treino
        
        **Dataset:** 2.149 amostras
        """)
    
    with tab3:
        st.markdown("### ℹ️ Sobre esta Aplicação")
        
        st.markdown("""
        Esta é uma ferramenta educacional de detecção de Alzheimer baseada em 
        Machine Learning. Utiliza um modelo Random Forest treinado com dados de pacientes.
        
        **⚠️ Aviso Legal:**
        - Esta ferramenta é apenas para fins educacionais e de pesquisa
        - Não substitui diagnóstico médico profissional
        - Sempre consulte um médico especialista para confirmação de diagnóstico
        - Os resultados não devem ser usados como base única para tomada de decisão
        
        **📚 Metodologia:**
        - Comparação de 6 algoritmos diferentes
        - Hiperparâmetros otimizados via Grid Search
        - Validação cruzada estratificada
        - SMOTE para balanceamento de classes
        - StandardScaler para normalização
        
        **👨‍💻 Desenvolvido com:**
        - Python 3.8+
        - Streamlit
        - Scikit-Learn
        - Pandas & NumPy
        """)
        
        st.divider()
        st.markdown("_Desenvolvido como parte de um projeto de Machine Learning para Detecção de Alzheimer_")

if __name__ == "__main__":
    main()
