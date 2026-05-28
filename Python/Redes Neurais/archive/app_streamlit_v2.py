"""
===============================================================================
APLICAÇÃO STREAMLIT - DETECÇÃO DE ALZHEIMER (VERSÃO SIMPLIFICADA)
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuração de página
st.set_page_config(
    page_title="Detecção de Alzheimer",
    page_icon="🧠",
    layout="wide"
)

# CSS Styling
st.markdown("""
    <style>
    .big-font {
        font-size:2rem !important;
        font-weight:bold !important;
        color:#667eea;
    }
    .metric-box {
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
    """Carrega o modelo, scaler e features."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        features = joblib.load(FEATURES_FILE)
        return model, scaler, features, True
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None, None, False

def fazer_predicao(model, scaler, features, dados):
    """Faz predição com os dados fornecidos."""
    try:
        # Converter para DataFrame
        df = pd.DataFrame([dados])
        
        # Normalizar
        dados_scaled = scaler.transform(df[features])
        
        # Predição
        predicao = model.predict(dados_scaled)[0]
        probabilidade = model.predict_proba(dados_scaled)[0]
        
        return predicao, probabilidade, True
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return None, None, False

def main():
    # Header
    st.markdown('<p class="big-font">🧠 Detecção de Alzheimer</p>', unsafe_allow_html=True)
    st.markdown("Sistema de Prognóstico baseado em Machine Learning")
    st.divider()
    
    # Carregar modelo
    model, scaler, features = carregar_modelo()[:3]
    
    if model is None:
        st.error("Modelo não carregado. Verifique se os arquivos estão em models/")
        st.stop()
    
    # Abas
    tab1, tab2, tab3 = st.tabs(["Prognóstico", "Informações", "Sobre"])
    
    with tab1:
        st.subheader("Preencha os dados do paciente")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Idade", min_value=18, max_value=100, value=65, step=1)
            gender = st.selectbox("Genero", ["Masculino", "Feminino"])
            ethnicity = st.selectbox("Etnia", ["Caucasiano", "Afro-americano", "Hispanico", "Asiatico"])
            education_level = st.selectbox("Educacao", ["Primaria", "Secundaria", "Terciaria"])
            bmi = st.number_input("IMC", min_value=15.0, max_value=50.0, value=25.0, step=0.5)
        
        with col2:
            smoking = st.selectbox("Fumante?", ["Nao", "Sim"])
            alcohol = st.selectbox("Consumo de Alcool", ["Nao", "Leve", "Moderado", "Severo"])
            physical_activity = st.number_input("Atividade Fisica (horas/semana)", min_value=0.0, max_value=40.0, value=3.0, step=0.5)
            diet_quality = st.number_input("Qualidade da Dieta (0-10)", min_value=0, max_value=10, value=5, step=1)
            sleep_quality = st.number_input("Qualidade do Sono (0-10)", min_value=0, max_value=10, value=7, step=1)
        
        with col3:
            family_history = st.selectbox("Historico Familiar Alzheimer?", ["Nao", "Sim"])
            cardiovascular = st.selectbox("Doenca Cardiovascular?", ["Nao", "Sim"])
            diabetes = st.selectbox("Diabetes?", ["Nao", "Sim"])
            depression = st.selectbox("Depressao?", ["Nao", "Sim"])
            head_injury = st.selectbox("Lesao Craniana Previa?", ["Nao", "Sim"])
        
        # Row 2
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hypertension = st.selectbox("Hipertensao?", ["Nao", "Sim"])
            systolic_bp = st.number_input("PA Sistolica (mmHg)", min_value=60, max_value=200, value=120, step=1)
            diastolic_bp = st.number_input("PA Diastolica (mmHg)", min_value=40, max_value=120, value=80, step=1)
        
        with col2:
            cholesterol_total = st.number_input("Colesterol Total (mg/dL)", min_value=100, max_value=400, value=200, step=10)
            cholesterol_ldl = st.number_input("Colesterol LDL (mg/dL)", min_value=20, max_value=200, value=100, step=10)
            cholesterol_hdl = st.number_input("Colesterol HDL (mg/dL)", min_value=20, max_value=100, value=50, step=10)
        
        with col3:
            cholesterol_triglycerides = st.number_input("Triglicerides (mg/dL)", min_value=50, max_value=400, value=150, step=10)
            mmse = st.number_input("MMSE (0-30)", min_value=0, max_value=30, value=28, step=1)
            functional_assessment = st.number_input("Avaliacao Funcional (0-10)", min_value=0, max_value=10, value=8, step=1)
        
        # Row 3
        col1, col2 = st.columns(2)
        
        with col1:
            memory_complaints = st.selectbox("Queixas de Memoria?", ["Nao", "Sim"])
            behavioral_problems = st.selectbox("Problemas Comportamentais?", ["Nao", "Sim"])
            adl = st.selectbox("ADL (Atividades Diarias)?", ["Independente", "Dependente"])
            confusion = st.selectbox("Confusao?", ["Nao", "Sim"])
        
        with col2:
            disorientation = st.selectbox("Desorientacao?", ["Nao", "Sim"])
            personality_changes = st.selectbox("Mudancas de Personalidade?", ["Nao", "Sim"])
            difficulty_tasks = st.selectbox("Dificuldade em Tarefas?", ["Nao", "Sim"])
            forgetfulness = st.selectbox("Esquecimento?", ["Nao", "Sim"])
        
        # Preparar dados
        dados = {
            'Age': age,
            'Gender': 0 if gender == "Masculino" else 1,
            'Ethnicity': ["Caucasiano", "Afro-americano", "Hispanico", "Asiatico"].index(ethnicity),
            'EducationLevel': ["Primaria", "Secundaria", "Terciaria"].index(education_level),
            'BMI': bmi,
            'Smoking': 1 if smoking == "Sim" else 0,
            'AlcoholConsumption': ["Nao", "Leve", "Moderado", "Severo"].index(alcohol),
            'PhysicalActivity': physical_activity,
            'DietQuality': diet_quality,
            'SleepQuality': sleep_quality,
            'FamilyHistoryAlzheimers': 1 if family_history == "Sim" else 0,
            'CardiovascularDisease': 1 if cardiovascular == "Sim" else 0,
            'Diabetes': 1 if diabetes == "Sim" else 0,
            'Depression': 1 if depression == "Sim" else 0,
            'HeadInjury': 1 if head_injury == "Sim" else 0,
            'Hypertension': 1 if hypertension == "Sim" else 0,
            'SystolicBP': systolic_bp,
            'DiastolicBP': diastolic_bp,
            'CholesterolTotal': cholesterol_total,
            'CholesterolLDL': cholesterol_ldl,
            'CholesterolHDL': cholesterol_hdl,
            'CholesterolTriglycerides': cholesterol_triglycerides,
            'MMSE': mmse,
            'FunctionalAssessment': functional_assessment,
            'MemoryComplaints': 1 if memory_complaints == "Sim" else 0,
            'BehavioralProblems': 1 if behavioral_problems == "Sim" else 0,
            'ADL': 0 if adl == "Independente" else 1,
            'Confusion': 1 if confusion == "Sim" else 0,
            'Disorientation': 1 if disorientation == "Sim" else 0,
            'PersonalityChanges': 1 if personality_changes == "Sim" else 0,
            'DifficultyCompletingTasks': 1 if difficulty_tasks == "Sim" else 0,
            'Forgetfulness': 1 if forgetfulness == "Sim" else 0,
        }
        
        # Botão de predição
        if st.button("Fazer Prognóstico", use_container_width=True, type="primary"):
            with st.spinner("Processando..."):
                predicao, probabilidade, sucesso = fazer_predicao(model, scaler, features, dados)
            
            if sucesso:
                st.divider()
                
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
                    fig, ax = plt.subplots(figsize=(6, 4))
                    labels = ["Saudável", "Alzheimer"]
                    colors = ["#28a745", "#dc3545"]
                    bars = ax.barh(labels, probabilidade, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                    
                    for i, (bar, prob) in enumerate(zip(bars, probabilidade)):
                        ax.text(prob + 0.02, i, f"{prob*100:.1f}%", va='center', fontweight='bold')
                    
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Probabilidade", fontsize=12, fontweight='bold')
                    ax.set_title("Resultado da Predição", fontsize=14, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
    
    with tab2:
        st.subheader("Informações do Modelo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                '<div class="metric-box">'
                '<div style="font-size: 2rem; font-weight: bold;">91.82%</div>'
                '<div>Acurácia</div>'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                '<div class="metric-box">'
                '<div style="font-size: 2rem; font-weight: bold;">87.78%</div>'
                '<div>F1-Score</div>'
                '</div>',
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                '<div class="metric-box">'
                '<div style="font-size: 2rem; font-weight: bold;">93.81%</div>'
                '<div>AUC-ROC</div>'
                '</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        st.markdown("""
        **Algoritmo:** Random Forest com 200 estimadores
        
        **Características:** 32 features clínicas e de estilo de vida
        
        **Dataset:** 2.149 amostras de pacientes
        
        **Validação:** 5-fold Stratified Cross-Validation
        
        **Balanceamento:** SMOTE aplicado no treino
        """)
    
    with tab3:
        st.subheader("Sobre esta Aplicação")
        
        st.warning("""
        ⚠️ **DISCLAIMER:**
        
        Esta ferramenta é apenas para fins **educacionais e de pesquisa**.
        
        **NÃO substitui diagnóstico médico profissional.**
        
        Sempre consulte um médico especialista para confirmação de diagnóstico.
        """)
        
        st.markdown("""
        **Metodologia:**
        - Comparação de 6 algoritmos diferentes
        - Hiperparâmetros otimizados via Grid Search
        - SMOTE para balanceamento de classes (apenas no treino)
        - StandardScaler para normalização
        
        **Desenvolvido com:**
        - Python 3.8+
        - Streamlit
        - Scikit-Learn
        - Pandas & NumPy
        """)

if __name__ == "__main__":
    main()
