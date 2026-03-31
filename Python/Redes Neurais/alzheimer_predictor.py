# ==============================================================================
# SISTEMA DE PREDIÇÃO DE ALZHEIMER COM INTERFACE GRADIO
# ==============================================================================
# Este script cria uma interface web interativa para predição de Alzheimer
# baseada em dados clínicos do paciente.
# ==============================================================================

import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS
# ==============================================================================
RANDOM_STATE = 42
MODEL_PATH = "alzheimer_model.pkl"
SCALER_PATH = "alzheimer_scaler.pkl"

# ==============================================================================
# DEFINIÇÃO DAS FEATURES E SUAS DESCRIÇÕES
# ==============================================================================
FEATURES_INFO = {
    'Age': {'default': 75, 'label': 'Idade', 'description': 'Idade do paciente (anos)'},
    'Gender': {'choices': ['Masculino', 'Feminino'], 'default': 'Masculino', 'label': 'Gênero'},
    'Ethnicity': {'choices': ['Caucasiano', 'Afro-Americano', 'Asiático', 'Outro'], 'default': 'Caucasiano', 'label': 'Etnia'},
    'EducationLevel': {'choices': ['Nenhuma', 'Ensino Médio', 'Ensino Superior', 'Pós-Graduação'], 'default': 'Ensino Superior', 'label': 'Nível de Educação'},
    'BMI': {'default': 27.0, 'label': 'IMC (Índice de Massa Corporal)', 'description': 'IMC em kg/m²'},
    'Smoking': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Fumante'},
    'AlcoholConsumption': {'default': 10.0, 'label': 'Consumo de Álcool', 'description': 'Unidades por semana'},
    'PhysicalActivity': {'default': 5.0, 'label': 'Atividade Física', 'description': 'Horas por semana'},
    'DietQuality': {'default': 5.0, 'label': 'Qualidade da Dieta', 'description': 'Score de 0-10'},
    'SleepQuality': {'default': 7.0, 'label': 'Qualidade do Sono', 'description': 'Score de 4-10'},
    'FamilyHistoryAlzheimers': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Histórico Familiar de Alzheimer'},
    'CardiovascularDisease': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Doença Cardiovascular'},
    'Diabetes': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Diabetes'},
    'Depression': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Depressão'},
    'HeadInjury': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Lesão na Cabeça'},
    'Hypertension': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Hipertensão'},
    'SystolicBP': {'default': 120, 'label': 'Pressão Sistólica', 'description': 'mmHg (pode selecionar "Não sei")'},
    'DiastolicBP': {'default': 80, 'label': 'Pressão Diastólica', 'description': 'mmHg (pode selecionar "Não sei")'},
    'CholesterolTotal': {'default': 200.0, 'label': 'Colesterol Total', 'description': 'mg/dL (pode selecionar "Não sei")'},
    'CholesterolLDL': {'default': 100.0, 'label': 'Colesterol LDL', 'description': 'mg/dL (pode selecionar "Não sei")'},
    'CholesterolHDL': {'default': 50.0, 'label': 'Colesterol HDL', 'description': 'mg/dL (pode selecionar "Não sei")'},
    'CholesterolTriglycerides': {'default': 150.0, 'label': 'Triglicerídeos', 'description': 'mg/dL (pode selecionar "Não sei")'},
    'MMSE': {'default': 25.0, 'label': 'MMSE (Mini Exame do Estado Mental)', 'description': 'Score de 0-30 (pode selecionar "Não sei")'},
    'FunctionalAssessment': {'default': 5.0, 'label': 'Avaliação Funcional', 'description': 'Score de 0-10 (pode selecionar "Não sei")'},
    'MemoryComplaints': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Queixas de Memória'},
    'BehavioralProblems': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Problemas Comportamentais'},
    'ADL': {'default': 5.0, 'label': 'ADL (Atividades da Vida Diária)', 'description': 'Score de 0-10 (pode selecionar "Não sei")'},
    'Confusion': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Confusão Mental'},
    'Disorientation': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Desorientação'},
    'PersonalityChanges': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Mudanças de Personalidade'},
    'DifficultyCompletingTasks': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Dificuldade para Completar Tarefas'},
    'Forgetfulness': {'choices': ['Não', 'Sim'], 'default': 'Não', 'label': 'Esquecimento'},
}

# Ordem das features conforme o modelo
FEATURE_ORDER = [
    'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
    'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
    'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE',
    'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL',
    'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
    'Forgetfulness'
]

# ==============================================================================
# CLASSE DO PREDITOR DE ALZHEIMER
# ==============================================================================
class AlzheimerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
    
    def load_or_train_model(self, csv_path=None):
        """Carrega modelo existente ou treina um novo."""
        # Tenta carregar modelo salvo
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                print("Modelo carregado com sucesso!")
                return True
            except Exception as e:
                print(f"Erro ao carregar modelo: {e}")
        
        # Se não existe ou falhou, treina novo modelo
        if csv_path and os.path.exists(csv_path):
            return self.train_model(csv_path)
        
        return False
    
    def train_model(self, csv_path):
        """Treina o modelo com o dataset fornecido."""
        print("Treinando novo modelo...")
        
        try:
            # Carregar dados
            df = pd.read_csv(csv_path)
            
            # Remover colunas não necessárias
            columns_to_remove = ['PatientID', 'DoctorInCharge']
            df = df.drop(columns=[c for c in columns_to_remove if c in df.columns])
            
            # Separar features e target
            X = df.drop(columns=['Diagnosis'])
            y = df['Diagnosis']
            
            # Balancear com SMOTE
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Dividir treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced,
                test_size=0.25,
                random_state=RANDOM_STATE,
                stratify=y_balanced
            )
            
            # Normalizar
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Treinar MLP
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=500,
                random_state=RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.15
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Salvar modelo
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.is_trained = True
            print("Modelo treinado e salvo com sucesso!")
            return True
            
        except Exception as e:
            print(f"Erro ao treinar modelo: {e}")
            return False
    
    def predict(self, patient_data):
        """Faz predição para um paciente."""
        if not self.is_trained:
            return None, None
        
        # Converter para array na ordem correta
        features = np.array([[patient_data[f] for f in FEATURE_ORDER]])
        
        # Normalizar
        features_scaled = self.scaler.transform(features)
        
        # Predição
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probability

# ==============================================================================
# INSTÂNCIA GLOBAL DO PREDITOR
# ==============================================================================
predictor = AlzheimerPredictor()

# ==============================================================================
# FUNÇÕES DE CONVERSÃO
# ==============================================================================
def convert_binary(value):
    """Converte Sim/Não para 1/0."""
    return 1 if value == 'Sim' else 0

def convert_gender(value):
    """Converte gênero para valor numérico."""
    return 1 if value == 'Masculino' else 0

def convert_ethnicity(value):
    """Converte etnia para valor numérico."""
    mapping = {'Caucasiano': 0, 'Afro-Americano': 1, 'Asiático': 2, 'Outro': 3}
    return mapping.get(value, 0)

def convert_education(value):
    """Converte nível de educação para valor numérico."""
    mapping = {'Nenhuma': 0, 'Ensino Médio': 1, 'Ensino Superior': 2, 'Pós-Graduação': 3}
    return mapping.get(value, 1)

def convert_with_nao_sei(value, default):
    """Converte valor ou retorna default se 'Não sei' for selecionado."""
    if value == "Não sei" or value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# ==============================================================================
# FUNÇÃO DE PREDIÇÃO PARA GRADIO
# ==============================================================================
def make_prediction(
    age, gender, ethnicity, education_level, bmi, smoking,
    alcohol, physical_activity, diet_quality, sleep_quality,
    family_history, cardiovascular, diabetes, depression,
    head_injury, hypertension, systolic_bp, diastolic_bp,
    cholesterol_total, cholesterol_ldl, cholesterol_hdl, cholesterol_triglycerides,
    mmse, functional_assessment, memory_complaints, behavioral_problems,
    adl, confusion, disorientation, personality_changes,
    difficulty_tasks, forgetfulness
):
    """Processa os dados do paciente e retorna a predição."""
    
    if not predictor.is_trained:
        return "Modelo não treinado. Por favor, carregue o arquivo CSV primeiro.", "", ""
    
    # Montar dicionário de dados
    patient_data = {
        'Age': age,
        'Gender': convert_gender(gender),
        'Ethnicity': convert_ethnicity(ethnicity),
        'EducationLevel': convert_education(education_level),
        'BMI': bmi,
        'Smoking': convert_binary(smoking),
        'AlcoholConsumption': alcohol,
        'PhysicalActivity': physical_activity,
        'DietQuality': diet_quality,
        'SleepQuality': sleep_quality,
        'FamilyHistoryAlzheimers': convert_binary(family_history),
        'CardiovascularDisease': convert_binary(cardiovascular),
        'Diabetes': convert_binary(diabetes),
        'Depression': convert_binary(depression),
        'HeadInjury': convert_binary(head_injury),
        'Hypertension': convert_binary(hypertension),
        'SystolicBP': convert_with_nao_sei(systolic_bp, 120),
        'DiastolicBP': convert_with_nao_sei(diastolic_bp, 80),
        'CholesterolTotal': convert_with_nao_sei(cholesterol_total, 200.0),
        'CholesterolLDL': convert_with_nao_sei(cholesterol_ldl, 100.0),
        'CholesterolHDL': convert_with_nao_sei(cholesterol_hdl, 50.0),
        'CholesterolTriglycerides': convert_with_nao_sei(cholesterol_triglycerides, 150.0),
        'MMSE': convert_with_nao_sei(mmse, 25.0),
        'FunctionalAssessment': convert_with_nao_sei(functional_assessment, 5.0),
        'MemoryComplaints': convert_binary(memory_complaints),
        'BehavioralProblems': convert_binary(behavioral_problems),
        'ADL': convert_with_nao_sei(adl, 5.0),
        'Confusion': convert_binary(confusion),
        'Disorientation': convert_binary(disorientation),
        'PersonalityChanges': convert_binary(personality_changes),
        'DifficultyCompletingTasks': convert_binary(difficulty_tasks),
        'Forgetfulness': convert_binary(forgetfulness),
    }
    
    # Fazer predição
    prediction, probability = predictor.predict(patient_data)
    
    if prediction is None:
        return "Erro na predição.", "", ""
    
    # Formatar resultado
    prob_alzheimer = probability[1] * 100
    prob_saudavel = probability[0] * 100
    
    if prediction == 1:
        resultado = f"POSITIVO PARA ALZHEIMER"
        cor_resultado = "red"
        emoji = "⚠️"
    else:
        resultado = f"NEGATIVO PARA ALZHEIMER"
        cor_resultado = "green"
        emoji = "✅"
    
    # Análise de risco
    if prob_alzheimer >= 80:
        nivel_risco = "MUITO ALTO"
        recomendacao = "Recomenda-se consulta urgente com neurologista e exames complementares (ressonância magnética, PET scan)."
    elif prob_alzheimer >= 60:
        nivel_risco = "ALTO"
        recomendacao = "Recomenda-se acompanhamento médico especializado e avaliação neuropsicológica completa."
    elif prob_alzheimer >= 40:
        nivel_risco = "MODERADO"
        recomendacao = "Recomenda-se monitoramento regular e consulta com geriatra ou neurologista."
    elif prob_alzheimer >= 20:
        nivel_risco = "BAIXO"
        recomendacao = "Manter hábitos saudáveis e check-ups regulares. Considerar avaliação cognitiva preventiva."
    else:
        nivel_risco = "MUITO BAIXO"
        recomendacao = "Continue mantendo um estilo de vida saudável com exercícios físicos e mentais."
    
    resultado_html = f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {'#ffebee' if prediction == 1 else '#e8f5e9'};">
        <h1 style="color: {cor_resultado}; margin-bottom: 10px;">{emoji} {resultado}</h1>
        <h2 style="color: #333;">Probabilidade de Alzheimer: {prob_alzheimer:.1f}%</h2>
        <h3 style="color: #666;">Probabilidade de Saudável: {prob_saudavel:.1f}%</h3>
        <hr style="margin: 20px 0;">
        <h3 style="color: {'#c62828' if nivel_risco in ['MUITO ALTO', 'ALTO'] else '#ff8f00' if nivel_risco == 'MODERADO' else '#2e7d32'};">
            Nível de Risco: {nivel_risco}
        </h3>
        <p style="font-size: 14px; color: #555; margin-top: 15px; text-align: left; padding: 10px; background-color: #f5f5f5; border-radius: 5px;">
            <strong>Recomendação:</strong> {recomendacao}
        </p>
    </div>
    """
    
    # Barra de probabilidade
    barra_html = f"""
    <div style="margin-top: 20px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="color: green;">Saudável ({prob_saudavel:.1f}%)</span>
            <span style="color: red;">Alzheimer ({prob_alzheimer:.1f}%)</span>
        </div>
        <div style="width: 100%; height: 30px; background: linear-gradient(to right, #4caf50 {prob_saudavel}%, #f44336 {prob_saudavel}%); border-radius: 15px;">
        </div>
    </div>
    """
    
    # Detalhes dos fatores de risco
    fatores_risco = []
    if convert_binary(family_history) == 1:
        fatores_risco.append("Histórico familiar de Alzheimer")
    if age >= 75:
        fatores_risco.append(f"Idade avançada ({age} anos)")
    if mmse < 24:
        fatores_risco.append(f"MMSE baixo ({mmse:.1f})")
    if convert_binary(memory_complaints) == 1:
        fatores_risco.append("Queixas de memória")
    if convert_binary(confusion) == 1:
        fatores_risco.append("Confusão mental")
    if convert_binary(disorientation) == 1:
        fatores_risco.append("Desorientação")
    if convert_binary(depression) == 1:
        fatores_risco.append("Depressão")
    if convert_binary(cardiovascular) == 1:
        fatores_risco.append("Doença cardiovascular")
    if convert_binary(diabetes) == 1:
        fatores_risco.append("Diabetes")
    if convert_binary(head_injury) == 1:
        fatores_risco.append("Histórico de lesão na cabeça")
    
    fatores_html = ""
    if fatores_risco:
        fatores_html = f"""
        <div style="margin-top: 20px; padding: 15px; background-color: #fff3e0; border-radius: 10px;">
            <h4 style="color: #e65100; margin-bottom: 10px;">Fatores de Risco Identificados:</h4>
            <ul style="margin: 0; padding-left: 20px;">
                {"".join(f'<li style="margin-bottom: 5px;">{fator}</li>' for fator in fatores_risco)}
            </ul>
        </div>
        """
    
    return resultado_html, barra_html, fatores_html

def load_csv(file):
    """Carrega o CSV e treina o modelo."""
    if file is None:
        return "Por favor, selecione um arquivo CSV."
    
    # Salvar arquivo temporariamente
    csv_path = file.name if hasattr(file, 'name') else str(file)
    
    if predictor.train_model(csv_path):
        return "Modelo treinado com sucesso! Agora você pode fazer predições."
    else:
        return "Erro ao treinar o modelo. Verifique se o arquivo CSV está no formato correto."

# ==============================================================================
# INTERFACE GRADIO
# ==============================================================================
def create_interface():
    """Cria a interface Gradio."""
    
    with gr.Blocks(title="Preditor de Alzheimer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🧠 Sistema de Predição de Alzheimer
        
        Este sistema utiliza uma **Rede Neural Artificial (MLP)** treinada com dados clínicos 
        para estimar a probabilidade de um paciente desenvolver Alzheimer.
        
        **Importante:** Este é um sistema de apoio à decisão médica. Os resultados devem ser 
        interpretados por profissionais de saúde qualificados e não substituem diagnóstico médico.
        
        ---
        """)
        
        # Seção de carregamento do modelo
        with gr.Accordion("📁 Carregar Dataset (Necessário na primeira vez)", open=not predictor.is_trained):
            gr.Markdown("Carregue o arquivo `alzheimers_disease_data.csv` para treinar o modelo.")
            with gr.Row():
                csv_input = gr.File(label="Selecione o arquivo CSV", file_types=[".csv"])
                load_btn = gr.Button("Treinar Modelo", variant="primary")
            load_status = gr.Textbox(label="Status", interactive=False)
            load_btn.click(load_csv, inputs=[csv_input], outputs=[load_status])
        
        gr.Markdown("---")
        gr.Markdown("## 📋 Dados do Paciente")
        
        # Dados Demográficos
        with gr.Accordion("👤 Dados Demográficos", open=True):
            with gr.Row():
                age = gr.Number(value=75, label="Idade (anos)", precision=0)
                gender = gr.Radio(["Masculino", "Feminino"], value="Masculino", label="Gênero")
            with gr.Row():
                ethnicity = gr.Dropdown(["Caucasiano", "Afro-Americano", "Asiático", "Outro"], 
                                       value="Caucasiano", label="Etnia")
                education = gr.Dropdown(["Nenhuma", "Ensino Médio", "Ensino Superior", "Pós-Graduação"],
                                       value="Ensino Superior", label="Nível de Educação")
        
        # Estilo de Vida
        with gr.Accordion("🏃 Estilo de Vida", open=True):
            with gr.Row():
                bmi = gr.Number(value=27.0, label="IMC (kg/m²)")
                smoking = gr.Radio(["Não", "Sim"], value="Não", label="Fumante")
            with gr.Row():
                alcohol = gr.Number(value=10.0, label="Consumo de Álcool (unidades/semana)")
                physical_activity = gr.Number(value=5.0, label="Atividade Física (horas/semana)")
            with gr.Row():
                diet_quality = gr.Number(value=5.0, label="Qualidade da Dieta (0-10)")
                sleep_quality = gr.Number(value=7.0, label="Qualidade do Sono (4-10)")
        
        # Histórico Médico
        with gr.Accordion("🏥 Histórico Médico", open=True):
            with gr.Row():
                family_history = gr.Radio(["Não", "Sim"], value="Não", label="Histórico Familiar de Alzheimer")
                cardiovascular = gr.Radio(["Não", "Sim"], value="Não", label="Doença Cardiovascular")
            with gr.Row():
                diabetes = gr.Radio(["Não", "Sim"], value="Não", label="Diabetes")
                depression = gr.Radio(["Não", "Sim"], value="Não", label="Depressão")
            with gr.Row():
                head_injury = gr.Radio(["Não", "Sim"], value="Não", label="Lesão na Cabeça")
                hypertension = gr.Radio(["Não", "Sim"], value="Não", label="Hipertensão")
        
        # Sinais Vitais
        with gr.Accordion("💓 Sinais Vitais", open=True):
            gr.Markdown("*Selecione 'Não sei' se não souber o valor. Um valor médio será usado.*")
            with gr.Row():
                systolic_bp = gr.Dropdown(["Não sei"] + [str(i) for i in range(90, 181, 5)], 
                                         value="120", label="Pressão Sistólica (mmHg)")
                diastolic_bp = gr.Dropdown(["Não sei"] + [str(i) for i in range(60, 121, 5)], 
                                          value="80", label="Pressão Diastólica (mmHg)")
            with gr.Row():
                cholesterol_total = gr.Dropdown(["Não sei"] + [str(i) for i in range(100, 351, 10)], 
                                               value="200", label="Colesterol Total (mg/dL)")
                cholesterol_ldl = gr.Dropdown(["Não sei"] + [str(i) for i in range(30, 201, 10)], 
                                             value="100", label="Colesterol LDL (mg/dL)")
            with gr.Row():
                cholesterol_hdl = gr.Dropdown(["Não sei"] + [str(i) for i in range(20, 101, 5)], 
                                             value="50", label="Colesterol HDL (mg/dL)")
                cholesterol_triglycerides = gr.Dropdown(["Não sei"] + [str(i) for i in range(50, 451, 25)], 
                                                       value="150", label="Triglicerídeos (mg/dL)")
        
        # Avaliação Cognitiva
        with gr.Accordion("🧠 Avaliação Cognitiva e Funcional", open=True):
            gr.Markdown("*Selecione 'Não sei' se não souber o valor ou não tiver feito o teste.*")
            with gr.Row():
                mmse = gr.Dropdown(["Não sei"] + [str(i) for i in range(0, 31)], 
                                  value="25", label="MMSE - Mini Exame do Estado Mental (0-30, maior=melhor)")
                functional_assessment = gr.Dropdown(["Não sei"] + [str(i) for i in range(0, 11)], 
                                                   value="5", label="Avaliação Funcional (0-10)")
            with gr.Row():
                adl = gr.Dropdown(["Não sei"] + [str(i) for i in range(0, 11)], 
                                 value="5", label="ADL - Atividades da Vida Diária (0-10)")
        
        # Sintomas
        with gr.Accordion("⚠️ Sintomas", open=True):
            with gr.Row():
                memory_complaints = gr.Radio(["Não", "Sim"], value="Não", label="Queixas de Memória")
                behavioral_problems = gr.Radio(["Não", "Sim"], value="Não", label="Problemas Comportamentais")
            with gr.Row():
                confusion = gr.Radio(["Não", "Sim"], value="Não", label="Confusão Mental")
                disorientation = gr.Radio(["Não", "Sim"], value="Não", label="Desorientação")
            with gr.Row():
                personality_changes = gr.Radio(["Não", "Sim"], value="Não", label="Mudanças de Personalidade")
                difficulty_tasks = gr.Radio(["Não", "Sim"], value="Não", label="Dificuldade para Completar Tarefas")
            with gr.Row():
                forgetfulness = gr.Radio(["Não", "Sim"], value="Não", label="Esquecimento")
        
        gr.Markdown("---")
        
        # Botão de predição
        predict_btn = gr.Button("🔍 Realizar Predição", variant="primary", size="lg")
        
        # Resultados
        gr.Markdown("## 📊 Resultados")
        
        with gr.Row():
            with gr.Column(scale=2):
                resultado_output = gr.HTML(label="Diagnóstico")
            with gr.Column(scale=1):
                barra_output = gr.HTML(label="Probabilidade")
        
        fatores_output = gr.HTML(label="Fatores de Risco")
        
        # Conectar botão à função de predição
        predict_btn.click(
            make_prediction,
            inputs=[
                age, gender, ethnicity, education, bmi, smoking,
                alcohol, physical_activity, diet_quality, sleep_quality,
                family_history, cardiovascular, diabetes, depression,
                head_injury, hypertension, systolic_bp, diastolic_bp,
                cholesterol_total, cholesterol_ldl, cholesterol_hdl, cholesterol_triglycerides,
                mmse, functional_assessment, memory_complaints, behavioral_problems,
                adl, confusion, disorientation, personality_changes,
                difficulty_tasks, forgetfulness
            ],
            outputs=[resultado_output, barra_output, fatores_output]
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ Informações sobre as Métricas
        
        - **MMSE (Mini Mental State Examination):** Teste cognitivo padrão. Scores < 24 podem indicar comprometimento.
        - **ADL (Activities of Daily Living):** Capacidade de realizar atividades diárias de forma independente.
        - **Avaliação Funcional:** Avaliação geral da capacidade funcional do paciente.
        
        ### ⚖️ Disclaimer
        
        Este sistema é apenas uma ferramenta de apoio e **não substitui avaliação médica profissional**. 
        Consulte sempre um neurologista ou geriatra para diagnóstico definitivo.
        """)
    
    return interface

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # Tentar carregar modelo existente
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Verificar se existe modelo salvo
    predictor.load_or_train_model()
    
    # Criar e lançar interface
    interface = create_interface()
    interface.launch(
        share=False,  # Mudar para True se quiser compartilhar publicamente
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )
