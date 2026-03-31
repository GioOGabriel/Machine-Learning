# ==============================================================================
# SISTEMA DE PREDICAO DE ALZHEIMER - INTERFACE DE TERMINAL
# ==============================================================================
# Este script cria uma interface interativa no terminal para predicao de Alzheimer
# O usuario insere seus dados e o algoritmo retorna a tendencia de desenvolver Alzheimer.
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
import os
import pickle
import sys
from colorama import init, Fore, Back, Style
import tkinter as tk
from tkinter import filedialog

# Inicializar colorama para cores no terminal Windows
init()

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURACOES GLOBAIS
# ==============================================================================
RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "alzheimer_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "alzheimer_scaler.pkl")

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
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                return True
            except Exception as e:
                print(f"{Fore.RED}Erro ao carregar modelo: {e}{Style.RESET_ALL}")
        
        if csv_path and os.path.exists(csv_path):
            return self.train_model(csv_path)
        
        return False
    
    def train_model(self, csv_path):
        """Treina o modelo com o dataset fornecido."""
        print(f"\n{Fore.CYAN}Treinando modelo... Por favor, aguarde.{Style.RESET_ALL}")
        
        try:
            df = pd.read_csv(csv_path)
            
            columns_to_remove = ['PatientID', 'DoctorInCharge']
            df = df.drop(columns=[c for c in columns_to_remove if c in df.columns])
            
            X = df.drop(columns=['Diagnosis'])
            y = df['Diagnosis']
            
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced,
                test_size=0.25,
                random_state=RANDOM_STATE,
                stratify=y_balanced
            )
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
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
            
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.is_trained = True
            print(f"{Fore.GREEN}Modelo treinado e salvo com sucesso!{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Erro ao treinar modelo: {e}{Style.RESET_ALL}")
            return False
    
    def predict(self, patient_data):
        """Faz predicao para um paciente."""
        if not self.is_trained:
            return None, None
        
        features = np.array([[patient_data[f] for f in FEATURE_ORDER]])
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probability


# ==============================================================================
# FUNCOES DE INTERFACE
# ==============================================================================
def limpar_tela():
    """Limpa a tela do terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_cabecalho():
    """Imprime o cabecalho do sistema."""
    limpar_tela()
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{' '*15}SISTEMA DE PREDICAO DE ALZHEIMER")
    print(f"{' '*10}Interface de Avaliacao de Risco Cognitivo")
    print(f"{'='*70}{Style.RESET_ALL}\n")

def print_secao(titulo):
    """Imprime um titulo de secao."""
    print(f"\n{Fore.YELLOW}{'-'*50}")
    print(f"  {titulo}")
    print(f"{'-'*50}{Style.RESET_ALL}\n")

def input_numero(prompt, tipo=float, default=None):
    """Solicita um numero sem limite de intervalo."""
    while True:
        try:
            default_str = f" [{default}]" if default is not None else ""
            entrada = input(f"{Fore.WHITE}{prompt}{default_str}: {Style.RESET_ALL}")
            
            if entrada.strip() == "" and default is not None:
                return tipo(default)
            
            valor = tipo(entrada)
            return valor
        except ValueError:
            print(f"{Fore.RED}  Entrada invalida. Por favor, insira um numero.{Style.RESET_ALL}")

def input_numero_ou_nao_sei(prompt, tipo=float, default=None, valor_nao_sei=None):
    """Solicita um numero ou aceita 'nao sei' como resposta."""
    while True:
        try:
            default_str = f" [{default}]" if default is not None else ""
            entrada = input(f"{Fore.WHITE}{prompt}{default_str} (ou 'ns' se nao souber): {Style.RESET_ALL}").strip().lower()
            
            if entrada == "" and default is not None:
                return tipo(default)
            
            if entrada in ["ns", "nao sei", "não sei", "n/s", "?"]:
                return tipo(valor_nao_sei) if valor_nao_sei is not None else tipo(default)
            
            valor = tipo(entrada)
            return valor
        except ValueError:
            print(f"{Fore.RED}  Entrada invalida. Por favor, insira um numero ou 'ns' se nao souber.{Style.RESET_ALL}")

def input_sim_nao(prompt, default="N"):
    """Solicita uma resposta Sim/Nao."""
    while True:
        default_str = "S/n" if default.upper() == "S" else "s/N"
        entrada = input(f"{Fore.WHITE}{prompt} [{default_str}]: {Style.RESET_ALL}").strip().upper()
        
        if entrada == "":
            return 1 if default.upper() == "S" else 0
        elif entrada in ["S", "SIM", "Y", "YES", "1"]:
            return 1
        elif entrada in ["N", "NAO", "NO", "0"]:
            return 0
        else:
            print(f"{Fore.RED}  Por favor, responda S (sim) ou N (nao).{Style.RESET_ALL}")

def input_opcao(prompt, opcoes, default=0):
    """Solicita a selecao de uma opcao."""
    print(f"{Fore.WHITE}{prompt}:{Style.RESET_ALL}")
    for i, opcao in enumerate(opcoes):
        marcador = f"{Fore.GREEN}*{Style.RESET_ALL}" if i == default else " "
        print(f"  {marcador} {i+1}. {opcao}")
    
    while True:
        entrada = input(f"{Fore.WHITE}  Escolha (1-{len(opcoes)}) [{default+1}]: {Style.RESET_ALL}").strip()
        
        if entrada == "":
            return default
        
        try:
            escolha = int(entrada)
            if 1 <= escolha <= len(opcoes):
                return escolha - 1
            else:
                print(f"{Fore.RED}  Por favor, escolha uma opcao entre 1 e {len(opcoes)}.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}  Entrada invalida.{Style.RESET_ALL}")

def coletar_dados_paciente():
    """Coleta todos os dados do paciente via terminal."""
    dados = {}
    
    # ===== DADOS DEMOGRAFICOS =====
    print_secao("DADOS DEMOGRAFICOS")
    
    dados['Age'] = input_numero("Idade (anos)", int, 75)
    
    genero = input_opcao("Genero", ["Masculino", "Feminino"], 0)
    dados['Gender'] = genero  # 0 = Masculino, 1 = Feminino (invertido para o modelo)
    # Na verdade o modelo usa: Masculino = 1, Feminino = 0
    dados['Gender'] = 1 if genero == 0 else 0
    
    etnia = input_opcao("Etnia", ["Caucasiano", "Afro-Americano", "Asiatico", "Outro"], 0)
    dados['Ethnicity'] = etnia
    
    educacao = input_opcao("Nivel de Educacao", 
                          ["Nenhuma", "Ensino Medio", "Ensino Superior", "Pos-Graduacao"], 2)
    dados['EducationLevel'] = educacao
    
    # ===== ESTILO DE VIDA =====
    print_secao("ESTILO DE VIDA")
    
    dados['BMI'] = input_numero("Indice de Massa Corporal (IMC)", float, 27.0)
    dados['Smoking'] = input_sim_nao("Voce fuma?", "N")
    dados['AlcoholConsumption'] = input_numero("Consumo de alcool (unidades/semana)", float, 5.0)
    dados['PhysicalActivity'] = input_numero("Atividade fisica (horas/semana)", float, 5.0)
    dados['DietQuality'] = input_numero("Qualidade da dieta (0=pessima, 10=excelente)", float, 5.0)
    dados['SleepQuality'] = input_numero("Qualidade do sono (4=pessima, 10=excelente)", float, 7.0)
    
    # ===== HISTORICO MEDICO =====
    print_secao("HISTORICO MEDICO")
    
    dados['FamilyHistoryAlzheimers'] = input_sim_nao("Historico familiar de Alzheimer?", "N")
    dados['CardiovascularDisease'] = input_sim_nao("Possui doenca cardiovascular?", "N")
    dados['Diabetes'] = input_sim_nao("Possui diabetes?", "N")
    dados['Depression'] = input_sim_nao("Possui ou ja teve depressao?", "N")
    dados['HeadInjury'] = input_sim_nao("Ja teve lesao na cabeca?", "N")
    dados['Hypertension'] = input_sim_nao("Possui hipertensao?", "N")
    
    # ===== SINAIS VITAIS =====
    print_secao("SINAIS VITAIS E EXAMES")
    
    print(f"{Fore.CYAN}  Se voce nao souber algum valor, digite 'ns' para usar um valor medio.{Style.RESET_ALL}\n")
    
    dados['SystolicBP'] = input_numero_ou_nao_sei("Pressao arterial sistolica (mmHg)", int, 120, 120)
    dados['DiastolicBP'] = input_numero_ou_nao_sei("Pressao arterial diastolica (mmHg)", int, 80, 80)
    dados['CholesterolTotal'] = input_numero_ou_nao_sei("Colesterol total (mg/dL)", float, 200.0, 200.0)
    dados['CholesterolLDL'] = input_numero_ou_nao_sei("Colesterol LDL (mg/dL)", float, 100.0, 100.0)
    dados['CholesterolHDL'] = input_numero_ou_nao_sei("Colesterol HDL (mg/dL)", float, 50.0, 50.0)
    dados['CholesterolTriglycerides'] = input_numero_ou_nao_sei("Triglicerideos (mg/dL)", float, 150.0, 150.0)
    
    # ===== AVALIACAO COGNITIVA =====
    print_secao("AVALIACAO COGNITIVA")
    
    print(f"{Fore.CYAN}  O MMSE (Mini Exame do Estado Mental) avalia funcoes cognitivas.")
    print(f"  Pontuacao: 0-30 (maior = melhor). Valores < 24 podem indicar comprometimento.")
    print(f"  Se nao souber, digite 'ns' para usar um valor medio.{Style.RESET_ALL}\n")
    
    dados['MMSE'] = input_numero_ou_nao_sei("Pontuacao MMSE (0-30)", float, 25.0, 25.0)
    dados['FunctionalAssessment'] = input_numero_ou_nao_sei("Avaliacao funcional (0-10)", float, 5.0, 5.0)
    dados['ADL'] = input_numero_ou_nao_sei("Atividades da Vida Diaria - ADL (0-10)", float, 5.0, 5.0)
    
    # ===== SINTOMAS =====
    print_secao("SINTOMAS ATUAIS")
    
    print(f"{Fore.CYAN}  Responda sobre sintomas que voce tem percebido recentemente:{Style.RESET_ALL}\n")
    
    dados['MemoryComplaints'] = input_sim_nao("Tem queixas de memoria?", "N")
    dados['BehavioralProblems'] = input_sim_nao("Apresenta problemas comportamentais?", "N")
    dados['Confusion'] = input_sim_nao("Apresenta confusao mental?", "N")
    dados['Disorientation'] = input_sim_nao("Apresenta desorientacao (tempo/espaco)?", "N")
    dados['PersonalityChanges'] = input_sim_nao("Notou mudancas de personalidade?", "N")
    dados['DifficultyCompletingTasks'] = input_sim_nao("Tem dificuldade para completar tarefas?", "N")
    dados['Forgetfulness'] = input_sim_nao("Tem esquecimento frequente?", "N")
    
    return dados

def mostrar_resultado(prediction, probability, dados):
    """Mostra o resultado da predicao de forma formatada."""
    prob_alzheimer = probability[1] * 100
    prob_saudavel = probability[0] * 100
    
    print_cabecalho()
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{' '*20}RESULTADO DA AVALIACAO")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    # Resultado principal
    if prediction == 1:
        print(f"{Back.RED}{Fore.WHITE}{' '*70}")
        print(f"{'  RESULTADO: TENDENCIA POSITIVA PARA ALZHEIMER':<70}")
        print(f"{' '*70}{Style.RESET_ALL}")
    else:
        print(f"{Back.GREEN}{Fore.WHITE}{' '*70}")
        print(f"{'  RESULTADO: TENDENCIA NEGATIVA PARA ALZHEIMER':<70}")
        print(f"{' '*70}{Style.RESET_ALL}")
    
    # Probabilidades
    print(f"\n{Fore.WHITE}  PROBABILIDADES:")
    print(f"  {'-'*40}{Style.RESET_ALL}")
    
    # Barra visual de probabilidade
    tamanho_barra = 40
    preenchido_alzheimer = int((prob_alzheimer / 100) * tamanho_barra)
    
    barra = f"{Fore.GREEN}{'█' * (tamanho_barra - preenchido_alzheimer)}{Fore.RED}{'█' * preenchido_alzheimer}{Style.RESET_ALL}"
    
    print(f"\n  {Fore.GREEN}Saudavel{Style.RESET_ALL} [{barra}] {Fore.RED}Alzheimer{Style.RESET_ALL}")
    print(f"  {prob_saudavel:>7.1f}%{' '*(tamanho_barra-6)}{prob_alzheimer:>7.1f}%")
    
    # Nivel de risco
    print(f"\n{Fore.WHITE}  NIVEL DE RISCO:{Style.RESET_ALL}")
    print(f"  {'-'*40}")
    
    if prob_alzheimer >= 80:
        nivel_risco = "MUITO ALTO"
        cor_risco = Fore.RED
        recomendacao = """
  - Consulta URGENTE com neurologista
  - Exames complementares (ressonancia magnetica, PET scan)
  - Avaliacao neuropsicologica completa
  - Apoio familiar e psicologico"""
    elif prob_alzheimer >= 60:
        nivel_risco = "ALTO"
        cor_risco = Fore.RED
        recomendacao = """
  - Acompanhamento medico especializado
  - Avaliacao neuropsicologica
  - Monitoramento cognitivo regular
  - Considerar mudancas no estilo de vida"""
    elif prob_alzheimer >= 40:
        nivel_risco = "MODERADO"
        cor_risco = Fore.YELLOW
        recomendacao = """
  - Monitoramento regular
  - Consulta com geriatra ou neurologista
  - Exercicios cognitivos preventivos
  - Manutencao de habitos saudaveis"""
    elif prob_alzheimer >= 20:
        nivel_risco = "BAIXO"
        cor_risco = Fore.GREEN
        recomendacao = """
  - Manter habitos saudaveis
  - Check-ups regulares
  - Exercicios fisicos e mentais
  - Dieta equilibrada"""
    else:
        nivel_risco = "MUITO BAIXO"
        cor_risco = Fore.GREEN
        recomendacao = """
  - Continue com estilo de vida saudavel
  - Mantenha atividades cognitivas
  - Vida social ativa
  - Check-ups preventivos anuais"""
    
    print(f"\n  {cor_risco}>>> NIVEL DE RISCO: {nivel_risco} <<<{Style.RESET_ALL}")
    
    # Fatores de risco identificados
    print(f"\n{Fore.WHITE}  FATORES DE RISCO IDENTIFICADOS:{Style.RESET_ALL}")
    print(f"  {'-'*40}")
    
    fatores = []
    if dados['FamilyHistoryAlzheimers'] == 1:
        fatores.append("Historico familiar de Alzheimer")
    if dados['Age'] >= 75:
        fatores.append(f"Idade avancada ({dados['Age']} anos)")
    if dados['MMSE'] < 24:
        fatores.append(f"MMSE baixo ({dados['MMSE']:.1f})")
    if dados['MemoryComplaints'] == 1:
        fatores.append("Queixas de memoria")
    if dados['Confusion'] == 1:
        fatores.append("Confusao mental")
    if dados['Disorientation'] == 1:
        fatores.append("Desorientacao")
    if dados['Depression'] == 1:
        fatores.append("Depressao")
    if dados['CardiovascularDisease'] == 1:
        fatores.append("Doenca cardiovascular")
    if dados['Diabetes'] == 1:
        fatores.append("Diabetes")
    if dados['HeadInjury'] == 1:
        fatores.append("Historico de lesao na cabeca")
    if dados['Hypertension'] == 1:
        fatores.append("Hipertensao")
    if dados['Smoking'] == 1:
        fatores.append("Tabagismo")
    if dados['BMI'] > 30:
        fatores.append(f"Obesidade (IMC: {dados['BMI']:.1f})")
    if dados['PhysicalActivity'] < 2:
        fatores.append("Baixa atividade fisica")
    if dados['SleepQuality'] < 5:
        fatores.append("Qualidade do sono ruim")
    if dados['Forgetfulness'] == 1:
        fatores.append("Esquecimento frequente")
    if dados['DifficultyCompletingTasks'] == 1:
        fatores.append("Dificuldade para completar tarefas")
    if dados['PersonalityChanges'] == 1:
        fatores.append("Mudancas de personalidade")
    
    if fatores:
        for i, fator in enumerate(fatores, 1):
            print(f"  {Fore.YELLOW}{i}. {fator}{Style.RESET_ALL}")
    else:
        print(f"  {Fore.GREEN}Nenhum fator de risco significativo identificado.{Style.RESET_ALL}")
    
    # Recomendacoes
    print(f"\n{Fore.WHITE}  RECOMENDACOES:{Style.RESET_ALL}")
    print(f"  {'-'*40}")
    print(f"{Fore.CYAN}{recomendacao}{Style.RESET_ALL}")
    
    # Disclaimer
    print(f"\n{Fore.RED}{'='*70}")
    print("  AVISO IMPORTANTE")
    print(f"{'='*70}{Style.RESET_ALL}")
    print(f"""
  Este sistema e apenas uma ferramenta de apoio e NAO substitui
  avaliacao medica profissional. Consulte sempre um neurologista
  ou geriatra para diagnostico definitivo.

  Os resultados sao baseados em modelos estatisticos e podem nao
  refletir com precisao sua condicao de saude real.
""")

def menu_principal(predictor):
    """Menu principal do sistema."""
    while True:
        print_cabecalho()
        
        if not predictor.is_trained:
            print(f"{Fore.YELLOW}  ATENCAO: Modelo nao carregado!{Style.RESET_ALL}")
            print(f"  Voce precisa treinar o modelo primeiro.\n")
        else:
            print(f"{Fore.GREEN}  Modelo carregado e pronto para uso.{Style.RESET_ALL}\n")
        
        print(f"{Fore.WHITE}  MENU PRINCIPAL:{Style.RESET_ALL}")
        print(f"  {'-'*40}")
        print("  1. Realizar avaliacao de risco")
        print("  2. Treinar/Atualizar modelo (requer CSV)")
        print("  3. Informacoes sobre o sistema")
        print("  0. Sair")
        print()
        
        escolha = input(f"{Fore.WHITE}  Escolha uma opcao: {Style.RESET_ALL}").strip()
        
        if escolha == "1":
            if not predictor.is_trained:
                print(f"\n{Fore.RED}  Erro: Modelo nao treinado. Por favor, treine o modelo primeiro (opcao 2).{Style.RESET_ALL}")
                input("\n  Pressione ENTER para continuar...")
                continue
            
            print_cabecalho()
            print(f"{Fore.CYAN}  Vamos coletar seus dados para avaliacao.{Style.RESET_ALL}")
            print(f"  Pressione ENTER para usar valores padrao quando disponivel.\n")
            input("  Pressione ENTER para iniciar...")
            
            dados = coletar_dados_paciente()
            
            print(f"\n{Fore.CYAN}  Processando dados...{Style.RESET_ALL}")
            prediction, probability = predictor.predict(dados)
            
            if prediction is not None:
                mostrar_resultado(prediction, probability, dados)
            else:
                print(f"\n{Fore.RED}  Erro ao processar os dados.{Style.RESET_ALL}")
            
            input("\n  Pressione ENTER para voltar ao menu...")
            
        elif escolha == "2":
            print_cabecalho()
            print(f"{Fore.WHITE}  TREINAR MODELO{Style.RESET_ALL}")
            print(f"  {'-'*40}\n")
            
            print("  O arquivo CSV deve conter os dados de pacientes com as seguintes colunas:")
            print("  Age, Gender, Ethnicity, EducationLevel, BMI, etc.")
            print("  E a coluna 'Diagnosis' com 0 (saudavel) ou 1 (Alzheimer).\n")
            
            print(f"{Fore.CYAN}  Abrindo explorador de arquivos...{Style.RESET_ALL}")
            
            # Criar janela oculta do tkinter
            root = tk.Tk()
            root.withdraw()  # Esconder a janela principal
            root.attributes('-topmost', True)  # Manter o dialogo no topo
            
            # Abrir dialogo para selecionar arquivo CSV
            csv_path = filedialog.askopenfilename(
                title="Selecione o arquivo CSV de treinamento",
                filetypes=[
                    ("Arquivos CSV", "*.csv"),
                    ("Todos os arquivos", "*.*")
                ],
                initialdir=os.path.expanduser("~")
            )
            
            root.destroy()  # Fechar a janela do tkinter
            
            if csv_path:
                print(f"\n{Fore.GREEN}  Arquivo selecionado: {csv_path}{Style.RESET_ALL}")
                predictor.train_model(csv_path)
            else:
                print(f"\n{Fore.YELLOW}  Nenhum arquivo selecionado.{Style.RESET_ALL}")
            
            input("\n  Pressione ENTER para continuar...")
            
        elif escolha == "3":
            print_cabecalho()
            print(f"{Fore.WHITE}  SOBRE O SISTEMA{Style.RESET_ALL}")
            print(f"  {'-'*40}\n")
            print(f"""
  Este sistema utiliza uma Rede Neural Artificial (MLP - Multi-Layer
  Perceptron) para prever a tendencia de desenvolvimento de Alzheimer
  com base em dados clinicos e demograficos do paciente.

  {Fore.CYAN}CARACTERISTICAS DO MODELO:{Style.RESET_ALL}
  - Arquitetura: MLP com 3 camadas ocultas (128, 64, 32 neuronios)
  - Funcao de ativacao: ReLU
  - Otimizador: Adam
  - Balanceamento de dados: SMOTE
  - Normalizacao: StandardScaler

  {Fore.CYAN}VARIAVEIS ANALISADAS:{Style.RESET_ALL}
  - Dados demograficos (idade, genero, etnia, educacao)
  - Estilo de vida (IMC, tabagismo, alcool, atividade fisica, dieta, sono)
  - Historico medico (Alzheimer familiar, doencas cardiovasculares, etc.)
  - Sinais vitais (pressao arterial, colesterol)
  - Avaliacao cognitiva (MMSE, ADL)
  - Sintomas atuais (memoria, confusao, desorientacao, etc.)

  {Fore.CYAN}O QUE E O MMSE?{Style.RESET_ALL}
  O Mini Exame do Estado Mental (MMSE) e um teste padronizado que
  avalia funcoes cognitivas como orientacao, memoria, atencao,
  calculo e linguagem. Pontuacao maxima: 30 pontos.
  - 24-30: Funcao cognitiva normal
  - 19-23: Comprometimento leve
  - 10-18: Comprometimento moderado
  - 0-9: Comprometimento grave
""")
            input("\n  Pressione ENTER para voltar ao menu...")
            
        elif escolha == "0":
            print(f"\n{Fore.CYAN}  Obrigado por usar o sistema. Ate logo!{Style.RESET_ALL}\n")
            break
        else:
            print(f"\n{Fore.RED}  Opcao invalida.{Style.RESET_ALL}")
            input("  Pressione ENTER para continuar...")


# ==============================================================================
# EXECUCAO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # Inicializar preditor
    predictor = AlzheimerPredictor()
    
    # Tentar carregar modelo existente
    print(f"{Fore.CYAN}Inicializando sistema...{Style.RESET_ALL}")
    predictor.load_or_train_model()
    
    # Iniciar interface
    menu_principal(predictor)
