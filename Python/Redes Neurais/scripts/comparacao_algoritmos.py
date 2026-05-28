"""
===============================================================================
COMPARAÇÃO DE ALGORITMOS DE MACHINE LEARNING PARA DETECÇÃO DE ALZHEIMER
===============================================================================

Este script compara 6 algoritmos diferentes em suas versões BASE e OTIMIZADA:
1. MLP (Multi-Layer Perceptron)
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Logistic Regression
5. Random Forest
6. Support Vector Machine (SVM)

Dataset: Alzheimer's Disease Detection
Problema: Classificação binária (Saudável vs Alzheimer)

Autor: Comparação Automática
===============================================================================
"""

# ==============================================================================
# IMPORTAÇÃO DE BIBLIOTECAS
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
warnings.filterwarnings('ignore')

# Sklearn - Modelos
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Sklearn - Pré-processamento e validação
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)

# Balanceamento
from imblearn.over_sampling import SMOTE

# Utils KaggleHub
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.utils_kagglehub import carregar_dataset_kaggle

# Configuração de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Seed para reprodutibilidade
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("COMPARAÇÃO DE ALGORITMOS DE ML PARA DETECÇÃO DE ALZHEIMER")
print("=" * 80)


# ==============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ==============================================================================

def carregar_dados():
    """Carrega e pré-processa os dados do dataset de Alzheimer."""
    
    print("\n[1] CARREGAMENTO DOS DADOS")
    print("-" * 40)
    
    # Tentar carregar de caminhos comuns
    caminhos_possiveis = [
        'alzheimers_disease_data.csv',
        './alzheimers_disease_data.csv',
        '../alzheimers_disease_data.csv',
    ]
    
    df = None
    for caminho in caminhos_possiveis:
        try:
            df = pd.read_csv(caminho)
            print(f"Dataset carregado localmente: {caminho}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        # Se não encontrar localmente, baixar do Kaggle
        try:
            print("Arquivo local não encontrado. Baixando do Kaggle...")
            _, df = carregar_dataset_kaggle("rabieelkharoua/alzheimers-disease-dataset")
            print("Dataset carregado do Kaggle com sucesso!")
        except Exception as e:
            raise FileNotFoundError(f"Não foi possível carregar o dataset: {e}")
    
    print(f"Dimensões originais: {df.shape[0]} amostras x {df.shape[1]} features")
    
    # Remoção de colunas irrelevantes
    colunas_remover = ['PatientID', 'DoctorInCharge']
    df = df.drop(columns=[col for col in colunas_remover if col in df.columns])
    print(f"Colunas removidas: {colunas_remover}")
    
    # Separação features/target
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    
    # Distribuição das classes
    print(f"\nDistribuição das classes:")
    print(f"  - Saudável (0): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  - Alzheimer (1): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    
    return X, y


def preparar_dados(X, y):
     """Divide em treino/teste, aplica SMOTE apenas no treino e normaliza."""
     
     print("\n[2] PREPARAÇÃO DOS DADOS")
     print("-" * 40)
     
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


# ==============================================================================
# DEFINIÇÃO DOS ALGORITMOS (BASE E OTIMIZADO)
# ==============================================================================

def get_algoritmos_base():
    """Retorna dicionário com todos os algoritmos em configuração BASE."""
    
    return {
        'MLP': MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=500
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5  # padrão
        ),
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            random_state=RANDOM_STATE,
            probability=True
        )
    }


def get_algoritmos_otimizados():
    """Retorna dicionário com todos os algoritmos em configuração OTIMIZADA."""
    
    return {
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=RANDOM_STATE
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            criterion='gini',
            random_state=RANDOM_STATE
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski',
            p=2
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=RANDOM_STATE
        )
    }


def get_param_grids():
    """Retorna grids de hiperparâmetros para GridSearchCV."""
    
    return {
        'MLP': {
            'hidden_layer_sizes': [(64, 32), (128, 64, 32), (100, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        },
        'Decision Tree': {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'SVM': {
            'kernel': ['rbf', 'poly', 'linear'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    }


# ==============================================================================
# FUNÇÕES DE TREINAMENTO E AVALIAÇÃO
# ==============================================================================

def treinar_e_avaliar(modelo, X_train, X_test, y_train, y_test, nome):
    """Treina um modelo e retorna métricas de avaliação."""
    
    inicio = time()
    modelo.fit(X_train, y_train)
    tempo_treino = time() - inicio
    
    # Predições
    y_pred = modelo.predict(X_test)
    
    # Probabilidades (para ROC)
    if hasattr(modelo, 'predict_proba'):
        y_proba = modelo.predict_proba(X_test)[:, 1]
    else:
        y_proba = modelo.decision_function(X_test)
    
    # Métricas
    metricas = {
        'Algoritmo': nome,
        'Acurácia': accuracy_score(y_test, y_pred),
        'Precisão': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba),
        'Tempo (s)': tempo_treino
    }
    
    return metricas, y_pred, y_proba


def otimizar_com_gridsearch(modelo, param_grid, X_train, y_train, nome):
    """Otimiza hiperparâmetros usando GridSearchCV."""
    
    print(f"\n  Otimizando {nome}...", end=" ")
    inicio = time()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        modelo,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    tempo = time() - inicio
    
    print(f"OK ({tempo:.1f}s)")
    print(f"    Melhores parâmetros: {grid_search.best_params_}")
    
    return grid_search.best_estimator_


# ==============================================================================
# VISUALIZAÇÕES
# ==============================================================================

def plotar_comparacao_metricas(df_base, df_otimizado):
    """Plota comparação de métricas entre versões base e otimizada."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
    cores = ['#3498db', '#e74c3c']  # Azul para base, vermelho para otimizado
    
    for idx, metrica in enumerate(metricas):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(df_base))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_base[metrica], width, 
                       label='Base', color=cores[0], edgecolor='black')
        bars2 = ax.bar(x + width/2, df_otimizado[metrica], width,
                       label='Otimizado', color=cores[1], edgecolor='black')
        
        ax.set_xlabel('Algoritmo')
        ax.set_ylabel(metrica)
        ax.set_title(f'Comparação de {metrica}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_base['Algoritmo'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Adicionar valores nas barras
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Comparação de Métricas: Base vs Otimizado', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparacao_metricas.png', dpi=150, bbox_inches='tight')
    plt.show()


def plotar_curvas_roc(resultados_base, resultados_otim, y_test):
    """Plota curvas ROC para todos os algoritmos."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC - Modelos Base
    ax = axes[0]
    for nome, dados in resultados_base.items():
        fpr, tpr, _ = roc_curve(y_test, dados['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{nome} (AUC = {roc_auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curvas ROC - Modelos BASE', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # ROC - Modelos Otimizados
    ax = axes[1]
    for nome, dados in resultados_otim.items():
        fpr, tpr, _ = roc_curve(y_test, dados['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{nome} (AUC = {roc_auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curvas ROC - Modelos OTIMIZADOS', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('curvas_roc_comparacao.png', dpi=150, bbox_inches='tight')
    plt.show()


def plotar_heatmap_melhoria(df_base, df_otimizado):
    """Plota heatmap mostrando a melhoria percentual de cada métrica."""
    
    metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
    algoritmos = df_base['Algoritmo'].values
    
    # Calcular melhoria percentual
    melhoria = pd.DataFrame(index=algoritmos, columns=metricas)
    
    for metrica in metricas:
        for i, alg in enumerate(algoritmos):
            base = df_base.loc[i, metrica]
            otim = df_otimizado.loc[i, metrica]
            melhoria.loc[alg, metrica] = ((otim - base) / base) * 100 if base > 0 else 0
    
    melhoria = melhoria.astype(float)
    
    plt.figure(figsize=(12, 8))
    
    # Criar heatmap
    cmap = sns.diverging_palette(10, 130, as_cmap=True)  # Vermelho para negativo, verde para positivo
    
    ax = sns.heatmap(
        melhoria,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Melhoria (%)'}
    )
    
    plt.title('Melhoria Percentual: Base → Otimizado', fontsize=14, fontweight='bold')
    plt.xlabel('Métrica')
    plt.ylabel('Algoritmo')
    plt.tight_layout()
    plt.savefig('heatmap_melhoria.png', dpi=150, bbox_inches='tight')
    plt.show()


def plotar_ranking_final(df_base, df_otimizado):
    """Plota ranking final dos algoritmos baseado no F1-Score."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ranking Base
    df_base_sorted = df_base.sort_values('F1-Score', ascending=True)
    colors_base = plt.cm.Blues(np.linspace(0.3, 0.9, len(df_base_sorted)))
    
    ax = axes[0]
    bars = ax.barh(df_base_sorted['Algoritmo'], df_base_sorted['F1-Score'], 
                   color=colors_base, edgecolor='black')
    ax.set_xlabel('F1-Score')
    ax.set_title('Ranking - Modelos BASE', fontweight='bold')
    ax.set_xlim(0, 1)
    
    for bar, val in zip(bars, df_base_sorted['F1-Score']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    # Ranking Otimizado
    df_otim_sorted = df_otimizado.sort_values('F1-Score', ascending=True)
    colors_otim = plt.cm.Reds(np.linspace(0.3, 0.9, len(df_otim_sorted)))
    
    ax = axes[1]
    bars = ax.barh(df_otim_sorted['Algoritmo'], df_otim_sorted['F1-Score'],
                   color=colors_otim, edgecolor='black')
    ax.set_xlabel('F1-Score')
    ax.set_title('Ranking - Modelos OTIMIZADOS', fontweight='bold')
    ax.set_xlim(0, 1)
    
    for bar, val in zip(bars, df_otim_sorted['F1-Score']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ranking_f1score.png', dpi=150, bbox_inches='tight')
    plt.show()


def plotar_matrizes_confusao(resultados, y_test, titulo_grupo):
    """Plota matrizes de confusão para todos os algoritmos."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (nome, dados) in enumerate(resultados.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_test, dados['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Saudável', 'Alzheimer'],
                   yticklabels=['Saudável', 'Alzheimer'])
        ax.set_title(f'{nome}', fontweight='bold')
        ax.set_ylabel('Real')
        ax.set_xlabel('Previsto')
    
    plt.suptitle(f'Matrizes de Confusão - {titulo_grupo}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    nome_arquivo = f'matrizes_confusao_{titulo_grupo.lower().replace(" ", "_")}.png'
    plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
    plt.show()


def plotar_tempo_treinamento(df_base, df_otimizado):
    """Plota comparação de tempo de treinamento."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_base))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_base['Tempo (s)'], width,
                   label='Base', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, df_otimizado['Tempo (s)'], width,
                   label='Otimizado', color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Algoritmo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tempo de Treinamento (segundos)', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Tempo de Treinamento', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_base['Algoritmo'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('tempo_treinamento.png', dpi=150, bbox_inches='tight')
    plt.show()


def plotar_radar_desempenho(df_base, df_otimizado):
    """Plota gráfico radar comparando desempenho dos algoritmos."""
    
    from math import pi
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
    
    categorias = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
    
    # Função para plotar radar
    def plotar_radar_algoritmo(ax, df, titulo):
        N = len(categorias)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categorias, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)
        
        cores = plt.cm.Set2(np.linspace(0, 1, len(df)))
        
        for idx, (_, row) in enumerate(df.iterrows()):
            valores = [row['Acurácia'], row['Precisão'], row['Recall'], row['F1-Score'], row['AUC-ROC']]
            valores += valores[:1]
            ax.plot(angles, valores, 'o-', linewidth=2, label=row['Algoritmo'], color=cores[idx])
            ax.fill(angles, valores, alpha=0.15, color=cores[idx])
        
        ax.set_title(titulo, fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plotar_radar_algoritmo(axes[0], df_base, 'Desempenho dos Modelos BASE')
    plotar_radar_algoritmo(axes[1], df_otimizado, 'Desempenho dos Modelos OTIMIZADOS')
    
    plt.tight_layout()
    plt.savefig('radar_desempenho.png', dpi=150, bbox_inches='tight')
    plt.show()


def plotar_comparacao_scatter(df_base, df_otimizado):
    """Plota scatter plot F1-Score vs Acurácia."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plotar modelos base
    scatter1 = ax.scatter(df_base['Acurácia'], df_base['F1-Score'],
                         s=200, alpha=0.6, color='#3498db', edgecolor='black',
                         label='Base', linewidth=2)
    
    # Plotar modelos otimizados
    scatter2 = ax.scatter(df_otimizado['Acurácia'], df_otimizado['F1-Score'],
                         s=200, alpha=0.6, color='#e74c3c', edgecolor='black',
                         label='Otimizado', linewidth=2, marker='s')
    
    # Adicionar nomes dos algoritmos
    for idx, row in df_base.iterrows():
        ax.annotate(f"{row['Algoritmo']}\n(B)", 
                   xy=(row['Acurácia'], row['F1-Score']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    for idx, row in df_otimizado.iterrows():
        ax.annotate(f"{row['Algoritmo']}\n(O)", 
                   xy=(row['Acurácia'], row['F1-Score']),
                   xytext=(5, -15), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Acurácia', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Relação Acurácia vs F1-Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.5, 1.05)
    
    plt.tight_layout()
    plt.savefig('scatter_acuracia_f1.png', dpi=150, bbox_inches='tight')
    plt.show()


def plotar_box_plot_metricas(df_base, df_otimizado):
    """Plota box plots para distribuição de métricas."""
    
    metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    
    for idx, metrica in enumerate(metricas):
        ax = axes[idx]
        
        dados = [df_base[metrica].values, df_otimizado[metrica].values]
        bp = ax.boxplot(dados, labels=['Base', 'Otimizado'], patch_artist=True)
        
        for patch, cor in zip(bp['boxes'], ['#3498db', '#e74c3c']):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metrica, fontweight='bold')
        ax.set_title(f'Distribuição - {metrica}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0.4, 1.05)
    
    plt.suptitle('Box Plot de Métricas: Base vs Otimizado', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('boxplot_metricas.png', dpi=150, bbox_inches='tight')
    plt.show()


def plotar_summary_performance(df_base, df_otimizado):
    """Plota um sumário visual de desempenho."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Melhor F1-Score
    ax1 = fig.add_subplot(gs[0, 0])
    melhor_base = df_base.loc[df_base['F1-Score'].idxmax()]
    melhor_otim = df_otimizado.loc[df_otimizado['F1-Score'].idxmax()]
    
    nomes = [melhor_base['Algoritmo'], melhor_otim['Algoritmo']]
    scores = [melhor_base['F1-Score'], melhor_otim['F1-Score']]
    cores = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(nomes, scores, color=cores, edgecolor='black', linewidth=2)
    ax1.set_ylabel('F1-Score', fontweight='bold')
    ax1.set_title('Melhor F1-Score', fontweight='bold')
    ax1.set_ylim(0, 1)
    
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Melhor Acurácia
    ax2 = fig.add_subplot(gs[0, 1])
    melhor_base_acc = df_base.loc[df_base['Acurácia'].idxmax()]
    melhor_otim_acc = df_otimizado.loc[df_otimizado['Acurácia'].idxmax()]
    
    nomes = [melhor_base_acc['Algoritmo'], melhor_otim_acc['Algoritmo']]
    scores = [melhor_base_acc['Acurácia'], melhor_otim_acc['Acurácia']]
    
    bars = ax2.bar(nomes, scores, color=cores, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Acurácia', fontweight='bold')
    ax2.set_title('Melhor Acurácia', fontweight='bold')
    ax2.set_ylim(0, 1)
    
    for bar, score in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Tempo Médio
    ax3 = fig.add_subplot(gs[0, 2])
    tempo_base = df_base['Tempo (s)'].mean()
    tempo_otim = df_otimizado['Tempo (s)'].mean()
    
    bars = ax3.bar(['Base', 'Otimizado'], [tempo_base, tempo_otim],
                   color=cores, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Tempo (s)', fontweight='bold')
    ax3.set_title('Tempo Médio de Treino', fontweight='bold')
    
    for bar, tempo in zip(bars, [tempo_base, tempo_otim]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{tempo:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # 4-8. Comparação por métrica
    metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
    positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    
    for idx, (metrica, pos) in enumerate(zip(metricas, positions)):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        
        media_base = df_base[metrica].mean()
        media_otim = df_otimizado[metrica].mean()
        std_base = df_base[metrica].std()
        std_otim = df_otimizado[metrica].std()
        
        x_pos = np.arange(2)
        ax.bar(x_pos, [media_base, media_otim], 
               yerr=[std_base, std_otim],
               color=cores, edgecolor='black', linewidth=2, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Base', 'Otimizado'])
        ax.set_ylabel(metrica, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(f'{metrica} (média ± std)', fontweight='bold', fontsize=10)
    
    # Último gráfico: Ganho médio
    ax8 = fig.add_subplot(gs[2, 2])
    
    ganho = []
    for metrica in metricas:
        media_base = df_base[metrica].mean()
        media_otim = df_otimizado[metrica].mean()
        ganho_pct = ((media_otim - media_base) / media_base * 100) if media_base > 0 else 0
        ganho.append(ganho_pct)
    
    cores_ganho = ['green' if g >= 0 else 'red' for g in ganho]
    bars = ax8.barh(metricas, ganho, color=cores_ganho, edgecolor='black', linewidth=2)
    ax8.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax8.set_xlabel('Ganho (%)', fontweight='bold')
    ax8.set_title('Ganho Percentual (Otimizado - Base)', fontweight='bold')
    
    for i, (bar, g) in enumerate(zip(bars, ganho)):
        ax8.text(g + (1 if g > 0 else -1), i, f'{g:+.1f}%', 
                va='center', ha='left' if g > 0 else 'right', fontweight='bold')
    
    plt.suptitle('Resumo de Desempenho: Base vs Otimizado', fontsize=16, fontweight='bold')
    plt.savefig('summary_performance.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    """Função principal que executa toda a comparação."""
    
    # Carregar e preparar dados
    X, y = carregar_dados()
    X_train, X_test, y_train, y_test, scaler = preparar_dados(X, y)
    
    # =========================================================================
    # TREINAMENTO - MODELOS BASE
    # =========================================================================
    print("\n" + "=" * 80)
    print("[3] TREINAMENTO DOS MODELOS BASE")
    print("=" * 80)
    
    algoritmos_base = get_algoritmos_base()
    resultados_base = {}
    metricas_base = []
    
    for nome, modelo in algoritmos_base.items():
        print(f"\n  Treinando {nome}...", end=" ")
        metricas, y_pred, y_proba = treinar_e_avaliar(
            modelo, X_train, X_test, y_train, y_test, nome
        )
        print(f"OK (Acc: {metricas['Acurácia']:.2%}, F1: {metricas['F1-Score']:.2%})")
        
        metricas_base.append(metricas)
        resultados_base[nome] = {
            'modelo': modelo,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'metricas': metricas
        }
    
    df_base = pd.DataFrame(metricas_base)
    
    # =========================================================================
    # TREINAMENTO - MODELOS OTIMIZADOS
    # =========================================================================
    print("\n" + "=" * 80)
    print("[4] TREINAMENTO DOS MODELOS OTIMIZADOS")
    print("=" * 80)
    
    algoritmos_otimizados = get_algoritmos_otimizados()
    resultados_otim = {}
    metricas_otim = []
    
    for nome, modelo in algoritmos_otimizados.items():
        print(f"\n  Treinando {nome} (otimizado)...", end=" ")
        metricas, y_pred, y_proba = treinar_e_avaliar(
            modelo, X_train, X_test, y_train, y_test, nome
        )
        print(f"OK (Acc: {metricas['Acurácia']:.2%}, F1: {metricas['F1-Score']:.2%})")
        
        metricas_otim.append(metricas)
        resultados_otim[nome] = {
            'modelo': modelo,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'metricas': metricas
        }
    
    df_otimizado = pd.DataFrame(metricas_otim)
    
    # =========================================================================
    # RESULTADOS CONSOLIDADOS
    # =========================================================================
    print("\n" + "=" * 80)
    print("[5] RESULTADOS CONSOLIDADOS")
    print("=" * 80)
    
    print("\n--- MODELOS BASE ---")
    print(df_base.to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x))
    
    print("\n--- MODELOS OTIMIZADOS ---")
    print(df_otimizado.to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x))
    
    # Melhor modelo
    print("\n" + "-" * 40)
    melhor_base = df_base.loc[df_base['F1-Score'].idxmax()]
    melhor_otim = df_otimizado.loc[df_otimizado['F1-Score'].idxmax()]
    
    print(f"\nMelhor modelo BASE: {melhor_base['Algoritmo']}")
    print(f"  F1-Score: {melhor_base['F1-Score']:.4f}")
    print(f"  Acurácia: {melhor_base['Acurácia']:.4f}")
    print(f"  Recall: {melhor_base['Recall']:.4f}")
    
    print(f"\nMelhor modelo OTIMIZADO: {melhor_otim['Algoritmo']}")
    print(f"  F1-Score: {melhor_otim['F1-Score']:.4f}")
    print(f"  Acurácia: {melhor_otim['Acurácia']:.4f}")
    print(f"  Recall: {melhor_otim['Recall']:.4f}")
    
    # =========================================================================
    # VISUALIZAÇÕES
    # =========================================================================
    print("\n" + "=" * 80)
    print("[6] GERANDO VISUALIZAÇÕES")
    print("=" * 80)
    
    # 1. Comparação de métricas
    print("\n  Gerando gráfico de comparação de métricas...")
    plotar_comparacao_metricas(df_base, df_otimizado)
    
    # 2. Curvas ROC
    print("  Gerando curvas ROC...")
    plotar_curvas_roc(resultados_base, resultados_otim, y_test)
    
    # 3. Heatmap de melhoria
    print("  Gerando heatmap de melhoria...")
    plotar_heatmap_melhoria(df_base, df_otimizado)
    
    # 4. Ranking final
    print("  Gerando ranking final...")
    plotar_ranking_final(df_base, df_otimizado)
    
    # 5. Matrizes de confusão
    print("  Gerando matrizes de confusão...")
    plotar_matrizes_confusao(resultados_base, y_test, "Modelos Base")
    plotar_matrizes_confusao(resultados_otim, y_test, "Modelos Otimizados")
    
    # 6. Tempo de treinamento
    print("  Gerando gráfico de tempo de treinamento...")
    plotar_tempo_treinamento(df_base, df_otimizado)
    
    # 7. Gráfico Radar
    print("  Gerando gráfico radar...")
    plotar_radar_desempenho(df_base, df_otimizado)
    
    # 8. Scatter plot
    print("  Gerando scatter plot...")
    plotar_comparacao_scatter(df_base, df_otimizado)
    
    # 9. Box plots
    print("  Gerando box plots...")
    plotar_box_plot_metricas(df_base, df_otimizado)
    
    # 10. Summary performance
    print("  Gerando sumário de desempenho...")
    plotar_summary_performance(df_base, df_otimizado)
    
    # =========================================================================
    # SALVAR RESULTADOS
    # =========================================================================
    print("\n" + "=" * 80)
    print("[7] SALVANDO RESULTADOS")
    print("=" * 80)
    
    # Salvar DataFrames
    df_base.to_csv('resultados_base.csv', index=False)
    df_otimizado.to_csv('resultados_otimizados.csv', index=False)
    
    # Criar relatório comparativo
    df_comparativo = df_base.copy()
    df_comparativo['Versão'] = 'Base'
    df_otimizado_temp = df_otimizado.copy()
    df_otimizado_temp['Versão'] = 'Otimizado'
    df_completo = pd.concat([df_comparativo, df_otimizado_temp], ignore_index=True)
    df_completo.to_csv('resultados_completos.csv', index=False)
    
    print("  Arquivos salvos:")
    print("    - resultados_base.csv")
    print("    - resultados_otimizados.csv")
    print("    - resultados_completos.csv")
    print("    - comparacao_metricas.png")
    print("    - curvas_roc_comparacao.png")
    print("    - heatmap_melhoria.png")
    print("    - ranking_f1score.png")
    print("    - matrizes_confusao_modelos_base.png")
    print("    - matrizes_confusao_modelos_otimizados.png")
    print("    - tempo_treinamento.png")
    print("    - radar_desempenho.png")
    print("    - scatter_acuracia_f1.png")
    print("    - boxplot_metricas.png")
    print("    - summary_performance.png")
    
    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESUMO FINAL")
    print("=" * 80)
    
    print("""
    Este script comparou 6 algoritmos de Machine Learning para detecção de Alzheimer:
    
    1. MLP (Multi-Layer Perceptron) - Rede Neural
    2. Decision Tree - Árvore de Decisão
    3. KNN - K-Nearest Neighbors
    4. Logistic Regression - Regressão Logística
    5. Random Forest - Floresta Aleatória
    6. SVM - Support Vector Machine
    
    Cada algoritmo foi testado em duas versões:
    - BASE: Parâmetros padrão do sklearn
    - OTIMIZADO: Hiperparâmetros ajustados para melhor desempenho
    
    Técnicas utilizadas:
    - SMOTE para balanceamento de classes
    - StandardScaler para normalização
    - Divisão estratificada 75/25 (treino/teste)
    - Validação cruzada 5-fold para otimização
    """)
    
    print("=" * 80)
    print("COMPARAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 80)
    
    return df_base, df_otimizado, resultados_base, resultados_otim


# ==============================================================================
# EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    df_base, df_otimizado, resultados_base, resultados_otim = main()
