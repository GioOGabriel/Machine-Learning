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
            print(f"Dataset carregado: {caminho}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        # Se não encontrar, usar tkinter para selecionar
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            csv_path = filedialog.askopenfilename(
                title='Selecione o arquivo alzheimers_disease_data.csv',
                filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
            )
            root.destroy()
            
            if csv_path:
                df = pd.read_csv(csv_path)
                print(f"Dataset carregado: {csv_path}")
            else:
                raise FileNotFoundError("Nenhum arquivo selecionado!")
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
    """Aplica SMOTE, divide em treino/teste e normaliza."""
    
    print("\n[2] PREPARAÇÃO DOS DADOS")
    print("-" * 40)
    
    # Balanceamento com SMOTE
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"Após SMOTE: {len(y_balanced)} amostras (balanceado)")
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_balanced
    )
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Normalização aplicada (StandardScaler)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


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
