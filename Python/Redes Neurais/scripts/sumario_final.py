"""
===============================================================================
SUMÁRIO DE COMPARAÇÃO - ALGORITMOS DE REDES NEURAIS PARA ALZHEIMER
===============================================================================
"""

import pandas as pd
import numpy as np
import os

# Carregar dados
df_base = pd.read_csv('resultados_base.csv')
df_otimizado = pd.read_csv('resultados_otimizados.csv')

print("\n" + "="*100)
print(" "*25 + "COMPARACAO DE ALGORITMOS - DETECCAO DE ALZHEIMER")
print("="*100)

# ============================================================================
# RESUMO GERAL
# ============================================================================
print("\n[RESUMO GERAL]")
print("-"*100)

print(f"\nTotal de algoritmos comparados: {len(df_base)}")
print(f"Dataset: Alzheimer's Disease Detection")
print(f"Tamanho do conjunto: 2.149 amostras | 33 features")
print(f"Classes: Saudavel (64.6%) vs Alzheimer (35.4%)")

# ============================================================================
# RANKING FINAL
# ============================================================================
print("\n[RANKING FINAL (F1-Score)]")
print("-"*100)

ranking_base = df_base.sort_values('F1-Score', ascending=False).reset_index(drop=True)
ranking_otim = df_otimizado.sort_values('F1-Score', ascending=False).reset_index(drop=True)

print("\nMODELOS BASE:")
for idx, row in ranking_base.iterrows():
    medal = ["[1o]", "[2o]", "[3o]", "[4o]", "[5o]", "[6o]"][idx]
    print(f"{medal} {row['Algoritmo']:20} | F1: {row['F1-Score']:.4f} | Acuracia: {row['Acurácia']:.4f} | AUC: {row['AUC-ROC']:.4f}")

print("\nMODELOS OTIMIZADOS:")
for idx, row in ranking_otim.iterrows():
    medal = ["[1o]", "[2o]", "[3o]", "[4o]", "[5o]", "[6o]"][idx]
    print(f"{medal} {row['Algoritmo']:20} | F1: {row['F1-Score']:.4f} | Acuracia: {row['Acurácia']:.4f} | AUC: {row['AUC-ROC']:.4f}")

# ============================================================================
# MELHORES SEGUNDO CRITÉRIOS
# ============================================================================
print("\n[MELHORES ALGORITMOS POR CRITERIO]")
print("-"*100)

criterios = {
    'F1-Score': 'F1-Score',
    'Acurácia': 'Acurácia',
    'Precisão': 'Precisão',
    'Recall': 'Recall',
    'AUC-ROC': 'AUC-ROC',
    'Tempo (s)': 'Tempo (s)'
}

for nome_criterio, coluna in criterios.items():
    if coluna == 'Tempo (s)':
        melhor_base = df_base.loc[df_base[coluna].idxmin()]
        melhor_otim = df_otimizado.loc[df_otimizado[coluna].idxmin()]
        print(f"\n{nome_criterio} (menor eh melhor):")
    else:
        melhor_base = df_base.loc[df_base[coluna].idxmax()]
        melhor_otim = df_otimizado.loc[df_otimizado[coluna].idxmax()]
        print(f"\n{nome_criterio} (maior eh melhor):")
    
    print(f"  Base: {melhor_base['Algoritmo']:20} -> {melhor_base[coluna]:.4f}")
    print(f"  Otimizado: {melhor_otim['Algoritmo']:20} -> {melhor_otim[coluna]:.4f}")

# ============================================================================
# ANÁLISE DE MELHORIA
# ============================================================================
print("\n[ANALISE DE MELHORIA (Base -> Otimizado)]")
print("-"*100)

print("\nMelhorias por Algoritmo:")
for idx, alg in enumerate(df_base['Algoritmo']):
    base_f1 = df_base.loc[idx, 'F1-Score']
    otim_f1 = df_otimizado.loc[idx, 'F1-Score']
    melhoria_f1 = ((otim_f1 - base_f1) / base_f1) * 100
    
    base_acc = df_base.loc[idx, 'Acurácia']
    otim_acc = df_otimizado.loc[idx, 'Acurácia']
    melhoria_acc = ((otim_acc - base_acc) / base_acc) * 100
    
    base_tempo = df_base.loc[idx, 'Tempo (s)']
    otim_tempo = df_otimizado.loc[idx, 'Tempo (s)']
    melhoria_tempo = ((otim_tempo - base_tempo) / base_tempo) * 100
    
    sinal_f1 = "UP" if melhoria_f1 > 0 else "DOWN"
    sinal_acc = "UP" if melhoria_acc > 0 else "DOWN"
    sinal_tempo = "DOWN" if melhoria_tempo < 0 else "UP"
    
    print(f"{alg:20} | F1: {sinal_f1:4}({abs(melhoria_f1):5.2f}%) | Acc: {sinal_acc:4}({abs(melhoria_acc):5.2f}%) | Tempo: {sinal_tempo:4}({abs(melhoria_tempo):5.2f}%)")

# ============================================================================
# ESTATÍSTICAS GERAIS
# ============================================================================
print("\n[ESTATISTICAS GERAIS]")
print("-"*100)

print("\nMODELOS BASE:")
metricas_base = {
    'Acurácia': (df_base['Acurácia'].mean(), df_base['Acurácia'].std()),
    'F1-Score': (df_base['F1-Score'].mean(), df_base['F1-Score'].std()),
    'AUC-ROC': (df_base['AUC-ROC'].mean(), df_base['AUC-ROC'].std()),
    'Tempo': (df_base['Tempo (s)'].mean(), df_base['Tempo (s)'].std())
}

for metrica, (media, std) in metricas_base.items():
    print(f"  {metrica:12} -> Media: {media:.4f} +/- Std: {std:.4f}")

print("\nMODELOS OTIMIZADOS:")
metricas_otim = {
    'Acurácia': (df_otimizado['Acurácia'].mean(), df_otimizado['Acurácia'].std()),
    'F1-Score': (df_otimizado['F1-Score'].mean(), df_otimizado['F1-Score'].std()),
    'AUC-ROC': (df_otimizado['AUC-ROC'].mean(), df_otimizado['AUC-ROC'].std()),
    'Tempo': (df_otimizado['Tempo (s)'].mean(), df_otimizado['Tempo (s)'].std())
}

for metrica, (media, std) in metricas_otim.items():
    print(f"  {metrica:12} -> Media: {media:.4f} +/- Std: {std:.4f}")

# ============================================================================
# RECOMENDAÇÕES
# ============================================================================
print("\n[RECOMENDACOES FINAIS]")
print("-"*100)

recomendacoes = """
1. PARA PRODUCAO (Melhor Acuracia):
   - Algoritmo: Random Forest
   - F1-Score: 0.9080 (Base) / 0.9074 (Otimizado)
   - Acuracia: 0.9122 (91.22%)
   - Recomendacao: Use versao BASE (mais rapida com desempenho similar)

2. PARA INTERPRETABILIDADE (Explicavel):
   - Algoritmo: Decision Tree
   - F1-Score: 0.8314 (Base) / 0.8588 (Otimizado)
   - Acuracia: 0.8302 (Base) / 0.8604 (Otimizado)
   - Recomendacao: Use OTIMIZADO com max_depth=10

3. PARA VELOCIDADE (Menor Latencia):
   - Algoritmo: KNN
   - Tempo: 0.0013s (extremamente rapido)
   - F1-Score: 0.7546 (Base) / 0.7687 (Otimizado)
   - Recomendacao: Ideal para aplicacoes em tempo real

4. PARA EQUILIBRIO (Trade-off Ideal):
   - Algoritmo: SVM
   - F1-Score: 0.8608 (Base) / 0.8588 (Otimizado)
   - AUC-ROC: 0.9362 (excelente discriminacao)
   - Recomendacao: Bom desempenho com velocidade aceitavel

CONCLUSAO GERAL:
>>> Random Forest eh o melhor algoritmo para este problema,
>>> oferecendo a melhor acuracia (91.22%) com desempenho
>>> consistente. Para casos onde interpretabilidade eh critica,
>>> use Decision Tree com otimizacao.
"""

print(recomendacoes)

# ============================================================================
# ARQUIVOS GERADOS
# ============================================================================
print("[ARQUIVOS GERADOS]")
print("-"*100)

files_generated = {
    'CSV': [f for f in os.listdir('.') if f.startswith('resultados') and f.endswith('.csv')],
    'PNG': [f for f in os.listdir('.') if f.endswith('.png')],
    'HTML': [f for f in os.listdir('relatorios') if f.endswith('.html')] if os.path.exists('relatorios') else []
}

print("\nArquivos de Dados:")
for f in files_generated['CSV']:
    print(f"  [OK] {f}")

print("\nGraficos Gerados:")
for f in sorted(files_generated['PNG']):
    print(f"  [OK] {f}")

print("\nRelatorios:")
for f in files_generated['HTML']:
    print(f"  [OK] relatorios/{f}")

print("\n" + "="*100)
print(" "*30 + "COMPARACAO CONCLUIDA COM SUCESSO!")
print("="*100 + "\n")
