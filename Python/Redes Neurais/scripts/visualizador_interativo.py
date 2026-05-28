"""
===============================================================================
VISUALIZADOR INTERATIVO DE RESULTADOS
===============================================================================
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def listar_arquivos():
    """Lista todos os arquivos gerados."""
    print("\n" + "="*80)
    print("ARQUIVOS DISPONÍVEIS")
    print("="*80)
    
    # Gráficos
    png_files = sorted([f for f in os.listdir('.') if f.endswith('.png')])
    print("\nGRÁFICOS (PNG):")
    for i, f in enumerate(png_files, 1):
        size = os.path.getsize(f) / 1024  # KB
        print(f"  {i}. {f} ({size:.1f} KB)")
    
    # CSV
    csv_files = [f for f in os.listdir('.') if f.startswith('resultados') and f.endswith('.csv')]
    print("\nDADOS (CSV):")
    for i, f in enumerate(csv_files, 1):
        size = os.path.getsize(f) / 1024  # KB
        print(f"  {i}. {f} ({size:.1f} KB)")
    
    # HTML
    if os.path.exists('relatorios'):
        html_files = [f for f in os.listdir('relatorios') if f.endswith('.html')]
        print("\nRELATÓRIOS (HTML):")
        for i, f in enumerate(html_files, 1):
            size = os.path.getsize(f'relatorios/{f}') / 1024  # KB
            print(f"  {i}. {f} ({size:.1f} KB)")
    
    return png_files, csv_files

def visualizar_grafico(numero):
    """Visualiza um gráfico específico."""
    png_files = sorted([f for f in os.listdir('.') if f.endswith('.png')])
    
    if 1 <= numero <= len(png_files):
        arquivo = png_files[numero - 1]
        print(f"\nAbrindo: {arquivo}")
        
        # Usar matplotlib para exibir
        img = plt.imread(arquivo)
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(img)
        ax.axis('off')
        plt.title(arquivo, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    else:
        print("Número inválido!")

def visualizar_dados(numero):
    """Visualiza dados de um CSV."""
    csv_files = [f for f in os.listdir('.') if f.startswith('resultados') and f.endswith('.csv')]
    
    if 1 <= numero <= len(csv_files):
        arquivo = csv_files[numero - 1]
        print(f"\nLendo: {arquivo}\n")
        
        df = pd.read_csv(arquivo)
        print(df.to_string(index=False))
    else:
        print("Número inválido!")

def resumo_analise():
    """Exibe um resumo da análise."""
    print("\n" + "="*80)
    print("RESUMO RÁPIDO DA ANÁLISE")
    print("="*80)
    
    df_base = pd.read_csv('resultados_base.csv')
    df_otim = pd.read_csv('resultados_otimizados.csv')
    
    print("\n[MELHOR DESEMPENHO]")
    melhor_idx = df_base['F1-Score'].idxmax()
    melhor = df_base.loc[melhor_idx]
    print(f"  {melhor['Algoritmo']}: F1={melhor['F1-Score']:.4f}, Acc={melhor['Acurácia']:.4f}")
    
    print("\n[MAIS RÁPIDO]")
    rapido_idx = df_base['Tempo (s)'].idxmin()
    rapido = df_base.loc[rapido_idx]
    print(f"  {rapido['Algoritmo']}: {rapido['Tempo (s)']:.4f}s")
    
    print("\n[MELHOR RECALL]")
    recall_idx = df_base['Recall'].idxmax()
    recall = df_base.loc[recall_idx]
    print(f"  {recall['Algoritmo']}: {recall['Recall']:.4f}")
    
    print("\n[MELHOR AUC-ROC]")
    auc_idx = df_base['AUC-ROC'].idxmax()
    auc = df_base.loc[auc_idx]
    print(f"  {auc['Algoritmo']}: {auc['AUC-ROC']:.4f}")
    
    # Médias
    print("\n[MÉDIAS GERAIS]")
    print(f"  Acurácia (Base): {df_base['Acurácia'].mean():.2%}")
    print(f"  Acurácia (Otim): {df_otim['Acurácia'].mean():.2%}")
    print(f"  F1-Score (Base): {df_base['F1-Score'].mean():.2%}")
    print(f"  F1-Score (Otim): {df_otim['F1-Score'].mean():.2%}")

def abrir_relatorio_html():
    """Abre o relatório HTML no navegador padrão."""
    if os.path.exists('relatorios/comparacao_algoritmos.html'):
        import webbrowser
        path = os.path.abspath('relatorios/comparacao_algoritmos.html')
        webbrowser.open('file://' + path)
        print("Abrindo relatório HTML no navegador...")
    else:
        print("Relatório HTML não encontrado!")

def menu_principal():
    """Menu principal do visualizador."""
    while True:
        print("\n" + "="*80)
        print("VISUALIZADOR INTERATIVO - COMPARAÇÃO DE ALGORITMOS")
        print("="*80)
        print("\n1. Listar arquivos disponíveis")
        print("2. Ver resumo rápido da análise")
        print("3. Abrir relatório HTML no navegador")
        print("4. Visualizar um gráfico (PNG)")
        print("5. Ver dados de um CSV")
        print("0. Sair")
        
        opcao = input("\nEscolha uma opção: ").strip()
        
        if opcao == '1':
            png_files, csv_files = listar_arquivos()
        
        elif opcao == '2':
            resumo_analise()
        
        elif opcao == '3':
            abrir_relatorio_html()
        
        elif opcao == '4':
            png_files = sorted([f for f in os.listdir('.') if f.endswith('.png')])
            print(f"\nTotal de {len(png_files)} gráficos disponíveis")
            try:
                num = int(input("Qual gráfico deseja ver (número)? "))
                visualizar_grafico(num)
            except ValueError:
                print("Entrada inválida!")
        
        elif opcao == '5':
            csv_files = [f for f in os.listdir('.') if f.startswith('resultados') and f.endswith('.csv')]
            print(f"\nTotal de {len(csv_files)} arquivos de dados disponíveis")
            try:
                num = int(input("Qual arquivo deseja ver (número)? "))
                visualizar_dados(num)
            except ValueError:
                print("Entrada inválida!")
        
        elif opcao == '0':
            print("\nAté logo!")
            break
        
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\n\nPrograma interrompido pelo usuário.")
        sys.exit(0)
