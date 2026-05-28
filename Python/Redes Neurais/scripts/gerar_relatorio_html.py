"""
===============================================================================
GERADOR DE RELATÓRIO HTML - COMPARAÇÃO DE ALGORITMOS DE ALZHEIMER
===============================================================================

Este script gera um relatório HTML interativo com todos os resultados
da comparação de algoritmos.
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# Carregar dados
df_base = pd.read_csv('resultados_base.csv')
df_otimizado = pd.read_csv('resultados_otimizados.csv')

# Criar diretório para relatórios se não existir
Path('relatorios').mkdir(exist_ok=True)

# Calcular estatísticas
def calcular_melhorias(df_base, df_otimizado):
    """Calcula as melhorias percentuais."""
    melhorias = {}
    for i, alg in enumerate(df_base['Algoritmo']):
        melhorias[alg] = {
            'acuracia': ((df_otimizado.loc[i, 'Acurácia'] - df_base.loc[i, 'Acurácia']) / df_base.loc[i, 'Acurácia'] * 100),
            'f1': ((df_otimizado.loc[i, 'F1-Score'] - df_base.loc[i, 'F1-Score']) / df_base.loc[i, 'F1-Score'] * 100),
            'tempo': ((df_otimizado.loc[i, 'Tempo (s)'] - df_base.loc[i, 'Tempo (s)']) / df_base.loc[i, 'Tempo (s)'] * 100),
        }
    return melhorias

melhorias = calcular_melhorias(df_base, df_otimizado)

# Melhor algoritmo
melhor_base_idx = df_base['F1-Score'].idxmax()
melhor_otim_idx = df_otimizado['F1-Score'].idxmax()

melhor_base = df_base.loc[melhor_base_idx]
melhor_otim = df_otimizado.loc[melhor_otim_idx]

# Template HTML
html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparação de Algoritmos - Detecção de Alzheimer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        section {{
            margin-bottom: 40px;
        }}
        
        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: #f8f9fa;
            border-left: 5px solid #667eea;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        .stat-card h3 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        
        .stat-card .unit {{
            color: #999;
            font-size: 0.8em;
            margin-top: 5px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        table thead {{
            background: #667eea;
            color: white;
        }}
        
        table th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        table tbody tr:hover {{
            background: #f8f9fa;
        }}
        
        table tbody tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .positive {{
            color: #27ae60;
            font-weight: bold;
        }}
        
        .negative {{
            color: #e74c3c;
            font-weight: bold;
        }}
        
        .neutral {{
            color: #95a5a6;
        }}
        
        .images {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }}
        
        .image-container {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        .image-container img {{
            width: 100%;
            height: auto;
            border-radius: 3px;
        }}
        
        .image-container h3 {{
            margin-top: 10px;
            color: #667eea;
            font-size: 1em;
            text-align: center;
        }}
        
        .highlight {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 3px;
        }}
        
        .highlight strong {{
            color: #333;
        }}
        
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        
        .comparison-item {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border: 2px solid #eee;
        }}
        
        .comparison-item.base {{
            border-left: 4px solid #3498db;
        }}
        
        .comparison-item.otimizado {{
            border-left: 4px solid #e74c3c;
        }}
        
        .comparison-item h3 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        .comparison-item p {{
            margin: 8px 0;
            font-size: 0.95em;
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .metric:last-child {{
            border-bottom: none;
        }}
        
        .metric-name {{
            color: #666;
        }}
        
        .metric-value {{
            font-weight: bold;
            color: #333;
        }}
        
        footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
        }}
        
        .badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 3px 8px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-left: 10px;
        }}
        
        .badge.best {{
            background: #27ae60;
        }}
        
        @media (max-width: 768px) {{
            .comparison {{
                grid-template-columns: 1fr;
            }}
            
            .images {{
                grid-template-columns: 1fr;
            }}
            
            header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Comparação de Algoritmos de Redes Neurais</h1>
            <p>Detecção de Alzheimer - Machine Learning</p>
            <p style="margin-top: 10px; opacity: 0.8;">Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</p>
        </header>
        
        <div class="content">
            <!-- RESUMO EXECUTIVO -->
            <section>
                <h2>📊 Resumo Executivo</h2>
                
                <div class="highlight">
                    <strong>Melhor Algoritmo (BASE):</strong> {melhor_base['Algoritmo']} 
                    <span class="badge best">F1-Score: {melhor_base['F1-Score']:.4f}</span>
                </div>
                
                <div class="highlight">
                    <strong>Melhor Algoritmo (OTIMIZADO):</strong> {melhor_otim['Algoritmo']}
                    <span class="badge best">F1-Score: {melhor_otim['F1-Score']:.4f}</span>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <h3>Acurácia Média (Base)</h3>
                        <div class="value">{df_base['Acurácia'].mean():.1%}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Acurácia Média (Otimizado)</h3>
                        <div class="value">{df_otimizado['Acurácia'].mean():.1%}</div>
                    </div>
                    <div class="stat-card">
                        <h3>F1-Score Médio (Base)</h3>
                        <div class="value">{df_base['F1-Score'].mean():.1%}</div>
                    </div>
                    <div class="stat-card">
                        <h3>F1-Score Médio (Otimizado)</h3>
                        <div class="value">{df_otimizado['F1-Score'].mean():.1%}</div>
                    </div>
                </div>
            </section>
            
            <!-- TABELA DE RESULTADOS BASE -->
            <section>
                <h2>📈 Resultados - Modelos BASE</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Algoritmo</th>
                            <th>Acurácia</th>
                            <th>Precisão</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>AUC-ROC</th>
                            <th>Tempo (s)</th>
                        </tr>
                    </thead>
                    <tbody>
"""

# Adicionar linhas da tabela base
for idx, row in df_base.iterrows():
    html_content += f"""
                        <tr>
                            <td><strong>{row['Algoritmo']}</strong></td>
                            <td>{row['Acurácia']:.4f}</td>
                            <td>{row['Precisão']:.4f}</td>
                            <td>{row['Recall']:.4f}</td>
                            <td><strong>{row['F1-Score']:.4f}</strong></td>
                            <td>{row['AUC-ROC']:.4f}</td>
                            <td>{row['Tempo (s)']:.4f}</td>
                        </tr>
"""

html_content += """
                    </tbody>
                </table>
            </section>
            
            <!-- TABELA DE RESULTADOS OTIMIZADOS -->
            <section>
                <h2>🚀 Resultados - Modelos OTIMIZADOS</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Algoritmo</th>
                            <th>Acurácia</th>
                            <th>Precisão</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>AUC-ROC</th>
                            <th>Tempo (s)</th>
                        </tr>
                    </thead>
                    <tbody>
"""

# Adicionar linhas da tabela otimizada
for idx, row in df_otimizado.iterrows():
    html_content += f"""
                        <tr>
                            <td><strong>{row['Algoritmo']}</strong></td>
                            <td>{row['Acurácia']:.4f}</td>
                            <td>{row['Precisão']:.4f}</td>
                            <td>{row['Recall']:.4f}</td>
                            <td><strong>{row['F1-Score']:.4f}</strong></td>
                            <td>{row['AUC-ROC']:.4f}</td>
                            <td>{row['Tempo (s)']:.4f}</td>
                        </tr>
"""

html_content += """
                    </tbody>
                </table>
            </section>
            
            <!-- TABELA DE MELHORIAS -->
            <section>
                <h2>📊 Análise de Melhorias (Base → Otimizado)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Algoritmo</th>
                            <th>Melhoria Acurácia</th>
                            <th>Melhoria F1-Score</th>
                            <th>Variação Tempo</th>
                        </tr>
                    </thead>
                    <tbody>
"""

for alg, melhoria in melhorias.items():
    acuracia_class = 'positive' if melhoria['acuracia'] >= 0 else 'negative'
    f1_class = 'positive' if melhoria['f1'] >= 0 else 'negative'
    tempo_class = 'positive' if melhoria['tempo'] <= 0 else 'negative'
    
    html_content += f"""
                        <tr>
                            <td><strong>{alg}</strong></td>
                            <td class="{acuracia_class}">{melhoria['acuracia']:+.2f}%</td>
                            <td class="{f1_class}">{melhoria['f1']:+.2f}%</td>
                            <td class="{tempo_class}">{melhoria['tempo']:+.2f}%</td>
                        </tr>
"""

html_content += """
                    </tbody>
                </table>
            </section>
            
            <!-- VISUALIZAÇÕES -->
            <section>
                <h2>📊 Visualizações</h2>
                <div class="images">
"""

# Adicionar imagens dos gráficos
imagens = [
    ('comparacao_metricas.png', 'Comparação de Métricas'),
    ('ranking_f1score.png', 'Ranking Final'),
    ('curvas_roc_comparacao.png', 'Curvas ROC'),
    ('heatmap_melhoria.png', 'Heatmap de Melhoria'),
    ('tempo_treinamento.png', 'Tempo de Treinamento'),
    ('radar_desempenho.png', 'Gráfico Radar'),
    ('scatter_acuracia_f1.png', 'Scatter Plot'),
    ('boxplot_metricas.png', 'Box Plot de Métricas'),
    ('summary_performance.png', 'Sumário de Desempenho'),
    ('matrizes_confusao_modelos_base.png', 'Matrizes Confusão (Base)'),
    ('matrizes_confusao_modelos_otimizados.png', 'Matrizes Confusão (Otimizado)'),
]

for imagem, titulo in imagens:
    if os.path.exists(imagem):
        html_content += f"""
                    <div class="image-container">
                        <img src="../{imagem}" alt="{titulo}">
                        <h3>{titulo}</h3>
                    </div>
"""

html_content += """
                </div>
            </section>
            
            <!-- CONCLUSÕES -->
            <section>
                <h2>💡 Conclusões e Recomendações</h2>
                <div class="highlight">
                    <p><strong>Algoritmo mais recomendado:</strong> Random Forest apresentou o melhor desempenho geral, 
                    com uma acurácia de 91.22% e F1-Score de 90.80% no modelo base.</p>
                </div>
                
                <div class="highlight">
                    <p><strong>Equilibrio entre desempenho e velocidade:</strong> Decision Tree oferece um bom equilíbrio, 
                    com acurácia de 83.02% e tempo de treinamento muito reduzido (0.035s).</p>
                </div>
                
                <div class="highlight">
                    <p><strong>Para aplicações em tempo real:</strong> KNN é extremamente rápido (0.0013s), 
                    mas com menor acurácia (71.37%). Pode ser útil em cenários com restrições de latência.</p>
                </div>
                
                <h3 style="margin-top: 25px; color: #667eea;">Recomendações Finais:</h3>
                <ul style="margin-left: 20px; margin-top: 10px; line-height: 2;">
                    <li><strong>Produção:</strong> Use Random Forest com 200 estimadores para melhor acurácia</li>
                    <li><strong>Interpretabilidade:</strong> Use Decision Tree com profundidade máxima de 10</li>
                    <li><strong>Velocidade crítica:</strong> Use KNN com k=7 e weights='distance'</li>
                    <li><strong>Trade-off:</strong> Use SVM com kernel RBF para bom desempenho geral</li>
                </ul>
            </section>
            
            <!-- METODOLOGIA -->
            <section>
                <h2>🔬 Metodologia</h2>
                <p><strong>Dataset:</strong> Alzheimer's Disease Detection (2.149 amostras, 33 features)</p>
                <p><strong>Divisão de dados:</strong> 75% treino, 25% teste (estratificado)</p>
                <p><strong>Balanceamento:</strong> SMOTE para equilibrar classes</p>
                <p><strong>Normalização:</strong> StandardScaler</p>
                <p><strong>Validação:</strong> 5-fold Stratified Cross-Validation</p>
                <p><strong>Métricas:</strong> Acurácia, Precisão, Recall, F1-Score, AUC-ROC</p>
                
                <h3 style="margin-top: 20px; color: #667eea;">Algoritmos Comparados:</h3>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>MLP (Multi-Layer Perceptron)</strong> - Rede Neural</li>
                    <li><strong>Decision Tree</strong> - Árvore de Decisão</li>
                    <li><strong>KNN</strong> - K-Nearest Neighbors</li>
                    <li><strong>Logistic Regression</strong> - Regressão Logística</li>
                    <li><strong>Random Forest</strong> - Floresta Aleatória</li>
                    <li><strong>SVM</strong> - Support Vector Machine</li>
                </ul>
            </section>
        </div>
        
        <footer>
            <p>© 2026 - Comparação de Algoritmos para Detecção de Alzheimer</p>
            <p style="margin-top: 5px; font-size: 0.9em;">Relatório gerado automaticamente - {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""

# Salvar arquivo HTML
output_path = 'relatorios/comparacao_algoritmos.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Relatorio HTML gerado com sucesso: {output_path}")
print(f"Abra o arquivo em seu navegador para visualizar!")
