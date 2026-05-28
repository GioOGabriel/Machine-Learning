@echo off
REM ================================================================
REM Script para iniciar a aplicacao Streamlit
REM ================================================================

cls
echo.
echo =====================================================
echo   APLICACAO STREAMLIT - DETECCAO DE ALZHEIMER
echo =====================================================
echo.

REM Verificar se Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no sistema!
    echo Instale Python de: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Verificar se streamlit esta instalado
python -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo [INFO] Instalando dependencias...
    python -m pip install -r requirements.txt
)

REM Verificar se modelo existe
if not exist "models\alzheimer_rf_model.pkl" (
    echo [INFO] Modelo nao encontrado. Treinando...
    python scripts/treinar_e_exportar_modelo.py
    if errorlevel 1 (
        echo [ERRO] Falha ao treinar o modelo!
        pause
        exit /b 1
    )
)

echo [OK] Iniciando aplicacao...
echo.
echo Abrindo navegador em: http://localhost:8501
echo.
echo Pressione Ctrl+C para parar a aplicacao
echo.

REM Iniciar Streamlit (versao corrigida)
streamlit run app_streamlit.py

pause
