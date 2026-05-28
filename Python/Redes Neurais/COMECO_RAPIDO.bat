@echo off
REM ============================================================
REM COMECO RAPIDO - Deteccao de Alzheimer
REM ============================================================

echo.
echo ====================================================
echo   DETECCAO DE ALZHEIMER - COMECO RAPIDO
echo ====================================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado!
    echo Instale Python 3.8+ de: https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python encontrado
echo.

REM Instalar dependencias
echo [2/4] Instalando dependencias...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERRO ao instalar dependencias!
    pause
    exit /b 1
)

echo [3/4] Dependencias instaladas
echo.

REM Testar Kaggle
echo [4/4] Testando Kaggle...
python tests\teste_kaggle.py
if errorlevel 1 (
    echo.
    echo AVISO: Kaggle nao esta configurado
    echo Veja: docs\GUIA_KAGGLEHUB.md
    echo.
)

echo.
echo ====================================================
echo   COMECO RAPIDO CONCLUIDO!
echo ====================================================
echo.
echo Proximos passos:
echo   1. python config\launcher_streamlit.py
echo   2. Abra: http://localhost:8501
echo.
echo Documentacao: README.md
echo.
pause
