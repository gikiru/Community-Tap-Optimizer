@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo   TAP PLACEMENT APP - SMART INSTALLER
echo ========================================
echo.
echo Checking system requirements...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python is installed
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available
    echo.
    echo Trying to install pip...
    python -m ensurepip --upgrade
    if errorlevel 1 (
        echo [ERROR] Could not install pip
        pause
        exit /b 1
    )
)

echo [OK] pip is available
pip --version
echo.

echo ========================================
echo   FULL INSTALLATION
echo ========================================
echo.
echo Installing packages... This may take 5-10 minutes.
echo.

echo [1/4] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Could not upgrade pip, continuing anyway...
)

echo.
echo [2/4] Installing core dependencies...
pip install streamlit pandas numpy scikit-learn geopy folium streamlit-folium tqdm
if errorlevel 1 (
    echo [ERROR] Core dependencies installation failed
    pause
    exit /b 1
)

echo.
echo [3/4] Installing visualization libraries...
pip install plotly matplotlib
if errorlevel 1 (
    echo [WARNING] Visualization libraries partially failed
)

echo.
echo [4/4] Installing advanced features...
pip install scipy shapely
if errorlevel 1 (
    echo [WARNING] Some advanced features may not work
)

echo.
echo [SUCCESS] Full installation complete!
echo.
echo You can now run: run_app.bat
goto end

:end
echo.
echo ========================================
echo   INSTALLATION SUMMARY
echo ========================================
echo.

REM Check installed packages
echo Verifying installations...
echo.

python -c "import streamlit; print('  [OK] Streamlit:', streamlit.__version__)" 2>nul || echo   [MISSING] Streamlit
python -c "import pandas; print('  [OK] Pandas:', pandas.__version__)" 2>nul || echo   [MISSING] Pandas
python -c "import numpy; print('  [OK] NumPy:', numpy.__version__)" 2>nul || echo   [MISSING] NumPy
python -c "import sklearn; print('  [OK] Scikit-learn:', sklearn.__version__)" 2>nul || echo   [MISSING] Scikit-learn
python -c "import folium; print('  [OK] Folium:', folium.__version__)" 2>nul || echo   [MISSING] Folium

if "%choice%"=="1" (
    python -c "import plotly; print('  [OK] Plotly:', plotly.__version__)" 2>nul || echo   [MISSING] Plotly
    python -c "import scipy; print('  [OK] SciPy:', scipy.__version__)" 2>nul || echo   [MISSING] SciPy
    python -c "import shapely; print('  [OK] Shapely:', shapely.__version__)" 2>nul || echo   [MISSING] Shapely
)

if "%choice%"=="2" (
    python -c "import matplotlib; print('  [OK] Matplotlib:', matplotlib.__version__)" 2>nul || echo   [MISSING] Matplotlib
)

echo.
echo ========================================
echo.
echo Installation complete!
echo.
echo NEXT STEPS:
echo   1. Prepare your CSV files with lat/lon columns
echo   2. Run run_app.bat
echo   3. Upload your data and start analyzing!
echo.
echo For help, see README.md
echo.
pause
