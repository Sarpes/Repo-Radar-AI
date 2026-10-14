@echo off
setlocal EnableExtensions
REM ===========================================================
REM  panel_v2.bat  ^|  RepoRadarAI SUNUM PANELI
REM ===========================================================
REM  panel_v2\app.py'i 8611 portunda acar.
REM  Grafiklerin cogu calisma aninda artefaktlardan cizilir;
REM  PNG figurler figures\panel\ klasorunden okunur.
REM
REM  Canli demo bolumu icin gecit.bat ayrica acik olmali.
REM ===========================================================

set "KOK=%~dp0"
set "PY=%USERPROFILE%\anaconda3\python.exe"
set "PORT=8611"

echo.
echo ============================================================
echo  DS21 SUNUM PANELI v2
echo ============================================================

if not exist "%PY%" (echo   [RET ] python yok: %PY% & pause & exit /b 1)
echo   [ OK ] python

for %%f in (outputs\model_rapor.json outputs\features.parquet outputs\repo_scores.parquet outputs\scores_today.parquet outputs\label_compare.parquet outputs\rag_sweep.csv Data\github_ai_intelligence_2026.csv panel_v2\app.py) do (
  if not exist "%KOK%%%f" (echo   [RET ] %%f yok & pause & exit /b 1)
)
echo   [ OK ] artefaktlar

if not exist "%KOK%figures\panel\F19_karisiklik_matrisi.png" (
  echo   [UYAR] figures\panel eksik - once cizdir: python panel_v2\figur_uret.py
) else (
  echo   [ OK ] figures\panel
)

for /f %%c in ('curl -s -m 3 -o nul -w "%%{http_code}" http://127.0.0.1:8787/saglik') do set "GC=%%c"
if "%GC%"=="200" (echo   [ OK ] gecit 8787 - canli demo hazir) else (echo   [UYAR] gecit kapali - demo icin gecit.bat ac)

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo   [RET ] %PORT% portu dolu - panel zaten acik olabilir: http://localhost:%PORT%
  start "" http://localhost:%PORT%
  exit /b 1
)

echo.
echo   Panel v2: http://localhost:%PORT%    Durdurmak icin Ctrl+C
echo.
start "" cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:%PORT%"
cd /d "%KOK%panel_v2"
"%PY%" -m streamlit run app.py --server.port %PORT%
if errorlevel 1 (echo. & echo   [RET ] streamlit hata ile kapandi - yukaridaki satirlara bak & pause)
