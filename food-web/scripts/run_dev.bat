@echo off
REM Launch FastAPI dev server (auto-reload)
setlocal
cd /d "%~dp0\.."

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

set "HOST=127.0.0.1"
set "PORT=8000"

echo Using interpreter: %PY%
echo Starting uvicorn at http://%HOST%:%PORT%
echo.

"%PY%" -m uvicorn server:app --reload --host %HOST% --port %PORT%

if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with error.
    pause
)
endlocal
