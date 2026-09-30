@echo off
REM Start a Cloudflare Quick Tunnel pointing at the local FastAPI server.
REM Prereq:
REM   1) Run scripts\install_cloudflared.bat once to download bin\cloudflared.exe
REM   2) Run scripts\run_dev.bat in another window to start the FastAPI server on port 8000
REM
REM This script will print a public URL like:
REM   https://something-random-words.trycloudflare.com
REM Anyone with that URL can use your model from the public internet, with no
REM account required. The URL is throwaway - it disappears when you Ctrl+C.

setlocal
cd /d "%~dp0\.."

set "CF=bin\cloudflared.exe"
set "LOCAL_URL=http://localhost:8000"

if not exist "%CF%" (
    echo [ERROR] %CF% not found.
    echo         Please run scripts\install_cloudflared.bat first.
    exit /b 1
)

echo ============================================================
echo  Cloudflare Quick Tunnel - free, ephemeral public URL
echo ============================================================
echo  Local target:  %LOCAL_URL%
echo  Make sure your FastAPI server is already running there!
echo  (Run scripts\run_dev.bat in another window if not.)
echo.
echo  Look for a line like:
echo    Your quick Tunnel has been created! Visit it at:
echo    https://xxxxx-xxxxx-xxxxx-xxxxx.trycloudflare.com
echo.
echo  Press Ctrl+C to stop the tunnel.
echo ============================================================
echo.

"%CF%" tunnel --url %LOCAL_URL% --no-autoupdate

endlocal
