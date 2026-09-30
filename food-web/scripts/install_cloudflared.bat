@echo off
REM Download cloudflared.exe for Windows into bin/
REM Tries direct GitHub first, falls back to ghproxy mirror if the first fails

setlocal
cd /d "%~dp0\.."

set "BIN_DIR=bin"
set "TARGET=%BIN_DIR%\cloudflared.exe"
set "URL_DIRECT=https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
set "URL_MIRROR=https://ghproxy.com/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

if exist "%TARGET%" (
    echo [INFO] %TARGET% already exists. Delete it first if you want to redownload.
    "%TARGET%" --version
    goto :end
)

echo [INFO] Downloading cloudflared.exe ...
echo        from: %URL_DIRECT%

powershell -NoProfile -Command "try { $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%URL_DIRECT%' -OutFile '%TARGET%' -UseBasicParsing -TimeoutSec 60; exit 0 } catch { exit 1 }"

if not exist "%TARGET%" (
    echo [WARN] Direct GitHub download failed, trying ghproxy mirror ...
    echo        from: %URL_MIRROR%
    powershell -NoProfile -Command "try { $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%URL_MIRROR%' -OutFile '%TARGET%' -UseBasicParsing -TimeoutSec 90; exit 0 } catch { exit 1 }"
)

if not exist "%TARGET%" (
    echo.
    echo [ERROR] Both downloads failed. Please download manually:
    echo   %URL_DIRECT%
    echo   and place it at:  %CD%\%TARGET%
    pause
    exit /b 1
)

echo.
echo [OK] Saved to %CD%\%TARGET%
"%TARGET%" --version

:end
endlocal
