@echo off
cd /d "%~dp0"
title "AI Render Studio Launcher - 192.168.10.168 [2x P102-100]"

set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_VISIBLE_DEVICES=0,1"

if not exist "loras" mkdir "loras"
if not exist "output_img" mkdir "output_img"
if not exist "output_vid" mkdir "output_vid"

taskkill /IM sd-server.exe /F >nul 2>&1

if not exist "sd-server.exe" (
    echo [ERROR] Missing core file: sd-server.exe
    pause
    exit /b 1
)
if not exist "ggml-cuda.dll" (
    echo [ERROR] Missing core file: ggml-cuda.dll
    pause
    exit /b 1
)
if not exist "studio.py" (
    echo [ERROR] Missing core file: studio.py
    pause
    exit /b 1
)

cls
echo ========================================================================
echo   AI RENDER STUDIO - 2x P102-100 [STRICT VRAM MODE]
echo   Open in Browser : http://192.168.10.168:8082
echo   VRAM at startup : 0 GB (Models load on demand from Web UI)
echo ========================================================================
echo.
echo [INFO] Press Ctrl+C in this window to stop everything and free VRAM.
echo.

python studio.py

echo.
echo [INFO] Shutting down backend and freeing VRAM...
taskkill /IM sd-server.exe /F >nul 2>&1
timeout /t 2 >nul
