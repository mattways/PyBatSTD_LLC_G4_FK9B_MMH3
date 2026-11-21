@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title LSG4 AI Launcher (2x Zotac P102-100 20GB - 100%% GPU + MTP)
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_SCALE_LAUNCH_QUEUES=4

:menu
cls
echo ========================================================================
echo  LLAMA-SERVER GEMMA 4 LAUNCHER ^| 2x Zotac P102-100 (20GB VRAM)
echo ========================================================================
echo.
echo  1. Gemma 4 12B      - Single GPU 0 (100%% GPU + MTP + Vision, 8K ctx)
echo  2. Gemma 4 26B-A4B  - Dual GPU (MoE + MTP Q2_K, Zero DDR3, 16K ctx)
echo  3. Gemma 4 4B       - Single GPU 0 (100%% GPU 8K ctx)
echo  5. Check GPU Status - Show nvidia-smi (Temp, Power, Bus ID, VRAM)
echo llama-server.exe -lv 4 -t 4 -tb 4 -m "%MODEL%" --mmproj "%MMPROJ%" --model-draft "%DRAFT%" -dev CUDA0 --spec-draft-device CUDA0 -sm none -ot ".+=CUDA0" --override-tensor-draft ".+=CUDA0" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 -ngl all --n-gpu-layers-draft all --no-host -lm none --cache-ram 0 -fit off -c 16000 -b 2048 -ub 1152 -np 1 -ctk q8_0 -ctv q8_0 --jinja --reasoning off --host 127.0.0.1 --temp 1.0 --top-p 0.95 --top-k 64 --port 8080
echo.
echo ========================================================================
set /p m="Select mode [1-2, 5]: "
if "%m%"=="1" goto mode1
if "%m%"=="2" goto mode2
if "%m%"=="3" goto mode3
if "%m%"=="5" goto status
goto menu

:mode1
set "MODEL=Gemma4-12B-QAT-Q4-K-M.gguf"
set "DRAFT=gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf"
set "MMPROJ=mmproj-Gemma4-12B-QAT-BF16.gguf"
if not exist "%MODEL%" goto missing
if not exist "%DRAFT%" goto missing_draft
if not exist "%MMPROJ%" goto missing_mmproj
echo [INFO] Starting Gemma 4 12B + MTP + Vision on GPU 0 (100%% VRAM, Zero DDR3)...
start "" /min cmd /c "timeout /t 12 /nobreak >nul && start http://127.0.0.1:8080"
llama-server.exe -lv 4 -t 4 -tb 4 -m "%MODEL%" --model-draft "%DRAFT%" -dev CUDA0,CUDA1 --spec-draft-device CUDA0 -sm layer --tensor-split 46,54 -ot "token_embd.weight=CUDA0" -ot "output.weight=CUDA0" --override-tensor-draft "token_embd.weight=CUDA0" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 -ngl all --n-gpu-layers-draft all --no-host -lm none --cache-ram 0 -fit off -c 100000 -b 2048 -ub 1152 -np 1 -ctk q8_0 -ctv q8_0 --jinja --reasoning on --reasoning-budget 3072 --temp 1.0 --top-p 0.95 --top-k 64 --host 127.0.0.1 --port 8080
goto end

:mode2
set "MODEL=gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
set "DRAFT=gemma-4-26B-A4B-it-qat-assistant-MTP-Q8_0.gguf"
if not exist "%MODEL%" goto missing
if not exist "%DRAFT%" goto missing_draft
echo [INFO] Starting Gemma 4 26B-A4B MoE + MTP Q2_K on Dual GPU (Zero DDR3)...
start "" /min cmd /c "timeout /t 18 /nobreak >nul && start http://127.0.0.1:8080"
llama-server.exe -lv 4 -t 4 -tb 4 -m "%MODEL%" --model-draft "%DRAFT%" -dev CUDA0,CUDA1 --spec-draft-device CUDA0 -sm layer --tensor-split 46,54 -ot "token_embd.weight=CUDA0" -ot "output.weight=CUDA0" --override-tensor-draft "token_embd.weight=CUDA0" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 -ngl all --n-gpu-layers-draft all --no-host -lm none --cache-ram 0 -fit off -c 12288 -b 2048 -ub 1152 -np 1 -ctk q8_0 -ctv q8_0 --jinja --reasoning on --reasoning-budget 3072 --temp 1.0 --top-p 0.95 --top-k 64 --host 127.0.0.1 --port 8080
goto end

:mode3
set "MODEL=Gemma-4-E2B-IQ3_M.gguf"
if not exist "%MODEL%" goto missing
echo [INFO] Starting Gemma 4 4B
start "" /min cmd /c "timeout /t 12 /nobreak >nul && start http://127.0.0.1:8080"
llama-server.exe -lv 4 -t 4 -tb 4 -m "%MODEL%" -dev CUDA0 --spec-draft-device CUDA0 -sm none -ot ".+=CUDA0" --override-tensor-draft ".+=CUDA0" --spec-draft-p-min 0.75 -ngl all --n-gpu-layers-draft all --no-host -lm none --cache-ram 0 -fit off -c 2048 -b 2048 -ub 1152 -np 1 -ctk q8_0 -ctv q8_0 --jinja --reasoning off --host 127.0.0.1 --port 8080
goto end

:status
cls
nvidia-smi
echo.
pause
goto menu

:missing
echo.
echo [ERROR] Model file not found: "%MODEL%"
pause
goto menu

:missing_draft
echo.
echo [ERROR] MTP draft file not found: "%DRAFT%"
pause
goto menu

:missing_mmproj
echo.
echo [ERROR] Vision mmproj file not found: "%MMPROJ%"
pause
goto menu

:end
echo.
echo [INFO] Server stopped.
pause
