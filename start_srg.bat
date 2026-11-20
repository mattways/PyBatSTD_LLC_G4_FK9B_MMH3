@echo off
if "%~1"=="tts_worker" goto tts_loop

cd /d "%~dp0"
title Game Screen Reader + Silero TTS Launcher
echo ==================================================
echo Starting Silero TTS Server (Port 8001) + Screen Reader
echo ==================================================

start "Silero TTS Server" cmd /k ""%~f0" tts_worker"

echo Waiting 2 seconds for Silero model to load...
timeout /t 2 /nobreak >nul

echo Launching Screen Reader...
python screen_reader.py
pause
exit /b

:tts_loop
cd /d "%~dp0"
title Silero TTS Server (Port 8001)
python silero_tts.py
echo [!] Restarting Silero TTS in 1 second...
timeout /t 1 /nobreak >nul
goto tts_loop
