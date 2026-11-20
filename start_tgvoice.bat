@echo off
cd /d "%~dp0"
title Silero TTS Server (Port 8001)
python silero_tts.py
echo [!] Restarting Silero TTS in 1 second...
timeout /t 1 /nobreak >nul