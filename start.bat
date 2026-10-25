@echo off
title RAG Chatbot Launcher
echo ==============================================
echo   Starting RAG Document Chatbot (Parallel)
echo ==============================================

:: Navigate to script directory
cd /d "%~dp0"

:: Start Backend in a separate window
echo [1/2] Launching Backend on http://127.0.0.1:8000...
start "RAG Chatbot - Backend" cmd /k ".\venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"

:: Start Frontend in a separate window
echo [2/2] Launching Frontend on http://localhost:5173...
start "RAG Chatbot - Frontend" cmd /k "cd frontend && npm run dev"

:: Wait 3 seconds and launch the browser
timeout /t 3 /nobreak >nul
start http://localhost:5173

echo.
echo Both servers are running in parallel!
echo To stop them, simply close the Backend and Frontend windows.
echo ==============================================
