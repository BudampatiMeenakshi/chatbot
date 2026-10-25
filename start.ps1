$root = $PSScriptRoot
if (-not $root) { $root = Get-Location }

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Starting RAG Document Chatbot (Parallel)    " -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# 1. Start Backend in a new window
Write-Host "[1/2] Starting Backend (FastAPI)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; .\venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"

# 2. Start Frontend in a new window
Write-Host "[2/2] Starting Frontend (React + Vite)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"

# 3. Open browser
Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"

Write-Host "`nBoth Backend and Frontend are running in parallel!" -ForegroundColor Green
Write-Host "Browser opened at http://localhost:5173" -ForegroundColor Green
