@echo off
title ScreenSafe Dev Launcher
echo ========================================
echo  ScreenSafe Development Launcher
echo ========================================
echo.

REM Start Python backend in new window
echo [1/2] Starting Python backend...
start "ScreenSafe Python Backend" cmd /k "cd /d %~dp0python && python main.py"

REM Wait a moment for backend to start
timeout /t 2 /nobreak > nul

REM Start Tauri dev server
echo [2/2] Starting Tauri app...
start "ScreenSafe App" cmd /k "cd /d %~dp0 && npm run tauri dev"

echo.
echo Both services starting in separate windows!
echo Close this window when done.
pause
