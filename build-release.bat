@echo off
setlocal enabledelayedexpansion

echo ================================================
echo ScreenSafe Release Build Script (Windows)
echo ================================================
echo.

:: Get version from tauri.conf.json
for /f "tokens=2 delims=:," %%a in ('findstr /C:"\"version\"" src-tauri\tauri.conf.json') do (
    set VERSION=%%~a
    set VERSION=!VERSION: =!
    set VERSION=!VERSION:"=!
)
echo Building version: v%VERSION%
echo.

:: Clean previous builds
echo [1/4] Cleaning previous builds...
if exist "release" rmdir /s /q "release"
mkdir release\ScreenSafe

:: Build the Tauri app with NSIS installer
echo [2/4] Building Tauri app with installer (this may take a few minutes)...
call npm run tauri build -- --bundles nsis
if %ERRORLEVEL% neq 0 (
    echo ERROR: Tauri build failed!
    exit /b 1
)

:: Copy files for portable version
echo [3/4] Creating portable package...
copy "src-tauri\target\release\ScreenSafe.exe" "release\ScreenSafe\" >nul

:: Copy Python sidecar scripts (dependencies install on first launch)
xcopy "python" "release\ScreenSafe\python\" /E /I /Q >nul

:: Copy assets
if exist "assets" xcopy "assets" "release\ScreenSafe\assets\" /E /I /Q >nul

:: Copy NSIS installer
for %%f in (src-tauri\target\release\bundle\nsis\*.exe) do (
    copy "%%f" "release\" >nul
    echo Found installer: %%~nxf
)

:: Create portable zip
echo [4/4] Creating portable ZIP...
cd release
powershell -Command "Compress-Archive -Path 'ScreenSafe\*' -DestinationPath 'ScreenSafe-v%VERSION%-windows-x64-portable.zip' -Force"
cd ..

echo.
echo ================================================
echo Build Complete!
echo ================================================
echo.
echo Release files created in: release\
echo.
dir release\*.zip release\*.exe /b 2>nul
echo.
echo Note: First-time users will see a setup dialog that installs
echo Python dependencies automatically. This takes ~3-5 minutes
echo on first launch, then subsequent launches are instant.
echo.
echo Ready to upload to GitHub release!

endlocal
