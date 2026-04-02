@echo off
REM Comprehensive build script for Peak Analysis Tool
REM This builds both the web UI and the backend service into a single deployable package

echo ================================
echo Peak Analysis Tool - Full Build
echo ================================
echo.

REM Step 1: Build the Next.js web UI
echo [1/3] Building web UI...
cd ui-web
call npm install
if errorlevel 1 (
    echo ERROR: npm install failed
    exit /b 1
)

call npm run build
if errorlevel 1 (
    echo ERROR: npm build failed
    exit /b 1
)
cd ..
echo Web UI built successfully!
echo.

REM Ensure documentation PDFs are present in the static export
echo [Docs] Copying user manual and mathematical reference into ui-web\out...
if not exist "ui-web\out\docs" mkdir "ui-web\out\docs"
robocopy "ui-web\public\docs" "ui-web\out\docs" "*.pdf" /NFL /NDL /NJH /NJS /NC /NS /NP >nul
if errorlevel 8 (
    echo ERROR: Failed to copy documentation PDFs into ui-web\out\docs
    exit /b 1
)
if not exist "ui-web\out\docs\user_manual.pdf" (
    echo ERROR: user_manual.pdf missing after copy
    exit /b 1
)
if not exist "ui-web\out\docs\mathematical_reference.pdf" (
    echo ERROR: mathematical_reference.pdf missing after copy
    exit /b 1
)
echo Documentation copied to ui-web\out\docs
echo.

REM Step 2: Build the FastAPI backend into an EXE
echo [2/3] Building backend service...
pyinstaller --noconfirm --clean --onefile ^
    --name PeakService ^
    --add-data "config;config" ^
    --add-data "ui-web/out;ui-web/out" ^
    --hidden-import "uvicorn.logging" ^
    --hidden-import "uvicorn.loops" ^
    --hidden-import "uvicorn.loops.auto" ^
    --hidden-import "uvicorn.protocols" ^
    --hidden-import "uvicorn.protocols.http" ^
    --hidden-import "uvicorn.protocols.http.auto" ^
    --hidden-import "uvicorn.protocols.websockets" ^
    --hidden-import "uvicorn.protocols.websockets.auto" ^
    --hidden-import "uvicorn.lifespan" ^
    --hidden-import "uvicorn.lifespan.on" ^
    service/app.py

if errorlevel 1 (
    echo ERROR: PyInstaller build failed
    exit /b 1
)
echo Backend service built successfully!
echo.

REM Step 3: Create distribution directory
echo [3/3] Creating distribution package...
if not exist "dist\PeakTool" mkdir "dist\PeakTool"
copy "dist\PeakService.exe" "dist\PeakTool\"
copy "launcher\launch_ui.py" "dist\PeakTool\"
copy "README.md" "dist\PeakTool\README.txt"

echo.
echo ================================
echo Build Complete!
echo ================================
echo.
echo Output directory: dist\PeakTool
echo   - PeakService.exe (Backend + Web UI)
echo   - launch_ui.py (Optional launcher)
echo.
echo To run: Just execute PeakService.exe
echo It will start the server and open your browser automatically.
echo.









