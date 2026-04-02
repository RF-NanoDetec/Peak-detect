@echo off
REM Build the FastAPI backend into a single EXE using PyInstaller
REM Requires: pip install pyinstaller

set NAME=PeakService
pyinstaller --noconfirm --clean --onefile --name %NAME% --add-data "config;config" service/app.py

echo Build complete. EXE at "dist\%NAME%.exe"


