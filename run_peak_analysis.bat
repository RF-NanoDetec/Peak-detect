@echo off
echo Starting Peak Analysis Tool...
python app.py
if errorlevel 1 (
    echo Error occurred during execution.
    echo Check log files for details.
    pause
    exit /b 1
)
exit /b 0 