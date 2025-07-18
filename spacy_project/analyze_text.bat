@echo off
REM Text Analysis Helper Script
REM This script automatically converts Windows paths to Docker container paths

echo Text Analysis Helper
echo -------------------

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is required but not found in PATH
    exit /b 1
)

REM Run the Python helper script with all arguments passed to this batch file
REM Determine the directory of this script
set SCRIPT_DIR=%~dp0

REM Execute the helper Python script located in the src folder
python "%SCRIPT_DIR%src\run_analysis.py" %*

echo Analysis complete!
