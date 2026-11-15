@echo off
setlocal
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0vision\dashboard.py"
    exit /b
)
where pyw.exe >nul 2>&1
if not errorlevel 1 (
    py -3.14 -c "import tkinter" >nul 2>&1
    if not errorlevel 1 (
        start "" pyw.exe -3.14 "%~dp0vision\dashboard.py"
        exit /b
    )
)
where pythonw.exe >nul 2>&1
if not errorlevel 1 (
    start "" pythonw.exe "%~dp0vision\dashboard.py"
    exit /b
)
echo Python with Tkinter is required. Install Python or activate the project environment.
pause
exit /b 1
