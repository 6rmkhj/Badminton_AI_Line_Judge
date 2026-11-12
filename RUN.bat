@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
"%~dp0runtime\python.exe" -X utf8 "%~dp0portable_start.py"
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" pause
exit /b %APP_EXIT%
