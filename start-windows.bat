@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
  py -3 -X utf8 start.py %*
) else (
  python -X utf8 start.py %*
)
pause
