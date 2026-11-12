@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if exist ".package-ready" goto launch
echo Preparing the application. Please wait...
powershell.exe -NoLogo -NoProfile -Command "$ErrorActionPreference='Stop'; Get-ChildItem -LiteralPath 'payload' -Filter '*.zip' | Sort-Object Name | ForEach-Object { Expand-Archive -LiteralPath $_.FullName -DestinationPath . -Force }"
if errorlevel 1 goto failed
type nul > ".package-ready"
:launch
call "%~dp0RUN.bat"
exit /b %ERRORLEVEL%
:failed
echo Could not extract the package. Extract the downloaded ZIP completely and try again.
pause
exit /b 1
