@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if exist ".package-ready-0.5.4" goto launch
echo Preparing Court Review 0.5.4. Please wait...
powershell.exe -NoLogo -NoProfile -Command "$ErrorActionPreference='Stop'; $items=@(Get-ChildItem -LiteralPath 'payload' -Filter '*.zip' | Sort-Object Name); $timer=[Diagnostics.Stopwatch]::StartNew(); for($i=0;$i -lt $items.Count;$i++){ Write-Host ('[{0}/{1}] {2}' -f ($i+1),$items.Count,$items[$i].Name); Write-Progress -Activity 'Preparing Court Review' -Status $items[$i].Name -PercentComplete (($i/$items.Count)*100); Expand-Archive -LiteralPath $items[$i].FullName -DestinationPath . -Force }; Write-Progress -Activity 'Preparing Court Review' -Completed; Write-Host ('Ready in {0:N1} seconds.' -f $timer.Elapsed.TotalSeconds)"
if errorlevel 1 goto failed
type nul > ".package-ready-0.5.4"
:launch
call "%~dp0RUN.bat"
exit /b %ERRORLEVEL%
:failed
echo Could not extract the package. Extract the downloaded ZIP completely and try again.
pause
exit /b 1
