@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
  py -3 -X utf8 scripts\setup_gpu.py --backend cuda %*
) else (
  python -X utf8 scripts\setup_gpu.py --backend cuda %*
)
set "GPU_SETUP_EXIT=%errorlevel%"
if "%GPU_SETUP_EXIT%"=="0" (
  echo GPU setup completed. Run start-pair-windows.bat to select the GPU.
) else (
  echo GPU setup failed. See the error above.
)
pause
exit /b %GPU_SETUP_EXIT%
