@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0START.bat" --prepare-only
if errorlevel 1 exit /b 1
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PATH=%~dp0tools;%~dp0runtime;%PATH%"
cd /d "%~dp0app"
"%~dp0runtime\python.exe" -X utf8 "%~dp0app\scripts\setup_gpu.py" --backend cuda %*
set "GPU_SETUP_EXIT=%ERRORLEVEL%"
if "%GPU_SETUP_EXIT%"=="0" (
  echo GPU 준비가 완료됐습니다. START.bat를 실행하고 장치 설정에서 GPU를 선택하세요.
) else (
  echo GPU 준비에 실패했습니다. 위의 오류와 NVIDIA 드라이버를 확인하세요.
)
pause
exit /b %GPU_SETUP_EXIT%
