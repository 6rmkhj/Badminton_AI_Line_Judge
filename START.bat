@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo 배드민턴 AI 0.5.8 구성 파일을 확인합니다.
powershell.exe -NoLogo -NoProfile -Command "$ErrorActionPreference='Stop'; $manifest='PAYLOAD_CHECKSUMS.json'; $items=@(Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json); if($items.Count -ne 9){throw 'Expected nine payload archives'}; $seen=@{}; foreach($item in $items){ if($item.file -notmatch '^(0[1-9])-[a-z_-]+[.]zip$' -or $seen.ContainsKey($item.file)){throw 'Invalid payload name'}; $seen[$item.file]=$true; $path=Join-Path 'payload' $item.file; if(!(Test-Path -LiteralPath $path -PathType Leaf)){throw ('Missing '+$path)}; if((Get-Item -LiteralPath $path).Length -ne $item.bytes){throw ('Size mismatch: '+$path)}; if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $item.sha256){throw ('SHA256 mismatch: '+$path)} }; $marker='.package-ready-0.5.8'; $digest=(Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash; if(!(Test-Path -LiteralPath $marker) -or (Get-Content -LiteralPath $marker -Raw).Trim() -ne $digest){ $i=0; foreach($item in $items){$i++; Write-Host ('[{0}/{1}] {2}' -f $i,$items.Count,$item.file); Expand-Archive -LiteralPath (Join-Path 'payload' $item.file) -DestinationPath . -Force }; Set-Content -LiteralPath $marker -Value $digest -Encoding ASCII }"
if errorlevel 1 goto failed
if /I "%~1"=="--prepare-only" exit /b 0
call "%~dp0RUN.bat" %*
exit /b %ERRORLEVEL%
:failed
echo 준비하지 못했습니다. ZIP을 새 폴더에 전부 압축 해제한 뒤 다시 실행하세요.
pause
exit /b 1
