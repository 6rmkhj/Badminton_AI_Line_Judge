정상 패키지에서도 `START.bat` 실행 시 `Expected nine payload archives`로 중단되던 Windows PowerShell 5.1 호환성 오류를 수정했습니다. JSON 배열을 한 겹 더 감싸 ZIP을 1개로 세던 문제와 PowerShell 7에서 상속된 모듈 경로 때문에 기본 명령을 찾지 못하는 문제를 보완했습니다. 구성 ZIP 개수·크기·SHA256 검사는 유지합니다.

**기존 0.5.10 사용자는 전체 패키지를 다시 받을 필요가 없습니다.** 아래 작은 패치를 풀고, 안의 `START.bat`를 기존 `PAYLOAD_CHECKSUMS.json`과 `payload` 폴더가 있는 위치에 덮어쓴 뒤 실행하세요. 기존 영상·분석 결과·GPU 환경은 그대로 사용합니다.

- [0.5.10용 START 교체 패치](https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/download/v0.5.11/badminton-windows-0.5.10-start-fix.zip)
- [0.5.11 전체 Windows 패키지](https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/download/v0.5.11/badminton-windows-0.5.11.zip): 새 폴더에 전부 압축 해제하고 `START.bat` 실행

Windows Server 2022 / Windows PowerShell 5.1에서 **13개 회귀 테스트가 통과**했습니다. 기존 오류를 재현했고, 한글·공백 경로의 압축 해제·재실행과 누락·크기·SHA256 손상 차단을 확인했습니다. [Windows 회귀 테스트 로그](https://github.com/6rmkhj/Badminton_AI_Line_Judge/actions/runs/43577373911).

릴리스 공개 전에 **새 전체 패키지**와 **실제 공개된 0.5.10 ZIP에 START만 교체한 패키지**를 각각 Windows에서 검사했습니다. 9개 구성 ZIP 검증·압축 해제·재실행, 포함된 Python·OpenCV·NumPy·SciPy·Filelock·FFmpeg·FFprobe를 통한 시작, A/B 화면 HTTP 200과 경기 방식 선택 표시를 확인합니다. 두 실행의 상세 JSON을 아래 자산에 첨부했습니다. GPU 실기나 경기 판정 정확도를 검증한 시험은 아닙니다.

소스: [`39b2b9e`](https://github.com/6rmkhj/Badminton_AI_Line_Judge/commit/39b2b9e97c7e22100f3124a397d3d1d3dda5ec15). [변경 내용과 적용 방법](https://github.com/6rmkhj/Badminton_AI_Line_Judge/blob/v0.5.11/docs/WINDOWS_START_FIX.md).

전체 ZIP SHA256:
```text
ef98b4d51142621e4ad1c74ef78dedbb6b36cd541cd114fae326a72ae4302dd3
```
전체 ZIP과 패치 ZIP의 `.sha256` 파일을 함께 제공합니다. 업로드한 모든 파일의 크기·SHA256이 검증본과 일치할 때만 릴리스를 공개합니다.
