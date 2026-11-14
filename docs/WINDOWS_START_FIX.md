# Windows PowerShell 5.1 START 오류 수정 — 0.5.11

0.5.10의 `START.bat`는 정상 패키지에서도 `Expected nine payload archives`를 출력하며 중단할 수 있다. Windows PowerShell 5.1의 `ConvertFrom-Json`이 최상위 JSON 배열을 하나의 파이프라인 객체로 반환하는데, 기존 `@(... | ConvertFrom-Json)`이 그 배열을 한 겹 더 감싸 `Count`를 1로 만들었다.

JSON 변환 결과를 변수에 먼저 대입하고 그 변수의 요소를 배열로 정규화하도록 수정했다. 정상적인 9개 ZIP의 개수·이름·크기·SHA256 검사, 압축 해제 전 검증, 준비 완료 표시와 재실행 동작은 유지한다. `-NoEnumerate`처럼 PowerShell 5.1에서 지원하지 않는 옵션을 사용하지 않는다.

PowerShell 7에서 실행할 때 전달되는 모듈 검색 경로 때문에 Windows PowerShell 5.1이 `Get-FileHash` 같은 기본 명령을 찾지 못하는 경우도 보완했다. 배치 파일의 `setlocal` 범위에서 모듈 경로를 초기화해 5.1이 자신의 기본 모듈을 로드하게 하며, 부모 셸이나 시스템 설정은 변경하지 않는다.

## 기존 0.5.10 사용자

[0.5.11 릴리스](https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/tag/v0.5.11)의 `badminton-windows-0.5.10-start-fix.zip`을 내려받아 압축을 푼다. 안의 `START.bat`를 기존 0.5.10 패키지에서 `PAYLOAD_CHECKSUMS.json`과 `payload` 폴더가 있는 위치에 덮어쓴 뒤 실행한다. 패치는 시작 파일 한 개만 바꾸며, 기존 영상·분석 결과·GPU 환경·9개 구성 ZIP은 그대로 사용한다.

완전한 새 패키지를 원하는 경우 `badminton-windows-0.5.11.zip`을 새 폴더에 전부 압축 해제하고 `START.bat`를 실행한다. 기존 폴더 위에 새 전체 패키지를 섞어 풀지 않는다. 이전 0.5.10 전체 ZIP은 변경하지 않으며 패치와 새 버전은 별도 파일로 배포한다.

## 검증 범위

`tests/test_windows_package.py`는 Windows PowerShell 5.1과 실제 `cmd.exe → START.bat` 실행을 사용한다. 기존 배열 개수가 1이고 직접 대입한 배열 개수가 9임을 재현하며, 한글·공백 경로에서 9개 ZIP 추출·재실행, 개수 오류·누락·크기 오류·같은 크기의 SHA256 손상을 검사한다. 준비 완료 표시가 있어도 손상 파일을 다시 거부하는지 확인한다.

별도 배포 검증은 실제 구성 ZIP을 포함한 전체 패키지를 Windows에서 준비하고 재실행한다. 포함된 Python·분석 의존성과 FFmpeg를 로드한 뒤 A/B 서버를 실행해 `/pair` 응답과 경기 방식 선택 화면을 확인한다. 이 절차와 0.5.10 패치 적용 검증에 통과한 뒤 릴리스를 공개하며 결과 JSON을 릴리스에 첨부한다. GPU 실기, 사용자 PC의 모든 환경과 경기 판정 정확도는 이 시작 호환성 시험의 범위가 아니다.

소스 CI: [Windows launcher compatibility](https://github.com/6rmkhj/Badminton_AI_Line_Judge/actions/workflows/windows-launcher.yml). 배포 검증의 실행 링크와 상세 결과는 릴리스 노트를 참고한다.
