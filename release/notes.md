Windows에서 한글 사용자 이름이나 한글 폴더 경로로 실행할 때 근거 사진 저장이 실패하여, A/B 분석이 `근거 영상 생성` 단계의 `OSError`로 중단되던 문제를 수정했습니다. 이미지 파일은 Windows의 Unicode 경로를 지원하는 방식으로 읽고 저장합니다.

근거 사진·영상 생성은 판정 계산과 분리했습니다. 저장이나 인코딩에 실패하더라도 이미 계산한 판정 결과를 저장하고, 실패한 근거 항목과 실제 오류를 화면에 표시합니다. 영상 추정은 직접 접촉이 확인된 판정과 계속 구분하며, IN/OUT의 근거 요건과 정확도 기준을 완화하지 않았습니다.

- [0.5.13 전체 Windows 패키지](https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/download/v0.5.13/badminton-windows-0.5.13.zip): 새 폴더에 전부 압축 해제하고 `START.bat` 실행
- [ZIP SHA256](https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/download/v0.5.13/badminton-windows-0.5.13.zip.sha256)
- [Windows 전체 검증 보고서](https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/download/v0.5.13/windows-0.5.13-validation.json)

Linux 전체 Python 테스트 1,329개 통과·12개 건너뜀 및 브라우저 검증을 완료했습니다.

릴리스 공개 전 Windows Server 2022 / Windows PowerShell 5.1에서 실제 전체 패키지의 `START.bat --prepare-only` 2회와 HTTP 화면을 확인하고, 번들 Python·OpenCV·NumPy·SciPy로 회귀 테스트 63개를 실행합니다. 추가로 한글·공백이 포함된 입력·출력·임시 경로에서 실제 A 복식·A 단식·B 복식 영상 분석을 끝까지 수행하여, 결과 JSON 저장과 근거 사진·영상 재생을 확인합니다. 이 검증에 실패하면 릴리스를 공개하지 않습니다.

A 영상은 저장소의 동명 원본이며 사용자 첨부와 바이트 동일성은 확인하지 못했습니다. B 영상은 사용자 첨부와 SHA256이 같은 저장소 원본입니다. 영상 시험은 해당 입력에서의 동작 확인이며, 독립 정답에 대한 경기 판정 정확도 측정이나 GPU 실기 시험은 아닙니다.

소스 태그 [`v0.5.13`](https://github.com/6rmkhj/Badminton_AI_Line_Judge/tree/v0.5.13)와 패키지 소스 커밋이 일치하고, 업로드한 ZIP·SHA256·Windows 보고서가 검증본과 일치할 때만 공개합니다.
