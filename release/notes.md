원본 영상의 연속 프레임과 착지 후보 구간을 다시 확인하여, 충분한 근거가 있는 경우 **영상 추정 IN/OUT**을 별도로 표시합니다. 직접 바닥 접촉과 대상 셔틀이 확인된 판정은 기존의 엄격한 검증을 유지합니다. 영상 추정은 최초 바닥 접촉이 직접 확인되었다는 뜻이 아니며, 화면에서 근거와 한계를 구분해 보여줍니다.

A/B 중 한 시점에서 조건을 충족하면 해당 시점의 결과를 표시하고, 서로 반대인 유효 결과나 서로 다른 사건이 확인되면 REVIEW로 남깁니다. 단식·복식 선택에 따른 코트 영역과 원본 영상 근거를 함께 사용합니다.

실제 새 A/B 분석에서 **복식 IN·단식 OUT**을 확인했습니다. 두 경우 모두 A의 원본 확대 19프레임 중 연속 물체 영역 10개를 사용하며 B는 REVIEW입니다. A는 첨부 전송 한도로 저장소의 동명 원본을 사용했고, 첨부와 바이트 동일성은 확인하지 못했습니다. B는 사용자 첨부 원본입니다.

전체 Python 테스트 1,316개 통과·12개 건너뜀, 실제 결과 화면의 모바일·데스크톱·근거 사진·구간 탐색 검증을 통과했습니다. 추가 기존 원본 4개와 부정 대조 5개는 REVIEW를 유지했습니다. 이는 독립 정답 정확도나 일반 영상의 판정 성공률을 측정한 결과가 아닙니다. [검증 기록과 실제 판정 사진](https://github.com/6rmkhj/Badminton_AI_Line_Judge/blob/v0.5.12/docs/VIDEO_ESTIMATES.md)을 확인하세요.

- [0.5.12 전체 Windows 패키지](https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/download/v0.5.12/badminton-windows-0.5.12.zip): 새 폴더에 전부 압축 해제하고 `START.bat` 실행
- [ZIP SHA256](https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/download/v0.5.12/badminton-windows-0.5.12.zip.sha256)
- [Windows 시작 검증 보고서](https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/download/v0.5.12/windows-0.5.12-validation.json)

릴리스 공개 조건은 Windows Server 2022 / Windows PowerShell 5.1에서 실제 전체 패키지의 `START.bat --prepare-only`를 두 번 실행하고, 구성 ZIP 9개의 검증과 압축 해제·재실행, 포함된 Python·OpenCV·NumPy·SciPy·Filelock·FFmpeg·FFprobe를 통한 시작, A/B 화면 `/pair`의 HTTP 200과 경기 방식 선택 표시를 확인하는 것입니다. 상세 결과는 첨부 JSON에 기록됩니다. 이 시작 검증은 GPU 실기나 경기 판정 정확도 시험을 포함하지 않습니다.

추가로 실제 번들 Python·NumPy·OpenCV·SciPy를 사용하여 원본 착지 후보, 축별 해상도 변환, 가능한 위치 영역의 IN/OUT 기하 판정 회귀 테스트 61개가 모두 통과해야 합니다. pytest만 임시 폴더에서 불러오며, 앱과 번들 의존성을 교체하지 않습니다. 실행 결과·모듈 버전·경로는 같은 Windows 보고서의 `native_geometry_tests`에 기록됩니다.

소스 태그 [`v0.5.12`](https://github.com/6rmkhj/Badminton_AI_Line_Judge/tree/v0.5.12)와 패키지의 소스 커밋이 일치하고, 업로드한 세 파일의 크기·SHA256이 검증본과 일치할 때만 공개합니다.
