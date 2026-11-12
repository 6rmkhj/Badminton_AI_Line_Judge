# 배드민턴 AI — Windows 실행 패키지 0.5.4

1. 이 브랜치의 **Code → Download ZIP**으로 다운로드합니다.
2. **새 폴더에 전부 압축 해제**합니다.
3. 맨 위 **START.bat**을 더블클릭합니다.
4. 브라우저에서 A/B 영상을 올리고 새 분석을 실행합니다.

Python, NumPy, SciPy, OpenCV, filelock, FFmpeg/ffprobe가 포함됩니다. 별도 설치나 실행 중 다운로드는 필요하지 않습니다. 다운로드 약 263MiB이며 첫 실행 후 약 900MiB 공간이 필요합니다.

변경 사항·실제 검증 범위·남은 실패 사례는 `app/docs/FALSE_POSITIVE_REMEDIATION.md`를 확인하세요. 근거가 부족하면 REVIEW를 유지하며 추정 위치를 확정 접촉이나 IN/OUT으로 취급하지 않습니다.

소스 `3f68c0b9b5d336921abda9bd76751e49f1d8dd12` (0.5.4). 전체 Python 테스트 684개 통과 (Linux). AI 시각 참조는 독립 사람이 확인한 정확도 평가가 아닙니다. Windows 실행 파일 자체는 Linux에서 직접 실행 검증하지 못했습니다.

사용법·라이선스·검증 범위는 `사용방법.txt`, `licenses/`, `PACKAGE_VALIDATION.txt`에 있습니다. 원본 영상·대용량 실행 데이터·학습된 모델은 미포함이며 검증 문서와 일부 근거 이미지는 포함합니다.

`payload/`의 ZIP들은 구성 요소입니다. START.bat이 자동으로 압축 해제합니다.
