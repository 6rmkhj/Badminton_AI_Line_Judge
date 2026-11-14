A/B 중 한 시점의 유효한 IN·OUT을 최종 결과로 사용하고, 단식·복식에 맞는 주 코트 경계를 선택하는 0.5.10입니다.

- A만 IN이면 최종 IN, B만 OUT이면 최종 OUT으로 표시하고 근거 시점을 보여줍니다. 상대 시점의 REVIEW·처리 실패·사건 연결 미확인이 유효한 단일 시점 판정을 막지 않습니다.
- 단식은 안쪽 사이드라인, 복식은 바깥쪽 사이드라인을 표시와 판정에 함께 적용합니다. 다른 경기 방식의 저장 결과는 재사용하지 않고 원본을 새로 분석합니다.
- 양쪽의 유효한 판정이 충돌하거나 서로 다른 랠리로 확인되면 REVIEW를 유지합니다. 양쪽 모두 접촉·식별 근거가 부족한 경우도 REVIEW입니다.

[Windows 64비트 패키지](https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/download/v0.5.10/badminton-windows-0.5.10.zip)를 **새 폴더에 전부 압축 해제**하고 `START.bat`를 실행하세요. A/B 영상과 경기 방식을 선택한 뒤, 기존 검수 근거 활용을 끄고 새 분석하면 변경된 탐지와 경계가 적용됩니다. Python·기본 분석 의존성·FFmpeg를 포함합니다.

전체 Python 회귀 테스트 1,192개 통과·GPU 관련 5개 건너뜀, 마지막 보완 후 관련 테스트 177개 통과를 확인했습니다. 패키지에서 추출한 소스에서도 177개 테스트가 통과했으며 앱 파일 754개의 원본 일치를 검사했습니다. 실제 B 원본에서 단식 경계 적용을 확인했지만 최초 접촉은 자동 확정되지 않아 결과는 REVIEW입니다. Windows 실행 파일과 GPU 실기 동작은 Linux 빌드 환경에서 직접 실행 검증하지 못했습니다.

소스 커밋: [`ef51298b44ebf0d957093b3fecc7ccf6573d23df`](https://github.com/6rmkhj/Badminton_AI_Line_Judge/commit/ef51298b44ebf0d957093b3fecc7ccf6573d23df). [변경·검증 상세](https://github.com/6rmkhj/Badminton_AI_Line_Judge/blob/v0.5.10/docs/GAME_MODES.md).

ZIP 크기: 285,878,163바이트. [SHA256 파일](https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/download/v0.5.10/badminton-windows-0.5.10.zip.sha256):

```text
9044da2d402ed2e799432df23f76900110a032d0004b469d159e6d6b89053177  badminton-windows-0.5.10.zip
```
