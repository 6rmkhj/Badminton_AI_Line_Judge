# Windows 실행 패키지 0.5.7

[Windows 64비트 ZIP 다운로드](https://github.com/6rmkhj/Badminton_AI_Line_Judge/archive/refs/heads/downloads/windows-portable-0.5.7.zip)

ZIP을 쓰기 가능한 새 폴더에 **전부 압축 해제**한 뒤 루트의 `START.bat`를 실행합니다. 첫 실행에서는 구성 ZIP의 크기·SHA256을 확인하고 포함된 Python·NumPy·OpenCV·SciPy·FFmpeg·FFprobe를 압축 해제합니다. CPU 기본 분석에는 별도 Python 설치나 인터넷 연결이 필요하지 않습니다. 이전 패키지 폴더와 섞지 마세요.

브라우저가 열리면 서버 컴퓨터의 장치 설정을 확인합니다. 처리 스레드·A/B 순차 또는 동시 처리·프레임 버퍼 예산을 정하고, A/B 영상을 각각 선택하면 FPS·해상도·확장자·코덱 등의 정보가 표시됩니다. `A/B 연결 분석`으로 분석을 시작합니다. 결과와 설정은 `app/runs/`에 저장됩니다. 실행 콘솔에서 `Ctrl+C`로 종료합니다.

## NVIDIA GPU 준비

서버를 종료한 뒤 루트의 `SETUP_GPU.bat`를 실행합니다. 이 설치에는 인터넷과 지원되는 NVIDIA 드라이버가 필요합니다. 포함된 Python으로 프로젝트 내부의 `app/.venv`를 만들고 CUDA용 PyTorch 2.10.0을 설치한 뒤 실제 GPU 연산을 확인합니다. 외부 Python 설치는 필요하지 않습니다. 완료 후 `START.bat`를 다시 실행하면 준비된 `app/.venv`를 우선 사용합니다.

ONNX CUDA 모델을 쓸 경우 명령 프롬프트에서 `SETUP_GPU.bat --onnx`를 실행합니다. 설치 계획만 보려면 `--dry-run`, 기존 설치의 연산 확인만 하려면 `--check-only`를 붙입니다. 소스 폴더 안의 `app/setup-gpu-windows.bat` 대신 이 패키지 루트의 `SETUP_GPU.bat`를 사용하세요. 루트 실행기가 포함된 Python과 FFmpeg 경로를 설정합니다.

학습된 모델 가중치는 포함하지 않습니다. 기본 GPU 경로는 후보 전처리를 가속하며 연결한 호환 모델도 선택한 장치에서 추론합니다. 영상 읽기·추적·낙하 후보 연결·코트 보정·판정은 CPU에서 계속 수행합니다. AMD/Intel GPU용 설치 경로는 제공하지 않습니다. GPU 실패 시 사유를 표시하므로 장치 설정에서 CPU를 선택하거나 원인을 해결한 뒤 다시 실행하세요.

GPU 설치 후 패키지 폴더를 다른 위치로 옮기면 `.venv`가 이전 Python 경로를 참조할 수 있습니다. 이동한 경우 새 위치의 패키지에서 GPU 환경을 다시 준비하세요.

## 포함 범위와 검증 한계

최신 0.5.7 소스, 장치 설정·영상 정보 화면, GPU 설치기, 첫 낙하 후보 복구·순서 개선과 검증 문서를 포함합니다. 저장소의 원본 경기 영상·과거 대용량 실행 결과·학습된 모델은 포함하지 않습니다. 브라우저에서 사용자의 원본 영상을 업로드하세요.

개발 AI 근사 참조 일치는 7/17 → 9/17, 대조군은 2/2 유지, 설정 동결 후 별도 검증은 0/1 → 0/1입니다. 검증한 38개 영상의 공식 판정은 모두 `REVIEW`입니다. 후보 위치를 찾는 것과 코르크의 최초 접촉·IN/OUT 확정은 구분합니다. [낙하 엔진 설계·실험 기록](LANDING_ENGINE.md)을 확인하세요.

Linux에서 전체 테스트 986개가 통과했고 실제 GPU 하드웨어가 필요한 5개는 실행하지 않았습니다. 빌더는 런타임과 ZIP의 CRC·SHA256·파일 경로·파일별 앱 해시를 검사합니다. Windows 실행 파일과 GPU 실기 동작은 이 Linux 빌드 환경에서 직접 실행 검증하지 못했습니다. 문제가 생기면 실행 콘솔과 루트의 `startup-error.log`를 확인하세요.

실행 시 payload를 다시 검증하며, `.package-ready-0.5.7`에 기록한 manifest 해시가 다르면 새로 압축을 풉니다. 이 검사는 손상을 확인하기 위한 것이며 배포자의 전자 서명을 대신하지 않습니다. 앱의 `.venv`·`runs`는 자동 삭제하지 않습니다. 완전히 새 환경으로 시작하려면 ZIP을 다른 새 폴더에 압축 해제하세요.

## 패키지 재생성

이전 `downloads/windows-portable-0.5.4` 배포 브랜치의 커밋 `330a6163eabae1a0fae0f9a93db4c212eb8f0557`을 별도 폴더에 준비합니다. 빌더는 기존 manifest에 기록된 9개 ZIP의 크기·SHA256·CRC를 검사하고, 런타임 `01~07`·`09`는 바이트 그대로 재사용합니다. `08-application.zip`만 현재 소스로 재생성하며 런타임 라이선스도 보존합니다.

```bash
python scripts/package_windows_portable.py \
  --runtime-base /path/to/windows-portable-0.5.4 \
  --output-dir /path/to/new-windows-0.5.7 \
  --zip /path/to/badminton-windows-0.5.7.zip
```

출력 폴더는 없거나 비어 있어야 하며 ZIP 경로는 출력 폴더 밖의 새 경로여야 합니다. 소스·런타임 입력과 출력은 겹칠 수 없고, symlink·ZIP 경로 탈출·Windows 대소문자 중복은 거부합니다. Git이 추적하는 `badminton/configs/docs/examples/scripts/tests/annotations`와 루트 README·프로젝트 메타데이터·실행기·계획 PDF를 포함합니다. 배치 파일 줄바꿈은 CRLF로 통일합니다. 빌더 자신과 이 문서는 첫 패키징 때 아직 추적되지 않았어도 명시적으로 포함합니다.

`PACKAGE_MANIFEST.json`은 소스 커밋, 작업 트리 변경 여부, 정확한 앱 소스 스냅샷 해시, 파일별 해시와 런타임 출처를 기록합니다. `APP_FILE_CHECKSUMS.json`은 앱과 루트 메타데이터·라이선스 파일의 해시를, `PAYLOAD_CHECKSUMS.json`은 구성 ZIP 해시를 기록합니다. 최종 ZIP은 이미 압축된 payload를 중복 압축하지 않도록 `ZIP_STORED`로 만들고 SHA256 파일을 함께 생성합니다. 최종 배포는 소스를 커밋한 깨끗한 작업 트리에서 생성하여 manifest의 커밋과 포함 소스를 일치시키세요.
