# Windows 실행 패키지 0.5.13

배포 ZIP 이름은 `badminton-windows-0.5.13.zip`입니다. 배포된 파일의 다운로드 위치와 SHA256은 해당 배포 안내를 확인하세요.

ZIP을 쓰기 가능한 새 폴더에 **전부 압축 해제**한 뒤 루트의 `START.bat`를 실행합니다. 첫 실행에서는 구성 ZIP의 크기·SHA256을 확인하고 포함된 Python·NumPy·OpenCV·SciPy·FFmpeg·FFprobe를 압축 해제합니다. CPU 기본 분석에는 별도 Python 설치나 인터넷 연결이 필요하지 않습니다. 이전 패키지 폴더와 섞지 마세요.

## PyTorch CPU 포함판

`badminton-windows-pytorch-cpu-0.5.13.zip`은 기본판에 **PyTorch 2.10.0+cpu와 필수 의존성**을 추가한 별도 패키지입니다. 사용법은 동일하게 새 폴더에 전체 압축 해제 후 `START.bat` 실행입니다. 첫 실행에서 10개 구성 ZIP을 확인하고 PyTorch CPU 연산도 검사합니다. Python·PyTorch를 따로 설치하거나 실행 시 인터넷으로 내려받을 필요가 없습니다. 기본판은 기존 9개 구성 ZIP을 유지합니다.

포함판에서는 호환 PyTorch 모델의 CPU 추론·학습을 사용할 수 있습니다. 학습된 모델 가중치는 포함하지 않으며, PyTorch 포함 자체로 기본 분석 방식이나 판정 정확도가 바뀌지는 않습니다. NVIDIA GPU를 사용할 때는 아래의 `SETUP_GPU.bat` 절차로 CUDA 환경을 별도로 준비하세요. torchvision·torchaudio는 이 앱에서 요구하지 않아 포함하지 않습니다.

두 패키지는 서로 다른 새 폴더에 풀어 사용하세요. 포함판은 기본판보다 용량이 크며, 정확한 크기·SHA256은 생성된 ZIP과 체크섬에서 확인합니다. Windows 검증 작업이 성공하면 GitHub Actions의 `badminton-windows-pytorch-cpu-0.5.13` 아티팩트에 ZIP·SHA256·`windows-pytorch-cpu-validation.json`을 함께 저장합니다. 아티팩트 생성과 공개 릴리스 등록은 별도입니다.

브라우저가 열리면 서버 컴퓨터의 장치 설정을 확인합니다. 처리 스레드·A/B 순차 또는 동시 처리·프레임 버퍼 예산을 정하고, A/B 영상을 각각 선택하면 FPS·해상도·확장자·코덱 등의 정보가 표시됩니다. 경기 방식에서 **단식 또는 복식**을 고르고 `A/B 연결 분석`으로 분석을 시작합니다. 한 시점만 유효한 IN·OUT이어도 최종 결과에 반영하며 근거 시점을 표시합니다. 원본 프레임의 착지 가능 영역으로 결정한 경우 **영상 추정** 배지와 원본 확대 근거·경계 여유를 함께 표시합니다. 직접 접촉 확인 결과와는 구분합니다. 결과와 설정은 `app/runs/`에 저장됩니다. 실행 콘솔에서 `Ctrl+C`로 종료합니다.

## Windows PowerShell 5.1 시작 오류 수정

0.5.11부터 정상적인 9개 구성 ZIP을 읽어도 `Expected nine payload archives`가 발생하던 배열 처리 오류를 수정합니다. 0.5.10 사용자는 릴리스의 작은 START 수정 패치를 기존 패키지 루트에 덮어쓸 수 있습니다. 자세한 적용 방법과 실제 Windows 검증은 [시작 오류 수정 기록](WINDOWS_START_FIX.md)을 확인하세요.

## 근거 영상 생성 오류 수정

0.5.13은 Windows 한글 경로의 이미지 저장·읽기를 수정하고, 선택적 근거 파일 생성 실패가 계산된 IN·OUT 결과를 없애지 않도록 분리합니다. 파일별 경고를 표시하며 분석 자체의 실패는 단계·오류 상세와 `app/runs/paired/jobs/<작업ID>/views/<A또는B>/analysis-error.log`에 기록합니다. [수정 및 실제 Windows 검증 절차](WINDOWS_EVIDENCE_FIX.md)를 참고하세요.

## 코트 라인 탐지 확인

0.5.13은 앞쪽 두 모서리가 보이는 주 코트 하나를 선택합니다. 초록색은 선택한 코트의 관측 외곽선, 원은 앞쪽 두 모서리, 하늘색은 선택한 코트의 내부 선입니다. 옅은 초록 영역은 IN 판정을 지원하는 확인된 내부 범위입니다. 임의로 닫은 먼 쪽 경계를 OUT 판정에 사용하지 않습니다. 부분 코트도 확인된 접촉점을 단식·복식 랠리의 IN·OUT으로 비교할 수 있습니다. 단식은 안쪽 사이드라인, 복식은 바깥쪽 사이드라인을 사용합니다. [적용 범위와 검증 기록](PRIMARY_COURT.md)을 확인하세요.

**기존 검수 재사용을 끄고 새 분석을 실행해야 개선된 탐지가 적용됩니다.** 이전 결과를 다시 여는 것만으로 분석을 갱신하지 않습니다. 카메라 이동이 확인되면 고정된 라인은 진단 이미지에서만 표시하고 재생 영상에 덧그리지 않습니다. 코트 보정이 되어도 최초 접촉·셔틀 식별·위치 오차 조건이 부족하면 최종 판정은 `REVIEW`일 수 있습니다.

## NVIDIA GPU 준비

서버를 종료한 뒤 루트의 `SETUP_GPU.bat`를 실행합니다. 이 설치에는 인터넷과 지원되는 NVIDIA 드라이버가 필요합니다. 포함된 Python으로 프로젝트 내부의 `app/.venv`를 만들고 CUDA용 PyTorch 2.10.0을 설치한 뒤 실제 GPU 연산을 확인합니다. 외부 Python 설치는 필요하지 않습니다. 완료 후 `START.bat`를 다시 실행하면 준비된 `app/.venv`를 우선 사용합니다.

ONNX CUDA 모델을 쓸 경우 명령 프롬프트에서 `SETUP_GPU.bat --onnx`를 실행합니다. 설치 계획만 보려면 `--dry-run`, 기존 설치의 연산 확인만 하려면 `--check-only`를 붙입니다. 소스 폴더 안의 `app/setup-gpu-windows.bat` 대신 이 패키지 루트의 `SETUP_GPU.bat`를 사용하세요. 루트 실행기가 포함된 Python과 FFmpeg 경로를 설정합니다.

학습된 모델 가중치는 포함하지 않습니다. 기본 GPU 경로는 후보 전처리를 가속하며 연결한 호환 모델도 선택한 장치에서 추론합니다. 영상 읽기·추적·낙하 후보 연결·코트 보정·판정은 CPU에서 계속 수행합니다. AMD/Intel GPU용 설치 경로는 제공하지 않습니다. GPU 실패 시 사유를 표시하므로 장치 설정에서 CPU를 선택하거나 원인을 해결한 뒤 다시 실행하세요.

GPU 설치 후 패키지 폴더를 다른 위치로 옮기면 `.venv`가 이전 Python 경로를 참조할 수 있습니다. 이동한 경우 새 위치의 패키지에서 GPU 환경을 다시 준비하세요.

## 포함 범위와 검증 한계

0.5.13 소스, 착지 사건 통합·원본 픽셀 근거·가능 영역 전체의 IN·OUT 추정, 단일 시점 최종 판정·단식/복식 선택, 주 코트 선택·부분 코트 경계 판정·진단 화면, 장치 설정·영상 정보 화면, GPU 설치기, 첫 낙하 후보 복구·순서 개선과 검증 문서를 포함합니다. 저장소의 원본 경기 영상·과거 대용량 실행 결과·학습된 모델은 포함하지 않습니다. 브라우저에서 사용자의 원본 영상을 업로드하세요.

이번 변경과 검증은 [원본 착지 범위의 IN·OUT 추정](VIDEO_ESTIMATES.md)에 있습니다. [단일 시점 판정·경기 방식 기록](GAME_MODES.md)도 참고하세요. 기존 코트 탐지 변경의 실제 영상 비교와 회귀 테스트는 [코트 캘리브레이션 검증 기록](CALIBRATION_REPORT.md)에 있습니다. 기존 낙하 엔진의 실험은 [낙하 엔진 설계·실험 기록](LANDING_ENGINE.md)에 별도로 남아 있습니다. 후보 위치나 라인 선분을 찾는 것과 코르크의 최초 접촉·IN/OUT 확정은 구분합니다.

빌더는 런타임과 ZIP의 CRC·SHA256·파일 경로·파일별 앱 해시를 검사합니다. 자동 테스트·영상 평가는 패키지 빌더가 별도로 실행하지 않으며 해당 소스의 검증 기록을 참고해야 합니다. Linux 빌더는 Windows 실행 파일을 직접 실행하지 않습니다. 0.5.13 배포는 별도 Windows 실행 환경에서 START 압축 해제·재실행·A/B 화면 응답과 한글 경로의 실제 A/B 영상 분석·근거 파일 생성을 검사한 뒤 공개합니다. 배포에 첨부한 `windows-0.5.13-validation.json`에 실행 결과를 기록합니다. GPU 실기 동작과 사용자 PC의 모든 환경은 검증 범위에 포함하지 않습니다. 문제가 생기면 실행 콘솔과 루트의 `startup-error.log`를 확인하세요.

실행 시 payload를 다시 검증하며, `.package-ready-0.5.13`에 기록한 manifest 해시가 다르면 새로 압축을 풉니다. 이 검사는 손상을 확인하기 위한 것이며 배포자의 전자 서명을 대신하지 않습니다. 앱의 `.venv`·`runs`는 자동 삭제하지 않습니다. 완전히 새 환경으로 시작하려면 ZIP을 다른 새 폴더에 압축 해제하세요.

## 패키지 재생성

이전 `downloads/windows-portable-0.5.4` 배포 브랜치의 커밋 `330a6163eabae1a0fae0f9a93db4c212eb8f0557`을 별도 폴더에 준비합니다. 빌더는 기존 manifest에 기록된 9개 ZIP의 크기·SHA256·CRC를 검사하고, 런타임 `01~07`·`09`는 바이트 그대로 재사용합니다. `08-application.zip`만 현재 소스로 재생성하며 런타임 라이선스도 보존합니다.

```bash
python scripts/package_windows_portable.py \
  --runtime-base /path/to/windows-portable-0.5.4 \
  --output-dir /path/to/new-windows-0.5.13 \
  --zip /path/to/badminton-windows-0.5.13.zip \
  --validation-report docs/WINDOWS_EVIDENCE_FIX.md
```

출력 폴더는 없거나 비어 있어야 하며 ZIP 경로는 출력 폴더 밖의 새 경로여야 합니다. 소스·런타임 입력과 출력은 겹칠 수 없고, symlink·ZIP 경로 탈출·Windows 대소문자 중복은 거부합니다. Git이 추적하는 `badminton/configs/docs/examples/scripts/tests/annotations`와 루트 README·프로젝트 메타데이터·실행기·계획 PDF를 포함합니다. 배치 파일 줄바꿈은 CRLF로 통일합니다. 빌더 자신과 이 문서는 첫 패키징 때 아직 추적되지 않았어도 명시적으로 포함합니다.

### PyTorch 포함판 생성

기존 0.5.13 릴리스 ZIP을 SHA256 확인 후 압축 해제한 폴더도 `--runtime-base`로 사용할 수 있습니다. 빌드 환경에는 Python 3.12와 `packaging`이 필요합니다. 아래 명령으로 빈 wheel 폴더에 Windows x64용 파일을 준비합니다. 다운로드는 빌드할 때만 필요합니다.

```bash
python -m pip install packaging
python -m pip download --dest /path/to/wheels \
  --platform win_amd64 --python-version 3.12 --implementation cp --abi cp312 \
  --only-binary=:all: --no-deps --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.10.0+cpu'
python -m pip download --dest /path/to/wheels \
  --platform win_amd64 --python-version 3.12 --implementation cp --abi cp312 \
  --only-binary=:all: --no-deps --index-url https://pypi.org/simple \
  typing_extensions==4.16.0 sympy==1.14.0 networkx==3.7 Jinja2==3.1.6 \
  fsspec==2026.9.0 mpmath==1.3.0 MarkupSafe==3.0.3 setuptools==78.1.0
python scripts/package_windows_portable.py \
  --runtime-base /path/to/badminton-windows-0.5.13 \
  --pytorch-wheels /path/to/wheels \
  --output-dir /path/to/new-windows-pytorch-cpu-0.5.13 \
  --zip /path/to/badminton-windows-pytorch-cpu-0.5.13.zip
```

다운로드 명령의 `--no-deps`는 고정한 파일만 준비하기 위한 옵션입니다. 빌더가 Windows Python 3.12 대상 메타데이터를 읽어 전이 의존성 전체를 검사하며, 기존 런타임의 filelock은 재사용합니다. 빠진 의존성·잘못된 버전·다른 플랫폼 wheel·파일 충돌은 빌드를 실패시킵니다. 추가 파일은 `10-pytorch_cpu.zip`에 저장하고 원본 wheel 해시·라이선스·재사용 의존성을 manifest에 기록합니다. torchrun 등의 콘솔 명령 래퍼는 생성하지 않으며 앱과 Python API용 런타임입니다.

기본 NumPy payload에 들어 있는 Microsoft 런타임 DLL도 PyTorch가 요구하는 `torch/lib/msvcp140.dll` 이름으로 함께 넣습니다. 원본 파일의 SHA256·저작권·출처를 기록하고 기존 Python 배포본의 Microsoft 재배포 조건을 보존합니다. Windows 검증은 시스템에 설치된 DLL이 대신 쓰이지 않도록 실제 로드 경로까지 확인합니다.

Windows 검증은 최종 ZIP을 한글·공백 경로에 새로 압축 해제하고, 번들 Python으로 PyTorch 버전·NumPy 변환·CPU convolution·역전파·optimizer 갱신을 실행합니다. `START.bat` 준비·재실행과 실제 A/B 화면 HTTP 응답도 확인합니다. 검증 중 패키지 인덱스와 외부 HTTP 프록시는 사용할 수 없도록 설정합니다. 이 확인은 CUDA 실기 시험이나 경기 판정 정확도 시험이 아닙니다.

`--validation-report`는 Git으로 추적되고 패키지에 포함되는 `docs/` 파일이어야 합니다. 지정하지 않으면 이 문서를 참조합니다. 테스트 개수나 이전 버전의 영상 성능 수치를 빌더가 새 결과처럼 복사하지 않습니다.

`PACKAGE_MANIFEST.json`은 소스 커밋, 작업 트리 변경 여부, 정확한 앱 소스 스냅샷 해시, 파일별 해시와 런타임 출처를 기록합니다. 런타임 입력이 Git 체크아웃이면 실제 커밋을 기록하며 압축 해제본이면 커밋은 `null`로 두고 입력 manifest 해시와 런타임 패키지별 해시로 출처를 남깁니다. `APP_FILE_CHECKSUMS.json`은 앱과 루트 메타데이터·라이선스 파일의 해시를, `PAYLOAD_CHECKSUMS.json`은 구성 ZIP 해시를 기록합니다. 최종 ZIP은 이미 압축된 payload를 중복 압축하지 않도록 `ZIP_STORED`로 만들고 SHA256 파일을 함께 생성합니다. 최종 배포는 소스를 커밋한 깨끗한 작업 트리에서 생성하여 manifest의 커밋과 포함 소스를 일치시키세요.
