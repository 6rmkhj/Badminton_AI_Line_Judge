"""Open the included PoC review results on the local computer (Python 3.11+)."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import threading
import venv
import webbrowser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--review', action='store_true', help='Open the portable AI visual review report')
    mode.add_argument('--pair', action='store_true', help='Open the A/B input and combined verdict workspace')
    parser.add_argument('--video-dir', default='video')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not sys.flags.utf8_mode:
        # Korean filenames and saved JSON must have the same encoding on Windows.
        os.execv(sys.executable, [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), *sys.argv[1:]])
    project = Path(__file__).resolve().parent
    if sys.version_info < (3, 11):
        parser.error('Python 3.11 이상이 필요합니다.')
    if not args.review and not all(importlib.util.find_spec(name) for name in ('numpy', 'cv2', 'scipy', 'filelock')):
        environment = project / '.venv'
        python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
        if not python.is_file():
            print('로컬 Python 환경을 준비합니다.', flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        print('처음 실행에 필요한 기본 패키지를 설치합니다.', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '-e', str(project)], check=True)
        return subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    default_run = 'runs/ai-reviewed-poc' if args.review else 'runs/pair-workspace' if args.pair else 'runs/integrated-poc'
    run = (project / (args.run or default_run)).resolve()
    required_index = 'index.html' if args.review else 'index.json'
    if not args.pair and not (run / required_index).is_file():
        parser.error(f'분석 결과가 없습니다: {run}. README의 analyze 명령을 먼저 실행하세요.')
    if args.pair:
        run.mkdir(parents=True, exist_ok=True)
    if args.review:
        # Load the bundled stdlib-only handler directly, including under Python
        # -I where this extracted project's directory is absent from sys.path.
        spec = importlib.util.spec_from_file_location('_badminton_static_review', project / 'badminton' / 'static_review.py')
        static_review = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(static_review)
        server = static_review.create_static_review_server(run, port=args.port)
    else:
        from badminton.web import create_server
        server = create_server(run, port=args.port, video_dir=(project / args.video_dir).resolve())
    url = f'http://127.0.0.1:{server.server_address[1]}' + ('/pair' if args.pair else '')
    print(f'검수 화면: {url}\n종료: Ctrl+C', flush=True)
    if not args.no_browser:
        timer = threading.Timer(.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
