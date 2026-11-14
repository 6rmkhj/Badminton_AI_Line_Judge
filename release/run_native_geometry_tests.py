"""Run native geometry regressions using only the package's native runtime."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

TEST_FILES = ['tests/test_native_landing.py', 'tests/test_native_scale.py',
              'tests/test_video_estimate.py']
EXPECTED_TESTS = 63


class TestCounts:
    def __init__(self):
        self.collected = self.passed = self.failed = self.skipped = self.errors = 0

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_collectreport(self, report):
        if report.failed:
            self.errors += 1
        elif report.skipped:
            self.skipped += 1

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.skipped += 1
        elif report.failed:
            if report.when == 'call':
                self.failed += 1
            else:
                self.errors += 1
        elif report.when == 'call' and report.passed:
            self.passed += 1


def run(package, dependencies, report):
    runtime = package / 'runtime'
    app = package / 'app'
    if Path(sys.executable).resolve() != (runtime / 'python.exe').resolve():
        raise RuntimeError('Native tests must execute with the bundled runtime/python.exe')
    if dependencies.is_relative_to(package):
        raise RuntimeError('pytest dependencies must be isolated outside the package')
    if not (dependencies / 'pytest/__init__.py').is_file():
        raise RuntimeError('Isolated pytest dependencies are missing')
    for name in TEST_FILES:
        if not (app / name).is_file():
            raise RuntimeError('Packaged native regression is missing: ' + name)

    sys.dont_write_bytecode = True
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    os.environ.pop('PYTEST_ADDOPTS', None)
    os.environ.pop('PYTEST_PLUGINS', None)
    # Keep the embedded interpreter's own site-packages. Only the pure-Python
    # test harness and exact packaged application are added to the import path.
    sys.path[:0] = [str(app), str(dependencies), str(app / 'tests')]
    os.chdir(app)
    native = {}
    for name in ('numpy', 'cv2', 'scipy', 'filelock'):
        module = importlib.import_module(name)
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(runtime):
            raise RuntimeError('Native/application dependency was not bundled: ' + name)
        native[name] = {'version': module.__version__, 'file': str(module_path)}
    pytest = importlib.import_module('pytest')
    if not Path(pytest.__file__).resolve().is_relative_to(dependencies):
        raise RuntimeError('pytest did not load from isolated testing dependencies')
    if pytest.__version__ != '8.4.2':
        raise RuntimeError('Unexpected pytest version')
    report.update(runtime_executable=str(Path(sys.executable).resolve()),
                  runtime_dependencies=native, bundled_runtime_verified=True,
                  pytest_version=pytest.__version__, pytest_dependencies_isolated=True)
    counts = TestCounts()
    code = int(pytest.main(['-q', '-p', 'no:cacheprovider', '--color=no', *TEST_FILES],
                           plugins=[counts]))
    report.update(returncode=code, **vars(counts))
    if (code != 0 or counts.collected != EXPECTED_TESTS or counts.passed != EXPECTED_TESTS
            or counts.failed or counts.skipped or counts.errors):
        raise RuntimeError('Native geometry tests did not all pass (expected 63, no skips)')
    report['status'] = 'passed'


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--pytest-deps', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    output = args.report.resolve()
    report = {'status': 'failed', 'test_files': TEST_FILES, 'expected_tests': EXPECTED_TESTS}
    started = time.monotonic()
    try:
        run(args.package.resolve(), args.pytest_deps.resolve(), report)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        print(report['error'], file=sys.stderr)
    report['elapsed_seconds'] = round(time.monotonic() - started, 3)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
