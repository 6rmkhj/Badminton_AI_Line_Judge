"""Server hardware discovery and explicit, bounded analysis resource settings.

Thread counts are library limits, not an operating-system CPU percentage cap.
The memory setting bounds retained decoded BGR frames, not total process RAM.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import threading

MIB = 1024 ** 2
_runtime_lock = threading.RLock()
_decoder_threads = 2
_SETTING_FIELDS = {"device", "cpu_threads", "parallel_views", "frame_memory_mb"}


class ResourceSettingsError(ValueError):
    """A requested resource policy cannot be applied safely."""


def _text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def _command(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True,
                                timeout=3, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return ""


def _positive_integer(value):
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _cgroup_limits():
    """Honor the effective container limits exposed at the cgroup mount."""
    quota = None
    cpu = _text("/sys/fs/cgroup/cpu.max").split()
    if len(cpu) == 2 and cpu[0] != "max":
        numerator, denominator = map(_positive_integer, cpu)
        if numerator and denominator:
            quota = max(1, math.floor(numerator / denominator))
    if quota is None:
        numerator = _positive_integer(_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
        denominator = _positive_integer(_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
        if numerator and denominator:
            quota = max(1, math.floor(numerator / denominator))
    limit = _positive_integer(_text("/sys/fs/cgroup/memory.max"))
    usage_text = _text("/sys/fs/cgroup/memory.current")
    if limit is None:
        limit = _positive_integer(_text("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        usage_text = _text("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    usage = int(usage_text) if usage_text.isdigit() else None
    return quota, limit, usage


def _memory_bytes(system):
    total = available = None
    if system == "Linux":
        values = {}
        for line in _text("/proc/meminfo").splitlines():
            match = re.match(r"(\w+):\s+(\d+)\s+kB", line)
            if match:
                values[match[1]] = int(match[2]) * 1024
        total, available = values.get("MemTotal"), values.get("MemAvailable")
    elif system == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                        ("total_physical", ctypes.c_uint64), ("available_physical", ctypes.c_uint64),
                        ("total_page", ctypes.c_uint64), ("available_page", ctypes.c_uint64),
                        ("total_virtual", ctypes.c_uint64), ("available_virtual", ctypes.c_uint64),
                        ("available_extended", ctypes.c_uint64)]
        try:
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total, available = status.total_physical, status.available_physical
        except (AttributeError, OSError):
            pass
    elif system == "Darwin":
        total = _positive_integer(_command(["sysctl", "-n", "hw.memsize"]))
        output = _command(["vm_stat"])
        page = re.search(r"page size of (\d+) bytes", output)
        values = re.findall(r"Pages (?:free|inactive|speculative):\s+(\d+)\.", output)
        if page and values:
            available = int(page[1]) * sum(map(int, values))
    if system == "Linux":
        _, limit, usage = _cgroup_limits()
        if limit is not None:
            total = min(total, limit) if total is not None else limit
            if usage is not None:
                remaining = max(0, limit - usage)
                available = min(available, remaining) if available is not None else remaining
    if total is not None and available is not None:
        available = min(total, available)
    return total, available


def _hardware_accelerators(system):
    """Inventory names are not CUDA runtime ordinals or proof of execution."""
    devices = []
    output = _command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    for line in output.splitlines():
        name, separator, memory = line.rpartition(",")
        if separator and name.strip():
            devices.append({"id": f"display:nvidia:{len(devices)}", "name": name.strip(),
                            "memory_mb": _positive_integer(memory.strip())})
    if not devices and system == "Linux":
        # Display controllers are inventory only; their presence does not imply
        # that a compatible compute runtime or driver is installed.
        for line in _command(["lspci", "-mm"]).splitlines():
            try:
                fields = shlex.split(line)
            except ValueError:
                continue
            if len(fields) >= 4 and any(kind in fields[1].lower() for kind in ("vga", "3d controller", "display")):
                devices.append({"id": f"display:{fields[0]}", "name": " ".join(fields[2:4]), "memory_mb": None})
    if not devices and system == "Windows":
        output = _command(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                           "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"])
        devices = [{"id": f"display:{index}", "name": name.strip(), "memory_mb": None}
                   for index, name in enumerate(output.splitlines()) if name.strip()]
    elif not devices and system == "Darwin":
        output = _command(["system_profiler", "SPDisplaysDataType", "-json"])
        try:
            rows = json.loads(output).get("SPDisplaysDataType", [])
            devices = [{"id": f"display:{index}", "name": row.get("sppci_model") or row.get("_name"),
                        "memory_mb": None} for index, row in enumerate(rows)
                       if row.get("sppci_model") or row.get("_name")]
        except (ValueError, TypeError, AttributeError):
            pass
    for device in devices:
        device.update(selectable=False, reason="GPU는 감지했지만 실행 가능한 PyTorch 장치를 확인하지 못했습니다. GPU 런타임·드라이버 설치 후 서버를 다시 시작하세요.")
    return devices


def _torch_accelerators():
    from .inference import available_torch_devices
    return available_torch_devices()


def _accelerators(system):
    # Runtime ordinals honor CUDA_VISIBLE_DEVICES; nvidia-smi's physical order
    # must never be used to route inference to a potentially different GPU.
    devices = [dict(device) for device in _torch_accelerators()]
    remaining_names = [str(device.get("name", "")).strip().lower() for device in devices]
    for device in _hardware_accelerators(system):
        name = device["name"].strip().lower()
        if system == "Darwin" and name.startswith("apple ") and any(item.get("id") == "mps" for item in devices):
            continue
        if name in remaining_names:
            remaining_names.remove(name)
        else:
            devices.append(device)
    return devices


def detect_environment():
    """Inspect the machine running Python, never the browser's client device."""
    system = platform.system()
    logical = max(1, os.cpu_count() or 1)
    available = logical
    try:
        available = min(available, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    if system == "Linux":
        quota, _, _ = _cgroup_limits()
        if quota:
            available = min(available, quota)
    name = platform.processor().strip()
    if system == "Linux":
        match = re.search(r"^model name\s*:\s*(.+)$", _text("/proc/cpuinfo"), re.MULTILINE)
        name = match[1].strip() if match else name
    elif system == "Darwin":
        name = _command(["sysctl", "-n", "machdep.cpu.brand_string"]) or name
    total, free = _memory_bytes(system)
    environment = {"system": {"os": system, "release": platform.release(), "machine": platform.machine()},
                   "cpu": {"name": name or platform.machine() or "CPU", "logical_cores": logical,
                           "available_cores": max(1, available)},
                   "memory": {"total_mb": total // MIB if total is not None else None,
                              "available_mb": free // MIB if free is not None else None},
                   "accelerators": _accelerators(system)}
    environment["limits"] = {"cpu_threads": {"min": 1, "max": max(1, available)},
                             "parallel_views": {"min": 1, "max": 2},
                             "frame_memory_mb": {"min": 64, "max": max(64, min(total // MIB if total else 16384, 65536))}}
    # Exclude transient available memory from the stable environment identity.
    identity = {key: environment[key] for key in ("system", "cpu", "accelerators", "limits")}
    environment["fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    environment["defaults"] = recommend_settings(environment)
    return environment


def recommend_settings(environment):
    memory = environment["memory"]
    available = memory.get("available_mb")
    total = memory.get("total_mb")
    baseline = available if available is not None else total if total is not None else 2048
    maximum = environment["limits"]["frame_memory_mb"]["max"]
    return {"device": "cpu", "cpu_threads": min(2, environment["cpu"]["available_cores"]),
            "parallel_views": 1, "frame_memory_mb": max(64, min(maximum, 4096, int(baseline * .4)))}


def validate_settings(settings, environment):
    if (not isinstance(settings, dict) or not _SETTING_FIELDS.issubset(settings)
            or set(settings) - _SETTING_FIELDS - {"gpu_memory_percent"}):
        raise ResourceSettingsError("장치, CPU 스레드, 동시 분석 수, 프레임 버퍼 예산을 모두 지정하세요.")
    device = settings["device"]
    if not isinstance(device, str):
        raise ResourceSettingsError("사용할 추론 장치를 선택하세요.")
    if device != "cpu" and not any(item.get("id") == device and item.get("selectable") is True
                                   for item in environment.get("accelerators", [])):
        raise ResourceSettingsError("선택한 GPU에서 추론을 실행할 수 없습니다. GPU 런타임·드라이버를 확인하거나 CPU를 선택하세요.")
    normalized = {"device": device}
    for field, label in (("cpu_threads", "CPU 스레드"), ("parallel_views", "동시 분석 수"),
                         ("frame_memory_mb", "프레임 버퍼 예산")):
        value, bounds = settings[field], environment["limits"][field]
        if isinstance(value, bool) or not isinstance(value, int) or not bounds["min"] <= value <= bounds["max"]:
            raise ResourceSettingsError(f"{label}: {bounds['min']}~{bounds['max']} 사이의 정수를 지정하세요.")
        normalized[field] = value
    if device.startswith("cuda:"):
        percent = settings.get("gpu_memory_percent", 50)
        if isinstance(percent, bool) or not isinstance(percent, int) or not 10 <= percent <= 90:
            raise ResourceSettingsError("GPU 메모리 예산은 10~90 사이의 정수(%)여야 합니다.")
        normalized["gpu_memory_percent"] = percent
    elif "gpu_memory_percent" in settings:
        raise ResourceSettingsError("GPU 메모리 비율 설정은 CUDA 장치에서만 지원합니다.")
    return normalized


def apply_runtime(settings):
    """Call only when all server analysis queues are idle; OpenCV is global."""
    import cv2
    global _decoder_threads
    threads = settings["cpu_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ResourceSettingsError("CPU 스레드는 1 이상의 정수여야 합니다.")
    with _runtime_lock:
        from .inference import configure_device, InferenceDeviceError
        try:
            # This repeats the runtime check when saving; stale hardware
            # discovery must not turn a missing GPU into silent CPU execution.
            configure_device(settings.get("device", "cpu"),
                             gpu_memory_percent=settings.get("gpu_memory_percent", 50))
        except InferenceDeviceError as error:
            raise ResourceSettingsError(str(error)) from error
        cv2.setNumThreads(threads)
        _decoder_threads = threads
        return {"opencv_threads": cv2.getNumThreads(), "decoder_threads": threads}


def decoder_threads():
    """Thread count used by each video decoder/encoder, not a process-wide cap."""
    with _runtime_lock:
        return _decoder_threads


def frame_budget_bytes(settings):
    return settings["frame_memory_mb"] * MIB // settings["parallel_views"]


def validate_frame_budget(metadata, analysis_width, budget_bytes):
    """Reject an oversized full video; never silently drop analysis frames."""
    values = [metadata.get(key) for key in ("width", "height", "frame_count")]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ResourceSettingsError("프레임 버퍼 계산에 필요한 영상 크기·프레임 수를 확인할 수 없습니다.")
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes <= 0:
        raise ResourceSettingsError("프레임 버퍼 예산은 0보다 커야 합니다.")
    if isinstance(analysis_width, bool) or not isinstance(analysis_width, int) or analysis_width < 160:
        raise ResourceSettingsError("분석 영상 너비는 160픽셀 이상의 정수여야 합니다.")
    source_width, source_height, count = values
    width = min(analysis_width, source_width)
    if width < 160:
        raise ResourceSettingsError("분석 영상 너비는 160픽셀 이상이어야 합니다.")
    width -= width % 2
    height = max(2, round(source_height * width / source_width / 2) * 2)
    required = width * height * 3 * count
    if required > budget_bytes:
        raise ResourceSettingsError(f"영상 프레임 버퍼에 {math.ceil(required / MIB):,} MiB가 필요하지만 "
                                    f"시점당 예산은 {budget_bytes // MIB:,} MiB입니다. "
                                    "동시 분석 수를 줄이거나 프레임 버퍼 예산을 늘리세요.")
    return required
