from copy import deepcopy
import json

import pytest

from badminton import resources


@pytest.fixture
def environment(monkeypatch):
    monkeypatch.setattr(resources.platform, "system", lambda: "Linux")
    monkeypatch.setattr(resources.platform, "processor", lambda: "")
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _: set(range(8)))
    records = {"/proc/cpuinfo": "model name : Example CPU\n",
               "/proc/meminfo": "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n",
               "/sys/fs/cgroup/cpu.max": "250000 100000",
               "/sys/fs/cgroup/memory.max": str(8 * 1024 ** 3),
               "/sys/fs/cgroup/memory.current": str(2 * 1024 ** 3)}
    monkeypatch.setattr(resources, "_text", lambda path: records.get(path, ""))
    monkeypatch.setattr(resources, "_command", lambda args: "")
    return resources.detect_environment()


def test_detects_server_affinity_container_quota_and_memory(environment):
    assert environment["cpu"] == {"name": "Example CPU", "logical_cores": 16, "available_cores": 2}
    assert environment["memory"] == {"total_mb": 8192, "available_mb": 6144}
    assert environment["defaults"] == {"device": "cpu", "cpu_threads": 2,
                                       "parallel_views": 1, "frame_memory_mb": 2457}
    assert environment["accelerators"] == []
    json.dumps(environment, allow_nan=False)


def test_fingerprint_ignores_available_memory(environment, monkeypatch):
    original = resources._memory_bytes
    monkeypatch.setattr(resources, "_memory_bytes", lambda system: (original(system)[0], 1024 ** 3))
    assert resources.detect_environment()["fingerprint"] == environment["fingerprint"]


def test_accelerator_is_detected_without_promising_gpu_execution(monkeypatch):
    monkeypatch.setattr(resources, "_command", lambda args: "Example NVIDIA, 8192\nSecond GPU, 24576")
    found = resources._accelerators("Linux")
    assert [gpu["memory_mb"] for gpu in found] == [8192, 24576]
    assert all(gpu["selectable"] is False and gpu["reason"] for gpu in found)


def test_linux_inventory_recognizes_non_cuda_display_controllers(monkeypatch):
    monkeypatch.setattr(resources, "_command", lambda args:
        '00:02.0 "VGA compatible controller" "Intel Corporation" "Integrated Graphics"\n'
        '00:03.0 "Audio device" "Intel Corporation" "Audio"\n'
        '03:00.0 "Display controller" "Advanced Micro Devices, Inc." "Radeon"'
        if args[0] == "lspci" else "")
    found = resources._accelerators("Linux")
    assert len(found) == 2
    assert found[0]["name"] == "Intel Corporation Integrated Graphics"
    assert all(gpu["selectable"] is False for gpu in found)


def test_missing_hardware_information_has_bounded_defaults(monkeypatch):
    monkeypatch.setattr(resources.platform, "system", lambda: "Unknown")
    monkeypatch.setattr(resources.os, "cpu_count", lambda: None)
    monkeypatch.setattr(resources, "_command", lambda args: "")
    result = resources.detect_environment()
    assert result["cpu"]["available_cores"] == 1
    assert result["memory"] == {"total_mb": None, "available_mb": None}
    assert resources.validate_settings(result["defaults"], result) == result["defaults"]


@pytest.mark.parametrize("field,value", [("device", "cuda:0"), ("cpu_threads", 3),
    ("cpu_threads", True), ("cpu_threads", 1.5), ("parallel_views", 3),
    ("parallel_views", "1"), ("frame_memory_mb", 0), ("frame_memory_mb", 8193),
    ("frame_memory_mb", float("nan"))])
def test_invalid_settings_cannot_bypass_limits(environment, field, value):
    settings = {**environment["defaults"], field: value}
    with pytest.raises(resources.ResourceSettingsError):
        resources.validate_settings(settings, environment)


def test_settings_require_complete_known_fields(environment):
    settings = deepcopy(environment["defaults"])
    assert resources.validate_settings(settings, environment) == settings
    with pytest.raises(resources.ResourceSettingsError):
        resources.validate_settings({**settings, "gpu_percent": 50}, environment)
    settings.pop("cpu_threads")
    with pytest.raises(resources.ResourceSettingsError):
        resources.validate_settings(settings, environment)


def test_frame_budget_is_shared_across_concurrent_views(environment):
    settings = {**environment["defaults"], "parallel_views": 2, "frame_memory_mb": 100}
    assert resources.frame_budget_bytes(settings) == 50 * resources.MIB


def test_exact_full_video_buffer_limit_and_portrait_rounding():
    metadata = {"width": 2160, "height": 3840, "frame_count": 100}
    required = 960 * 1706 * 3 * 100
    assert resources.validate_frame_budget(metadata, 960, required) == required
    with pytest.raises(resources.ResourceSettingsError, match="시점당 예산"):
        resources.validate_frame_budget(metadata, 960, required - 1)
    assert metadata["frame_count"] == 100


@pytest.mark.parametrize("metadata", [
    {"width": 0, "height": 1080, "frame_count": 3},
    {"width": 1920, "height": 1080, "frame_count": None},
    {"width": 1920, "height": 1080, "frame_count": True},
    {"width": 159, "height": 1080, "frame_count": 3},
])
def test_unknown_frame_sizes_fail_before_allocation(metadata):
    with pytest.raises(resources.ResourceSettingsError):
        resources.validate_frame_budget(metadata, 960, 100 * resources.MIB)


def test_runtime_applies_opencv_and_decoder_threads(environment):
    import cv2
    previous_cv, previous_decoder = cv2.getNumThreads(), resources.decoder_threads()
    try:
        result = resources.apply_runtime({**environment["defaults"], "cpu_threads": 1})
        assert result == {"opencv_threads": 1, "decoder_threads": 1}
        assert cv2.getNumThreads() == 1
        assert resources.decoder_threads() == 1
    finally:
        resources.apply_runtime({"cpu_threads": previous_decoder})
        cv2.setNumThreads(previous_cv)
