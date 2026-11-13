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
    monkeypatch.setattr(resources, "_torch_accelerators", lambda: [])
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
    monkeypatch.setattr(resources, "_torch_accelerators", lambda: [])
    monkeypatch.setattr(resources, "_command", lambda args: "Example NVIDIA, 8192\nSecond GPU, 24576")
    found = resources._accelerators("Linux")
    assert [gpu["memory_mb"] for gpu in found] == [8192, 24576]
    assert all(gpu["selectable"] is False and gpu["reason"] for gpu in found)
    assert all(gpu["id"].startswith("display:") for gpu in found)


def test_linux_inventory_recognizes_non_cuda_display_controllers(monkeypatch):
    monkeypatch.setattr(resources, "_torch_accelerators", lambda: [])
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
    monkeypatch.setattr(resources, "_torch_accelerators", lambda: [])
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


def test_runtime_gpu_id_is_authoritative_not_inventory_order(monkeypatch):
    # CUDA_VISIBLE_DEVICES can expose physical GPU 1 as logical cuda:0.
    monkeypatch.setattr(resources, "_torch_accelerators", lambda: [
        {"id": "cuda:0", "name": "Second GPU", "memory_mb": 24576,
         "selectable": True, "reason": "실행 확인 완료"}])
    monkeypatch.setattr(resources, "_command", lambda args: "First GPU, 8192\nSecond GPU, 24576")
    devices = resources._accelerators("Linux")
    assert devices[0]["id"] == "cuda:0" and devices[0]["name"] == "Second GPU"
    assert len(devices) == 2
    assert devices[1]["id"] == "display:nvidia:0" and not devices[1]["selectable"]


@pytest.fixture
def gpu_environment(environment):
    environment["accelerators"] = [
        {"id": "cuda:1", "name": "Working GPU", "selectable": True},
        {"id": "display:0", "name": "Display only", "selectable": False},
        {"id": "mps", "name": "Apple GPU", "selectable": True}]
    return environment


def test_selectable_gpu_keeps_exact_device_and_default_budget(gpu_environment):
    policy = {**gpu_environment["defaults"], "device": "cuda:1"}
    selected = resources.validate_settings(policy, gpu_environment)
    assert selected == {**policy, "gpu_memory_percent": 50}
    assert resources.validate_settings({**policy, "gpu_memory_percent": 35}, gpu_environment)["gpu_memory_percent"] == 35


@pytest.mark.parametrize("device", ["cuda:0", "cuda:2", "display:0", "cuda", None, [], {}])
def test_unavailable_or_malformed_gpu_cannot_be_selected(gpu_environment, device):
    with pytest.raises(resources.ResourceSettingsError):
        resources.validate_settings({**gpu_environment["defaults"], "device": device}, gpu_environment)


@pytest.mark.parametrize("percent", [0, 9, 91, 100, True, 20.5, "50", None, float("nan")])
def test_gpu_memory_percent_is_bounded_and_typed(gpu_environment, percent):
    with pytest.raises(resources.ResourceSettingsError, match="GPU 메모리 예산"):
        resources.validate_settings({**gpu_environment["defaults"], "device": "cuda:1",
                                     "gpu_memory_percent": percent}, gpu_environment)


def test_cpu_and_mps_do_not_accept_fake_cuda_memory_cap(gpu_environment):
    for device in ("cpu", "mps"):
        selected = resources.validate_settings({**gpu_environment["defaults"], "device": device}, gpu_environment)
        assert "gpu_memory_percent" not in selected
        with pytest.raises(resources.ResourceSettingsError, match="CUDA"):
            resources.validate_settings({**selected, "gpu_memory_percent": 50}, gpu_environment)


def test_runtime_rechecks_gpu_before_mutating_cpu_settings(gpu_environment, monkeypatch):
    import cv2
    from badminton import inference
    before_cv, before_decoder = cv2.getNumThreads(), resources.decoder_threads()
    calls = []
    def unavailable(device, *, gpu_memory_percent):
        calls.append((device, gpu_memory_percent))
        raise inference.InferenceDeviceError("선택한 GPU 연결이 해제되었습니다.")
    monkeypatch.setattr(inference, "configure_device", unavailable)
    with pytest.raises(resources.ResourceSettingsError, match="연결이 해제"):
        resources.apply_runtime({**gpu_environment["defaults"], "device": "cuda:1",
                                 "cpu_threads": 1, "gpu_memory_percent": 40})
    assert calls == [("cuda:1", 40)]
    assert cv2.getNumThreads() == before_cv
    assert resources.decoder_threads() == before_decoder
