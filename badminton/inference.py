"""Explicit inference devices. GPU failures never silently select a CPU device.

CPU-only installs do not need PyTorch. Heavy video, geometric and tracking work
still runs on the CPU; GPU stages explicitly transfer their own inputs/outputs.
"""
from __future__ import annotations

import importlib
import os
import re
import sys


class InferenceDeviceError(RuntimeError):
    """The requested inference device cannot execute the requested operation."""


def device_error(device, error):
    detail = str(error).strip()[:600] or type(error).__name__
    memory = any(word in detail.lower() for word in ("out of memory", "memory allocation", "alloc_failed"))
    advice = ("GPU 메모리 예산이나 동시 분석 수를 조정하고 다른 GPU 프로그램을 종료한 뒤 다시 시도하세요."
              if memory else "GPU 드라이버와 실행 패키지를 확인하거나 장치 설정에서 CPU를 선택해 다시 시도하세요.")
    return InferenceDeviceError(f"{device} 추론을 실행하지 못했습니다. {advice} CPU로 자동 전환하지 않았습니다. ({detail})")


def _torch():
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise InferenceDeviceError("GPU 실행용 PyTorch를 불러오지 못했습니다. GPU 환경 설치 후 서버를 다시 시작하세요.") from error


def resolve_device(device="cpu", *, verify=False):
    """Resolve exact runtime IDs, respecting CUDA_VISIBLE_DEVICES.

    Availability checks are cheap enough at model boundaries. ``verify=True``
    also executes a small convolution and reads back a result; use at inventory,
    configuration and job start, rather than for every candidate crop.
    """
    if device == "cpu":
        return "cpu"
    if not isinstance(device, str) or not (re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) or device == "mps"):
        raise InferenceDeviceError("추론 장치는 cpu, cuda:번호 또는 mps여야 합니다.")
    torch = _torch()
    try:
        if device.startswith("cuda:"):
            index = int(device.split(":", 1)[1])
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise InferenceDeviceError(f"{device}를 실행할 수 없습니다. GPU 드라이버와 GPU용 PyTorch 설치를 확인하세요.")
        else:
            backend = getattr(torch.backends, "mps", None)
            if backend is None or not backend.is_available():
                raise InferenceDeviceError("Apple MPS GPU를 실행할 수 없습니다. 지원되는 Mac과 PyTorch 설치를 확인하세요.")
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
                raise InferenceDeviceError("MPS의 자동 CPU 대체가 켜져 있습니다. PYTORCH_ENABLE_MPS_FALLBACK=0으로 서버를 다시 시작하세요.")
        if verify:
            with torch.inference_mode():
                sample = torch.ones((1, 1, 4, 4), dtype=torch.float32, device=device)
                kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
                result = torch.nn.functional.conv2d(sample, kernel)
                if result.device.type != device.split(":")[0] or float(result.sum().cpu()) != 36.0:
                    raise InferenceDeviceError(f"{device} 실행 확인 결과가 올바르지 않습니다.")
        return device
    except InferenceDeviceError:
        raise
    except (RuntimeError, OSError, AssertionError) as error:
        raise device_error(device, error) from error


def available_torch_devices():
    """Inventory runtime-visible GPUs; a card name alone is not GPU support."""
    try:
        torch = _torch()
    except InferenceDeviceError:
        return []
    devices = []
    try:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for index in range(count):
            identifier = f"cuda:{index}"
            try:
                properties = torch.cuda.get_device_properties(index)
                item = {"id": identifier, "name": properties.name,
                        "memory_mb": int(properties.total_memory) // (1024 ** 2),
                        "runtime": "pytorch_rocm" if getattr(torch.version, "hip", None) else "pytorch_cuda"}
                try:
                    resolve_device(identifier, verify=True)
                    item.update(selectable=True, reason="GPU 후보 연산·연결된 모델 추론에 사용할 수 있습니다.")
                except InferenceDeviceError as error:
                    item.update(selectable=False, reason=str(error))
                devices.append(item)
            except (RuntimeError, OSError, AssertionError) as error:
                devices.append({"id": identifier, "name": identifier, "memory_mb": None,
                                "selectable": False, "reason": str(device_error(identifier, error))})
    except (RuntimeError, OSError, AssertionError):
        pass
    backend = getattr(torch.backends, "mps", None)
    if backend is not None and backend.is_available():
        item = {"id": "mps", "name": "Apple GPU · Metal", "memory_mb": None, "runtime": "pytorch_mps"}
        try:
            resolve_device("mps", verify=True)
            item.update(selectable=True, reason="공유 메모리를 사용하는 Apple GPU입니다.")
        except InferenceDeviceError as error:
            item.update(selectable=False, reason=str(error))
        devices.append(item)
    return devices


def configure_device(device="cpu", gpu_memory_percent=50):
    """Apply a PyTorch allocator budget while all analysis queues are idle.

    This is a per-process PyTorch allocation limit, not a total VRAM or GPU
    utilization cap. Other runtimes (e.g. ONNX Runtime) have their own allocator.
    """
    selected = resolve_device(device, verify=True)
    if selected.startswith("cuda:") and (isinstance(gpu_memory_percent, bool)
            or not isinstance(gpu_memory_percent, int) or not 10 <= gpu_memory_percent <= 90):
        raise InferenceDeviceError("GPU 메모리 예산은 10~90 사이의 정수(%)여야 합니다.")
    cached = sys.modules.get("badminton.contact_model")
    if cached is not None and hasattr(cached, "clear_contact_model_cache"):
        cached.clear_contact_model_cache()
    torch = sys.modules.get("torch")
    if torch is None:
        return selected
    if selected == "cpu":
        # A broken or disconnected GPU must not prevent choosing the CPU.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            backend = getattr(torch.backends, "mps", None)
            if backend is not None and backend.is_available() and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except (RuntimeError, OSError, AssertionError):
            pass
        return selected
    try:
        if selected.startswith("cuda:"):
            index = int(selected.split(":")[1])
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(gpu_memory_percent / 100, index)
        elif selected == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except (RuntimeError, OSError, AssertionError) as error:
        raise device_error(selected, error) from error
    return selected
