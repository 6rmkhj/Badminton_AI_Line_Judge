"""Explicit GPU routing contracts; mocked providers are not hardware validation."""
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest

from badminton import detection
from badminton.gpu_candidates import CudaOnnxSession


@pytest.fixture
def fake_ort(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = {}
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index:
                        calls.setdefault("property_device", index) is not None and SimpleNamespace(total_memory=8 * 1024 ** 3))

    class Options:
        def add_session_config_entry(self, key, value):
            calls.setdefault("options", {})[key] = value

    class Session:
        def __init__(self, path, *, sess_options, providers):
            calls.update(path=path, providers=providers)
        def disable_fallback(self):
            calls["fallback_disabled"] = True
        def get_providers(self):
            return calls.get("active_providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        def get_inputs(self):
            return [SimpleNamespace(name="images", type="tensor(float)")]
        def get_outputs(self):
            return [SimpleNamespace(name="detections")]
        def run(self, names, inputs):
            calls.update(output_names=names, inputs=inputs)
            if calls.get("run_error"):
                raise RuntimeError("CUDA out of memory")
            return [np.array([[[32], [32], [16], [16], [.9]]], np.float32)]

    module = SimpleNamespace(SessionOptions=Options, InferenceSession=Session,
                             get_available_providers=lambda: calls.get("available", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
                             preload_dlls=lambda: calls.update(preloaded=True))
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    return calls


def test_onnx_pins_cuda_index_memory_budget_and_disables_cpu_fallback(fake_ort):
    session = CudaOnnxSession("model.onnx", "cuda:1", 25)
    assert fake_ort["property_device"] == 1
    assert fake_ort["providers"] == [("CUDAExecutionProvider", {"device_id": 1, "gpu_mem_limit": 2 * 1024 ** 3})]
    assert fake_ort["options"] == {"session.disable_cpu_ep_fallback": "1"}
    assert fake_ort["fallback_disabled"] and fake_ort["preloaded"]
    blob = np.zeros((1, 3, 64, 64), np.float32)
    assert session.forward(blob).shape == (1, 5, 1)
    assert fake_ort["output_names"] == ["detections"]
    assert fake_ort["inputs"]["images"] is blob


def test_parallel_onnx_sessions_share_the_requested_memory_budget(fake_ort):
    CudaOnnxSession("model.onnx", "cuda:1", 50, parallel_views=2)
    assert fake_ort["providers"][0][1]["gpu_mem_limit"] == 2 * 1024 ** 3


@pytest.mark.parametrize("reported", [[], ["CPUExecutionProvider"]])
def test_missing_cuda_provider_cannot_silently_use_cpu(fake_ort, reported):
    from badminton.inference import InferenceDeviceError
    fake_ort["available"] = reported
    with pytest.raises(InferenceDeviceError, match="CUDA"):
        CudaOnnxSession("model.onnx", "cuda:0")
    assert "providers" not in fake_ort


def test_provider_initialization_fallback_is_rejected(fake_ort):
    from badminton.inference import InferenceDeviceError
    fake_ort["active_providers"] = ["CPUExecutionProvider"]
    with pytest.raises(InferenceDeviceError, match="자동 전환"):
        CudaOnnxSession("model.onnx", "cuda:0")


def test_cuda_runtime_failure_is_actionable_and_does_not_retry_cpu(fake_ort):
    from badminton.inference import InferenceDeviceError
    session = CudaOnnxSession("model.onnx", "cuda:0")
    fake_ort["run_error"] = True
    with pytest.raises(InferenceDeviceError, match="CPU로 자동 전환하지"):
        session.forward(np.zeros((1, 3, 64, 64), np.float32))
    assert fake_ort["providers"][0][0] == "CUDAExecutionProvider"


def test_mps_onnx_is_explicitly_unsupported():
    from badminton.inference import InferenceDeviceError
    with pytest.raises(InferenceDeviceError, match="CUDA 장치"):
        CudaOnnxSession("model.onnx", "mps")


def test_onnx_gpu_detector_uses_session_output_and_records_selected_device(tmp_path, monkeypatch, fake_ort):
    from badminton import inference
    monkeypatch.setattr(inference, "resolve_device", lambda device: device)
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda path: pytest.fail("GPU inference must not initialize the CPU net"))
    path = tmp_path / "test.onnx"
    path.write_bytes(b"provider contract fixture, not a trained model")
    detector = detection._OnnxDetector({"model_path": str(path), "inference_device": "cuda:1",
                                       "model_input_size": 64, "gpu_memory_percent": 25})
    rows = detector._tile(np.zeros((64, 64, 3), np.uint8), (0, 0))
    assert len(rows) == 1 and rows[0]["x"] == 32 and rows[0]["y"] == 32
    assert detector.provenance["inference_device"] == "cuda:1"
    assert detector.provenance["execution_provider"] == "CUDAExecutionProvider"
    assert fake_ort["inputs"]["images"].shape == (1, 3, 64, 64)


def test_default_gpu_route_runs_candidate_masks_without_any_model_weights(monkeypatch):
    """Routing is mocked; actual CUDA/MPS parity lives in hardware-gated tests."""
    pytest.importorskip("torch")
    from badminton import gpu_candidates, inference
    monkeypatch.setattr(inference, "resolve_device", lambda device: device)
    original, devices = gpu_candidates.appearance_masks, []

    def cpu_reference_for_routing(*args, device, **kwargs):
        devices.append(device)
        return original(*args, device="cpu", **kwargs)

    monkeypatch.setattr(gpu_candidates, "appearance_masks", cpu_reference_for_routing)
    frames = [np.zeros((48, 64, 3), np.uint8) for _ in range(3)]
    for index, frame in enumerate(frames):
        cv2.circle(frame, (16 + index * 7, 24), 2, (255, 255, 255), -1)
    cpu = detection.detect_sequence(frames, [0, .04, .08], {"person_backend": "none", "highres_enabled": False})
    gpu = detection.detect_sequence(frames, [0, .04, .08], {"person_backend": "none", "highres_enabled": False,
                                                          "inference_device": "cuda:1"})
    assert devices == ["cuda:1"] * len(frames)
    assert [row["detections"] for row in cpu] == [row["detections"] for row in gpu]
    assert all(row["detector"]["inference_device"] == "cuda:1" for row in gpu)
    assert all(row["detector"]["gpu_stages"] == ["candidate_masks"] for row in gpu)
    assert all(row["detector"]["trained_model"] is False for row in gpu)
