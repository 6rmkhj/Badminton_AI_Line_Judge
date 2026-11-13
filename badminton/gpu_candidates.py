"""Device-local candidate masks; component geometry remains on the CPU.

This preserves the heuristic proposal algorithm. It is not a trained detector.
Only the current frame/ROI is transferred, never the retained video buffer.
"""
from __future__ import annotations

import math
import numpy as np


def appearance_masks(gray, hsv, bright, motion, foreground, local_base, config,
                     exclusion, roi=None, *, native=False, device="cpu"):
    """Return (polarity, selected, contrast, cue) arrays from tensor operations.

    ``device='cpu'`` is useful for numerical contract tests. Production calls
    use the validated, explicit CUDA/MPS device selected for this analysis.
    Any device failure aborts the request rather than changing the device.
    """
    from .inference import InferenceDeviceError
    try:
        import torch
    except (ImportError, OSError) as error:
        raise InferenceDeviceError("GPU 후보 연산에 PyTorch가 필요합니다. GPU 실행 환경을 설치한 뒤 다시 시작하세요.") from error
    try:
        def tensor(value, dtype=None):
            return torch.as_tensor(np.ascontiguousarray(value), device=device, dtype=dtype)

        def integer_threshold(value):
            # NumPy compares uint8 values to fractional Python thresholds in
            # float64. Round upward before float32 GPU comparisons so 18 is
            # still rejected by 18.0000001 (MPS does not support float64).
            return math.ceil(value) if math.isfinite(value) else value

        with torch.inference_mode():
            current = tensor(gray, torch.float32)
            baseline = tensor(local_base, torch.float32)
            positive = (current - baseline).clamp(min=0, max=255)
            temporal = tensor(motion, torch.float32)
            distance = tensor(foreground, torch.float32)
            allowed = ~tensor(exclusion, torch.bool)
            if roi is not None:
                allowed &= tensor(roi) > 0
            channels = [("bright", tensor(bright) > 0, positive,
                         float(config.get("motion_threshold", 16)),
                         float(config.get("contrast_threshold", 8)))]
            if config.get("include_dark_components"):
                negative = (baseline - current).clamp(min=0, max=255)
                threshold = float(config.get("dark_contrast_threshold", 18))
                dark = ((negative >= integer_threshold(threshold))
                        & (tensor(hsv[:, :, 1]) <= int(config.get("dark_max_saturation", 115))))
                channels.append(("dark", dark, negative,
                                 float(config.get("dark_motion_threshold", config.get("motion_threshold", 16))),
                                 threshold))
            outputs = []
            for polarity, appearance, contrast, motion_threshold, contrast_threshold in channels:
                temporal_threshold = (integer_threshold(motion_threshold)
                                      if np.issubdtype(np.asarray(motion).dtype, np.integer) else motion_threshold)
                cue = ((temporal >= temporal_threshold)
                       & (distance >= integer_threshold(float(config.get("background_threshold", 8))))
                       & (contrast >= integer_threshold(contrast_threshold)))
                selected = appearance & allowed
                if not native or config.get("native_motion_components"):
                    selected &= cue
                elif config.get("native_local_components", bool(config.get("refinement_regions"))):
                    selected &= contrast >= integer_threshold(contrast_threshold)
                # Synchronizing the copies also surfaces asynchronous device
                # errors here, before the algorithm can emit CPU-only results.
                outputs.append((polarity, selected.cpu().numpy(),
                                contrast.to(torch.uint8).cpu().numpy(), cue.cpu().numpy()))
            return outputs
    except (RuntimeError, OSError) as error:
        raise InferenceDeviceError(
            f"{device} 후보 연산에 실패했습니다. GPU 메모리와 실행 환경을 확인하거나 CPU를 선택하세요. "
            "CPU로 자동 전환하지 않았습니다.") from error


class CudaOnnxSession:
    """YOLO ONNX execution pinned to one CUDA device without CPU fallback."""

    def __init__(self, path, device, gpu_memory_percent=50, parallel_views=1):
        from .inference import InferenceDeviceError
        if not device.startswith("cuda:"):
            raise InferenceDeviceError("ONNX GPU 추론은 CUDA 장치가 필요합니다. CPU를 선택하거나 PyTorch 모델을 사용하세요.")
        if (isinstance(gpu_memory_percent, bool) or not isinstance(gpu_memory_percent, int)
                or not 10 <= gpu_memory_percent <= 90):
            raise InferenceDeviceError("GPU 메모리 비율은 10~90 사이의 정수여야 합니다.")
        if isinstance(parallel_views, bool) or not isinstance(parallel_views, int) or parallel_views not in (1, 2):
            raise InferenceDeviceError("ONNX 동시 분석 수는 1 또는 2여야 합니다.")
        self.device = device
        try:
            import onnxruntime as ort
        except (ImportError, OSError) as error:
            raise InferenceDeviceError("ONNX GPU 추론에 onnxruntime-gpu가 필요합니다. GPU 실행 환경을 설치하세요.") from error
        try:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise InferenceDeviceError("ONNX Runtime에서 CUDA 실행기를 사용할 수 없습니다. onnxruntime-gpu와 CUDA 환경을 확인하세요.")
            # ORT 1.21+ can load the CUDA/cuDNN libraries shipped by PyTorch.
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            import torch
            device_index = int(device.split(":")[1])
            memory_limit = int(torch.cuda.get_device_properties(device_index).total_memory * gpu_memory_percent / 100) // parallel_views
            options = ort.SessionOptions()
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            self.session = ort.InferenceSession(str(path), sess_options=options,
                providers=[("CUDAExecutionProvider", {"device_id": device_index, "gpu_mem_limit": memory_limit})])
            self.session.disable_fallback()
            if not self.session.get_providers() or self.session.get_providers()[0] != "CUDAExecutionProvider":
                raise InferenceDeviceError("선택한 CUDA 실행기를 초기화하지 못했습니다. CPU로 자동 전환하지 않았습니다.")
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != "tensor(float)":
                raise InferenceDeviceError("ONNX 모델은 float32 영상 입력 1개와 YOLO 출력 1개여야 합니다.")
            self.input_name, self.output_name = inputs[0].name, outputs[0].name
        except InferenceDeviceError:
            raise
        except Exception as error:
            raise InferenceDeviceError(
                f"{device} ONNX 실행기를 초기화하지 못했습니다. 모델의 CUDA 지원과 GPU 메모리를 확인하세요. "
                "CPU로 자동 전환하지 않았습니다.") from error

    def forward(self, blob):
        from .inference import InferenceDeviceError
        try:
            return self.session.run([self.output_name], {self.input_name: blob})[0]
        except Exception as error:
            raise InferenceDeviceError(
                f"{self.device} ONNX 추론에 실패했습니다. GPU 메모리와 모델을 확인하거나 CPU를 선택하세요. "
                "CPU로 자동 전환하지 않았습니다.") from error
