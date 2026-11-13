"""Mask equivalence on CPU tensors; real accelerator checks are separate/skippable.

Synthetic image arrays verify segmentation semantics, not shuttle accuracy or
GPU performance. CPU-tensor tests do not establish execution on GPU hardware.
"""
from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch


def reference_masks(gray, hsv, bright, motion, foreground, local_base, config,
                    exclusion, roi=None, *, native=False):
    """The pre-existing OpenCV/NumPy segmentation contract, independently run."""
    channels = [("bright", bright > 0, cv2.subtract(gray, local_base),
                 config.get("motion_threshold", 16), config.get("contrast_threshold", 8))]
    if config.get("include_dark_components"):
        negative = cv2.subtract(local_base, gray)
        threshold = config.get("dark_contrast_threshold", 18)
        dark = (negative >= threshold) & (hsv[:, :, 1] <= config.get("dark_max_saturation", 115))
        channels.append(("dark", dark, negative,
                         config.get("dark_motion_threshold", config.get("motion_threshold", 16)), threshold))
    result = []
    for name, appearance, contrast, motion_threshold, contrast_threshold in channels:
        cue = ((motion >= motion_threshold)
               & (foreground >= config.get("background_threshold", 8))
               & (contrast >= contrast_threshold))
        selected = appearance & ~exclusion.astype(bool)
        if roi is not None:
            selected &= roi > 0
        if not native or config.get("native_motion_components"):
            selected &= cue
        elif config.get("native_local_components", bool(config.get("refinement_regions"))):
            selected &= contrast >= contrast_threshold
        result.append((name, selected, contrast, cue))
    return result


def example_inputs():
    rng = np.random.default_rng(20261113)
    shape = (17, 23)
    return {
        "gray": rng.integers(0, 256, shape, dtype=np.uint8),
        "hsv": rng.integers(0, 256, (*shape, 3), dtype=np.uint8),
        "bright": rng.choice(np.array([0, 255], dtype=np.uint8), size=shape),
        "motion": rng.uniform(0, 90, shape).astype(np.float32),
        "foreground": rng.integers(0, 256, shape, dtype=np.uint8),
        "local_base": rng.integers(0, 256, shape, dtype=np.uint8),
        "exclusion": rng.choice(np.array([0, 255], dtype=np.uint8), size=shape),
        "roi": rng.choice(np.array([0, 255], dtype=np.uint8), size=shape),
    }


def assert_identical(actual, expected):
    actual = list(actual)
    assert [row[0] for row in actual] == [row[0] for row in expected]
    for actual_row, expected_row in zip(actual, expected):
        for item, wanted in zip(actual_row[1:], expected_row[1:]):
            assert isinstance(item, np.ndarray)
            assert item.dtype == wanted.dtype
            np.testing.assert_array_equal(item, wanted)


@pytest.mark.parametrize("include_dark", [False, True])
@pytest.mark.parametrize("native,extra", [
    (False, {}),
    (True, {}),
    (True, {"native_motion_components": True}),
    (True, {"native_local_components": True}),
    (True, {"refinement_regions": [{"frame_start": 0}]}),
    (True, {"native_local_components": False, "refinement_regions": [{"frame_start": 0}]}),
])
def test_cpu_tensor_masks_match_existing_segmentation(include_dark, native, extra):
    from badminton.gpu_candidates import appearance_masks

    values = example_inputs()
    originals = {key: value.copy() for key, value in values.items()}
    config = {"include_dark_components": include_dark,
              "motion_threshold": 16.5, "background_threshold": 8.5,
              "contrast_threshold": 8.5, "dark_motion_threshold": 8.25,
              "dark_contrast_threshold": 18.5, "dark_max_saturation": 115,
              **extra}
    saved_config = deepcopy(config)
    expected = reference_masks(**values, config=config, native=native)
    actual = appearance_masks(**values, config=config, native=native, device="cpu")
    assert_identical(actual, expected)
    for key, value in values.items():
        np.testing.assert_array_equal(value, originals[key], err_msg=f"Mutated input {key}")
    assert config == saved_config


def test_saturated_subtraction_saturation_gate_roi_and_exclusions():
    from badminton.gpu_candidates import appearance_masks

    gray = np.array([[250, 0, 100, 120, 110, 130, 200, 90]], dtype=np.uint8)
    local_base = np.array([[0, 250, 120, 100, 100, 100, 100, 100]], dtype=np.uint8)
    hsv = np.zeros((1, 8, 3), dtype=np.uint8)
    hsv[0, 2, 1] = 255
    values = dict(gray=gray, hsv=hsv, local_base=local_base,
                  bright=np.full_like(gray, 255), motion=np.full(gray.shape, 100, dtype=np.float32),
                  foreground=np.full_like(gray, 100),
                  exclusion=np.array([[0, 0, 0, 0, 255, 0, 0, 0]], dtype=np.uint8),
                  roi=np.array([[255, 255, 255, 255, 255, 0, 255, 255]], dtype=np.uint8))
    actual = list(appearance_masks(**values, config={"include_dark_components": True}, device="cpu"))
    assert [row[0] for row in actual] == ["bright", "dark"]
    np.testing.assert_array_equal(actual[0][1], [[True, False, False, True, False, False, True, False]])
    np.testing.assert_array_equal(actual[1][1], [[False, True, False, False, False, False, False, False]])
    np.testing.assert_array_equal(actual[0][2], [[250, 0, 0, 20, 10, 30, 100, 0]])
    np.testing.assert_array_equal(actual[1][2], [[0, 250, 20, 0, 0, 0, 0, 10]])
    assert actual[0][2].dtype == np.uint8
    assert actual[1][2].dtype == np.uint8


def test_masks_allow_no_roi_and_boolean_exclusion():
    from badminton.gpu_candidates import appearance_masks

    values = example_inputs()
    values["roi"] = None
    values["exclusion"] = values["exclusion"] > 0
    config = {"include_dark_components": True, "native_local_components": True}
    assert_identical(appearance_masks(**values, config=config, native=True, device="cpu"),
                     reference_masks(**values, config=config, native=True))


def test_fractional_thresholds_near_integer_boundaries_do_not_round_down():
    from badminton.gpu_candidates import appearance_masks

    values = dict(gray=np.array([[18, 0]], dtype=np.uint8),
                  local_base=np.array([[0, 18]], dtype=np.uint8),
                  hsv=np.zeros((1, 2, 3), dtype=np.uint8),
                  bright=np.full((1, 2), 255, dtype=np.uint8),
                  motion=np.full((1, 2), 100, dtype=np.float32),
                  foreground=np.full((1, 2), 9, dtype=np.uint8),
                  exclusion=np.zeros((1, 2), dtype=np.uint8))
    config = {"include_dark_components": True, "contrast_threshold": 18.0000001,
              "dark_contrast_threshold": 18.0000001, "background_threshold": 8.0000001}
    assert_identical(appearance_masks(**values, config=config, device="cpu"),
                     reference_masks(**values, config=config))


def test_tensor_device_failure_is_reported_without_cpu_retry(monkeypatch):
    from badminton.gpu_candidates import appearance_masks
    from badminton.inference import InferenceDeviceError

    devices = []
    def fail_tensor(*args, **kwargs):
        devices.append(kwargs.get("device"))
        raise RuntimeError("synthetic device allocation failure")

    monkeypatch.setattr(torch, "as_tensor", fail_tensor)
    with pytest.raises(InferenceDeviceError, match="cuda:1.*후보 연산"):
        appearance_masks(**example_inputs(), config={}, device="cuda:1")
    assert devices == ["cuda:1"]


@pytest.mark.parametrize("device", ["cuda:0", "mps"])
def test_real_accelerator_masks_match_cpu_tensor_results_when_available(device):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available; CPU mask equivalence is tested separately")
    elif not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        pytest.skip("No MPS GPU available; CPU mask equivalence is tested separately")
    from badminton.gpu_candidates import appearance_masks

    values = example_inputs()
    config = {"include_dark_components": True, "native_motion_components": True,
              "dark_contrast_threshold": 18.5, "motion_threshold": 16.5}
    expected = appearance_masks(**values, config=config, native=True, device="cpu")
    actual = appearance_masks(**values, config=config, native=True, device=device)
    assert_identical(actual, expected)
