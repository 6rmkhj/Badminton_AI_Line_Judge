"""Inference-device routing; synthetic checkpoints are not trained shuttle models."""
import numpy as np
import pytest
import torch

from badminton import contact_model, events, learning
from badminton.inference import InferenceDeviceError


@pytest.fixture
def candidate_checkpoint(tmp_path):
    torch.manual_seed(7)
    model = learning._network().eval()
    path = tmp_path / "candidate.pt"
    metadata = {"architecture": learning.ARCHITECTURE, "patch_size": 24,
                "preprocessing": learning.PREPROCESSING, "analysis_sizes": [[64, 64]],
                "dataset_kind": "synthetic_software_fixture", "inference_device": "forged-device"}
    torch.save({"format_version": learning.FORMAT_VERSION, "metadata": metadata,
                "state_dict": model.state_dict()}, path)
    return path, model


@pytest.fixture
def contact_checkpoint(tmp_path):
    torch.manual_seed(11)
    model = contact_model._network().eval()
    path = tmp_path / "contact.pt"
    metadata = {"architecture": contact_model.ARCHITECTURE, "classes": list(contact_model.CLASSES),
                "patch_size": 24, "preprocessing": contact_model.SOURCE_PREPROCESSING,
                "dataset_kind": "synthetic_software_fixture", "analysis_sizes": [[64, 64]],
                "frame_interval_range_seconds": [1 / 30, 1 / 30]}
    torch.save({"format_version": 1, "metadata": metadata, "state_dict": model.state_dict()}, path)
    contact_model._load_checkpoint_cached.cache_clear()
    yield path, metadata
    contact_model._load_checkpoint_cached.cache_clear()


def test_candidate_cpu_inference_matches_forward_and_cannot_forge_provenance(candidate_checkpoint):
    path, model = candidate_checkpoint
    scorer = learning.load_candidate_scorer(path, device="cpu")
    patches = np.random.default_rng(4).integers(0, 256, (3, 24, 24, 3), dtype=np.uint8)
    with torch.inference_mode():
        expected = torch.sigmoid(model(torch.from_numpy(patches.transpose(0, 3, 1, 2).copy()).float() / 255).squeeze(1)).tolist()
    assert scorer.predict(patches) == pytest.approx(expected)
    assert next(scorer.model.parameters()).device.type == "cpu"
    assert scorer.provenance["inference_device"] == "cpu"
    assert scorer.last_inference_executed is True
    assert scorer.score(None, []) == []
    assert scorer.last_inference_executed is False
    scorer.predict(patches)
    assert scorer.predict(np.zeros((0, 24, 24, 3), np.uint8)) == []
    assert scorer.last_inference_executed is False


def test_gpu_load_failure_is_not_retried_on_cpu(monkeypatch):
    calls = []
    monkeypatch.setattr("badminton.inference.resolve_device", lambda value: value)

    class FailingModel:
        def eval(self):
            return self

        def to(self, device):
            calls.append(device)
            raise RuntimeError("synthetic allocation failure")

    with pytest.raises(InferenceDeviceError, match="cuda:1"):
        learning.CandidateScorer(FailingModel(), {"patch_size": 24}, "digest", device="cuda:1")
    assert calls == ["cuda:1"]


def test_gpu_forward_failure_does_not_silently_use_cpu(candidate_checkpoint, monkeypatch):
    path, _ = candidate_checkpoint
    scorer = learning.load_candidate_scorer(path)
    scorer.device = "cuda:1"  # Route test only: fail on transfer before any CPU forward.
    original_to = torch.Tensor.to
    requested = []

    def fail_gpu_transfer(tensor, device, *args, **kwargs):
        if str(device) == "cuda:1":
            requested.append(str(device))
            raise RuntimeError("synthetic GPU lost")
        return original_to(tensor, device, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fail_gpu_transfer)
    with pytest.raises(InferenceDeviceError, match="cuda:1"):
        scorer.predict(np.zeros((1, 24, 24, 3), np.uint8))
    assert requested == ["cuda:1"]


def test_gpu_output_copy_error_is_not_retried_on_cpu(candidate_checkpoint, contact_checkpoint, monkeypatch):
    candidate_path, _ = candidate_checkpoint
    scorer = learning.load_candidate_scorer(candidate_path)
    contact_path, metadata = contact_checkpoint
    model, _, digest = contact_model._checkpoint(contact_path)
    scorer.device = "cuda:0"
    original_to = torch.Tensor.to

    def emulate_transfer(tensor, device, *args, **kwargs):
        # Routing fixture only; this is not a hardware execution test.
        return original_to(tensor, "cpu" if device == "cuda:0" else device, *args, **kwargs)

    def delayed_failure(tensor, *args, **kwargs):
        raise RuntimeError("synthetic asynchronous GPU failure")

    monkeypatch.setattr(torch.Tensor, "to", emulate_transfer)
    monkeypatch.setattr(torch.Tensor, "cpu", delayed_failure)
    with pytest.raises(InferenceDeviceError, match="asynchronous"):
        scorer.predict(np.zeros((1, 24, 24, 3), np.uint8))
    assert scorer.last_inference_executed is False
    with pytest.raises(InferenceDeviceError, match="asynchronous"):
        contact_model._predict_patches(np.zeros((5, 24, 24, 3), np.uint8), [i / 30 for i in range(5)],
            {"frame_index": 2, "candidate_px": [32, 32]}, (64, 64), model, metadata, digest,
            contact_model.SOURCE_PREPROCESSING, device="cuda:0")


def test_missing_gpu_runtime_is_rejected_before_reading_checkpoint(monkeypatch):
    def missing_runtime():
        raise InferenceDeviceError("GPU runtime missing")

    monkeypatch.setattr("badminton.inference._torch", missing_runtime)
    with pytest.raises(InferenceDeviceError, match="runtime missing"):
        learning.load_candidate_scorer("does-not-exist.pt", device="cuda:0")


def test_detection_records_scorer_execution_per_frame_without_shared_state(monkeypatch):
    import cv2
    from badminton import detection, gpu_candidates
    monkeypatch.setattr("badminton.inference.resolve_device", lambda value: value)
    masks = gpu_candidates.appearance_masks
    monkeypatch.setattr(gpu_candidates, "appearance_masks", lambda *args, device, **kwargs:
                        masks(*args, device="cpu", **kwargs))

    class Scorer:
        provenance = {"inference_device": "cuda:1", "nested": {"value": 1}}
        last_inference_executed = False

        def score(self, frame, candidates, **kwargs):
            self.last_inference_executed = bool(candidates)
            return [{**candidate} for candidate in candidates]

    scorer = Scorer()
    monkeypatch.setattr(learning, "load_candidate_scorer", lambda *args, **kwargs: scorer)
    frames = [np.zeros((48, 64, 3), np.uint8) for _ in range(3)]
    cv2.circle(frames[1], (24, 24), 2, (255, 255, 255), -1)
    rows = detection.detect_sequence(frames, [0, .04, .08], {
        "person_backend": "none", "highres_enabled": False, "scorer_path": "fixture.pt",
        "inference_device": "cuda:1"})
    assert [row["candidate_scorer"]["inference_executed"] for row in rows] == [False, True, False]
    assert ["candidate_scorer" in row["detector"]["gpu_stages"] for row in rows] == [False, True, False]
    rows[1]["detector"]["gpu_stages"].append("caller-change")
    rows[1]["candidate_scorer"]["nested"]["value"] = 2
    assert "caller-change" not in rows[0]["detector"]["gpu_stages"]
    assert rows[0]["candidate_scorer"]["nested"]["value"] == 1
    assert scorer.provenance["nested"]["value"] == 1


def test_contact_cache_keeps_device_instances_separate(contact_checkpoint, monkeypatch):
    path, _ = contact_checkpoint
    requested = []
    monkeypatch.setattr("badminton.inference.resolve_device", lambda value: value)

    class RoutingModel:
        def load_state_dict(self, state, strict):
            assert strict

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            requested.append(device)
            return self

    monkeypatch.setattr(contact_model, "_network", RoutingModel)
    first = contact_model._checkpoint(path, "cuda:0")[0]
    second = contact_model._checkpoint(path, "cuda:1")[0]
    assert first is not second
    assert contact_model._checkpoint(path, "cuda:0")[0] is first
    assert (first.device, second.device) == ("cuda:0", "cuda:1")
    assert requested == ["cuda:0", "cuda:1"]
    # Reading preprocessing metadata only requests the CPU instance.
    contact_model.contact_checkpoint_metadata(path)
    assert requested[-1] == "cpu"
    assert first.device == "cuda:0"


def test_contact_cpu_predictions_remain_unqualified(contact_checkpoint):
    path, _ = contact_checkpoint
    result = contact_model.infer_contact_patches(np.zeros((5, 24, 24, 3), np.uint8),
        [i / 30 for i in range(5)], {"frame_index": 2, "candidate_px": [32, 32]}, path,
        image_size=(64, 64), preprocessing=contact_model.SOURCE_PREPROCESSING, device="cpu")
    assert result["provenance"]["inference_device"] == "cpu"
    assert result["provenance"]["inference_executed"] is True
    assert result["contact_observed"] is False
    assert "CONTACT_MODEL_UNQUALIFIED_HOLDOUT_EVIDENCE" in result["reason_codes"]
    assert set(result["probabilities"]) == set(contact_model.CLASSES)


def test_legacy_contact_device_error_is_not_downgraded_to_candidate_warning(monkeypatch):
    requested = []

    def fail(*args, **kwargs):
        requested.append(kwargs["device"])
        raise InferenceDeviceError("selected GPU unavailable")

    monkeypatch.setattr(contact_model, "infer_contact", fail)
    candidate = {"frame_index": 2, "candidate_px": [32, 32], "reason_codes": []}
    with pytest.raises(InferenceDeviceError, match="unavailable"):
        events._apply_contact_model(candidate, [np.zeros((64, 64, 3), np.uint8)] * 5,
                                    [i / 30 for i in range(5)], "model.pt", device="cuda:1")
    assert requested == ["cuda:1"]
    assert "contact_model" not in candidate


def test_native_contact_device_error_is_not_downgraded_to_review(monkeypatch):
    requested = []
    monkeypatch.setattr(contact_model, "contact_checkpoint_metadata", lambda path: {
        "preprocessing": contact_model.SOURCE_PREPROCESSING, "patch_size": 24})

    def fail(*args, **kwargs):
        requested.append(kwargs["device"])
        raise InferenceDeviceError("selected GPU unavailable")

    monkeypatch.setattr(contact_model, "infer_contact_patches", fail)
    frames = [np.zeros((64, 64, 3), np.uint8) for _ in range(5)]
    candidate = {"frame_index": 2, "time": 2 / 30, "candidate_px": [32, 32],
                 "contact_observed": False, "confidence": .5, "identity_confidence": .3,
                 "reason_codes": []}
    with pytest.raises(InferenceDeviceError, match="unavailable"):
        events.refine_source_contacts([candidate], frames, [i / 30 for i in range(5)], (64, 64),
                                      {"source_contact_model_path": "model.pt", "inference_device": "mps"})
    assert requested == ["mps"]


@pytest.mark.parametrize("device", ["cuda:0", "mps"])
def test_real_accelerator_matches_cpu_model_outputs_when_available(candidate_checkpoint, contact_checkpoint, device):
    available = torch.cuda.is_available() if device.startswith("cuda:") else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"Real {device} accelerator unavailable on this executor")
    path, _ = candidate_checkpoint
    patches = np.random.default_rng(8).integers(0, 256, (2, 24, 24, 3), dtype=np.uint8)
    cpu = learning.load_candidate_scorer(path)
    accelerated = learning.load_candidate_scorer(path, device=device)
    actual_device = next(accelerated.model.parameters()).device
    assert (str(actual_device) == device) if device.startswith("cuda:") else (actual_device.type == "mps")
    assert accelerated.predict(patches) == pytest.approx(cpu.predict(patches), abs=1e-4)
    contact_path, _ = contact_checkpoint
    kwargs = {"image_size": (64, 64), "preprocessing": contact_model.SOURCE_PREPROCESSING}
    args = (np.zeros((5, 24, 24, 3), np.uint8), [i / 30 for i in range(5)],
            {"frame_index": 2, "candidate_px": [32, 32]}, contact_path)
    reference = contact_model.infer_contact_patches(*args, **kwargs)
    actual = contact_model.infer_contact_patches(*args, device=device, **kwargs)
    assert actual["probabilities"] == pytest.approx(reference["probabilities"], abs=1e-4)
    assert actual["provenance"]["inference_device"] == device
    assert actual["contact_observed"] is False
