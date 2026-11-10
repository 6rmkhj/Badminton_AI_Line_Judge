"""Native contact crops test software geometry; they are not badminton labels."""
import hashlib
import json
import subprocess

import cv2
import numpy as np
import pytest
import torch

from badminton.contact_model import (ARCHITECTURE, CLASSES, SOURCE_PREPROCESSING,
    _network, build_contact_examples, infer_contact, native_contact_patch, train_contact_model)
from badminton.events import estimate_landings, refine_source_contacts, validate_contact_modes
from badminton.learning import PREPROCESSING, _patch
from badminton.media import SourceFrameSequence, load_frames, probe


class OrderedFrames:
    """Reject repeat/backward decoder reads to exercise the streaming contract."""
    def __init__(self, frames):
        self.frames, self.requested = frames, []

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        assert not self.requested or index > self.requested[-1]
        self.requested.append(index)
        return self.frames[index]


def candidate(index=3, point=(20, 15)):
    return {"frame_index": index, "time": index / 60, "candidate_px": list(point),
            "contact_observed": False, "contact_px": None, "identity_confidence": 0.3,
            "confidence": 0.5, "reason_codes": ["FIRST_CONTACT_NOT_OBSERVED", "CORK_CONTACT_REGION_UNRESOLVED"]}


def native_frames(count=8):
    frames = [np.zeros((120, 160, 3), np.uint8) for _ in range(count)]
    for i, frame in enumerate(frames):
        frame[60, 80] = [20, 70, 180 + i]
    return frames


def test_dual_contact_model_configuration_is_rejected_before_any_prediction():
    cfg = {"contact_model_path": "legacy.pt", "source_contact_model_path": "native.pt"}
    with pytest.raises(ValueError, match="not both"):
        validate_contact_modes({**cfg, "source_contact_enabled": False})
    with pytest.raises(ValueError, match="not both"):
        estimate_landings([], [], [], {}, cfg)
    frames = OrderedFrames(native_frames())
    with pytest.raises(ValueError, match="not both"):
        refine_source_contacts([candidate()], frames, [i/60 for i in range(8)], (40, 30), cfg)
    assert frames.requested == []


def test_requested_late_event_is_filtered_before_global_candidate_cap():
    points = [(10 + i * 4, 10 + i * 3) for i in range(7)] + [(34, 28)] * 5
    tracks = []
    for track_id, shift in ((1, 0), (2, 60)):
        tracks.append({"track_id": track_id, "observations": [
            {"frame_index": shift+i, "time": (shift+i)/60, "x": x, "y": y,
             "radius": 2, "confidence": 0.9} for i, (x, y) in enumerate(points)]})
    times = [i/60 for i in range(80)]
    result = estimate_landings(tracks, [], times, {}, {"event_window_seconds": [1, 1.3], "max_candidates": 1})
    assert len(result) == 1
    assert result[0]["track_id"] == 2
    assert 1 <= result[0]["time"] <= 1.3


def test_contiguous_observation_span_does_not_count_a_gap():
    track = {"track_id": 1, "observations": [
        {"frame_index": i, "time": time, "x": 10+i*4, "y": 10+i*3,
         "radius": 2, "confidence": .8} for i, time in enumerate([0, .02, .04, .5, .52, .54])]}
    result = estimate_landings([track], [], [i*.02 for i in range(30)], {}, {"min_observations": 3})
    assert result
    assert result[0]["contiguous_observed_before_seconds"] == .04
    assert result[0]["observed_before_seconds"] == .54


@pytest.mark.parametrize("point", [(12.5, 9.5), (0.1, 0.1), (18.8, 12.8)])
def test_native_crop_exactly_matches_training_rgb_padding(point):
    frame = np.arange(13 * 19 * 3, dtype=np.uint8).reshape(13, 19, 3)
    assert np.array_equal(native_contact_patch(frame, *point, 24), _patch(frame, *point, 24))


def test_native_temporal_geometry_and_single_decode_pass(tmp_path):
    frames = OrderedFrames(native_frames())
    original = [candidate(4), candidate(3)]
    result = refine_source_contacts(original, frames, [i/60 for i in range(8)], (40, 30),
                                    {"source_contact_patch_size": 24}, tmp_path)
    assert frames.requested == list(range(1, 7))
    evidence = result[0]["source_contact_evidence"]
    assert evidence["status"] == "complete"
    assert evidence["scale_xy"] == [4, 4]
    assert evidence["candidate_source_px"] == [80, 60]
    assert evidence["frames"][0]["source_roi"] == [68, 48, 92, 72]
    assert evidence["frame_indices"] == [2, 3, 4, 5, 6]
    strip = cv2.imread(str(tmp_path / evidence["image"]))
    assert strip.shape == (56, 120, 3)
    assert all(not item["contact_observed"] and item["contact_px"] is None for item in result)
    assert "source_contact_evidence" not in original[0]
    json.dumps(result, allow_nan=False)


def test_incomplete_or_irregular_window_never_promotes_contact(tmp_path):
    frames = native_frames()
    near_start = refine_source_contacts([candidate(0)], OrderedFrames(frames), [i/60 for i in range(8)],
                                       (40, 30), output_dir=tmp_path)[0]
    assert near_start["source_contact_evidence"]["status"] == "incomplete"
    assert near_start["contact_observed"] is False
    times = [0, .02, .04, .4, .42, .44, .46, .48]
    gap = refine_source_contacts([candidate(3)], OrderedFrames(frames), times, (40, 30))[0]
    assert gap["source_contact_evidence"]["reason"] == "timestamp_discontinuity"
    assert gap["contact_px"] is None


def test_missing_source_frames_fail_closed_without_losing_candidate():
    frames = native_frames()
    frames[3] = None
    result = refine_source_contacts([candidate(3)], OrderedFrames(frames), [i/60 for i in range(8)], (40, 30))[0]
    assert result["source_contact_evidence"]["status"] == "incomplete"
    assert "uint8" in result["source_contact_evidence"]["error"]
    assert result["candidate_px"] == [20, 15]
    assert result["contact_observed"] is False


def test_rotated_source_uses_display_geometry_and_preserves_center_frame(tmp_path):
    original, rotated = tmp_path / "landscape.mp4", tmp_path / "portrait.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30:duration=0.3",
                    "-c:v", "libx264", "-threads", "1", str(original)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(original),
                    "-c", "copy", str(rotated)], check=True)
    metadata = probe(rotated)
    assert (metadata["width"], metadata["height"]) == (64, 96)
    decoded = load_frames(rotated)
    assert decoded[3].shape[:2] == (96, 64)
    point = candidate(3, (16, 24))
    point["time"] = metadata["timestamps"][3]
    with SourceFrameSequence(rotated, len(metadata["timestamps"])) as source:
        result = refine_source_contacts([point], source, metadata["timestamps"], (32, 48),
                                        {"source_contact_patch_size": 24}, tmp_path / "evidence-check")[0]
    evidence = result["source_contact_evidence"]
    assert evidence["source_size"] == {"width": 64, "height": 96}
    assert evidence["candidate_source_px"] == [32, 48]
    assert evidence["frames"][2]["source_roi"] == [20, 36, 44, 60]
    assert evidence["frames"][2]["frame_index"] == result["frame_index"] == 3
    assert evidence["frames"][2]["time"] == result["time"]
    assert result["contact_px"] is None


def native_checkpoint(path, *, preprocessing=SOURCE_PREPROCESSING, qualified=True):
    """Artificial contract-only fixture, never a distributed trained model."""
    torch.set_num_threads(1)
    model = _network()
    for parameter in model.parameters():
        parameter.data.zero_()
    model.classes.bias.data[1] = 12
    model.offset.bias.data[:] = torch.tensor([0.1, -0.1])
    evidence = {"statistical_unit": "rally_group", "validation": {"correct": 100, "selected": 100},
                "test": {"correct": 100, "selected": 100}, "all_holdout_labels_have_independent_evidence": True,
                "localization": {"uncertainty_px": 2, "test_count": 100, "test_covered": 100}}
    metadata = {"architecture": ARCHITECTURE, "classes": list(CLASSES),
                "dataset_kind": "real_video" if qualified else "synthetic_software_fixture",
                "preprocessing": preprocessing, "patch_size": 32, "threshold": 0.99,
                "analysis_sizes": [[160, 120]], "frame_interval_range_seconds": [1/60, 1/60],
                "qualification_evidence": evidence,
                "split_groups": {"train": ["training"], "val": [f"v{i}" for i in range(100)],
                                  "test": [f"t{i}" for i in range(100)]}}
    torch.save({"format_version": 1, "metadata": metadata, "state_dict": model.state_dict()}, path)
    return path


def test_native_qualified_geometry_maps_back_without_changing_identity(tmp_path):
    path = native_checkpoint(tmp_path / "native.pt")
    result = refine_source_contacts([candidate()], OrderedFrames(native_frames()), [i/60 for i in range(8)],
                                    (40, 30), {"source_contact_model_path": str(path)})[0]
    expected = np.array([20, 15]) + np.tanh([0.1, -0.1]) * 4
    assert result["contact_observed"] is True
    assert np.allclose(result["contact_px"], expected)
    assert result["uncertainty_px"] == .75  # Native 3px uncertainty / 4x source scale.
    assert result["identity_confidence"] == .3
    assert result["source_contact_model"]["coordinate_system"] == "source_pixels"
    assert result["contact_model"]["coordinate_system"] == "analysis_pixels"
    assert result["source_contact_evidence"]["contact_observed"] is False


@pytest.mark.parametrize("preprocessing,qualified", [(PREPROCESSING, True), (SOURCE_PREPROCESSING, False)])
def test_legacy_or_unqualified_checkpoint_cannot_certify_native_contact(tmp_path, preprocessing, qualified):
    path = native_checkpoint(tmp_path / "wrong.pt", preprocessing=preprocessing, qualified=qualified)
    result = refine_source_contacts([candidate()], OrderedFrames(native_frames()), [i/60 for i in range(8)],
                                    (40, 30), {"source_contact_model_path": str(path)})[0]
    assert result["contact_observed"] is False
    assert result["contact_px"] is None
    assert result["source_contact_evidence"]["status"] == "complete"


def test_native_training_and_inference_use_same_source_scale(tmp_path):
    annotations = tmp_path / "labels"
    annotations.mkdir()
    videos = []
    for split in ("train", "val", "test"):
        for label in CLASSES:
            name = f"{split}_{label}"
            path = tmp_path / f"{name}.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60, (160, 120))
            assert writer.isOpened()
            for frame in native_frames(7):
                writer.write(frame)
            writer.release()
            # Add distinct content so the software fixture is group-disjoint.
            with path.open("ab") as handle:
                handle.write(name.encode())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            videos.append({"id": name, "path": str(path), "file": path.name, "sha256": digest,
                           "group_id": name, "split": split, "pair_key": name})
            row = {"video_id": name, "source": "human_review", "reviewer": "software fixture",
                   "source_sha256": digest, "contact_sequences_verified": True,
                   "analysis_size": {"width": 40, "height": 30},
                   "contact_sequences": [{"frame_index": 3, "x": 20, "y": 15, "label": label,
                                            "contact_observed": label == "first_contact",
                                            "contact_point_verified": label == "first_contact",
                                            "contact_px": [20.5, 15.5] if label == "first_contact" else None,
                                            "independent_evidence": "synthetic software fixture only"}]}
            (annotations / f"{name}.json").write_text(json.dumps(row))
    manifest = {"dataset_kind": "synthetic_software_fixture", "videos": videos}
    examples = build_contact_examples(manifest, annotations, patch_size=32, minimum_per_class=1,
                                      preprocessing=SOURCE_PREPROCESSING)
    assert examples["clips"].shape == (9, 5, 32, 32, 3)
    assert np.allclose(examples["offsets"][1], [.125, .125])  # 0.5 analysis px == 2 native px.
    assert examples["samples"][1]["model_candidate_px"] == [80, 60]
    capture = cv2.VideoCapture(videos[0]["path"])
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    assert np.array_equal(examples["clips"][0, 2], native_contact_patch(decoded[3], 80, 60, 32))
    torch.set_num_threads(1)
    path = tmp_path / "trained-native.pt"
    report = train_contact_model(manifest, annotations, None, path, epochs=1, patch_size=32, minimum_per_class=1,
                                 preprocessing=SOURCE_PREPROCESSING)
    assert report["preprocessing"] == SOURCE_PREPROCESSING
    assert report["analysis_sizes"] == [(160, 120)]
    assert report["coordinate_system"] == "source_pixels"
    assert not report["automatic_contact_qualified"]
    result = infer_contact(decoded, [i/60 for i in range(7)], {"frame_index": 3, "candidate_px": [80, 60]},
                           path, preprocessing=SOURCE_PREPROCESSING)
    assert "CONTACT_MODEL_PREPROCESSING_MISMATCH" not in result["reason_codes"]
    assert "CONTACT_MODEL_CAPTURE_DOMAIN_MISMATCH" not in result["reason_codes"]
    assert result["contact_observed"] is False
