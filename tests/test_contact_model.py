"""Temporal network software fixtures, not measured badminton accuracy."""
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from badminton.contact_model import (ARCHITECTURE, CLASSES, _network, _qualified, _contact_precision, _load_checkpoint_cached,
    build_contact_examples, contact_training_readiness, infer_contact, train_contact_model)
from badminton.learning import PREPROCESSING


def fixture_dataset(tmp_path):
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    videos = []
    for split_i, split in enumerate(("train", "val", "test")):
        for label_i, label in enumerate(CLASSES):
            video_id = f"{split}_{label}"
            path = tmp_path / f"{video_id}.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 64))
            assert writer.isOpened()
            for t in range(7):
                frame = np.full((64, 64, 3), 12 + split_i, dtype=np.uint8)
                cy = (16 + t * 2 if label_i == 0 else min(20 + t * 4, 33) if label_i == 1 else 33)
                cv2.circle(frame, (33, cy), 3, (250, 250, 250), -1)
                frame[-1, :, 0] = 30 + 10 * split_i + label_i
                writer.write(frame)
            writer.release()
            videos.append({"id": video_id, "file": path.name, "path": str(path),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "split": split, "group_id": video_id, "pair_key": video_id})
            point = {"frame_index": 3, "x": 32, "y": 32, "label": label,
                     "contact_observed": label == "first_contact", "contact_point_verified": label == "first_contact",
                     "contact_px": [33, 33] if label == "first_contact" else None,
                     "independent_evidence": "synthetic software fixture only"}
            annotation = {"video_id": video_id, "source": "human_review", "reviewer": "synthetic fixture",
                          "source_sha256": videos[-1]["sha256"],
                          "status": "needs_review", "contact_sequences_verified": True,
                          "analysis_size": {"width": 64, "height": 64}, "contact_sequences": [point]}
            (annotation_dir / f"{video_id}.json").write_text(json.dumps(annotation))
    return {"dataset_kind": "synthetic_software_fixture", "videos": videos}, annotation_dir


def test_contact_labels_are_explicit_and_group_disjoint(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    assert contact_training_readiness(manifest, annotations, 1)["ready"]
    file = annotations / "train_first_contact.json"
    data = json.loads(file.read_text())
    data["contact_sequences"][0]["contact_point_verified"] = False
    file.write_text(json.dumps(data))
    report = contact_training_readiness(manifest, annotations, 1)
    assert not report["ready"]
    assert report["excluded"]["first_contact_requires_observed_cork_contact_point"] == 1
    manifest["videos"][3]["group_id"] = manifest["videos"][0]["group_id"]
    assert any("leakage" in reason for reason in contact_training_readiness(manifest, annotations, 1)["reasons"])


def test_unreviewed_contact_drafts_cannot_train(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    for file in annotations.glob("*.json"):
        data = json.loads(file.read_text())
        data["source"] = "machine_draft"
        file.write_text(json.dumps(data))
    report = contact_training_readiness(manifest, annotations, 1)
    assert report["reviewed_sequences"] == 0 and not report["ready"]
    with pytest.raises(ValueError, match="not ready"):
        train_contact_model(manifest, annotations, tmp_path, tmp_path / "absent.pt", minimum_per_class=1)
    assert not (tmp_path / "absent.pt").exists()


def test_extract_sequences_and_localization_targets(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    examples = build_contact_examples(manifest, annotations, tmp_path, patch_size=32, minimum_per_class=1)
    assert examples["clips"].shape == (9, 5, 32, 32, 3)
    assert np.allclose(examples["offsets"][1], [1/16, 1/16])
    file = annotations / "train_airborne.json"
    data = json.loads(file.read_text())
    data["contact_sequences"][0]["frame_index"] = 6
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incomplete"):
        build_contact_examples(manifest, annotations, tmp_path, minimum_per_class=1)


def test_train_infer_is_functional_but_small_synthetic_holdouts_do_not_certify(tmp_path):
    torch.set_num_threads(1)
    manifest, annotations = fixture_dataset(tmp_path)
    destination = tmp_path / "temporal.pt"
    report = train_contact_model(manifest, annotations, tmp_path, destination, epochs=15,
                                 learning_rate=0.01, patch_size=32, minimum_per_class=1)
    assert destination.is_file() and destination.with_suffix(".pt.json").is_file()
    assert report["history"][-1]["train_loss"] < report["history"][0]["train_loss"]
    assert report["automatic_contact_qualified"] is False
    assert report["metrics"]["test"]["count"] == 3
    capture = cv2.VideoCapture(str(tmp_path / "test_first_contact.avi"))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok: break
        frames.append(frame)
    capture.release()
    times = [i/30 for i in range(7)]
    result = infer_contact(frames, times, {"frame_index": 3, "candidate_px": [32, 32]}, destination, preprocessing=PREPROCESSING)
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
    assert result["contact_observed"] is False and result["contact_px"] is None
    assert "CONTACT_MODEL_UNQUALIFIED_HOLDOUT_EVIDENCE" in result["reason_codes"]
    assert result["proposed_contact_px"] is not None
    bad_times = [0, 0.03, 0.06, 0.5, 0.53, 0.56, 0.59]
    assert "CONTACT_MODEL_TIMESTAMP_GAP" in infer_contact(frames, bad_times, {"frame_index": 3, "candidate_px": [32, 32]}, destination)["reason_codes"]


def test_qualified_model_path_requires_actual_threshold_and_capture_compatibility(tmp_path):
    """Unit-test a simulated qualifying checkpoint; not a real trained artifact."""
    torch.set_num_threads(1)
    model = _network()
    for parameter in model.parameters(): parameter.data.zero_()
    model.classes.bias.data[1] = 12
    evidence = {"statistical_unit": "rally_group", "validation": {"correct": 100, "selected": 100},
                "test": {"correct": 100, "selected": 100},
                "all_holdout_labels_have_independent_evidence": True,
                "localization": {"uncertainty_px": 2, "test_count": 100, "test_covered": 100}}
    metadata = {"architecture": ARCHITECTURE, "classes": list(CLASSES), "dataset_kind": "real_video",
                "preprocessing": PREPROCESSING,
                "patch_size": 32, "threshold": 0.99, "analysis_sizes": [[64, 64]],
                "frame_interval_range_seconds": [1/30, 1/30], "qualification_evidence": evidence,
                "split_groups": {"train": ["train1"], "val": [f"v{i}" for i in range(100)], "test": [f"t{i}" for i in range(100)]}}
    assert _qualified(metadata)
    path = tmp_path / "simulated_qualified.pt"
    torch.save({"format_version": 1, "metadata": metadata, "state_dict": model.state_dict()}, path)
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
    candidate = {"frame_index": 2, "candidate_px": [30, 30]}
    result = infer_contact(frames, [i/30 for i in range(5)], candidate, path, preprocessing=PREPROCESSING)
    assert result["contact_observed"] is True
    assert result["contact_px"] == [30, 30] and result["uncertainty_px"] == 3
    hits = _load_checkpoint_cached.cache_info().hits
    result["provenance"]["qualification_evidence"]["test"]["correct"] = 0
    second = infer_contact(frames, [i/30 for i in range(5)], candidate, path, preprocessing=PREPROCESSING)
    assert second["contact_observed"] is True  # Caller cannot mutate cached qualification.
    assert _load_checkpoint_cached.cache_info().hits == hits + 1
    mismatch = infer_contact(frames, [i/60 for i in range(5)], candidate, path, preprocessing=PREPROCESSING)
    assert mismatch["contact_observed"] is False
    assert "CONTACT_MODEL_CAPTURE_DOMAIN_MISMATCH" in mismatch["reason_codes"]
    unknown_preprocessing = infer_contact(frames, [i/30 for i in range(5)], candidate, path)
    assert not unknown_preprocessing["contact_observed"]
    assert "CONTACT_MODEL_PREPROCESSING_MISMATCH" in unknown_preprocessing["reason_codes"]
    compressed = infer_contact(frames, [i/30 for i in range(5)], candidate, path, preprocessing="h264_crf21_preview")
    assert not compressed["contact_observed"]
    assert "CONTACT_MODEL_PREPROCESSING_MISMATCH" in compressed["reason_codes"]
    evidence["test"]["correct"] = 80
    assert not _qualified(metadata)


def test_missing_window_does_not_create_contact(tmp_path):
    model = _network()
    path = tmp_path / "unqualified.pt"
    torch.save({"format_version": 1, "metadata": {"architecture": ARCHITECTURE,
               "classes": list(CLASSES), "patch_size": 32}, "state_dict": model.state_dict()}, path)
    frames = [np.zeros((64, 64, 3), dtype=np.uint8)] * 3
    result = infer_contact(frames, [0, 0.03, 0.06], {"frame_index": 1, "candidate_px": [30, 30]}, path)
    assert result["contact_observed"] is False
    assert result["reason_codes"] == ["CONTACT_MODEL_INCOMPLETE_WINDOW"]


def test_correlated_frames_do_not_inflate_qualification_samples():
    labels = np.ones(100, dtype=int)
    probabilities = np.tile([0.001, 0.998, 0.001], (100, 1))
    result = _contact_precision(labels, probabilities, 0.99, ["same_rally"] * 100)
    assert result["selected"] == 1 and result["correct"] == 1
    assert result["wilson_95_lower"] < 0.95


def test_verified_verdict_does_not_verify_unchecked_temporal_labels(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    file = annotations / "train_first_contact.json"
    annotation = json.loads(file.read_text())
    annotation.update(status="verified", verdict="IN", contact_sequences_verified=False)
    file.write_text(json.dumps(annotation))
    report = contact_training_readiness(manifest, annotations, 1)
    assert not report["ready"]
    assert report["excluded"]["unverified_contact_sequences"] == 1
