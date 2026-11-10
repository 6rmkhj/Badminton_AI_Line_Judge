"""Synthetic software fixtures only; these are not badminton performance results."""
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from badminton.learning import (PREPROCESSING, load_candidate_scorer, prepare_training_samples,
                                 train_candidate_scorer, training_readiness)


def fixture_dataset(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    videos = []
    for index, split in enumerate(("train", "val", "test")):
        video_id = f"clip_{index}"
        path = tmp_path / f"{video_id}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
        assert writer.isOpened()
        labels = []
        for frame_index in range(8):
            frame = np.full((64, 64, 3), 16 + index * 3 + frame_index, dtype=np.uint8)
            cv2.circle(frame, (16, 16), 6, (250, 250, 250), -1)
            frame[-1, :, 0] = 40 + index * 20
            writer.write(frame)
            labels.extend([{"frame_index": frame_index, "x": 16, "y": 16, "label": "shuttle"},
                           {"frame_index": frame_index, "x": 48, "y": 48, "label": "background"}])
        writer.release()
        videos.append({"id": video_id, "file": path.name, "path": str(path),
                       "group_id": f"rally_{index}", "pair_key": f"pair_{index}", "split": split,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        annotation = {"video_id": video_id, "source": "human_review", "reviewer": "synthetic fixture",
                      "source_sha256": videos[-1]["sha256"],
                      "detection_labels_verified": True,
                      "status": "verified", "analysis_size": {"width": 64, "height": 64},
                      "detection_labels": labels}
        (annotations / f"{video_id}.json").write_text(json.dumps(annotation))
    return {"dataset_kind": "synthetic_software_fixture", "videos": videos}, annotations


def test_drafts_are_not_training_truth(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    for file in annotations.glob("*.json"):
        annotation = json.loads(file.read_text())
        annotation.update(source="machine_draft", status="needs_review", detection_labels_verified=True)
        file.write_text(json.dumps(annotation))
    report = training_readiness(manifest, annotations)
    assert report["ready"] is False
    assert report["trained"] is False
    assert report["verified_patch_count"] == 0
    with pytest.raises(ValueError, match="Training data not ready"):
        train_candidate_scorer(manifest, annotations, tmp_path, tmp_path / "never.pt")
    assert not (tmp_path / "never.pt").exists()


@pytest.mark.parametrize("field", ["group_id", "sha256", "pair_key"])
def test_group_content_and_pair_leakage_rejected(tmp_path, field):
    manifest, annotations = fixture_dataset(tmp_path)
    manifest["videos"][1][field] = manifest["videos"][0][field]
    report = training_readiness(manifest, annotations)
    assert report["ready"] is False
    assert any("leakage" in reason for reason in report["reasons"])


def test_explicit_patch_review_independent_of_unknown_verdict(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    file = annotations / "clip_0.json"
    data = json.loads(file.read_text())
    data.update(status="needs_review", verdict="UNKNOWN", detection_labels_verified=True)
    file.write_text(json.dumps(data))
    assert training_readiness(manifest, annotations)["ready"] is True
    data["detection_labels_verified"] = False
    file.write_text(json.dumps(data))
    assert training_readiness(manifest, annotations)["ready"] is False


def test_prepare_frames_and_reject_changed_source(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    prepared = prepare_training_samples(manifest, annotations, tmp_path)
    assert prepared["patches"].shape == (48, 32, 32, 3)
    assert prepared["patches"].dtype == np.uint8
    assert prepared["samples"][0]["label"] == 1
    assert prepared["patches"][0].mean() > prepared["patches"][1].mean()
    with (tmp_path / "clip_0.avi").open("ab") as handle:
        handle.write(b"changed source")
    with pytest.raises(ValueError, match="content changed"):
        prepare_training_samples(manifest, annotations, tmp_path)


def test_contradictory_label_and_missing_coordinate_dimensions(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    file = annotations / "clip_0.json"
    data = json.loads(file.read_text())
    data["detection_labels"].append({**data["detection_labels"][0], "label": "background"})
    file.write_text(json.dumps(data))
    assert any("Contradictory" in reason for reason in training_readiness(manifest, annotations)["reasons"])
    del data["analysis_size"]
    file.write_text(json.dumps(data))
    assert training_readiness(manifest, annotations)["excluded"]["missing_analysis_size"] > 0


def test_train_checkpoint_and_candidate_scoring_on_synthetic_fixture(tmp_path):
    torch.set_num_threads(1)
    manifest, annotations = fixture_dataset(tmp_path)
    model_path = tmp_path / "candidate.pt"
    report = train_candidate_scorer(manifest, annotations, tmp_path, model_path, epochs=40, learning_rate=0.01)
    assert report["dataset_kind"] == "synthetic_software_fixture"
    assert report["metrics"]["test"]["accuracy"] >= 0.9
    assert set(report["split_groups"]["train"]).isdisjoint(report["split_groups"]["test"])
    assert report["weights_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert model_path.with_suffix(".pt.json").is_file()
    scorer = load_candidate_scorer(model_path)
    frame = np.full((64, 64, 3), 20, dtype=np.uint8)
    cv2.circle(frame, (16, 16), 6, (250, 250, 250), -1)
    candidates = [{"x": 16, "y": 16, "confidence": 0.4}, {"x": 48, "y": 48, "confidence": 0.7}]
    scored = scorer.score(frame, candidates, preprocessing=PREPROCESSING)
    assert scored[0]["confidence"] == 0.4 and scored[1]["confidence"] == 0.7
    assert scored[0]["learned_confidence"] > scored[1]["learned_confidence"]
    assert scored[0]["learned_score_calibrated"] is False
    assert "learned_confidence" not in candidates[0]
    assert scorer.score(frame, []) == []
    with pytest.raises(ValueError, match="inside"):
        scorer.score(frame, [{"x": -1, "y": 10}], preprocessing=PREPROCESSING)
    with pytest.raises(ValueError, match="preprocessing"):
        scorer.score(frame, candidates)
    with pytest.raises(ValueError, match="preprocessing"):
        scorer.score(frame, candidates, preprocessing="h264_crf21_preview")
    with pytest.raises(ValueError, match="frame size"):
        scorer.score(np.zeros((80, 80, 3), dtype=np.uint8), candidates, preprocessing=PREPROCESSING)


def test_unsupported_checkpoint_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 999, "metadata": {}}, path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_candidate_scorer(path)


def test_stale_annotation_source_hash_cannot_train(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    file = annotations / "clip_0.json"
    annotation = json.loads(file.read_text())
    annotation["source_sha256"] = "0" * 64
    file.write_text(json.dumps(annotation))
    report = training_readiness(manifest, annotations)
    assert not report["ready"]
    assert report["excluded"]["annotation_source_hash_missing_or_mismatched"] == 1


def test_verified_verdict_does_not_verify_unchecked_detection_labels(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    file = annotations / "clip_0.json"
    annotation = json.loads(file.read_text())
    annotation.update(status="verified", verdict="IN", detection_labels_verified=False)
    file.write_text(json.dumps(annotation))
    assert not training_readiness(manifest, annotations)["ready"]


def test_ai_visual_supervision_requires_explicit_opt_in_and_keeps_provenance(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    for path in annotations.glob('*.json'):
        data = json.loads(path.read_text())
        data.update(source='ai_visual_review', status='reviewed_by_ai', human_verified=False,
                    detection_labels_verified_by_ai=True)
        path.write_text(json.dumps(data))
    assert not training_readiness(manifest, annotations)['ready']
    report = training_readiness(manifest, annotations, label_source='ai_visual_review')
    assert report['ready'] and report['label_source'] == 'ai_visual_review'
    assert report['independent_human_ground_truth'] is False
    torch.set_num_threads(1)
    model = train_candidate_scorer(manifest, annotations, tmp_path, tmp_path/'ai.pt',
                                   epochs=1, label_source='ai_visual_review')
    assert model['label_source'] == 'ai_visual_review'
    assert model['independent_human_ground_truth'] is False
    assert 'not verified accuracy' in model['metric_interpretation']
    data = json.loads((annotations/'clip_0.json').read_text())
    data['human_verified'] = True
    (annotations/'clip_0.json').write_text(json.dumps(data))
    assert not training_readiness(manifest, annotations, label_source='ai_visual_review')['ready']


def test_duplicate_content_patches_are_not_double_weighted(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    original = manifest['videos'][0]
    manifest['videos'].append({**original, 'id': 'same_bytes'})
    data = json.loads((annotations/'clip_0.json').read_text())
    data['video_id'] = 'same_bytes'
    (annotations/'same_bytes.json').write_text(json.dumps(data))
    report = training_readiness(manifest, annotations)
    assert report['ready']
    assert report['verified_patch_count'] == 48
    assert report['excluded']['duplicate_content_patch_label'] == 16
    data['detection_labels'][0]['label'] = 'background'
    (annotations/'same_bytes.json').write_text(json.dumps(data))
    report = training_readiness(manifest, annotations)
    assert not report['ready']
    assert 'Contradictory labels for duplicate video content' in report['reasons']


def test_portable_training_rebases_stale_source_paths_and_checks_bytes(tmp_path):
    manifest, annotations = fixture_dataset(tmp_path)
    for video in manifest['videos']:
        video['path'] = str(tmp_path/'missing-original-machine'/video['file'])
    prepared = prepare_training_samples(manifest, annotations, tmp_path)
    assert prepared['patches'].shape == (48, 32, 32, 3)
    with (tmp_path/manifest['videos'][0]['file']).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='content changed'):
        prepare_training_samples(manifest, annotations, tmp_path)
