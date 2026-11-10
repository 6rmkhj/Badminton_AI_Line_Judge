import json

from badminton.reporting import _summarize


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _run(tmp_path, pairs):
    _write(tmp_path / "index.json", {
        "status": "completed", "videos": [{"id": "v1", "file": "clip.mp4", "status": "completed"}],
        "pairs": pairs,
    })
    _write(tmp_path / "videos/v1/result.json", {
        "video_id": "v1", "source_sha256": "same-source", "decision": {"decision": "REVIEW"},
        "calibration": {"status": "review"}, "tracks": [], "landings": [],
    })
    _write(tmp_path / "manifest.json", {
        "file_count": 1, "videos": [{"id": "v1", "sha256": "same-source"}],
    })
    _write(tmp_path / "evaluation.json", {
        "video_count": 1, "eligible_ground_truth_count": 0,
        "decision": {"automatic_accuracy": None},
    })
    return _summarize(tmp_path)


def test_sync_counts_use_current_index_key_and_keep_legacy_compatibility(tmp_path):
    summary = _run(tmp_path, [
        {"sync": {"status": "review", "same_rally_verified": False}},
        {"sync": {"status": "ok", "same_rally_verified": True}},
        {"sync": {"status": "failed", "same_rally_verified": False}},
        {"synchronization": {"status": "review", "same_rally_verified": False}},
    ])
    assert summary["synchronization"]["status_counts"] == {"review": 2, "ok": 1, "failed": 1}
    assert summary["synchronization"]["accepted_pairs"] == 1
    assert summary["synchronization"]["same_rally_verified_pairs"] == 1
    assert summary["implementation"]["modules_present"]["contact_model"] is True
    assert summary["goal_status"]["real_automatic_accuracy_ready"] is False


def test_current_sync_does_not_inherit_stale_legacy_acceptance(tmp_path):
    summary = _run(tmp_path, [{
        "sync": {"status": "review", "same_rally_verified": False},
        "synchronization": {"status": "ok", "same_rally_verified": True},
    }])
    assert summary["synchronization"]["status_counts"] == {"review": 1}
    assert summary["synchronization"]["accepted_pairs"] == 0
    assert summary["synchronization"]["same_rally_verified_pairs"] == 0
