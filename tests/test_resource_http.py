"""Resource choices must be confirmed on the server before analysis can start."""
from contextlib import contextmanager
import http.client
import json
import threading

import pytest

from badminton.web import create_server


@contextmanager
def running_server(root):
    server = create_server(root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def request(server, method="GET", path="/api/resources", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    headers = dict(headers or {})
    if body is not None or method == "POST":
        body = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        payload = response.read()
        return response.status, json.loads(payload) if payload else None
    finally:
        connection.close()


def settings_for(server):
    status, snapshot = request(server)
    assert status == 200
    return snapshot["settings"]


def test_startup_requires_confirmation_before_either_analysis_route(tmp_path):
    with running_server(tmp_path / "run") as server:
        status, snapshot = request(server)
        assert status == 200
        assert snapshot["confirmed"] is False
        assert snapshot["busy"] is False
        assert snapshot["environment"]["cpu"]["available_cores"] >= 1
        assert snapshot["settings"]["device"] == "cpu"
        for route in ("/api/pair-jobs", "/api/analyze/anything"):
            status, error = request(server, "POST", route, {})
            assert status == 409, (route, error)
        assert not server.jobs
        workspace = server.get_pair_workspace()
        assert not workspace.jobs


def test_confirmation_survives_page_reads_but_requires_review_after_restart(tmp_path):
    root = tmp_path / "run"
    with running_server(root) as server:
        chosen = {**settings_for(server), "cpu_threads": 1, "parallel_views": 1, "frame_memory_mb": 64}
        status, saved = request(server, "POST", body=chosen)
        assert status == 200, saved
        assert saved["confirmed"] is True
        assert saved["settings"] == chosen
        assert request(server)[1]["confirmed"] is True
        assert (root / "resource-settings.json").is_file()
    with running_server(root) as restarted:
        status, snapshot = request(restarted)
        assert status == 200
        assert snapshot["settings"] == chosen
        assert snapshot["confirmed"] is False
        assert request(restarted, "POST", "/api/pair-jobs", {})[0] == 409


@pytest.mark.parametrize("change", [
    {"device": "cuda:0"}, {"cpu_threads": True}, {"cpu_threads": 0},
    {"cpu_threads": 1.5}, {"cpu_threads": 10 ** 6}, {"parallel_views": 3},
    {"frame_memory_mb": 0}, {"frame_memory_mb": "512"}, {"unexpected": 1},
])
def test_invalid_resource_choices_do_not_confirm_or_write(tmp_path, change):
    root = tmp_path / "run"
    with running_server(root) as server:
        original = settings_for(server)
        status, error = request(server, "POST", body={**original, **change})
        assert status == 400, error
        snapshot = request(server)[1]
        assert snapshot["settings"] == original
        assert snapshot["confirmed"] is False
        assert not (root / "resource-settings.json").exists()


@pytest.mark.parametrize("body", [[], None, {}, {"device": "cpu"}])
def test_missing_or_non_object_settings_are_rejected(tmp_path, body):
    with running_server(tmp_path / "run") as server:
        assert request(server, "POST", body=body)[0] == 400
        assert request(server)[1]["confirmed"] is False


def test_foreign_origin_cannot_change_local_machine_settings(tmp_path):
    with running_server(tmp_path / "run") as server:
        chosen = settings_for(server)
        status, _ = request(server, "POST", body=chosen, headers={"Origin": "https://other.example"})
        assert status == 403
        assert request(server)[1]["confirmed"] is False


@pytest.mark.parametrize("kind", ["legacy", "pair"])
@pytest.mark.parametrize("job_status", ["queued", "running"])
def test_active_jobs_lock_settings_without_changing_confirmed_choice(tmp_path, kind, job_status):
    with running_server(tmp_path / "run") as server:
        chosen = settings_for(server)
        assert request(server, "POST", body=chosen)[0] == 200
        owner = server if kind == "legacy" else server.get_pair_workspace()
        lock = server.job_lock if kind == "legacy" else owner.lock
        with lock:
            owner.jobs["in-flight"] = {"job_id": "in-flight", "status": job_status}
        assert request(server)[1]["busy"] is True
        status, _ = request(server, "POST", body={**chosen, "frame_memory_mb": 64})
        assert status == 409
        assert request(server)[1]["settings"] == chosen
        assert request(server)[1]["confirmed"] is True
        with lock:
            owner.jobs["in-flight"]["status"] = "completed"
        assert request(server)[1]["busy"] is False
        assert request(server, "POST", body={**chosen, "frame_memory_mb": 64})[0] == 200


@pytest.mark.parametrize("stored", ["not-json", '{"device":"cuda:99"}', "[]"])
def test_corrupt_or_stale_saved_choices_fall_back_without_confirming(tmp_path, stored):
    root = tmp_path / "run"
    root.mkdir()
    (root / "resource-settings.json").write_text(stored, encoding="utf-8")
    with running_server(root) as server:
        status, snapshot = request(server)
        assert status == 200
        assert snapshot["confirmed"] is False
        assert snapshot["settings"]["device"] == "cpu"
        assert snapshot["settings"]["cpu_threads"] >= 1


def test_submitted_job_persists_the_confirmed_resource_snapshot(tmp_path, monkeypatch):
    with running_server(tmp_path / "run") as server:
        chosen = {**settings_for(server), "cpu_threads": 1, "parallel_views": 1, "frame_memory_mb": 64}
        assert request(server, "POST", body=chosen)[0] == 200
        workspace = server.get_pair_workspace()
        calls = []
        monkeypatch.setattr(workspace.executor, "submit", lambda *args: calls.append(args))
        for camera in ("A", "B"):
            folder = workspace.root / "uploads" / camera
            folder.mkdir(parents=True)
            (folder / "upload.json").write_text(json.dumps({
                "upload_id": camera, "file": camera + ".mp4", "metadata": {"width": 640, "height": 360},
            }), encoding="utf-8")
        status, submitted = request(server, "POST", "/api/pair-jobs", {
            "views": {camera: {"upload_id": camera} for camera in ("A", "B")},
        })
        assert status == 202, submitted
        identifier = submitted["job_id"]
        assert len(calls) == 1
        assert workspace.jobs[identifier]["request"]["resource_settings"] == chosen
        saved = json.loads((workspace.root / "jobs" / identifier / "job.json").read_text(encoding="utf-8"))
        assert saved["request"]["resource_settings"] == chosen
        assert request(server, "POST", body={**chosen, "parallel_views": 2})[0] == 409
        assert saved["request"]["resource_settings"]["parallel_views"] == 1


def test_saved_settings_for_a_different_machine_are_not_reused(tmp_path):
    root = tmp_path / "run"
    with running_server(root) as server:
        chosen = {**settings_for(server), "cpu_threads": 1, "frame_memory_mb": 64}
        assert request(server, "POST", body=chosen)[0] == 200
    path = root / "resource-settings.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["fingerprint"] = "another-machine"
    saved["settings"]["device"] = "cuda:99"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with running_server(root) as restarted:
        snapshot = request(restarted)[1]
        assert snapshot["settings"] == snapshot["environment"]["defaults"]
        assert snapshot["confirmed"] is False


def test_failed_settings_save_rolls_back_runtime_and_preserves_confirmation(tmp_path, monkeypatch):
    import cv2
    from badminton.resources import decoder_threads
    root = tmp_path / "run"
    with running_server(root) as server:
        initial = request(server)[1]
        chosen = {**initial["settings"], "cpu_threads": 1, "parallel_views": 1, "frame_memory_mb": 64}
        assert request(server, "POST", body=chosen)[0] == 200
        persisted = (root / "resource-settings.json").read_bytes()
        replacement = {**chosen, "cpu_threads": min(2, initial["environment"]["limits"]["cpu_threads"]["max"]),
                       "parallel_views": 2}

        def fail_write(*args, **kwargs):
            # The attempted policy is active before persistence fails.
            assert cv2.getNumThreads() == replacement["cpu_threads"]
            assert decoder_threads() == replacement["cpu_threads"]
            raise OSError("disk full")

        monkeypatch.setattr("badminton.pair_workspace._write", fail_write)
        status, error = request(server, "POST", body=replacement)
        assert status == 500
        assert "저장" in error["error"]
        snapshot = request(server)[1]
        assert snapshot["confirmed"] is True
        assert snapshot["settings"] == chosen
        assert cv2.getNumThreads() == chosen["cpu_threads"]
        assert decoder_threads() == chosen["cpu_threads"]
        assert (root / "resource-settings.json").read_bytes() == persisted
