"""A copied run can reanalyse only its exact original local video contents."""
import hashlib
import http.client
import json
from pathlib import Path
import threading

import pytest

from badminton.web import RequestError, create_server


@pytest.fixture
def portable_review(tmp_path):
    run, video_dir = tmp_path / "copied-run", tmp_path / "local-video"
    run.mkdir()
    video_dir.mkdir()
    source = video_dir / "경기 A.mp4"
    source.write_bytes(b"original camera A recording")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    row = {"id": "v_a", "file": source.name, "sha256": digest,
           "source_path": "/unavailable-cloud-workspace/video/경기 A.mp4",
           "result_path": "videos/v_a/result.json"}
    result = {"video_id": "v_a", "source_sha256": digest, "config": {"analysis_width": 640}}
    (run / "videos/v_a").mkdir(parents=True)
    (run / "index.json").write_text(json.dumps({"videos": [row]}), encoding="utf-8")
    (run / row["result_path"]).write_text(json.dumps(result), encoding="utf-8")
    server = create_server(run, port=0, video_dir=video_dir)
    server.configure_resources(server.resources_status()["settings"])
    yield server, source, row, result
    server.server_close()


def test_copied_cloud_and_windows_paths_resolve_to_same_local_content(portable_review):
    server, source, row, result = portable_review
    assert server.resolve_original(row, result) == (source, result["source_sha256"])
    row["source_path"] = r"C:\old-PC\project\video\경기 A.mp4"
    assert server.resolve_original(row, result)[0] == source


def test_available_original_is_preferred_but_wrong_old_file_cannot_override_local_source(portable_review, tmp_path):
    server, source, row, result = portable_review
    original = tmp_path / "original.mp4"
    original.write_bytes(source.read_bytes())
    row["source_path"] = str(original)
    assert server.resolve_original(row, result)[0] == original
    original.write_bytes(b"different content at a formerly valid path")
    assert server.resolve_original(row, result)[0] == source


def test_missing_or_wrong_video_fails_before_reanalysis(portable_review):
    server, source, row, result = portable_review
    source.write_bytes(b"same filename but wrong recording")
    with pytest.raises(RequestError) as error:
        server.resolve_original(row, result)
    assert error.value.status == 409 and "SHA256" in error.value.message
    source.unlink()
    with pytest.raises(RequestError) as error:
        server.resolve_original(row, result)
    assert error.value.status == 404 and "--video-dir" in error.value.message
    assert server.jobs == {}


@pytest.mark.parametrize("name", ["../private.mp4", "/tmp/private.mp4", r"..\private.mp4",
                                 "C:/private.mp4", r"C:\private.mp4", "movie.mp4:stream"])
def test_fallback_rejects_posix_windows_traversal_and_absolute_paths(portable_review, name):
    server, source, row, result = portable_review
    row["file"] = name
    with pytest.raises(RequestError) as error:
        server.resolve_original(row, result)
    assert error.value.status == 403


def test_symlink_escape_is_rejected_even_if_content_hash_matches(portable_review, tmp_path):
    server, source, row, result = portable_review
    external = tmp_path / "external.mp4"
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)
    with pytest.raises(RequestError) as error:
        server.resolve_original(row, result)
    assert error.value.status == 403


def test_absent_or_conflicting_original_hashes_are_not_accepted(portable_review):
    server, source, row, result = portable_review
    row["sha256"] = "0" * 64
    with pytest.raises(RequestError) as error:
        server.resolve_original(row, result)
    assert error.value.status == 409
    row.pop("sha256")
    result.pop("source_sha256")
    with pytest.raises(RequestError) as error:
        server.resolve_original(row, result)
    assert error.value.status == 409


def _post(server):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    connection.request("POST", "/api/analyze/v_a", body=b"{}", headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    status, payload = response.status, json.loads(response.read())
    connection.close()
    return status, payload


def test_portable_reanalysis_queues_verified_local_source_and_rechecks_before_work(portable_review, monkeypatch):
    server, source, _, _ = portable_review
    callbacks, calls = [], []
    monkeypatch.setattr(server.executor, "submit", lambda callback: callbacks.append(callback))
    monkeypatch.setattr("badminton.pipeline.analyze_video", lambda *args, **kwargs: calls.append((args, kwargs)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = _post(server)
        assert status == 202 and len(callbacks) == 1
        callbacks.pop()()
        assert server.jobs[payload["job_id"]]["status"] == "completed"
        assert calls[0][0][0] == source
        assert calls[0][1]["config"]["analysis_width"] == 640
        status, payload = _post(server)
        assert status == 202
        source.write_bytes(b"changed while waiting in queue")
        callbacks.pop()()
        assert len(calls) == 1
        assert server.jobs[payload["job_id"]]["status"] == "failed"
        status, payload = _post(server)
        assert status == 409 and "SHA256" in payload["error"]
        assert len(callbacks) == 0
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_serve_cli_passes_local_video_directory(monkeypatch):
    from badminton.__main__ import main
    calls = []
    monkeypatch.setattr("badminton.web.serve", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr("sys.argv", ["badminton", "serve", "--run", "copied-run", "--video-dir", "local-originals"])
    assert main() == 0
    assert calls[0][0][0] == Path("copied-run")
    assert calls[0][1]["video_dir"] == Path("local-originals")


def test_reanalysis_retains_actionable_resource_budget_failure(portable_review, monkeypatch):
    from badminton.resources import ResourceSettingsError
    server, _, _, _ = portable_review
    callbacks = []
    monkeypatch.setattr(server.executor, "submit", lambda callback: callbacks.append(callback))
    message = "영상 프레임 버퍼에 96 MiB가 필요하지만 시점당 예산은 64 MiB입니다."

    def over_budget(*args, **kwargs):
        raise ResourceSettingsError(message)

    monkeypatch.setattr("badminton.pipeline.analyze_video", over_budget)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = _post(server)
        assert status == 202 and len(callbacks) == 1
        callbacks.pop()()
        job = server.jobs[payload["job_id"]]
        assert job["status"] == "failed"
        assert job["error"] == message
        assert server.resources_status()["busy"] is False
    finally:
        server.shutdown()
        thread.join(timeout=2)
