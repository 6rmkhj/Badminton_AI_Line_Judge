"""HTTP contract tests for paired input, separate from verdict-engine tests."""
from __future__ import annotations

import http.client
import json
import threading
from urllib.parse import quote

import pytest

from badminton.web import create_server


class FakePairWorkspace:
    """Expose adapter mistakes without running expensive video inference."""

    def __init__(self, directory):
        self.directory = directory
        self.calls = []
        self.closed = False
        (directory / 'preview.mp4').write_bytes(b'0123456789abcdef')

    def library(self):
        self.calls.append(('library',))
        return {'videos': [{'id': 'v_a', 'file': 'A.mp4'}, {'id': 'v_b', 'file': 'B.mp4'}]}

    def list_jobs(self):
        self.calls.append(('list_jobs',))
        return {'jobs': [{'job_id': 'saved', 'status': 'completed'}]}

    def get_job(self, identifier):
        from badminton.pair_workspace import PairWorkspaceError
        self.calls.append(('get_job', identifier))
        if identifier != 'saved':
            raise PairWorkspaceError(404, '작업을 찾을 수 없습니다.')
        return {'job_id': identifier, 'status': 'completed', 'result': {'decision': 'REVIEW'}}

    def submit(self, body):
        from badminton.pair_workspace import PairWorkspaceError
        self.calls.append(('submit', body))
        if body.get('force_busy'):
            raise PairWorkspaceError(409, '분석이 이미 진행 중입니다.')
        return {'job_id': 'new-job', 'status': 'queued'}

    def stage_upload(self, stream, length, filename):
        data = stream.read(length)
        self.calls.append(('stage_upload', length, filename, data))
        return {'upload_id': 'uploaded', 'file': filename, 'bytes': len(data)}

    def _asset(self, relative, scope):
        from badminton.pair_workspace import PairWorkspaceError
        self.calls.append((scope, relative))
        if relative != 'preview.mp4':
            raise PairWorkspaceError(403, '허용되지 않는 근거 경로입니다.')
        return self.directory / 'preview.mp4'

    def asset_path(self, relative):
        return self._asset(relative, 'asset_path')

    def review_asset_path(self, relative):
        return self._asset(relative, 'review_asset_path')

    def library_asset_path(self, relative):
        return self._asset(relative, 'library_asset_path')

    def close(self):
        self.closed = True


@pytest.fixture
def pair_server(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    (root / 'index.json').write_text(json.dumps({'videos': [], 'pairs': []}))
    fake = FakePairWorkspace(root)
    server = create_server(root, port=0)
    server.configure_resources(server.resources_status()['settings'])
    server.pair_workspace = fake
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, fake
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert fake.closed


def request(server, path, method='GET', body=None, headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=4)
    request_headers = dict(headers or {})
    if isinstance(body, dict):
        body = json.dumps(body).encode()
        request_headers.setdefault('Content-Type', 'application/json')
    try:
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        payload = response.read()
        response_headers = dict(response.getheaders())
        if payload and response_headers.get('Content-Type', '').startswith('application/json'):
            payload = json.loads(payload)
        return response.status, response_headers, payload
    finally:
        connection.close()


def upload_headers(filename='A.mp4'):
    return {'Content-Type': 'application/octet-stream', 'X-Filename': quote(filename, safe='')}


def test_pair_read_routes_return_existing_state_without_submitting_work(pair_server):
    server, fake = pair_server
    assert len(request(server, '/api/pair-library')[2]['videos']) == 2
    assert request(server, '/api/pair-jobs')[2]['jobs'][0]['job_id'] == 'saved'
    assert request(server, '/api/pair-jobs/saved')[2]['result']['decision'] == 'REVIEW'
    assert request(server, '/api/pair-jobs/saved', method='HEAD')[2] == b''
    assert request(server, '/api/pair-jobs/missing')[0] == 404
    assert request(server, '/api/pair-uploads')[0] == 404
    assert all(call[0] not in {'submit', 'stage_upload'} for call in fake.calls)


def test_pair_submission_preserves_both_sources_and_returns_job_handle(pair_server):
    server, fake = pair_server
    body = {'views': {'A': {'video_id': 'v_a'}, 'B': {'upload_id': 'upload-b'}},
            'same_event_confirmed': False, 'reuse_review': True}
    status, _, result = request(server, '/api/pair-jobs', 'POST', body)
    assert status == 202 and result == {'job_id': 'new-job', 'status': 'queued'}
    assert fake.calls == [('submit', body)]
    status, _, result = request(server, '/api/pair-jobs', 'POST', {'force_busy': True})
    assert status == 409 and '이미 진행' in result['error']


def test_pair_submission_preserves_optional_per_camera_region_without_rewriting_coordinates(pair_server):
    server, fake = pair_server
    body = {'views': {'A': {'upload_id':'upload-a', 'review_region':{'x':0,'y':.125,'width':1,'height':.75}},
                      'B': {'video_id':'v_b','review_region':None}},
            'reuse_review':False, 'refine_review_regions':True}
    status, _, result = request(server, '/api/pair-jobs', 'POST', body)
    assert status == 202 and result['job_id']=='new-job'
    assert fake.calls == [('submit', body)]


def test_raw_video_upload_stream_preserves_binary_bytes_and_unicode_name(pair_server):
    server, fake = pair_server
    payload = b'\x00\xff\x80binary\r\nvideo\x00'
    filename = 'A 시점 (원본) 100%.MP4'
    headers = upload_headers(filename)
    headers['Origin'] = f'http://127.0.0.1:{server.server_port}'
    status, _, body = request(server, '/api/pair-uploads', 'POST', payload, headers)
    assert status == 201 and body['file'] == filename
    assert fake.calls == [('stage_upload', len(payload), filename, payload)]


@pytest.mark.parametrize('extra,status', [
    ({'Content-Type': 'application/json'}, 415),
    ({'Origin': 'https://other.example'}, 403),
    ({'Sec-Fetch-Site': 'cross-site'}, 403),
    ({'Transfer-Encoding': 'chunked'}, 400),
    ({'Content-Encoding': 'gzip'}, 400),
    ({'Content-Length': '0'}, 413),
    ({'Content-Length': str(1024 * 1024 * 1024 + 1)}, 413),
    ({'Content-Length': '-1'}, 400),
    ({'Content-Length': 'NaN'}, 400),
])
def test_upload_rejected_before_backend_receives_body(pair_server, extra, status):
    server, fake = pair_server
    headers = {**upload_headers(), **extra}
    actual, _, payload = request(server, '/api/pair-uploads', 'POST', b'', headers)
    assert actual == status and isinstance(payload['error'], str)
    assert not fake.calls


@pytest.mark.parametrize('extra,status', [
    ({'Origin': 'https://other.example'}, 403),
    ({'Sec-Fetch-Site': 'cross-site'}, 403),
    ({'Content-Type': 'text/plain'}, 415),
])
def test_pair_job_writes_keep_existing_origin_and_json_guards(pair_server, extra, status):
    server, fake = pair_server
    assert request(server, '/api/pair-jobs', 'POST', {'views': {}}, extra)[0] == status
    assert not fake.calls


@pytest.mark.parametrize('prefix,method_name', [
    ('/pair-media/', 'asset_path'),
    ('/pair-review-assets/', 'review_asset_path'),
    ('/pair-library-assets/', 'library_asset_path'),
])
def test_pair_media_routes_honor_ranges_head_and_backend_path_denials(pair_server, prefix, method_name):
    server, fake = pair_server
    status, headers, body = request(server, prefix + 'preview.mp4', headers={'Range': 'bytes=2-5'})
    assert status == 206 and body == b'2345' and headers['Content-Range'] == 'bytes 2-5/16'
    status, headers, body = request(server, prefix + 'preview.mp4', method='HEAD')
    assert status == 200 and body == b'' and headers['Content-Length'] == '16'
    assert request(server, prefix + 'preview.mp4', headers={'Range': 'bytes=90-'})[0] == 416
    status, _, body = request(server, prefix + '%2e%2e/private.txt')
    assert status == 403 and body['error'] == '허용되지 않는 근거 경로입니다.'
    assert (method_name, '../private.txt') in fake.calls


def test_pair_responses_allow_local_upload_previews_and_preserve_security_headers(pair_server):
    server, _ = pair_server
    status, headers, _ = request(server, '/api/pair-library')
    assert status == 200
    assert "media-src 'self' blob:" in headers['Content-Security-Policy']
    assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
    assert headers['Cache-Control'] == 'no-store'
    assert headers['X-Content-Type-Options'] == 'nosniff'


def test_pair_workspace_page_and_assets_are_reachable_without_starting_analysis(pair_server):
    server, fake = pair_server
    for path in ('/pair', '/pair/', '/pair.html'):
        status, headers, body = request(server, path)
        assert status == 200 and 'text/html' in headers['Content-Type']
        assert b'id="pair-form"' in body and b'/pair.js' in body
    for path, content_type in (('/pair.js', 'javascript'), ('/pair.css', 'text/css')):
        status, headers, body = request(server, path)
        assert status == 200 and content_type in headers['Content-Type'] and body
    assert not fake.calls
