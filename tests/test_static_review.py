import http.client
import threading

import pytest

from badminton.static_review import create_static_review_server


@pytest.fixture
def server(tmp_path):
    (tmp_path/'preview.mp4').write_bytes(bytes(range(100)))
    (tmp_path/'index.html').write_text('<!doctype html><title>Review</title>')
    instance=create_static_review_server(tmp_path,port=0)
    thread=threading.Thread(target=instance.serve_forever,daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def request(server,path='/preview.mp4',method='GET',headers=None):
    connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
    try:
        connection.request(method,path,headers=headers or {})
        response=connection.getresponse()
        return response.status,response.headers,response.read()
    finally:
        connection.close()


def test_video_get_advertises_seek_ranges_and_preserves_bytes(server):
    status,headers,body=request(server)
    assert status==200 and body==bytes(range(100))
    assert headers['Accept-Ranges']=='bytes'
    assert headers['Content-Type']=='video/mp4'


@pytest.mark.parametrize('value,expected_range,expected_bytes',[
    ('bytes=20-29','bytes 20-29/100',bytes(range(20,30))),
    ('bytes=90-','bytes 90-99/100',bytes(range(90,100))),
    ('bytes=-7','bytes 93-99/100',bytes(range(93,100))),
    ('bytes=95-500','bytes 95-99/100',bytes(range(95,100))),
])
def test_video_partial_response_matches_exact_requested_bytes(server,value,expected_range,expected_bytes):
    status,headers,body=request(server,headers={'Range':value})
    assert status==206 and body==expected_bytes
    assert headers['Content-Range']==expected_range
    assert int(headers['Content-Length'])==len(expected_bytes)


@pytest.mark.parametrize('value',['bytes=100-','bytes=20-10','bytes=-0','bytes=','bytes=1-2,4-5'])
def test_unsatisfiable_or_unsupported_range_is_explicit(server,value):
    status,headers,body=request(server,headers={'Range':value})
    assert status==416 and body==b''
    assert headers['Content-Range']=='bytes */100'


def test_range_head_and_changed_if_range_validator(server):
    status,headers,body=request(server,method='HEAD',headers={'Range':'bytes=5-9'})
    assert status==206 and body==b''
    assert headers['Content-Length']=='5'
    status,headers,body=request(server,headers={'Range':'bytes=5-9','If-Range':'stale-validator'})
    assert status==200 and body==bytes(range(100))


def test_portable_report_index_still_loads(server):
    status,headers,body=request(server,path='/')
    assert status==200 and b'<title>Review</title>' in body
