from __future__ import annotations

import http.client
import importlib
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.lab.field_assistance import server

RESPONSE_METHODS = (
    "_send_bytes",
    "_send_json",
    "_error",
    "_stream_media",
    "_stream_media_response",
    "_send_media_headers",
    "_range_error",
)


def test_server_keeps_reader_alias_and_response_method_identity():
    transport = importlib.import_module("billiards_ai.lab.field_assistance.http_transport")
    assert server._ClosingRequestReader is transport._ClosingRequestReader
    assert server._RANGE_PATTERN is transport._RANGE_PATTERN
    for name in RESPONSE_METHODS:
        assert getattr(server._FieldAssistanceHandler, name) is getattr(
            transport.FieldHTTPResponseMixin, name
        )
        assert name not in server._FieldAssistanceHandler.__dict__


def test_transport_loads_without_server_or_product_dependencies(tmp_path):
    path = Path(server.__file__).with_name("http_transport.py")
    probe = """
import builtins
import importlib.util
import sys

original = builtins.__import__
def stdlib_only(name, globals=None, locals=None, fromlist=(), level=0):
    root = (globals or {}).get('__package__', '').split('.')[0] if level else name.split('.')[0]
    if root not in sys.stdlib_module_names:
        raise AssertionError('non-stdlib dependency: ' + name)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = stdlib_only
spec = importlib.util.spec_from_file_location('isolated_http_transport', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module._ClosingRequestReader.__module__ == 'isolated_http_transport'
assert module._RANGE_PATTERN.fullmatch('bytes=2-5')
assert module.FieldHTTPResponseMixin.__bases__ == (object,)
for name in ('__init__', 'setup', 'finish', 'do_GET', 'do_POST', 'server_close'):
    assert name not in module.FieldHTTPResponseMixin.__dict__
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe, str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@contextmanager
def running_handler(handler, artifact_path):
    state = SimpleNamespace(
        artifact_path=artifact_path,
        maintain_service=lambda: None,
        close_media_worker=lambda: None,
        close_projection=lambda: None,
        session_payload=lambda: {},
        shots_payload=lambda: {},
        report_payload=lambda: {},
    )
    service = server.FieldAssistanceHTTPServer(("127.0.0.1", 0), state)
    service.RequestHandlerClass = handler
    serving = threading.Thread(target=service.serve_forever, daemon=True)
    serving.start()
    try:
        yield service
    finally:
        service.shutdown()
        service.server_close()
        serving.join(2)
        assert not serving.is_alive()
        assert not service._request_connections


def request(service, method, path, *, body=None, headers=None):
    client = http.client.HTTPConnection(*service.server_address, timeout=3)
    try:
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        client.close()


def test_legacy_handler_subclass_can_override_and_inherit_transport(tmp_path):
    class ReadonlyHandler(server._FieldAssistanceHandler):
        def do_GET(self):
            if self.path == "/json":
                self._send_json(200, {"状态": "可播放"})
            else:
                super().do_GET()

        def do_POST(self):
            self._error(405, "review is read-only")

    with running_handler(ReadonlyHandler, tmp_path) as service:
        status, headers, body = request(service, "GET", "/json")
        assert status == 200
        assert body == '{"状态":"可播放"}'.encode()
        assert headers["Content-Length"] == str(len(body))
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert request(service, "GET", "/screen")[0] == 200
        assert request(service, "GET", "/missing")[0] == 404
        assert request(service, "POST", "/api/control", body=b"{}")[0] == 405


@pytest.mark.parametrize(
    "requested,status,body,content_range",
    [
        (None, 200, b"0123456789", None),
        ("bytes=2-5", 206, b"2345", "bytes 2-5/10"),
        ("bytes=8-", 206, b"89", "bytes 8-9/10"),
        ("bytes=-3", 206, b"789", "bytes 7-9/10"),
        ("bytes=8-99", 206, b"89", "bytes 8-9/10"),
        ("bytes=10-", 416, b"", "bytes */10"),
        ("bytes=-0", 416, b"", "bytes */10"),
        ("bytes=5-2", 416, b"", "bytes */10"),
        ("bytes=-", 416, b"", "bytes */10"),
        ("bytes=1-2,4-5", 416, b"", "bytes */10"),
    ],
)
def test_range_response_preserves_status_headers_and_bytes(
    tmp_path, requested, status, body, content_range
):
    (tmp_path / "presentation.mp4").write_bytes(b"0123456789")
    with running_handler(server._FieldAssistanceHandler, tmp_path) as service:
        actual_status, headers, actual_body = request(
            service,
            "GET",
            "/media/presentation.mp4",
            headers={} if requested is None else {"Range": requested},
        )
    assert actual_status == status
    assert actual_body == body
    assert headers["Content-Length"] == str(len(body))
    assert headers["Content-Type"] == "video/mp4"
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers.get("Content-Range") == content_range
