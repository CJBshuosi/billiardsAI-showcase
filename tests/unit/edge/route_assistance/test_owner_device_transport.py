"""Owned synthetic loopback HTTP; never contacts a configured cloud deployment."""

from __future__ import annotations

import importlib
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@contextmanager
def endpoint(mode, status=200):
    seen = []
    disconnected = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            seen.append(
                (
                    self.path,
                    dict(self.headers),
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
            )
            try:
                if callable(mode):
                    mode(self, seen)
                    return
                if mode == "headers":
                    time.sleep(0.8)
                self.send_response(status)
                if mode == "headerhuge":
                    self.send_header("X-Large", "a" * 17000)
                if mode == "redirect":
                    self.send_header("Location", "/followed")
                if mode == "huge":
                    self.send_header("Content-Length", "16385")
                self.end_headers()
                if mode == "drip":
                    for _ in range(80):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        time.sleep(0.04)
                else:
                    self.wfile.write(b"x" * (16385 if mode in ("huge", "chunkhuge") else 16384))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/sync", seen, disconnected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def transport(url, **kwargs):
    mod = importlib.import_module("billiards_ai.edge.route_assistance.owner_device_transport")
    return mod.OwnerDeviceMetadataHttpTransport(url, **kwargs)


@pytest.mark.parametrize("mode", ["headers", "drip"])
def test_absolute_deadline_kills_and_reaps_owned_network_process(mode):
    with endpoint(mode) as (url, seen, disconnected):
        tx = transport(url, timeout_seconds=0.45)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            tx.post_json("/v1/owner-media-deletions/claim", {}, b"{}")
        assert time.monotonic() - started < 0.8
        assert tx._active is None
        tx.close()
        assert len(seen) == 1
        if mode == "drip":
            assert disconnected.wait(1.5)


@pytest.mark.parametrize("status", [200, 503])
def test_oversized_success_and_error_are_bounded(status):
    with endpoint("huge", status) as (url, seen, _):
        tx = transport(url)
        with pytest.raises(ValueError, match="response"):
            tx.post_json("/v1/runtime-attestation", {}, b"{}")
        assert tx._active is None and len(seen) == 1
        tx.close()


def test_exact_limit_and_no_redirect_follow():
    with endpoint("normal") as (url, _, _):
        tx = transport(url)
        assert len(tx.post_json("/v1/runtime-attestation", {}, b"{}").body) == 16384
        tx.close()
    with endpoint("redirect", 302) as (url, seen, _):
        tx = transport(url)
        assert tx.post_json("/v1/runtime-attestation", {}, b"{}").status_code == 302
        assert len(seen) == 1
        tx.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://untrusted.example",
        "https://a.test/?x=y",
        "https://u:p@a.test",
        "https://a.test/#fragment",
    ],
)
def test_rejects_untrusted_or_ambiguous_origin(url):
    with pytest.raises(ValueError):
        transport(url)


def test_dns_equivalent_child_stall_is_killed_and_private_ipc_removed(monkeypatch):
    mod = importlib.import_module("billiards_ai.edge.route_assistance.owner_device_transport")
    actual = mod.subprocess.Popen
    processes, roots = [], []

    def spawn(argv, **kwargs):
        import os
        from pathlib import Path

        root = Path(argv[-1])
        roots.append(root)
        if os.name == "nt":
            from tests.support.windows_ipc_security import assert_private_windows_ipc

            evidence = assert_private_windows_ipc(root, root / "request")
            if destination := os.environ.get("RAYTHEON_TEST_IPC_ACL_EVIDENCE"):
                import json

                evidence_path = Path(destination)
                assert evidence_path.is_absolute(), "ACL evidence destination must be absolute"
                # The packet opts in with a fresh owned evidence path. Exclusive
                # creation preserves existing evidence and never reads IPC data.
                with evidence_path.open("x", encoding="utf-8") as stream:
                    json.dump(evidence, stream, indent=2)
        else:
            assert root.stat().st_mode & 0o777 == 0o700
            assert (root / "request").stat().st_mode & 0o777 == 0o600
        assert set(kwargs["env"]) <= {"SystemRoot", "WINDIR"}
        assert "synthetic-auth" not in str(argv)
        process = actual(argv, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(mod.subprocess, "Popen", spawn)
    monkeypatch.setattr(mod, "_CHILD", "import time; time.sleep(30)")
    tx = transport("http://127.0.0.1:1", timeout_seconds=0.2)
    with pytest.raises(TimeoutError):
        tx.post_json("/v1/runtime-attestation", {"x-signature": "synthetic-auth"}, b"{}")
    assert processes[0].poll() is not None
    assert tx._active is None and not roots[0].exists()
    tx.close()


def test_unconfirmed_kill_retains_child_and_ipc_and_blocks_replacement(monkeypatch):
    import subprocess
    from pathlib import Path

    mod = importlib.import_module("billiards_ai.edge.route_assistance.owner_device_transport")

    class Child:
        exited = False

        def poll(self):
            return 0 if self.exited else None

        def wait(self, timeout):
            if not self.exited:
                raise subprocess.TimeoutExpired("owned", timeout)
            return 0

        def kill(self):
            pass

    child = Child()
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **kw: child)
    tx = transport("http://127.0.0.1:1", timeout_seconds=0.2)
    with pytest.raises(mod.OwnerDeviceTransportOwnershipError):
        tx.post_json("/v1/runtime-attestation", {}, b"{}")
    root = Path(tx._temporary.name)
    assert root.exists() and tx._active is child
    with pytest.raises(mod.OwnerDeviceTransportOwnershipError):
        tx.post_json("/v1/runtime-attestation", {}, b"{}")
    with pytest.raises(mod.OwnerDeviceTransportOwnershipError):
        tx.close()
    assert root.exists() and tx._active is child
    child.exited = True
    tx.close()
    assert not root.exists() and tx._active is None


@pytest.mark.parametrize("mode", ["headerhuge", "chunkhuge"])
def test_unannounced_oversized_body_and_aggregate_headers_rejected(mode):
    with endpoint(mode) as (url, seen, _):
        tx = transport(url)
        with pytest.raises((ValueError, OSError)):
            tx.post_json("/v1/runtime-attestation", {}, b"{}")
        assert len(seen) == 1 and tx._active is None
        tx.close()


@pytest.mark.parametrize(
    "path,headers,body",
    [
        ("/v1/runtime-attestation", {}, b"x" * 16385),
        ("/v1/" + "x" * 257, {}, b"{}"),
        ("/v1/runtime-attestation", {"x-signature": "x" * 1025}, b"{}"),
    ],
)
def test_request_bounds_reject_before_child_creation(monkeypatch, path, headers, body):
    mod = importlib.import_module("billiards_ai.edge.route_assistance.owner_device_transport")
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not spawn"))
    tx = transport("http://127.0.0.1:1")
    with pytest.raises(ValueError):
        tx.post_json(path, headers, body)
    assert tx._temporary is None and tx._active is None
    tx.close()


def test_ipc_cleanup_failure_releases_lock_and_retains_resources_for_close_retry(monkeypatch):
    from pathlib import Path

    mod = importlib.import_module("billiards_ai.edge.route_assistance.owner_device_transport")
    tx = transport("http://127.0.0.1:1")
    saved = []

    class Exited:
        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    def spawn(argv, **kwargs):
        root = Path(argv[-1])
        (root / "response").write_bytes((200).to_bytes(4, "big") + b"{}")
        saved.append((root, tx._temporary.cleanup))

        def fail():
            raise OSError("synthetic private IPC cleanup failure")

        monkeypatch.setattr(tx._temporary, "cleanup", fail)
        return Exited()

    monkeypatch.setattr(mod.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="synthetic"):
        tx.post_json("/v1/runtime-attestation", {}, b"{}")
    root, cleanup = saved[0]
    assert root.exists() and tx._temporary is not None and tx._active is not None
    try:
        assert not tx._lock.locked()
        with pytest.raises((OSError, mod.OwnerDeviceTransportOwnershipError)):
            tx.post_json("/v1/runtime-attestation", {}, b"{}")
        assert len(saved) == 1 and root.exists() and tx._temporary is not None
        assert not tx._lock.locked()
    finally:
        monkeypatch.setattr(tx._temporary, "cleanup", cleanup)
        if tx._lock.locked():
            tx._lock.release()  # Own failed-RED fixture must still release its temp files.
        tx.close()
    assert not root.exists() and tx._active is None


def test_no_child_ipc_cleanup_failure_blocks_replacement_and_retries_close(monkeypatch):
    from pathlib import Path

    mod = importlib.import_module("billiards_ai.edge.route_assistance.owner_device_transport")
    tx = transport("http://127.0.0.1:1")
    saved = []

    def spawn(argv, **kwargs):
        temporary = tx._temporary
        saved.append((temporary, temporary.cleanup, Path(argv[-1])))

        def fail_cleanup():
            raise OSError("synthetic retained IPC cleanup failure")

        monkeypatch.setattr(temporary, "cleanup", fail_cleanup)
        raise OSError("synthetic child creation failure")

    monkeypatch.setattr(mod.subprocess, "Popen", spawn)
    try:
        with pytest.raises(OSError, match="cleanup failure"):
            tx.post_json("/v1/runtime-attestation", {}, b"{}")
        temporary, cleanup, root = saved[0]
        assert tx._active is None and tx._temporary is temporary and root.exists()
        assert not tx._lock.locked()
        with pytest.raises((OSError, mod.OwnerDeviceTransportOwnershipError)):
            tx.post_json("/v1/runtime-attestation", {}, b"{}")
        replaced = len(saved) != 1 or tx._temporary is not temporary
        try:
            tx.close()
        except OSError:
            close_failed = True
        else:
            close_failed = False
        assert (replaced, close_failed) == (False, True)
        assert root.exists() and tx._temporary is temporary and tx._active is None
        assert not tx._lock.locked()
        monkeypatch.setattr(temporary, "cleanup", cleanup)
        tx.close()
        assert not root.exists() and tx._temporary is None and tx._active is None
    finally:
        # Both old and replacement IPC directories must be removed after RED too.
        for temporary, cleanup, _ in saved:
            monkeypatch.setattr(temporary, "cleanup", cleanup)
            cleanup()
