"""Owned killable metadata HTTP; generic media transports stay unchanged.

The child owns DNS/socket/TLS/HTTP. Only signed headers and metadata cross private
IPC files; the long-term signing secret stays in the parent. Neither argv nor
logs contain protocol content. Unconfirmed child exit retains all resources.
"""

from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from billiards_ai.edge.route_assistance.sync import HttpResult

_CHILD = r"""
import base64, http.client, json, os, struct, sys, urllib.error, urllib.request
# Separate framing bound: status <=4KiB, aggregate header lines <=12KiB.
http.client._MAXLINE = 4096
def bounded_headers(fp):
    lines, count = [], 0
    while True:
        line = fp.readline(min(4097, 12289 - count))
        count += len(line)
        if len(line) > 4096 or count > 12288 or len(lines) >= 32:
            raise ValueError("invalid response headers")
        lines.append(line)
        if line in (b"\r\n", b"\n", b""):
            return lines
http.client._read_headers = bounded_headers
root = sys.argv[1]
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None
try:
    with open(os.path.join(root, 'request'), 'rb') as f:
        raw = f.read(32769)
    if len(raw) > 32768:
        sys.exit(3)
    p = json.loads(raw)
    request = urllib.request.Request(p['url'], data=base64.b64decode(p['body']),
                                     headers=p['headers'], method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        response = opener.open(request, timeout=p['timeout'])
    except urllib.error.HTTPError as error:
        response = error
    with response:
        length = response.headers.get('Content-Length')
        if length is not None and int(length) > p['max_bytes']:
            sys.exit(3)
        fd = os.open(os.path.join(root, 'response'), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(struct.pack('!I', response.code))
            count = 0
            while count < p['max_bytes']:
                chunk = response.read1(min(4096, p['max_bytes'] - count))
                if not chunk:
                    break
                f.write(chunk)
                count += len(chunk)
            if count == p['max_bytes'] and response.read1(1):
                sys.exit(3)
            if length is not None and count != int(length):
                sys.exit(2)
except (TimeoutError,):
    sys.exit(4)
except Exception:
    sys.exit(2)
"""


class OwnerDeviceTransportOwnershipError(RuntimeError):
    """Child exit remains unconfirmed; retain transport and temporary resources."""


class OwnerDeviceMetadataHttpTransport:
    def __init__(self, base_url, timeout_seconds=20, max_bytes=16384):
        parsed = urlsplit(base_url)
        if (
            not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or len(base_url) > 2048
            or (
                parsed.scheme != "https"
                and not (
                    parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
                )
            )
        ):
            raise ValueError("invalid owner metadata origin")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 20
            or type(max_bytes) is not int
            or not 0 < max_bytes <= 16384
        ):
            raise ValueError("invalid owner metadata bounds")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_bytes = max_bytes
        self._active = None
        self._temporary = None
        self._lock = threading.Lock()
        self._closed = False

    def _release(self):
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._active = None

    def _kill_and_reap(self, timeout):
        process = self._active
        if process is None:
            # IPC ownership exists independently of successful child creation.
            self._release()
            return
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=max(0, timeout))
        except subprocess.TimeoutExpired:
            raise OwnerDeviceTransportOwnershipError(
                "owner metadata child exit unconfirmed"
            ) from None
        self._release()

    def close(self):
        if not self._lock.acquire(timeout=1):
            raise OwnerDeviceTransportOwnershipError("owner metadata operation still active")
        try:
            self._closed = True
            self._kill_and_reap(1)
        finally:
            self._lock.release()

    def post_json(self, path, headers, body):
        if (
            not isinstance(path, str)
            or len(path) > 256
            or not path.startswith("/v1/")
            or "?" in path
            or "#" in path
            or ".." in path
            or not isinstance(body, bytes)
            or len(body) > self._max_bytes
        ):
            raise ValueError("invalid owner metadata request")
        if (
            len(headers) > 16
            or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or len(key) > 128
                or len(value) > 1024
                for key, value in headers.items()
            )
            or sum(len(key) + len(value) for key, value in headers.items()) > 8192
        ):
            raise ValueError("invalid owner metadata request")
        if not self._lock.acquire(blocking=False):
            raise OwnerDeviceTransportOwnershipError("owner metadata operation still active")
        deadline = time.monotonic() + self._timeout
        reserve = min(0.1, self._timeout / 4)
        try:
            if self._closed or self._active is not None or self._temporary is not None:
                raise OwnerDeviceTransportOwnershipError("owner metadata ownership unavailable")
            payload = json.dumps(
                dict(
                    url=self._base_url + path,
                    headers=dict(headers),
                    body=base64.b64encode(body).decode("ascii"),
                    timeout=self._timeout,
                    max_bytes=self._max_bytes,
                )
            ).encode("ascii")
            if len(payload) > 32768:
                raise ValueError("invalid owner metadata request")
            self._temporary = tempfile.TemporaryDirectory(prefix="owner-device-http-")
            root = Path(self._temporary.name)
            fd = os.open(root / "request", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as target:
                target.write(payload)
            environment = {k: os.environ[k] for k in ("SystemRoot", "WINDIR") if k in os.environ}
            self._active = subprocess.Popen(
                [sys.executable, "-I", "-c", _CHILD, str(root)],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                code = self._active.wait(timeout=max(0, deadline - time.monotonic() - reserve))
            except subprocess.TimeoutExpired:
                self._kill_and_reap(deadline - time.monotonic())
                raise TimeoutError("owner metadata deadline exceeded") from None
            if time.monotonic() >= deadline:
                raise TimeoutError("owner metadata deadline exceeded")
            if code == 3:
                raise ValueError("invalid owner metadata response size")
            if code == 4:
                raise TimeoutError("owner metadata deadline exceeded")
            if code != 0:
                raise OSError("owner metadata unavailable")
            with (root / "response").open("rb") as source:
                status = source.read(4)
                data = source.read(self._max_bytes)
                if len(status) != 4 or source.read(1):
                    raise ValueError("invalid owner metadata response")
            if time.monotonic() >= deadline:
                raise TimeoutError("owner metadata deadline exceeded")
            return HttpResult(int.from_bytes(status, "big"), data)
        finally:
            try:
                if self._active is None or self._active.poll() is not None:
                    self._release()
            finally:
                # Failed IPC cleanup retains ownership, but must permit close retry.
                self._lock.release()
