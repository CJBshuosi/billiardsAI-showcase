"""Standard-library HTTP transport primitives for field-assistance handlers."""

from __future__ import annotations

import io
import json
import re
import selectors
import time
from http import HTTPStatus

_RANGE_PATTERN = re.compile(r"bytes=(\d*)-(\d*)\Z")


class _ClosingRequestReader(io.RawIOBase):
    """Poll cancellation without relying on Windows makefile shutdown/EOF."""

    def __init__(self, connection, closing):
        super().__init__()
        self.connection, self.closing = connection, closing

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.closed:
            raise ValueError("read of closed request")
        if not len(buffer):
            return 0
        timeout = self.connection.gettimeout()
        deadline = None if timeout is None else time.monotonic() + timeout
        with selectors.DefaultSelector() as readiness:
            readiness.register(self.connection, selectors.EVENT_READ)
            while True:
                if self.closing.is_set():
                    raise ConnectionAbortedError("field service is closing")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("request read timed out")
                ready = readiness.select(0.25 if remaining is None else min(0.25, remaining))
                if ready:
                    if self.closing.is_set():
                        raise ConnectionAbortedError("field service is closing")
                    received = self.connection.recv_into(buffer)
                    if self.closing.is_set():
                        raise ConnectionAbortedError("field service is closing")
                    return received


class FieldHTTPResponseMixin:
    """Byte responses and MP4 ranges shared by the existing field handlers."""

    def _send_bytes(
        self,
        status: int,
        payload: bytes,
        *,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Browsers routinely cancel an older poll while navigating or refreshing.
            return

    def _send_json(self, status: int, payload: object) -> None:
        self._send_bytes(
            status,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
            content_type="application/json; charset=utf-8",
        )

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _stream_media(self, source, size: int, *, continue_check=None) -> None:
        try:
            self._stream_media_response(source, size, continue_check=continue_check)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            return

    def _stream_media_response(self, source, size: int, *, continue_check=None) -> None:
        requested = self.headers.get("Range")
        if requested is None:
            self._send_media_headers(HTTPStatus.OK, size)
            remaining = size
        else:
            match = _RANGE_PATTERN.fullmatch(requested.strip())
            if match is None or not any(match.groups()):
                self._range_error(size)
                return
            start_text, end_text = match.groups()
            try:
                if start_text:
                    start = int(start_text)
                    end = size - 1 if not end_text else int(end_text)
                else:
                    suffix = int(end_text)
                    if suffix <= 0:
                        raise ValueError
                    start = max(0, size - suffix)
                    end = size - 1
                if start < 0 or start >= size or end < start:
                    raise ValueError
                end = min(end, size - 1)
            except ValueError:
                self._range_error(size)
                return
            source.seek(start)
            remaining = end - start + 1
            self._send_media_headers(
                HTTPStatus.PARTIAL_CONTENT,
                remaining,
                content_range=f"bytes {start}-{end}/{size}",
            )
        while remaining:
            if continue_check is not None and not continue_check():
                # Headers may already be sent. Close this response, never append a 404.
                self.close_connection = True
                return
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            self.wfile.write(chunk)
            remaining -= len(chunk)

    def _send_media_headers(
        self,
        status: int,
        content_length: int,
        *,
        content_range: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if content_range is not None:
            self.send_header("Content-Range", content_range)
        self.end_headers()

    def _range_error(self, size: int) -> None:
        self._send_bytes(
            HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
            b"",
            content_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes */{size}",
            },
        )
