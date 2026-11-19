"""Content-free request summaries; no provider or workflow instrumentation."""

import json
import logging
import math
import os
import stat
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

CAPTURE_LIMIT = 256 * 1024
MAX_TOKENS = 2**53 - 1
STATUSES = frozenset(
    {
        "answered",
        "partial",
        "clarification",
        "unsupported",
        "insufficient_evidence",
        "timeout",
        "error",
        "limit_reached",
    }
)


def diagnostic():
    try:
        sys.stderr.write("OBSERVATION_WRITE_FAILED\n")
        sys.stderr.flush()
    except Exception:
        pass


@dataclass(frozen=True)
class RequestObservation:
    schema_version: int
    timestamp: str
    request_id: str
    route: str
    http_status: int | None
    mode: str | None
    business_status: str | None
    duration_ms: float
    tool_count: int | None
    input_tokens: int | None
    output_tokens: int | None
    usage_complete: bool


def token(value):
    return value if type(value) is int and 0 <= value <= MAX_TOKENS else None


def summary(raw):
    """Select metadata only; never return or retain content fields."""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        value = data.get("request_id")
        try:
            request_id = str(UUID(value)) if isinstance(value, str) else None
        except (ValueError, AttributeError):
            request_id = None
        mode = data.get("mode")
        status = data.get("status")
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        trace = data.get("tool_trace")
        return {
            "request_id": request_id,
            "mode": mode
            if isinstance(mode, str) and mode in ("offline", "live")
            else None,
            "business_status": status
            if isinstance(status, str) and status in STATUSES
            else None,
            "input_tokens": token(usage.get("input_tokens")),
            "output_tokens": token(usage.get("output_tokens")),
            "tool_count": len(trace)
            if isinstance(trace, list) and len(trace) <= 4
            else None,
        }
    except (ValueError, TypeError, RecursionError):
        return {}


class RequestObserver:
    """Forward the original ASGI messages, observing only bounded ask JSON."""

    def __init__(self, app, sink):
        self.app = app
        self.sink = sink

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        ask_routes = {
            "/api/ask",
            "/api/rules/ask",
            "/api/review/ask",
            "/api/assistant/ask",
        }
        route = (
            scope.get("path")
            if scope.get("method") == "POST" and scope.get("path") in ask_routes
            else "other"
        )
        started = time.monotonic()
        status = None
        capturing = False
        complete = False
        raw = bytearray()

        async def observe_send(message):
            nonlocal status, capturing, complete
            kind = message.get("type")
            if kind == "http.response.start":
                value = message.get("status")
                status = value if type(value) is int and 100 <= value <= 599 else None
                headers = {k.lower(): v.lower() for k, v in message.get("headers", ())}
                media_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip()
                capturing = (
                    route in ask_routes
                    and status == 200
                    and media_type == b"application/json"
                    and headers.get(b"content-encoding", b"identity") == b"identity"
                )
            elif kind == "http.response.body" and capturing:
                chunk = message.get("body", b"")
                if len(raw) + len(chunk) > CAPTURE_LIMIT:
                    raw.clear()
                    capturing = False
                else:
                    raw.extend(chunk)
            # Forward exactly, without waiting for the final body to reach the client.
            await send(message)
            if kind == "http.response.body" and not message.get("more_body", False):
                complete = True

        try:
            await self.app(scope, receive, observe_send)
        finally:
            info = summary(raw) if capturing and complete else {}
            raw.clear()
            duration = (time.monotonic() - started) * 1000
            record = RequestObservation(
                schema_version=1,
                timestamp=datetime.now(UTC).isoformat(),
                request_id=info.get("request_id") or str(uuid4()),
                route=route,
                http_status=status,
                mode=info.get("mode"),
                business_status=info.get("business_status"),
                duration_ms=round(max(0, duration), 3)
                if math.isfinite(duration)
                else 0,
                tool_count=info.get("tool_count"),
                input_tokens=info.get("input_tokens"),
                output_tokens=info.get("output_tokens"),
                usage_complete=info.get("input_tokens") is not None
                and info.get("output_tokens") is not None,
            )
            try:
                self.sink.write(record)
            except Exception:
                diagnostic()


class ObservationSink:
    """Single-process bounded JSONL, optional; failures always fall back safely."""

    def __init__(self, *, stream=None, file_path=None, max_bytes=1024 * 1024):
        self.stream = sys.stdout if stream is None else stream
        self.file_path = Path(file_path) if file_path else None
        self.max_bytes = min(max_bytes, 1024 * 1024)
        self.lock = threading.Lock()
        if self.file_path is not None:
            try:
                self._prepare()
            except Exception:
                self.file_path = None
                diagnostic()

    def _prepare(self):
        path = self.file_path
        if path.parent.resolve() != path.parent.absolute():
            raise ValueError("UNSAFE_OBSERVATION_DIRECTORY")
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        info = path.parent.stat()
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("UNSAFE_OBSERVATION_DIRECTORY")
        for candidate in (path, Path(str(path) + ".1"), Path(str(path) + ".2")):
            if candidate.is_symlink():
                raise ValueError("UNSAFE_OBSERVATION_FILE")
            if candidate.exists():
                info = candidate.stat()
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o600
                ):
                    raise ValueError("UNSAFE_OBSERVATION_FILE")

    def _open_log(self):
        fd = os.open(
            self.file_path,
            os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
            0o600,
        )
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise ValueError("UNSAFE_OBSERVATION_FILE")
            return fd
        except Exception:
            os.close(fd)
            raise

    def _append(self, encoded):
        self._prepare()
        path = self.file_path
        fd = self._open_log()
        try:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                # A prior partial write is not a complete JSONL record.
                # Bound recovery reads to the configured file-size budget.
                start = max(0, size - self.max_bytes)
                tail = os.pread(fd, size - start, start)
                newline = tail.rfind(b"\n")
                size = start + newline + 1 if newline >= 0 else 0
                os.ftruncate(fd, size)
            if size + len(encoded) > self.max_bytes:
                os.close(fd)
                fd = None
                last = Path(str(path) + ".2")
                if last.exists():
                    last.unlink()
                previous = Path(str(path) + ".1")
                if previous.exists():
                    previous.replace(last)
                path.replace(previous)
                fd = self._open_log()
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("OBSERVATION_WRITE_FAILED")
                view = view[written:]
        finally:
            if fd is not None:
                os.close(fd)

    def write(self, record: RequestObservation):
        # All callers supply the immutable projection produced above.
        line = (
            json.dumps(
                asdict(record),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        with self.lock:
            try:
                self.stream.write(line)
                self.stream.flush()
            except Exception:
                diagnostic()
            if self.file_path is not None:
                try:
                    if len(line.encode()) > self.max_bytes:
                        raise ValueError("OBSERVATION_TOO_LARGE")
                    self._append(line.encode())
                except Exception:
                    self.file_path = None
                    diagnostic()


class SafeDiagnosticFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(
            {
                "event": "RUNTIME_DIAGNOSTIC",
                "timestamp": datetime.now(UTC).isoformat(),
                "level": record.levelname
                if record.levelno in (10, 20, 30, 40, 50)
                else "UNKNOWN",
            }
        )


def configure_safe_logging():
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(SafeDiagnosticFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    for name in ("uvicorn", "uvicorn.error", "httpx", "httpcore", "sqlalchemy"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True
