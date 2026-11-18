"""Loopback-only server for exact ball-identity review."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from billiards_ai.lab.training.identity_review import IdentityReviewError, IdentityReviewSession
from billiards_ai.lab.training.identity_review_page import render_identity_review_page


class IdentityReviewHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        session: IdentityReviewSession,
        export_root: Path,
    ) -> None:
        if address[0] not in {"127.0.0.1", "localhost"}:
            raise ValueError("identity review server must bind to loopback")
        self.session = session
        self.export_root = export_root.resolve()
        self.media: dict[str, bytes] = {}
        for crop_id, record in session._crop_records.items():
            relative_path = record.get("relative_path")
            if not isinstance(relative_path, str):
                raise IdentityReviewError("candidate crop media is invalid")
            path = session.candidate_artifact / relative_path
            if not path.is_file() or path.is_symlink():
                raise IdentityReviewError("candidate crop media is missing or unsafe")
            self.media[crop_id] = path.read_bytes()
        super().__init__(address, IdentityReviewRequestHandler)

    def session_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.session.candidate_id,
            "crop_ids": list(self.session.crop_ids),
            "decisions": {
                crop_id: decision.model_dump(mode="json")
                for crop_id, decision in sorted(self.session.decisions.items())
            },
            "suggestions": {
                crop_id: decision.model_dump(mode="json")
                for crop_id, decision in sorted(self.session.suggestions.items())
            },
            "contexts": dict(sorted(self.session.contexts.items())),
            "reviewed_count": self.session.reviewed_count,
            "total_count": len(self.session.crop_ids),
            "revision": self.session.revision,
        }


class IdentityReviewRequestHandler(BaseHTTPRequestHandler):
    server: IdentityReviewHttpServer

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'; img-src 'self' data:",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _read_json(self) -> dict[str, object]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise IdentityReviewError("invalid request length") from exc
        if length <= 0 or length > 64 * 1024:
            raise IdentityReviewError("invalid request length")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise IdentityReviewError("invalid JSON request") from exc
        if not isinstance(value, dict):
            raise IdentityReviewError("invalid JSON request")
        return value

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(
                HTTPStatus.OK,
                render_identity_review_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/session":
            self._json(HTTPStatus.OK, self.server.session_payload())
            return
        if path.startswith("/media/"):
            name = unquote(path.removeprefix("/media/")).removesuffix(".png")
            body = self.server.media.get(name)
            if body is None or "/" in name or "\\" in name:
                self._error(HTTPStatus.NOT_FOUND, "media not found")
                return
            self._send(HTTPStatus.OK, body, "image/png")
            return
        self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/decision":
                payload = self._read_json()
                self.server.session.record(
                    crop_id=str(payload.get("crop_id") or ""),
                    group=str(payload.get("group") or ""),
                    exact_identity=(
                        str(payload["exact_identity"])
                        if payload.get("exact_identity") is not None
                        else None
                    ),
                    exact_evidence=(
                        str(payload["exact_evidence"])
                        if payload.get("exact_evidence") is not None
                        else None
                    ),
                    visibility=str(payload.get("visibility") or ""),
                    orientation=str(payload.get("orientation") or ""),
                    note=payload.get("note") if isinstance(payload.get("note"), str) else None,
                )
                self._json(HTTPStatus.OK, self.server.session_payload())
                return
            if path == "/api/export":
                verdict = self.server.session.export_verdict(self.server.export_root)
                self._json(HTTPStatus.OK, {"path": str(verdict)})
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except IdentityReviewError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))


def make_identity_review_server(
    *,
    session: IdentityReviewSession,
    export_root: Path,
    port: int,
    host: str = "127.0.0.1",
) -> IdentityReviewHttpServer:
    return IdentityReviewHttpServer(
        (host, port),
        session=session,
        export_root=export_root,
    )
