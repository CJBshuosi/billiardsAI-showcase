"""Loopback-only HTTP server for the billiards annotation page."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from pydantic import ValidationError

from billiards_ai.lab.annotation_review import (
    FrameDraft,
    ReviewConflictError,
    ReviewError,
    ReviewStore,
    build_review_session,
    export_reviewed_dataset,
)
from billiards_ai.lab.annotation_review_page import render_annotation_page


class ReviewHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        *,
        store: ReviewStore,
        review_root: Path,
        export_root: Path,
    ) -> None:
        super().__init__(address, ReviewRequestHandler)
        self.store = store
        self.review_root = review_root.resolve()
        self.export_root = export_root
        self.write_lock = threading.Lock()


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server: ReviewHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, object]:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(size))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ReviewError("请求内容不是有效 JSON") from exc
        if not isinstance(value, dict):
            raise ReviewError("请求内容必须是 JSON 对象")
        return value

    def _frame_id(self, path: str) -> str:
        value = path.rsplit("/", maxsplit=1)[-1].removesuffix(".jpg")
        if value not in {frame.frame_id for frame in self.server.store.manifest.frames}:
            raise ReviewError("未知审查帧")
        return value

    def do_GET(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            if path == "/":
                self._bytes(render_annotation_page().encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/session":
                frames = self.server.store.manifest.frames
                completed = sum(
                    self.server.store.load_draft(frame.frame_id).whole_frame_complete
                    for frame in frames
                )
                self._json(
                    {
                        "session_id": self.server.store.manifest.session_id,
                        "completed_count": completed,
                        "frames": [frame.model_dump(mode="json") for frame in frames],
                    }
                )
                return
            if path.startswith("/api/frame/"):
                draft = self.server.store.load_draft(self._frame_id(path))
                self._json(draft.model_dump(mode="json"))
                return
            if path.startswith("/media/"):
                frame_id = self._frame_id(path)
                frame = next(
                    item for item in self.server.store.manifest.frames if item.frame_id == frame_id
                )
                source = (self.server.review_root / frame.source_path).resolve()
                if not source.is_relative_to(self.server.review_root) or not source.is_file():
                    raise ReviewError("源图片不存在")
                self._bytes(source.read_bytes(), "image/jpeg")
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except ReviewError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_PUT(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            if not path.startswith("/api/frame/"):
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            frame_id = self._frame_id(path)
            body = self._body()
            draft = FrameDraft.model_validate(body.get("draft"))
            if draft.frame_id != frame_id:
                raise ReviewError("帧 ID 不匹配")
            expected_revision = int(body.get("expected_revision", -1))
            with self.server.write_lock:
                saved = self.server.store.save_draft(
                    draft, expected_revision=expected_revision, actor="Boyu"
                )
            self._json(saved.model_dump(mode="json"))
        except ReviewConflictError as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except (ReviewError, ValidationError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if urlparse(self.path).path != "/api/export":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            with self.server.write_lock:
                exported = export_reviewed_dataset(
                    store=self.server.store,
                    review_root=self.server.review_root,
                    output_root=self.server.export_root,
                )
            self._json(exported.model_dump(mode="json"))
        except ReviewError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def make_server(
    *, session_path: Path, review_root: Path, export_root: Path, port: int
) -> ReviewHTTPServer:
    return ReviewHTTPServer(
        ("127.0.0.1", port),
        store=ReviewStore(session_path),
        review_root=review_root,
        export_root=export_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local billiards annotation page")
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--sparse-run", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    session = build_review_session(
        review_root=args.review_root,
        sparse_run=args.sparse_run,
        output_root=args.artifact_root / "sessions",
        high_confidence=0.3,
        zero_candidate_limit=10,
        selection_mode="all",
    )
    server = make_server(
        session_path=session.path,
        review_root=args.review_root,
        export_root=args.artifact_root / "exports",
        port=args.port,
    )
    host, port = server.server_address
    print(f"ANNOTATION_REVIEW_URL=http://{host}:{port}", flush=True)
    print(f"SESSION={session.path} FRAMES={session.frame_count}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
