"""In-memory CloudBase boundary used by the offline end-to-end sync gate."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.sync import (
    FieldSyncClient,
    HttpResult,
    SyncMediaItem,
)
from billiards_ai.shared.models.miniprogram import (
    EdgeSyncEnvelopeV1,
    MiniProgramClipV1,
    canonical_sync_hash,
)


@dataclass(frozen=True, slots=True)
class FixtureSyncResult:
    session_id: str
    clips: tuple[MiniProgramClipV1, ...]
    duplicate_session_count: int
    sync_status: str


class _Credentials:
    def device_id(self) -> str:
        return "fixture-device"

    def secret(self) -> bytes:
        return b"fixture-secret"


class _InMemorySyncTransport:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, object]] = {}
        self.objects: dict[str, str] = {}
        self.duplicate_session_count = 0

    @staticmethod
    def _verify(headers: dict[str, str], path: str, body: bytes) -> None:
        body_hash = hashlib.sha256(body).hexdigest()
        assert headers["x-body-sha256"] == body_hash
        canonical = (
            f"POST\n{path}\n{headers['x-timestamp']}\n{headers['x-nonce']}\n{body_hash}"
        ).encode()
        expected = hmac.new(b"fixture-secret", canonical, hashlib.sha256).hexdigest()
        assert hmac.compare_digest(headers["x-signature"], expected)

    def post_json(self, path: str, headers: dict[str, str], body: bytes) -> HttpResult:
        self._verify(headers, path, body)
        if path == "/v1/sync/jobs":
            envelope = EdgeSyncEnvelopeV1.model_validate_json(body)
            key = headers["x-idempotency-key"]
            manifest_hash = headers["x-manifest-sha256"]
            assert manifest_hash == canonical_sync_hash(envelope)
            existing = self.jobs.get(key)
            if existing is None:
                self.jobs[key] = {
                    "job_id": "fixture-cloud-job",
                    "envelope": envelope,
                    "manifest_hash": manifest_hash,
                }
            else:
                assert existing["manifest_hash"] == manifest_hash
            return HttpResult(200, b'{"data":{"job_id":"fixture-cloud-job"}}')
        if path.endswith("/media-authorizations"):
            clip_id = str(json.loads(body)["clip_id"])
            return HttpResult(
                200,
                json.dumps(
                    {
                        "data": {
                            "method": "PUT",
                            "url": f"https://fixture.invalid/{clip_id}",
                            "headers": {"authorization": "fixture-upload-signature"},
                        }
                    },
                    separators=(",", ":"),
                ).encode(),
            )
        if path.endswith("/complete"):
            job = next(iter(self.jobs.values()))
            envelope = job["envelope"]
            assert isinstance(envelope, EdgeSyncEnvelopeV1)
            assert {clip.content_sha256 for clip in envelope.media} == set(self.objects.values())
            return HttpResult(200, b'{"data":{"status":"completed"}}')
        return HttpResult(404, b'{"error":{"code":"SYNC_INVALID"}}')

    def put_file(
        self, url: str, path: Path, content_type: str, headers: dict[str, str]
    ) -> HttpResult:
        assert url.startswith("https://fixture.invalid/")
        assert content_type == "video/mp4"
        assert headers == {"authorization": "fixture-upload-signature"}
        clip_id = url.rsplit("/", 1)[-1]
        self.objects[clip_id] = hashlib.sha256(path.read_bytes()).hexdigest()
        return HttpResult(200, b"")


def run_fixture_sync(tmp_path: Path, *, fixture: str) -> FixtureSyncResult:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "synthetic" / "miniprogram" / fixture
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    EdgeSyncEnvelopeV1.model_validate(payload)

    clip_bytes = b"synthetic-development-video-bytes"
    clip_path = (tmp_path / "clip-shot-0001.mp4").resolve()
    clip_path.write_bytes(clip_bytes)
    payload["media"][0]["content_sha256"] = hashlib.sha256(clip_bytes).hexdigest()
    payload["media"][0]["byte_size"] = len(clip_bytes)
    envelope = EdgeSyncEnvelopeV1.model_validate(payload)
    manifest_json = json.dumps(
        envelope.model_dump(mode="json", exclude_none=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest_hash = canonical_sync_hash(envelope)

    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: datetime(2026, 8, 26, 14, 0, tzinfo=UTC),
        session_id_factory=lambda: envelope.session.session_id,
        sync_job_id_factory=lambda: "fixture-sync-job",
    )
    store.open_or_resume_session(artifact_id=envelope.session.artifact_id)
    store.end_session(envelope.session.session_id)
    media = tuple(
        SyncMediaItem(
            clip_id=clip.clip_id,
            shot_id=clip.shot_id,
            local_path=clip_path,
            content_type=clip.media_type,
            content_sha256=clip.content_sha256,
            byte_size=clip.byte_size,
        )
        for clip in envelope.media
    )
    queued = store.enqueue_sync_job(
        session_id=envelope.session.session_id,
        manifest_hash=manifest_hash,
        manifest_json=manifest_json,
        media=media,
    )
    claimed = store.claim_due_sync_job(datetime(2026, 8, 26, 14, 0, tzinfo=UTC))
    assert claimed is not None and claimed.sync_job_id == queued.sync_job_id

    transport = _InMemorySyncTransport()
    client = FieldSyncClient(
        credentials=_Credentials(),
        transport=transport,
        now_seconds=lambda: 1_787_752_800,
        nonce_factory=(f"nonce-fixture-{index:04d}" for index in range(100)).__next__,
    )
    first = client.run_once(claimed)
    assert first.outcome == "complete"
    store.mark_sync_complete(claimed.sync_job_id)
    second = client.run_once(claimed)
    assert second.outcome == "complete"

    return FixtureSyncResult(
        session_id=envelope.session.session_id,
        clips=envelope.media,
        duplicate_session_count=transport.duplicate_session_count,
        sync_status=store.load_sync_job(claimed.sync_job_id).status,
    )
