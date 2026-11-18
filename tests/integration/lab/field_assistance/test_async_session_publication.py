from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.sync import FieldSyncClient, HttpResult
from billiards_ai.lab.field_assistance.cloud_gateway import (
    DevelopmentAttestedFieldCloudGateway,
    SignedFieldCloudGateway,
)
from billiards_ai.lab.mainline1_session_analysis import (
    DevelopmentSessionAnalysisPublisher,
    SQLiteSessionAnalysisOutbox,
)
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.shared.models import SessionAnalysisProjectionV1
from tests.unit.lab.test_mainline1_shot_worker import (
    NOW,
    _record_ended_usb_session,
    _worker,
)


class _RestartingGateway:
    def __init__(self) -> None:
        self.offline = True
        self.attempts: list[dict[str, object]] = []

    def publish_session_analysis(self, payload):
        self.attempts.append(json.loads(json.dumps(payload)))
        if self.offline:
            self.offline = False
            raise OSError("development gateway offline")
        return SessionAnalysisProjectionV1.model_validate(
            {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "frozen_shots_digest",
                    "frozen_shot_ids",
                    "published_shots",
                    "published_clips",
                }
            }
        )


def test_real_worker_restart_retries_exact_unpublished_projection(
    tmp_path: Path,
) -> None:
    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: NOW,
        session_id_factory=lambda: "session-1",
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    job = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "capture",
        session_id="session-1",
        shot_id="shot-1",
        job_id="job-1",
        epoch="epoch-1",
        artifact_id="field-artifact-1",
    )
    gateway = _RestartingGateway()

    first_publisher = DevelopmentSessionAnalysisPublisher(
        records=records,
        jobs=jobs,
        outbox=SQLiteSessionAnalysisOutbox(tmp_path / "jobs.sqlite3"),
        gateway=gateway,
    )
    assert (
        _worker(
            records,
            jobs,
            tmp_path / "artifacts",
            analysis_publisher=first_publisher,
        ).process_one("worker-before-restart", job.session_id, NOW)
        is not None
    )

    restarted_publisher = DevelopmentSessionAnalysisPublisher(
        records=records,
        jobs=SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3"),
        outbox=SQLiteSessionAnalysisOutbox(tmp_path / "jobs.sqlite3"),
        gateway=gateway,
    )
    assert (
        _worker(
            records,
            jobs,
            tmp_path / "artifacts",
            analysis_publisher=restarted_publisher,
        ).process_one("worker-after-restart", job.session_id, NOW + timedelta(seconds=31))
        is None
    )

    assert [payload["revision"] for payload in gateway.attempts] == [1, 1]
    assert gateway.attempts[0] == gateway.attempts[1]
    assert [
        entry.status
        for entry in SQLiteSessionAnalysisOutbox(tmp_path / "jobs.sqlite3").entries(job.session_id)
    ] == ["published"]


def test_real_worker_syncs_facts_only_and_records_unsaved_clip_for_local_review(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"

    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"development-secret"

    class CloudTransport:
        def __init__(self) -> None:
            self.events: list[tuple[str, str]] = []
            self.session_published = False
            self.analysis: list[dict[str, object]] = []

        def post_json(self, path, _headers, body):
            if path == "/v1/runtime-attestation":
                self.events.append(("attest", path))
                return HttpResult(
                    200,
                    b'{"ok":true,"data":{"schema_version":"1.0",'
                    b'"runtime_environment":"development",'
                    b'"cloudbase_environment_id":"cloud-development"}}',
                )
            self.events.append(("post", path))
            if path == "/v1/sync/jobs":
                envelope = json.loads(body)
                assert envelope["media"] == []
                assert envelope["session"]["evidence_source_kind"] == "usb_live"
                assert envelope["session"]["postgame_manifest_sha256"] == (
                    records.get_postgame_manifest("session-1").manifest_sha256
                )
                assert envelope["session"]["shots"][0]["live_job_id"] == "job-1"
                assert list(artifact_root.glob("*/overlay.mp4")) == []
                return HttpResult(200, b'{"data":{"job_id":"cloud-sync-1"}}')
            if path.endswith("/media-authorizations"):
                raise AssertionError(
                    "unsaved generated clips must not request upload authorization"
                )
            if path.endswith("/complete"):
                assert list(artifact_root.glob("*/overlay.mp4")) == []
                self.session_published = True
                return HttpResult(200, b'{"data":{"status":"completed"}}')
            raise AssertionError(path)

        def put_file(self, url, path, content_type, headers):
            raise AssertionError("unsaved generated clips must not upload")

        def put_json(self, path, _headers, body):
            payload = json.loads(body)
            self.events.append(("analysis", payload["status"]))
            assert payload["published_clips"] == []
            self.analysis.append(payload)
            public = {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "frozen_shots_digest",
                    "frozen_shot_ids",
                    "published_shots",
                    "published_clips",
                }
            }
            return HttpResult(200, json.dumps({"ok": True, "data": public}).encode())

    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: NOW,
        session_id_factory=lambda: "session-1",
        sync_job_id_factory=lambda: "sync-session-1",
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    job = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "capture",
        session_id="session-1",
        shot_id="shot-1",
        job_id="job-1",
        epoch="epoch-1",
        artifact_id="field-artifact-1",
    )
    credentials = Credentials()
    transport = CloudTransport()
    signed = SignedFieldCloudGateway(credentials=credentials, transport=transport)
    gateway = DevelopmentAttestedFieldCloudGateway(
        delegate=signed,
        expected_cloudbase_environment_id="cloud-development",
    )
    sync_client = FieldSyncClient(
        credentials=credentials,
        transport=transport,
        before_mutation=gateway.runtime_attestation,
    )
    publisher = DevelopmentSessionAnalysisPublisher(
        records=records,
        jobs=jobs,
        outbox=SQLiteSessionAnalysisOutbox(tmp_path / "jobs.sqlite3"),
        gateway=gateway,
        sync_client=sync_client,
        artifact_root=artifact_root,
    )

    result = _worker(
        records,
        jobs,
        artifact_root,
        analysis_publisher=publisher,
    ).process_one("worker-1", job.session_id, NOW)

    assert result is not None
    sync_job = records.load_sync_job("sync-session-1")
    assert sync_job.status == "complete"
    assert sync_job.media == ()
    assert [payload["status"] for payload in transport.analysis] == ["PROCESSING"]
    review_clip = records.list_postgame_clips(job.session_id)[0]
    assert review_clip.local_path == (result.artifact_path / "overlay.mp4").resolve()
    assert review_clip.save_status == "NOT_SAVED"
    assert records.list_saved_media_outbox() == ()
    complete_index = transport.events.index(("post", "/v1/sync/jobs/cloud-sync-1/complete"))
    processing_index = transport.events.index(("analysis", "PROCESSING"))
    assert complete_index < processing_index
    for index, event in enumerate(transport.events):
        if event[0] in {"post", "upload", "analysis"}:
            assert index > 0 and transport.events[index - 1][0] == "attest"
