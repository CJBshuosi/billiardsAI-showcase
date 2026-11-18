from __future__ import annotations

import hashlib
import http.client
import json
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import pytest

from billiards_ai.edge.route_assistance.local_media_cleanup import process_local_discard_once
from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.postgame import SavedMediaOutboxItem
from billiards_ai.edge.route_assistance.records import ownership_receipt_from_payload
from billiards_ai.edge.route_assistance.sync import (
    FieldSyncClient,
    HttpResult,
    SyncRequestToSign,
    sign_sync_request,
)
from billiards_ai.lab.field_assistance import server as field_service
from billiards_ai.lab.mainline1_session_analysis import (
    DevelopmentSessionAnalysisPublisher,
    SQLiteSessionAnalysisOutbox,
)
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.shared.models import SessionAnalysisProjectionV1
from scripts.algorithm.serve_mainline1_usb_field import _BackgroundSavedMediaSyncWorker
from tests.unit.lab.test_mainline1_shot_worker import NOW, _record_ended_usb_session, _worker

FIXTURE_PATH = (
    Path(__file__).parents[3]
    / "fixtures"
    / "synthetic"
    / "miniprogram"
    / "saved-postgame-clip-e2e-v2.json"
)


class _Credentials:
    def device_id(self) -> str:
        return "device-1"

    def secret(self) -> bytes:
        return b"development-device-secret"


class _Capture:
    def __init__(self, session_id, channel, source_epoch_id):
        self.session_id = session_id
        self.channel = channel
        self.source_epoch_id = source_epoch_id
        self.cancelled = threading.Event()
        self.validated = threading.Event()
        self.current_capture_frame = None
        self.calibration = None

    def run(self):
        self.validated.set()
        self.cancelled.wait(10)

    def close(self):
        self.cancelled.set()

    def wait_until_ready(self, *, timeout_seconds):
        return self.validated.wait(timeout_seconds) and not self.cancelled.is_set()


def _review_request(server, path="/api/postgame-review", payload=None, nonce=""):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(
        "GET" if payload is None else "POST",
        path,
        body=None if payload is None else json.dumps(payload),
        headers={"Content-Type": "application/json", "X-Kiosk-Nonce": nonce},
    )
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, json.loads(body)


def _projection(payload):
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


class _CloudContractTransport:
    def __init__(self, *, shot_count=2):
        self.shot_count = shot_count
        self.session_published = False
        self.session_envelope = None
        self.saved_descriptor = None
        self.saved_document = None
        self.uploaded_bytes = None
        self.create_count = 0
        self.authorize_count = 0
        self.upload_count = 0
        self.complete_count = 0
        self._lose_first_complete_ack = True

    @staticmethod
    def _assert_signature(path, headers, body):
        assert headers["x-body-sha256"] == hashlib.sha256(body).hexdigest()
        expected = sign_sync_request(
            SyncRequestToSign(
                method="POST",
                path=path,
                timestamp=headers["x-timestamp"],
                nonce=headers["x-nonce"],
                body_sha256=headers["x-body-sha256"],
            ),
            secret=b"development-device-secret",
        )
        assert headers["x-device-id"] == "device-1"
        assert headers["x-signature"] == expected.signature

    @staticmethod
    def _ack(descriptor, *, status):
        return json.dumps(
            {
                "data": {
                    "job_id": "saved-cloud-job-1",
                    "status": status,
                    "clip_id": descriptor["clip_id"],
                    "session_id": descriptor["session_id"],
                    "shot_id": descriptor["shot_id"],
                    "content_sha256": descriptor["content_sha256"],
                    "publication_kind": "saved_postgame_clip",
                }
            }
        ).encode()

    def post_json(self, path, headers, body):
        self._assert_signature(path, headers, body)
        if path == "/v1/sync/jobs":
            self.session_envelope = json.loads(body)
            assert self.session_envelope["media"] == []
            assert len(self.session_envelope["session"]["shots"]) == self.shot_count
            return HttpResult(200, b'{"data":{"job_id":"session-cloud-job-1"}}')
        if path == "/v1/sync/jobs/session-cloud-job-1/complete":
            self.session_published = True
            return HttpResult(200, b'{"data":{"status":"completed"}}')
        if path == "/v1/sync/saved-clips":
            assert self.session_published
            self.create_count += 1
            descriptor = json.loads(body)
            if self.saved_descriptor is None:
                self.saved_descriptor = descriptor
            else:
                assert descriptor == self.saved_descriptor
            status = "completed" if self.saved_document is not None else "pending"
            return HttpResult(200, self._ack(descriptor, status=status))
        if path.endswith("/media-authorization"):
            assert self.saved_descriptor is not None
            self.authorize_count += 1
            return HttpResult(
                200,
                b'{"data":{"method":"PUT","url":"https://upload.invalid/saved-1",'
                b'"headers":{"x-upload-token":"fixture"}}}',
            )
        if path.endswith("/complete"):
            assert self.saved_descriptor is not None
            assert self.uploaded_bytes is not None
            self.complete_count += 1
            self.saved_document = {
                **self.saved_descriptor,
                "storage_file_id": "cloud://development/private/saved-1.mp4",
                "upload_status": "published",
            }
            if self._lose_first_complete_ack:
                self._lose_first_complete_ack = False
                raise OSError("completion acknowledgement lost")
            return HttpResult(200, self._ack(self.saved_descriptor, status="completed"))
        raise AssertionError(path)

    def put_bytes(self, url, payload, content_type, headers):
        assert url == "https://upload.invalid/saved-1"
        assert content_type == "video/mp4"
        assert headers == {"x-upload-token": "fixture"}
        assert self.saved_descriptor is not None
        assert hashlib.sha256(payload).hexdigest() == self.saved_descriptor["content_sha256"]
        assert len(payload) == self.saved_descriptor["byte_size"]
        self.upload_count += 1
        self.uploaded_bytes = payload
        return HttpResult(200, b"")

    def put_file(self, *_args, **_kwargs):
        raise AssertionError("facts-only sync and saved-byte sync must not use put_file")

    def put_json(self, *_args, **_kwargs):
        raise AssertionError("this acceptance chain does not publish live facts here")


def test_ended_facts_only_session_saves_one_clip_once_and_omits_unsaved_sibling(tmp_path):
    database = tmp_path / "field.sqlite3"
    artifact_root = tmp_path / "artifacts"
    field_artifact_id = (
        "usb-field-" + hashlib.sha256(str(artifact_root.resolve()).encode()).hexdigest()
    )
    records = SQLiteFieldRecordStore(
        database,
        now_utc=lambda: NOW,
        session_id_factory=lambda: "session-1",
        sync_job_id_factory=lambda: "sync-session-1",
    )
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    first = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "capture-1",
        session_id="session-1",
        shot_id="shot-0001",
        job_id="job-1",
        epoch="epoch-1",
        artifact_id=field_artifact_id,
        end_session=False,
    )
    second = _record_ended_usb_session(
        records,
        jobs,
        tmp_path / "capture-2",
        session_id="session-1",
        shot_id="shot-0002",
        job_id="job-2",
        epoch="epoch-2",
        artifact_id=field_artifact_id,
        sequence=2,
        end_session=False,
    )
    records.end_session("session-1")
    transport = _CloudContractTransport()
    sync_client = FieldSyncClient(credentials=_Credentials(), transport=transport)

    class Gateway:
        def publish_session_analysis(self, payload):
            assert payload["published_clips"] == []
            return _projection(payload)

    publisher = DevelopmentSessionAnalysisPublisher(
        records=records,
        jobs=jobs,
        outbox=SQLiteSessionAnalysisOutbox(tmp_path / "jobs.sqlite3"),
        gateway=Gateway(),
        sync_client=sync_client,
        artifact_root=artifact_root,
    )
    analysis = _worker(
        records,
        jobs,
        artifact_root / "postgame" / "session-1",
        analysis_publisher=publisher,
    )
    assert analysis.process_one("analysis-1", "session-1", NOW) is not None
    assert analysis.process_one("analysis-2", "session-1", NOW + timedelta(seconds=1)) is not None
    assert transport.session_published
    assert {clip.shot_id for clip in records.list_postgame_clips("session-1")} == {
        first.shot_id,
        second.shot_id,
    }

    captures = []

    def runtime_factory(session_id, channel, _initial_shot_number, source_epoch_id):
        capture = _Capture(session_id, channel, source_epoch_id)
        captures.append(capture)
        return capture

    qr = tmp_path / "join.png"
    qr.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    server = field_service.build_live_field_assistance_server(
        artifact_path=artifact_root,
        state_db_path=database,
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=lambda: "epoch-live",
        calibration_id="cal-1",
        runtime_factory=runtime_factory,
        join_qr_asset_path=qr,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        status, review = _review_request(server)
        assert status == 200
        assert review["review_session_id"] == "session-1"
        assert review["phase"] == "POSTGAME_REVIEW"
        assert len(review["clips"]) == 2
        selected = next(clip for clip in review["clips"] if clip["shot_id"] == "shot-0001")
        payload = {
            "request_id": "large-screen-save-1",
            "session_id": "session-1",
            "shot_id": selected["shot_id"],
            "postgame_manifest_sha256": review["postgame_manifest_sha256"],
        }
        assert _review_request(server, "/api/postgame-review/save", payload, "wrong")[0] == 403
        stale = {**payload, "postgame_manifest_sha256": "0" * 64}
        assert (
            _review_request(server, "/api/postgame-review/save", stale, review["kiosk_nonce"])[0]
            == 409
        )
        status, receipt = _review_request(
            server, "/api/postgame-review/save", payload, review["kiosk_nonce"]
        )
        assert status == 200
        assert receipt["status"] == "SAVED"

        store = server.field_state.record_store
        rows = store.list_saved_media_outbox()
        assert len(rows) == 1
        assert rows[0].shot_id == "shot-0001"
        clock = {"now": datetime.now(UTC)}
        abandoned = store.claim_due_saved_media(clock["now"], lease_seconds=1)
        assert abandoned is not None
        clock["now"] += timedelta(seconds=2)
        first_worker = _BackgroundSavedMediaSyncWorker(
            store=store,
            client=sync_client,
            trusted_postgame_root=artifact_root / "postgame",
            wake_event=threading.Event(),
            now_utc=lambda: clock["now"],
        )
        assert first_worker.run_once()
        assert store.list_saved_media_outbox()[0].status == "pending"
        clock["now"] += timedelta(seconds=10)
        restarted_worker = _BackgroundSavedMediaSyncWorker(
            store=store,
            client=sync_client,
            trusted_postgame_root=artifact_root / "postgame",
            wake_event=threading.Event(),
            now_utc=lambda: clock["now"],
        )
        assert restarted_worker.run_once()
        assert store.list_saved_media_outbox()[0].status == "acknowledged"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(3)

    assert transport.create_count == 2
    assert transport.authorize_count == 1
    assert transport.upload_count == 1
    assert transport.complete_count == 1
    assert transport.saved_document["shot_id"] == "shot-0001"
    assert "shot-0002" not in json.dumps(transport.saved_document)
    assert len(captures) == 1 and captures[0].cancelled.is_set()


@pytest.mark.parametrize("attack", ["replacement", "symlink"])
def test_saved_clip_local_replacement_and_symlink_fail_before_cloud(tmp_path, attack):
    fixture = json.loads(FIXTURE_PATH.read_text())
    descriptor = fixture["descriptor"]
    media = tmp_path / "postgame" / "session-1" / "clip.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(bytes.fromhex(fixture["mp4_hex"]))
    if attack == "replacement":
        media.write_bytes(b"replacement")
    else:
        target = tmp_path / "outside.mp4"
        target.write_bytes(bytes.fromhex(fixture["mp4_hex"]))
        media.unlink()
        media.symlink_to(target)
    item = SavedMediaOutboxItem(
        **{
            key: value
            for key, value in descriptor.items()
            if key not in {"schema_version", "publication_kind", "ownership_receipt"}
        },
        ownership_receipt=ownership_receipt_from_payload(descriptor["ownership_receipt"]),
        local_path=media,
        status="leased",
        attempts=1,
        claim_token="claim-1",
        lease_until_utc="2026-10-28T15:00:00Z",
        next_attempt_at_utc="2026-10-28T14:00:00Z",
        last_error_code=None,
    )

    class NoCloud:
        def __getattr__(self, name):
            raise AssertionError(f"local attack reached cloud transport: {name}")

    result = FieldSyncClient(credentials=_Credentials(), transport=NoCloud()).run_saved_media_once(
        item,
        trusted_postgame_root=tmp_path / "postgame",
    )

    assert result.outcome == "blocked"
    assert result.error_code == "SYNC_LOCAL_MEDIA_INVALID"
    assert hashlib.sha256(item.descriptor_json.encode()).hexdigest() == fixture["descriptor_sha256"]


@contextmanager
def _running_review_server(tmp_path, database, artifact_root):
    captures = []

    def runtime_factory(session_id, channel, _initial_shot_number, source_epoch_id):
        capture = _Capture(session_id, channel, source_epoch_id)
        captures.append(capture)
        return capture

    qr = tmp_path / "join-cleanup.png"
    qr.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    server = field_service.build_live_field_assistance_server(
        artifact_path=artifact_root,
        state_db_path=database,
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=lambda: "epoch-live",
        calibration_id="cal-1",
        runtime_factory=runtime_factory,
        join_qr_asset_path=qr,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert not thread.is_alive()
        assert len(captures) == 1 and captures[0].cancelled.is_set()


def _media_request(server, path):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request("GET", path)
    response = connection.getresponse()
    result = response.status, response.read()
    connection.close()
    return result


@pytest.mark.parametrize("crash_point", ["after_detach", "after_unlink"])
def test_complete_review_reclaims_only_unsaved_media_and_recovers_after_crash(
    tmp_path, monkeypatch, crash_point
):
    """Catches premature disposal, lost selections, resurrection and facts/media coupling."""
    database = tmp_path / "field.sqlite3"
    artifact_root, jobs_root = tmp_path / "artifacts", tmp_path / "jobs"
    field_artifact_id = (
        "usb-field-" + hashlib.sha256(str(artifact_root.resolve()).encode()).hexdigest()
    )
    records = SQLiteFieldRecordStore(
        database,
        now_utc=lambda: NOW,
        session_id_factory=lambda: "session-1",
        sync_job_id_factory=lambda: "facts-session-1",
    )
    jobs_db = jobs_root / "analysis-jobs.sqlite3"
    jobs = SQLiteShotAnalysisJobStore(jobs_db)
    shots = tuple(
        _record_ended_usb_session(
            records,
            jobs,
            tmp_path / f"capture-{number}",
            session_id="session-1",
            shot_id=f"shot-{number:04d}",
            job_id=f"job-{number}",
            epoch=f"epoch-{number}",
            artifact_id=field_artifact_id,
            sequence=number,
            end_session=False,
            media_root=jobs_root / "media" / "session-1" / f"job-{number}",
        )
        for number in (1, 2, 3)
    )
    records.end_session("session-1")
    # Offline finalization stages immutable facts without network or cloud authority.
    outbox = SQLiteSessionAnalysisOutbox(jobs_db)
    publisher = DevelopmentSessionAnalysisPublisher(
        records=records, jobs=jobs, outbox=outbox, artifact_root=artifact_root
    )
    analysis = _worker(
        records, jobs, artifact_root / "postgame" / "session-1", analysis_publisher=publisher
    )
    assert analysis.process_one("analysis-1", "session-1", NOW) is not None
    sentinel = artifact_root / "unrelated-session.mp4"
    sentinel.write_bytes(b"unrelated media must survive")
    with _running_review_server(tmp_path, database, artifact_root) as server:
        status, partial = _review_request(server)
        assert status == 200
        assert partial["available_count"] == 1 and partial["expected_count"] == 3
        assert partial["can_complete"] is True
        complete = {
            "session_id": "session-1",
            "postgame_manifest_sha256": partial["postgame_manifest_sha256"],
        }
        # This scenario keeps reviewing until all outputs exist to exercise the
        # legacy erasure journal. S1 also permits finishing this partial review;
        # its pending-source path has an independent cancellation test.
        assert records.active_postgame_review(field_artifact_id) is not None
        assert (
            process_local_discard_once(records, jobs_db, artifact_root, jobs_root).outcome == "idle"
        )
        first_payload = {**complete, "request_id": "save-first", "shot_id": "shot-0001"}
        status, first_receipt = _review_request(
            server, "/api/postgame-review/save", first_payload, partial["kiosk_nonce"]
        )
        assert status == 200 and first_receipt["status"] == "SAVED"
        assert first_receipt["owner_user_id"] == "user-a"
        assert (
            analysis.process_one("analysis-2", "session-1", NOW + timedelta(seconds=1)) is not None
        )
        assert (
            analysis.process_one("analysis-3", "session-1", NOW + timedelta(seconds=2)) is not None
        )
        status, review = _review_request(server)
        assert status == 200 and review["can_complete"] is True
        assert review["available_count"] == 3
        for clip in review["clips"]:
            status, data = _media_request(server, clip["media_url"])
            assert status == 200 and data
        nonce = review["kiosk_nonce"]
        second_payload = {**complete, "request_id": "save-second", "shot_id": "shot-0002"}
        assert (
            _review_request(server, "/api/postgame-review/save", second_payload, "outsider")[0]
            == 403
        )
        # Exercise socket-peer rejection independently of a valid kiosk nonce.
        with monkeypatch.context() as peer:
            peer.setattr(
                field_service._FieldAssistanceHandler, "_is_loopback_peer", lambda _self: False
            )
            assert (
                _review_request(server, "/api/postgame-review/complete", complete, nonce)[0] == 403
            )
        for invalid in (
            {**complete, "session_id": "other-session"},
            {**complete, "postgame_manifest_sha256": "0" * 64},
        ):
            assert (
                _review_request(server, "/api/postgame-review/complete", invalid, nonce)[0] == 409
            )
            invalid_save = {**second_payload, **invalid}
            assert (
                _review_request(server, "/api/postgame-review/save", invalid_save, nonce)[0] == 409
            )
        status, second_receipt = _review_request(
            server, "/api/postgame-review/save", second_payload, nonce
        )
        assert status == 200 and second_receipt["status"] == "SAVED"
        clips = {clip.shot_id: clip for clip in records.list_postgame_clips("session-1")}
        overlays = {key: clip.local_path.read_bytes() for key, clip in clips.items()}
        raw_bytes = {
            job.shot_id: {
                job.media_root / name: (job.media_root / name).read_bytes()
                for name in json.loads((job.media_root / "interval.json").read_text())[
                    "segment_sha256"
                ]
            }
            for job in shots
        }
        assert all(raw_bytes.values())
        assert all(clip.local_path.is_file() for clip in clips.values())
        assert (
            process_local_discard_once(records, jobs_db, artifact_root, jobs_root).outcome == "idle"
        )
        # Completion commits the fence while old readers finish; disposal still
        # waits for the exclusive pin before touching either saved or unsaved media.
        with session_media_lock(artifact_root, "session-1", exclusive=False):
            status, completed = _review_request(
                server, "/api/postgame-review/complete", complete, nonce
            )
        assert status == 200 and completed["phase"] == "REVIEW_COMPLETE"
        assert records.is_postgame_discarded("session-1", "shot-0003")
        assert not records.is_postgame_discarded("session-1", "shot-0001")
        assert not records.is_postgame_discarded("session-1", "shot-0002")
        clock = datetime.now(UTC) + timedelta(seconds=1)
        with session_media_lock(artifact_root, "session-1", exclusive=False):
            retry = process_local_discard_once(
                records, jobs_db, artifact_root, jobs_root, now_utc=clock
            )
        assert retry.outcome == "retry"
        assert all(clip.local_path.is_file() for clip in clips.values())

    class PowerLoss(BaseException):
        pass

    def crash(point, _obj):
        if point == crash_point:
            raise PowerLoss()

    clock += timedelta(seconds=60)
    with pytest.raises(PowerLoss):
        process_local_discard_once(
            records, jobs_db, artifact_root, jobs_root, now_utc=clock, fault_hook=crash
        )
    # Fresh database/store/worker objects model process restart and expired lease recovery.
    records = SQLiteFieldRecordStore(database)
    jobs = SQLiteShotAnalysisJobStore(jobs_db)
    result = process_local_discard_once(
        records,
        jobs_db,
        artifact_root,
        jobs_root,
        now_utc=clock + timedelta(seconds=60),
        session_ids=("session-1",),
    )
    assert result.outcome == "completed"
    assert (result.session_id, result.shot_id) == ("session-1", "shot-0003")
    unsaved = clips["shot-0003"]
    assert not unsaved.local_path.exists()
    assert all(not path.exists() for path in raw_bytes["shot-0003"])
    expected_bytes = sum(len(value) for value in raw_bytes["shot-0003"].values())
    if crash_point == "after_detach":
        expected_bytes += len(overlays["shot-0003"])
    assert result.reclaimed_bytes == expected_bytes
    assert not list(artifact_root.rglob("*.detached"))
    assert not list(jobs_root.rglob("*.detached"))
    assert sentinel.read_bytes() == b"unrelated media must survive"
    for job in shots:
        assert (job.media_root / "interval.json").is_file()
        assert (clips[job.shot_id].local_path.parent / "manifest.json").is_file()
        assert (clips[job.shot_id].local_path.parent / "summary.json").is_file()
        assert jobs.snapshot(job.job_id).status.value == "available"
    assert records.is_postgame_finalized("session-1")
    assert len(records.list_shots("session-1")) == 3
    for shot_id in ("shot-0001", "shot-0002"):
        assert clips[shot_id].local_path.read_bytes() == overlays[shot_id]
        for path, data in raw_bytes[shot_id].items():
            assert path.read_bytes() == data
        playback = cv2.VideoCapture(str(clips[shot_id].local_path))
        try:
            assert playback.isOpened() and playback.read()[0]
        finally:
            playback.release()
    assert (
        process_local_discard_once(
            records, jobs_db, artifact_root, jobs_root, now_utc=clock + timedelta(seconds=120)
        ).outcome
        == "idle"
    )
    with pytest.raises(ValueError, match="discarded"):
        records.save_postgame_clip(
            **complete, shot_id="shot-0003", request_id="late-save-after-discard"
        )
    with pytest.raises(ValueError, match="discarded"):
        records.record_postgame_clip_available(
            **complete,
            shot_id=unsaved.shot_id,
            job_id=unsaved.job_id,
            artifact_id=unsaved.artifact_id,
            local_path=unsaved.local_path,
            content_sha256=unsaved.content_sha256,
            byte_size=unsaved.byte_size,
        )
    assert asdict(records.save_postgame_clip(**first_payload)) == first_receipt
    assert asdict(records.save_postgame_clip(**second_payload)) == second_receipt
    # Retry facts publication only after unsaved media has gone; no overlay re-verification.
    transport = _CloudContractTransport(shot_count=3)
    publisher = DevelopmentSessionAnalysisPublisher(
        records=records,
        jobs=jobs,
        outbox=SQLiteSessionAnalysisOutbox(jobs_db),
        sync_client=FieldSyncClient(credentials=_Credentials(), transport=transport),
        artifact_root=artifact_root,
    )
    restarted_analysis = _worker(
        records, jobs, artifact_root / "postgame" / "session-1", analysis_publisher=publisher
    )
    assert restarted_analysis.process_one("restart", "session-1", clock) is None
    assert transport.session_published
    assert transport.session_envelope["media"] == []
    assert {shot["shot_id"] for shot in transport.session_envelope["session"]["shots"]} == {
        "shot-0001",
        "shot-0002",
        "shot-0003",
    }
    assert not unsaved.local_path.exists()
    # Both durable selections upload after deletion; each transport is a synthetic contract peer.
    assert {item.shot_id for item in records.list_saved_media_outbox()} == {
        "shot-0001",
        "shot-0002",
    }
    peers = {
        shot_id: _CloudContractTransport(shot_count=3) for shot_id in ("shot-0001", "shot-0002")
    }
    for peer in peers.values():
        peer.session_published = transport.session_published

    class SelectedTransport:
        def post_json(self, path, headers, body):
            if path == "/v1/sync/saved-clips":
                self.peer = peers[json.loads(body)["shot_id"]]
            return self.peer.post_json(path, headers, body)

        def put_bytes(self, *args):
            return self.peer.put_bytes(*args)

    saved_client = FieldSyncClient(credentials=_Credentials(), transport=SelectedTransport())
    for _attempt in range(4):
        records = SQLiteFieldRecordStore(database)
        worker = _BackgroundSavedMediaSyncWorker(
            store=records,
            client=saved_client,
            trusted_postgame_root=artifact_root / "postgame",
            wake_event=threading.Event(),
            now_utc=lambda now=clock: now,
        )
        assert worker.run_once()
        clock += timedelta(seconds=60)
    for shot_id, saved_transport in peers.items():
        assert saved_transport.saved_document["shot_id"] == shot_id
        assert saved_transport.saved_document["owner_user_id"] == "user-a"
        assert saved_transport.uploaded_bytes == overlays[shot_id]
        assert saved_transport.upload_count == 1
    assert all(item.status == "acknowledged" for item in records.list_saved_media_outbox())


def test_real_ended_screen_allows_postgame_worker_before_next_match(tmp_path):
    import os
    import time

    from scripts.algorithm.run_mainline1_shot_worker import AuditedLivePriorityGate

    # Real SQLite, USBLiveProjectionSource, local end control and HTTP screen;
    # the synthetic capture has no detector/model or physical device.
    with _running_review_server(
        tmp_path, tmp_path / "field.sqlite3", tmp_path / "artifacts"
    ) as server:
        state = server.field_state
        # This case exercises a terminal screen, not unmanaged timed rollover.
        state.monotonic = lambda: 0.0
        state.payload()
        target = state.session_id
        state.mutate({"command": "end_session"})
        # A paused/slow test must still query the original ended projection.
        time.sleep(3.05)
        gate = AuditedLivePriorityGate(
            records=state.record_store,
            target_session_id=target,
            live_url=f"http://127.0.0.1:{server.server_port}/api/screen",
            server_pid=os.getpid(),
            audit_path=tmp_path / f"live-resource-{target}.jsonl",
        )
        assert gate.may_run_heavy_analysis() is True
        state.record_store.open_or_resume_session(artifact_id="new-artifact")
        assert gate.may_run_heavy_analysis() is False
