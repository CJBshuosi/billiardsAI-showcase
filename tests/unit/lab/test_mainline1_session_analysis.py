from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from billiards_ai.edge.route_assistance.postgame import PostgameSessionManifest
from billiards_ai.lab.mainline1_session_analysis import (
    DevelopmentSessionAnalysisPublisher,
    SQLiteSessionAnalysisOutbox,
)
from billiards_ai.lab.mainline1_shot_jobs import (
    ShotAnalysisJob,
    ShotAnalysisJobSnapshot,
    ShotAnalysisJobStatus,
)
from billiards_ai.shared.models import SessionAnalysisProjectionV1

NOW = datetime(2026, 9, 6, 12, tzinfo=UTC)


def test_frozen_manifest_digest_is_canonical_and_binds_ordered_job_identity() -> None:
    manifest = PostgameSessionManifest(
        "session-1", NOW, NOW + timedelta(seconds=120), ("job-1", "job-2")
    )
    # Equal instants with a different offset must produce the same save identity.
    from datetime import timezone

    equivalent = replace(manifest, ended_at_utc=NOW.astimezone(timezone(timedelta(hours=2))))
    assert len(manifest.manifest_sha256) == 64
    assert equivalent.manifest_sha256 == manifest.manifest_sha256
    for changed in (
        replace(manifest, session_id="session-2"),
        replace(manifest, expected_jobs=("job-2", "job-1")),
        replace(manifest, ended_at_utc=NOW + timedelta(seconds=1)),
        replace(manifest, deadline_at_utc=NOW + timedelta(seconds=121)),
    ):
        assert changed.manifest_sha256 != manifest.manifest_sha256


def _job(job_id: str, shot_id: str) -> ShotAnalysisJob:
    return ShotAnalysisJob(
        job_id=job_id,
        session_id="session-1",
        shot_id=shot_id,
        source_epoch_id="epoch-1",
        start_frame_id=0,
        end_frame_id=1,
        media_root=Path("/tmp") / job_id,
        media_sha256="a" * 64,
        capture_setup_id="capture-1",
        calibration_artifact_id="calibration-1",
        config_sha256="b" * 64,
        model_package_id="model-1",
    )


class _Records:
    def __init__(self) -> None:
        self.finalized = False
        self.manifest = PostgameSessionManifest(
            session_id="session-1",
            ended_at_utc=NOW,
            deadline_at_utc=NOW + timedelta(seconds=120),
            expected_jobs=("job-1", "job-2"),
        )

    def get_postgame_manifest(self, session_id: str):
        assert session_id == "session-1"
        return self.manifest

    def is_postgame_finalized(self, session_id: str) -> bool:
        assert session_id == "session-1"
        return self.finalized

    def list_shots(self, session_id: str):
        assert session_id == "session-1"
        return ()


class _Jobs:
    def __init__(self) -> None:
        self.snapshots = {
            "job-1": ShotAnalysisJobSnapshot(
                _job("job-1", "shot-1"), ShotAnalysisJobStatus.AVAILABLE, "artifact-1"
            ),
            "job-2": ShotAnalysisJobSnapshot(
                _job("job-2", "shot-2"),
                ShotAnalysisJobStatus.FAILED_RETRYING,
                None,
            ),
        }

    def snapshot(self, job_id: str) -> ShotAnalysisJobSnapshot:
        return self.snapshots[job_id]


class _Gateway:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.attempts: list[dict[str, object]] = []

    def publish_session_analysis(self, payload):
        self.attempts.append(json.loads(json.dumps(payload)))
        if self.failures:
            self.failures -= 1
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


def test_unpublished_local_results_retry_exactly_and_never_claim_cloud_availability(
    tmp_path: Path,
) -> None:
    records = _Records()
    jobs = _Jobs()
    gateway = _Gateway(failures=1)
    outbox = SQLiteSessionAnalysisOutbox(tmp_path / "jobs.sqlite3")
    publisher = DevelopmentSessionAnalysisPublisher(
        records=records,
        jobs=jobs,
        outbox=outbox,
        gateway=gateway,
    )

    first = publisher.refresh_and_publish("session-1", NOW)
    duplicate = publisher.refresh_and_publish("session-1", NOW + timedelta(seconds=1))
    retried = publisher.refresh_and_publish("session-1", NOW + timedelta(seconds=31))

    assert first.status == "FAILED_RETRYING"
    assert duplicate.revision == first.revision == retried.revision == 1
    assert first.report_available is False
    assert (first.total_shots, first.available_shots, first.retrying_shots) == (2, 0, 1)
    assert len(gateway.attempts) == 2
    assert gateway.attempts[0] == gateway.attempts[1]
    assert outbox.entries("session-1")[0].status == "published"

    jobs.snapshots["job-2"] = ShotAnalysisJobSnapshot(
        _job("job-2", "shot-2"), ShotAnalysisJobStatus.AVAILABLE, "artifact-2"
    )
    records.finalized = True
    available = publisher.refresh_and_publish("session-1", NOW + timedelta(seconds=32))

    assert available.revision == 2
    assert available.status == "PROCESSING"
    assert available.report_available is False
    assert gateway.attempts[-1]["published_shots"] == []
    assert [entry.revision for entry in outbox.entries("session-1")] == [1, 2]


def test_processing_projection_keeps_unknown_public_counts_and_no_media(tmp_path: Path) -> None:
    records = _Records()
    jobs = _Jobs()
    jobs.snapshots = {
        job_id: ShotAnalysisJobSnapshot(
            _job(job_id, f"shot-{index}"), ShotAnalysisJobStatus.PENDING, None
        )
        for index, job_id in enumerate(("job-1", "job-2"), start=1)
    }
    gateway = _Gateway()
    publisher = DevelopmentSessionAnalysisPublisher(
        records=records,
        jobs=jobs,
        outbox=SQLiteSessionAnalysisOutbox(tmp_path / "jobs.sqlite3"),
        gateway=gateway,
    )

    projection = publisher.refresh_and_publish("session-1", NOW)

    assert projection.status == "PROCESSING"
    assert projection.total_shots is None
    assert projection.available_shots is None
    assert projection.retrying_shots is None

    expired = publisher.refresh_and_publish("session-1", NOW + timedelta(seconds=121))

    assert expired.status == "FAILED_RETRYING"
    assert (expired.total_shots, expired.available_shots, expired.retrying_shots) == (2, 0, 2)
    assert gateway.attempts[0]["published_shots"] == []
    assert gateway.attempts[0]["published_clips"] == []


def test_partially_generated_projection_remains_processing_while_work_is_pending(
    tmp_path: Path,
) -> None:
    records = _Records()
    jobs = _Jobs()
    jobs.snapshots["job-2"] = ShotAnalysisJobSnapshot(
        _job("job-2", "shot-2"), ShotAnalysisJobStatus.PENDING, None
    )
    publisher = DevelopmentSessionAnalysisPublisher(
        records=records,
        jobs=jobs,
        outbox=SQLiteSessionAnalysisOutbox(tmp_path / "jobs.sqlite3"),
        gateway=_Gateway(),
    )

    projection = publisher.refresh_and_publish("session-1", NOW)

    assert projection.status == "PROCESSING"
    assert projection.total_shots is None
    assert projection.available_shots is None
    assert projection.retrying_shots is None

    expired = publisher.refresh_and_publish("session-1", NOW + timedelta(seconds=121))

    assert expired.status == "FAILED_RETRYING"
    assert (expired.total_shots, expired.available_shots, expired.retrying_shots) == (2, 0, 1)
