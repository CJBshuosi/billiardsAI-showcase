"""Software lifecycle only: synthetic rule facts, capture and CPU vision/media.

No camera, model, cloud, blackbox perception or playback-quality acceptance.
The unmanaged fixture holds rollover time fixed to exercise the original ended
screen; managed birth/resource recovery is outside this control.
"""

import json
import os
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from billiards_ai.edge.route_assistance.control import RackPhase
from billiards_ai.edge.route_assistance.local_media_cleanup import process_local_discard_once
from billiards_ai.lab.field_assistance import live_source
from billiards_ai.lab.field_assistance import server as service
from billiards_ai.lab.mainline1_capture_hub import CaptureHub
from billiards_ai.lab.mainline1_field_runtime import RuntimeHandle
from billiards_ai.lab.mainline1_product_realtime_runtime import (
    LiveSettledShot,
    ProductRealtimeEvent,
)
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame, RealtimeCaptureConfig
from billiards_ai.lab.mainline1_realtime_ring import MjpegSegmentRing
from billiards_ai.lab.mainline1_shot_jobs import (
    ShotAnalysisJob,
    SQLiteShotAnalysisJobStore,
    materialize_frozen_interval,
)
from billiards_ai.lab.shot_local.models import BallGroup
from scripts.algorithm.run_mainline1_shot_worker import AuditedLivePriorityGate
from tests.integration.lab.field_assistance.test_postgame_screen_saved_clips import (
    _media_request,
    _review_request,
)
from tests.unit.lab.field_assistance.test_live_match_profiled import (
    _match,
    _synthetic_fact_build,
    _terminal_facts,
)
from tests.unit.lab.test_mainline1_shot_worker import (
    CONFIG_SHA256,
    IDENTITY_PACKAGE_SHA256,
    MODEL_PACKAGE_ID,
    _FakeDetector,
    _jpeg,
    _worker,
)


def _until(probe):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        value = probe()
        if value:
            return value
        time.sleep(0.01)
    pytest.fail("synthetic lifecycle did not reach the expected boundary")


class _ControlledFiniteSource:
    """The real hub sees natural iterator exhaustion, not a fabricated health flag."""

    def __init__(self):
        self.eof = threading.Event()
        self.cancelled = threading.Event()

    def frames(self, stop):
        yield CaptureFrame("epoch-1", 0, 0, 160, 120, _jpeg(0))
        while not self.eof.wait(0.01):
            if stop.is_set():
                return

    def cancel(self):
        self.cancelled.set()
        self.eof.set()


class _OffsetDetector(_FakeDetector):
    """Reuse the eight-frame UNIT detector at distinct absolute capture spans."""

    def infer(self, frame):
        observation = super().infer(frame.model_copy(update={"frame_id": frame.frame_id % 100}))
        return observation.model_copy(update={"frame_id": frame.frame_id})


def _job(tmp_path, jobs_root, jobs, session_id, number):
    offset = number * 100
    ring = MjpegSegmentRing(
        tmp_path / f"unit-ring-{number}",
        expected_fps=60,
        pre_roll_frames=2,
        post_roll_frames=2,
        maximum_shot_frames=12,
        maximum_bytes=8 * 1024 * 1024,
    )
    try:
        for index, timestamp_ms in enumerate((0, 40, 80, 120, 160, 240, 280, 320)):
            ring.append(
                CaptureFrame(
                    "epoch-1",
                    offset + index,
                    (offset * 10 + timestamp_ms) * 1_000_000,
                    160,
                    120,
                    _jpeg(index),
                )
            )
            if index == 2:
                ring.begin_shot(offset + index)
            if index == 5:
                ring.settle_shot(offset + index)
        interval = materialize_frozen_interval(
            ring.freeze_shot(), jobs_root / "media" / session_id / f"unit-job-{number}"
        )
    finally:
        ring.close()
    job = ShotAnalysisJob(
        f"unit-job-{number}",
        session_id,
        f"shot-{number}",
        "epoch-1",
        interval.start_frame_id,
        interval.end_frame_id,
        interval.root,
        interval.media_sha256,
        "capture-1",
        "calibration-1",
        CONFIG_SHA256,
        MODEL_PACKAGE_ID,
        "identity-1",
        IDENTITY_PACKAGE_SHA256,
    )
    jobs.enqueue(job)
    return job


def test_natural_terminal_eof_keeps_delayed_postgame_replay_save_and_cleanup(tmp_path, monkeypatch):
    source = _ControlledFiniteSource()
    hub = CaptureHub(
        source=source, capture_config=RealtimeCaptureConfig(width=160, height=120, fps=60)
    )
    handles = []

    def runtime_factory(*_):
        class UnitAnalysis:
            def __init__(self, subscription):
                self.subscription = subscription

            def run(self):
                for _frame in self.subscription.frames(threading.Event()):
                    pass

        handle = RuntimeHandle(source, UnitAnalysis, capture_hub=hub)
        handles.append(handle)
        return handle

    artifact_root, jobs_root = tmp_path / "artifacts", tmp_path / "jobs"
    qr = tmp_path / "unit-qr.png"
    qr.write_bytes(b"\x89PNG\r\n\x1a\nunit-placeholder")
    server = service.build_live_field_assistance_server(
        artifact_path=artifact_root,
        state_db_path=tmp_path / "field.sqlite3",
        host="127.0.0.1",
        port=0,
        control_token="unit-only",
        source_epoch_factory=lambda: "epoch-1",
        calibration_id="calibration-1",
        runtime_factory=runtime_factory,
        join_qr_asset_path=qr,
        monotonic=lambda: 0.0,
        rule_profile_id=_terminal_facts().profile_id,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        state, store = server.field_state, server.field_state.record_store
        sid, projection = state.session_id, state.projection_source
        assert projection.wait_until_ready(timeout_seconds=2)
        before = store.load_session(sid).match
        store.save_match(
            sid, replace(_match(), revision=before.revision + 1), expected_revision=before.revision
        )
        jobs_db = jobs_root / "analysis-jobs.sqlite3"
        jobs = SQLiteShotAnalysisJobStore(jobs_db)
        prepared = [_job(tmp_path, jobs_root, jobs, sid, number) for number in (1, 2, 3)]
        facts = {}
        for number, job in enumerate(prepared, 1):
            start, settle = number * 100 + 2, number * 100 + 5
            terminal = _terminal_facts()
            if number < 3:
                terminal = replace(
                    terminal,
                    first_contact_group=BallGroup.SOLID,
                    pots=(replace(terminal.pots[0], group=BallGroup.SOLID),),
                    before_counts={**terminal.before_counts, BallGroup.SOLID: 3 - number},
                    after_counts={**terminal.before_counts, BallGroup.SOLID: 2 - number},
                )
            facts[job.shot_id] = replace(
                terminal,
                session_id=sid,
                shot_id=job.shot_id,
                match_revision_at_start=store.load_session(sid).match.revision,
                start_frame_id=start,
                settled_frame_id=settle,
                pots=(
                    replace(
                        terminal.pots[0],
                        last_seen_frame_id=settle - 2,
                        first_missing_frame_id=settle - 1,
                        confirmed_through_frame_id=settle,
                        evidence_frame_ids=(settle - 2, settle - 1, settle),
                    ),
                ),
                evidence_frame_ids=(start, settle),
                digest=str(number) * 64,
            )
            monkeypatch.setattr(
                live_source,
                "build_realtime_rule_facts_with_preimage",
                lambda shot, **context: _synthetic_fact_build(
                    facts[shot.shot_id],
                    required_cue_ball_placement=context["required_cue_ball_placement"],
                    event_config_digest=shot.event_config_digest,
                ),
            )
            projection.channel.publish(
                ProductRealtimeEvent(
                    "SHOT_STARTED",
                    shot_id=job.shot_id,
                    frame_id=start,
                    started_ms=number * 1000 + 80,
                )
            )
            projection.channel.publish(
                ProductRealtimeEvent(
                    "SHOT_SETTLED",
                    shot_id=job.shot_id,
                    frame_id=settle,
                    settled=LiveSettledShot(
                        job.shot_id,
                        start,
                        settle,
                        (number * 1000 + 240) * 1_000_000,
                        (),
                        True,
                        None,
                        {},
                        "epoch-1",
                        job.job_id,
                    ),
                )
            )
            _until(lambda job=job: store.get_rule_decision(sid, job.shot_id))
            if number == 3:
                assert store.load_session(sid).status.value == "open"
                assert store.load_session(sid).match.rack_phase is RackPhase.ENDED
                assert projection.current().frame.payload["rule_status"] == "EVALUATING"
                assert store.get_postgame_manifest(sid) is None
            projection.channel.publish(
                ProductRealtimeEvent(
                    "SHOT_ENQUEUED",
                    shot_id=job.shot_id,
                    frame_id=settle,
                    job=job,
                )
            )
            _until(lambda job=job: store.get_live_shot_media(sid, job.shot_id))
        _until(lambda: store.load_session(sid).status.value == "ended")
        assert store.load_session(sid).match.terminal_reason == "legal_eight"
        assert store.get_postgame_manifest(sid).expected_jobs == tuple(j.job_id for j in prepared)
        gate = AuditedLivePriorityGate(
            records=store,
            target_session_id=sid,
            live_url=f"http://127.0.0.1:{server.server_port}/api/screen",
            server_pid=os.getpid(),
            audit_path=tmp_path / f"live-resource-{sid}.jsonl",
        )
        worker = _worker(store, jobs, artifact_root / "postgame" / sid)
        worker._deps = replace(worker._deps, vision_backend=_OffsetDetector(), resource_gate=gate)
        now = datetime.now(UTC)
        assert worker.process_one("unit-analysis", sid, now) is not None
        assert _review_request(server)[1]["available_count"] == 1
        source.eof.set()  # Real hub iterator exhaustion after durable natural end.
        _until(lambda: hub.failure_reason)
        screen = _review_request(server, "/api/screen")[1]
        assert screen["video_health"]["status"] == "degraded"
        assert "capture source ended unexpectedly" in screen["video_health"]["reason"]
        assert screen["session"]["status"] == "ended"
        assert screen["frame"]["rule_status"] == "SESSION_ENDED"
        assert gate.may_run_heavy_analysis() is True
        for delay in (1, 2):
            # Actual CPU worker/SQLite publication; model identities are UNIT fakes.
            assert worker.process_one("unit-analysis", sid, now + timedelta(seconds=delay))
        review = _review_request(server)[1]
        assert review["available_count"] == 3 and review["can_complete"] is True
        clips = review["clips"]
        for clip in clips[1:]:
            status, content = _media_request(server, clip["media_url"])
            assert status == 200 and content
            payload = dict(
                session_id=sid,
                postgame_manifest_sha256=review["postgame_manifest_sha256"],
                shot_id=clip["shot_id"],
                request_id="save-" + clip["shot_id"],
            )
            first = _review_request(
                server, "/api/postgame-review/save", payload, review["kiosk_nonce"]
            )
            assert first[0] == 200 and first[1]["status"] == "SAVED"
            assert (
                _review_request(server, "/api/postgame-review/save", payload, review["kiosk_nonce"])
                == first
            )
        assert len(store.list_saved_media_outbox()) == 2
        paths = {c.shot_id: c.local_path for c in store.list_postgame_clips(sid)}
        complete = dict(session_id=sid, postgame_manifest_sha256=review["postgame_manifest_sha256"])
        status, completed = _review_request(
            server, "/api/postgame-review/complete", complete, review["kiosk_nonce"]
        )
        assert status == 200 and completed["phase"] == "REVIEW_COMPLETE"
        assert (
            process_local_discard_once(store, jobs_db, artifact_root, jobs_root).outcome
            == "completed"
        )
        assert not paths["shot-1"].exists()
        assert all(paths[shot].is_file() for shot in ("shot-2", "shot-3"))
        assert gate.may_run_heavy_analysis() is True
        store.open_or_resume_session(artifact_id="unit-new-live-match")
        assert gate.may_run_heavy_analysis() is False
        assert (
            json.loads(gate.audit_path.read_text().splitlines()[-1])["reason"]
            == "ended_session_not_idle"
        )
        assert thread.is_alive() and len(handles) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert not thread.is_alive() and source.cancelled.is_set()
