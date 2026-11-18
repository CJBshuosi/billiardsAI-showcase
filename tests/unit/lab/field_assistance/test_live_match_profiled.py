from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, replace

import pytest

from billiards_ai.edge.route_assistance.control import (
    BallGroupAssignment,
    CueBallPlacement,
    MatchControlSnapshot,
    MatchFormat,
    MatchMode,
    RackPhase,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.rule_facts import (
    FactTruth,
    RealtimePotFact,
    RealtimeRuleFacts,
    RealtimeRuleFactsBuild,
    RuleFactDeadlineExceeded,
    validate_realtime_rule_fact_preimage,
)
from billiards_ai.lab.field_assistance import live_match, live_source
from billiards_ai.lab.field_assistance.live_match import LiveMatchCoordinator
from billiards_ai.lab.mainline1_product_realtime_runtime import (
    LiveSettledShot,
    ProductRealtimeEvent,
)
from billiards_ai.lab.mainline1_shot_jobs import ShotAnalysisJob
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
)


def _match() -> MatchControlSnapshot:
    return MatchControlSnapshot(
        red_user_id="alice",
        blue_user_id="bob",
        breaker=WristbandRole.RED,
        active_role=WristbandRole.RED,
        red_group=BallGroupAssignment.SOLID,
        blue_group=BallGroupAssignment.STRIPE,
        revision=1,
        rule_profile_id=PROFILE_ID,
        rack_phase=RackPhase.ASSIGNED,
    )


def _shot() -> LiveSettledShot:
    frame = ObservationFrame(
        20,
        200,
        (
            CalibratedBallObservation("cue", 500, 500, 0.95, BallGroup.CUE, 0.95),
            CalibratedBallObservation("solid", 1500, 650, 0.95, BallGroup.SOLID, 0.95),
            CalibratedBallObservation("stripe", 1900, 900, 0.95, BallGroup.STRIPE, 0.95),
        ),
    )
    return LiveSettledShot(
        "shot-1",
        10,
        20,
        200_000_000,
        (frame,),
        True,
        None,
        {},
        "epoch-1",
        "job-1",
    )


def _facts(*, abstain: bool = False) -> RealtimeRuleFacts:
    return RealtimeRuleFacts(
        session_id="session-profiled",
        shot_id="shot-1",
        source_epoch_id="epoch-1",
        match_revision_at_start=1,
        actor_role=WristbandRole.RED,
        start_frame_id=10,
        settled_frame_id=20,
        settled_evidence_digest="e" * 64,
        first_contact_group=BallGroup.SOLID,
        pots=(),
        explicit_no_pot=FactTruth.TRUE,
        cue_ball_scratch=FactTruth.FALSE,
        post_contact_rail_or_pot=FactTruth.TRUE,
        distinct_break_object_rail_count=None,
        before_counts={
            BallGroup.CUE: 1,
            BallGroup.EIGHT: 1,
            BallGroup.SOLID: 7,
            BallGroup.STRIPE: 7,
        },
        after_counts={
            BallGroup.CUE: 1,
            BallGroup.EIGHT: 1,
            BallGroup.SOLID: 7,
            BallGroup.STRIPE: 7,
        },
        object_ball_off_table=FactTruth.FALSE,
        evidence_frame_ids=(9, 10, 20, 21),
        abstention_reasons=("tracker_uncertain:solid-1",) if abstain else (),
        digest=("a" if abstain else "f") * 64,
        profile_id=PROFILE_ID,
    )


def _terminal_facts() -> RealtimeRuleFacts:
    return replace(
        _facts(),
        first_contact_group=BallGroup.EIGHT,
        pots=(
            RealtimePotFact(
                track_id="eight-track",
                group=BallGroup.EIGHT,
                pocket_id="top-left",
                last_seen_frame_id=18,
                first_missing_frame_id=19,
                confirmed_through_frame_id=20,
                evidence_frame_ids=(18, 19, 20),
                confidence=0.99,
            ),
        ),
        explicit_no_pot=FactTruth.FALSE,
        before_counts={
            BallGroup.CUE: 1,
            BallGroup.EIGHT: 1,
            BallGroup.SOLID: 0,
            BallGroup.STRIPE: 7,
        },
        after_counts={
            BallGroup.CUE: 1,
            BallGroup.EIGHT: 0,
            BallGroup.SOLID: 0,
            BallGroup.STRIPE: 7,
        },
        digest="b" * 64,
    )


def _synthetic_fact_build(
    facts: RealtimeRuleFacts,
    *,
    required_cue_ball_placement: CueBallPlacement = CueBallPlacement.NONE,
    event_config_digest: str | None = None,
) -> RealtimeRuleFactsBuild:
    """Explicit synthetic-rule fixture, never a claim about actual perception."""
    payload = json.loads(json.dumps(asdict(facts)))
    payload.pop("digest")
    payload.update(
        required_cue_ball_placement=required_cue_ball_placement.value,
        event_config_digest=event_config_digest,
    )
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return RealtimeRuleFactsBuild(
        replace(facts, digest=hashlib.sha256(raw.encode()).hexdigest()), raw
    )


@pytest.fixture
def coordinator(tmp_path) -> LiveMatchCoordinator:
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        session_id_factory=lambda: "session-profiled",
    )
    session = store.open_or_resume_session(
        artifact_id="usb-profiled",
        rule_profile_id=PROFILE_ID,
    )
    store.save_match(session.session_id, _match(), expected_revision=0)
    return LiveMatchCoordinator(
        store=store,
        session_id=session.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
    )


def test_profiled_settle_atomically_switches_turn_and_replans(coordinator) -> None:
    coordinator.shot_started(shot_id="shot-1", started_ms=100)

    projection = coordinator.settle_profiled(_shot(), _facts(), elapsed_ms=8.5)

    assert projection.rule_status == "READY"
    assert projection.match.active_role is WristbandRole.BLUE
    assert projection.match.revision == 2
    assert projection.confirmation_ref is None
    assert projection.rule_pending_reason is None
    assert projection.guidance is not None


def test_profiled_settle_forwards_preimage_without_reencoding(coordinator, monkeypatch) -> None:
    coordinator.shot_started(shot_id="shot-1", started_ms=100)
    seen = []
    original = coordinator._store.settle_and_apply_automatic_rule_decision

    def capture_forwarding(*args, **kwargs):
        # UNIT forwarding sentinel only; never submit fake basis to persistence.
        seen.append(kwargs.pop("fact_preimage_json"))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        coordinator._store, "settle_and_apply_automatic_rule_decision", capture_forwarding
    )
    raw = '{"UNIT": "exact forwarding only"}'
    coordinator.settle_profiled(_shot(), _facts(), elapsed_ms=8.5, fact_preimage_json=raw)
    assert seen == [raw]


def test_observation_mode_settles_and_updates_turn_without_invoking_route_planner(
    tmp_path, monkeypatch
) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "observation.sqlite3",
        session_id_factory=lambda: "session-profiled",
    )
    session = store.open_or_resume_session(artifact_id="usb-profiled", rule_profile_id=PROFILE_ID)
    store.save_match(
        session.session_id,
        replace(
            _match(),
            mode=MatchMode.OBSERVATION,
            match_format=MatchFormat.SINGLE,
            race_target=None,
            ranking_intent=True,
            settings_revision=1,
        ),
        expected_revision=0,
    )
    planner = monkeypatch.setattr(
        live_match,
        "build_match_field_guidance",
        lambda **_kwargs: pytest.fail("observation mode invoked route planner"),
    )
    coordinator = LiveMatchCoordinator(
        store=store,
        session_id=session.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
    )
    coordinator.shot_started(shot_id="shot-1", started_ms=100)

    projection = coordinator.settle_profiled(_shot(), _facts(), elapsed_ms=8.5)

    assert planner is None
    assert projection.match.active_role is WristbandRole.BLUE
    assert projection.guidance is None
    assert projection.route_abstained_reason is None
    fact = store.get_live_shot_fact("session-profiled", "shot-1")
    assert fact is not None and fact.mode == "observation"


def test_observation_mode_stable_scene_never_invokes_route_planner(tmp_path, monkeypatch) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "observation-stable.sqlite3",
        session_id_factory=lambda: "session-profiled",
    )
    session = store.open_or_resume_session(artifact_id="usb-profiled", rule_profile_id=PROFILE_ID)
    store.save_match(
        session.session_id,
        replace(
            _match(),
            mode=MatchMode.OBSERVATION,
            match_format=MatchFormat.SINGLE,
            ranking_intent=True,
            settings_revision=1,
        ),
        expected_revision=0,
    )
    monkeypatch.setattr(
        live_match,
        "build_match_field_guidance",
        lambda **_kwargs: pytest.fail("observation mode invoked route planner"),
    )
    coordinator = LiveMatchCoordinator(
        store=store,
        session_id=session.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
    )

    projection = coordinator.observe_stable(
        _shot().observations[-1],
        source_epoch_id="epoch-1",
    )

    assert projection.guidance is None
    assert projection.route_abstained_reason is None


def test_profiled_settle_replay_is_idempotent_after_match_revision_advances(
    coordinator,
) -> None:
    coordinator.shot_started(shot_id="shot-1", started_ms=100)

    first = coordinator.settle_profiled(_shot(), _facts(), elapsed_ms=8.5)
    replay = coordinator.settle_profiled(_shot(), _facts(), elapsed_ms=8.5)

    assert replay == first
    assert replay.rule_status == "READY"
    assert replay.match.revision == 2
    assert len(coordinator._store.list_command_ledger("session-profiled")) == 1


def test_profiled_settle_rolls_back_shot_when_rule_receipt_fails(coordinator, monkeypatch) -> None:
    coordinator.shot_started(shot_id="shot-1", started_ms=100)
    store = coordinator._store
    apply = store.apply_automatic_rule_decision

    def fail_apply(*_args, **_kwargs):
        raise RuntimeError("injected receipt failure")

    monkeypatch.setattr(store, "apply_automatic_rule_decision", fail_apply)
    with pytest.raises(RuntimeError, match="receipt failure"):
        coordinator.settle_profiled(_shot(), _facts(), elapsed_ms=8.5)

    persisted = store.list_shots("session-profiled")[-1]
    assert persisted.settled_ms is None
    assert store.get_rule_decision("session-profiled", "shot-1") is None

    monkeypatch.setattr(store, "apply_automatic_rule_decision", apply)
    assert coordinator.settle_profiled(_shot(), _facts(), elapsed_ms=8.5).rule_status == "READY"


def test_profiled_abstention_closes_operational_shot_and_allows_next_without_guessing(
    coordinator,
) -> None:
    coordinator.shot_started(shot_id="shot-1", started_ms=100)

    projection = coordinator.settle_profiled(_shot(), _facts(abstain=True), elapsed_ms=9.0)

    assert projection.rule_status == "RULES_PENDING"
    assert projection.match.active_role is WristbandRole.RED
    assert projection.match.revision == 1
    assert projection.guidance is None
    following = coordinator.shot_started(shot_id="shot-2", started_ms=300)
    assert following.shot_id == "shot-2"
    assert following.match.active_role is WristbandRole.RED
    assert following.match.revision == 1


def test_profiled_assistance_bootstraps_route_from_current_stable_observation(
    coordinator,
) -> None:
    observation = _shot().observations[-1]

    projection = coordinator.observe_stable(
        observation,
        source_epoch_id="epoch-1",
    )

    assert projection.rule_status == "READY"
    assert projection.guidance is not None
    assert projection.route_abstained_reason is None


def test_profiled_stable_observation_rejects_stale_source_epoch(coordinator) -> None:
    with pytest.raises(ValueError, match="epoch"):
        coordinator.observe_stable(
            _shot().observations[-1],
            source_epoch_id="epoch-stale",
        )


def test_profiled_abstention_retries_route_from_next_stable_scene_without_rule_mutation(
    coordinator,
) -> None:
    coordinator.shot_started(shot_id="shot-1", started_ms=100)
    abstained = coordinator.settle_profiled(_shot(), _facts(abstain=True), elapsed_ms=9.0)
    next_scene = replace(
        _shot().observations[-1],
        frame_id=30,
        timestamp_ms=300,
    )

    recovered = coordinator.observe_stable(next_scene, source_epoch_id="epoch-1")

    assert recovered.rule_status == "RULES_PENDING"
    assert recovered.guidance is not None
    assert recovered.match.active_role is abstained.match.active_role is WristbandRole.RED
    assert recovered.match.revision == abstained.match.revision == 1
    assert recovered.rule_pending_reason == "RULE_DECISION_ABSTAINED"


def test_profiled_settle_rejects_fact_lineage_mismatch(coordinator) -> None:
    coordinator.shot_started(shot_id="shot-1", started_ms=100)

    with pytest.raises(ValueError, match="lineage"):
        coordinator.settle_profiled(
            _shot(),
            replace(_facts(), source_epoch_id="epoch-stale"),
            elapsed_ms=5.0,
        )


def test_usb_source_routes_settled_event_through_profiled_rules(tmp_path, monkeypatch) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "source.sqlite3",
        session_id_factory=lambda: "session-profiled",
    )
    session = store.open_or_resume_session(
        artifact_id="usb-profiled",
        rule_profile_id=PROFILE_ID,
    )
    store.save_match(session.session_id, _match(), expected_revision=0)

    class Capture:
        def __init__(self) -> None:
            self.cancelled = threading.Event()
            self.ready = threading.Event()

        def run(self) -> None:
            self.ready.set()
            self.cancelled.wait(3)

        def close(self) -> None:
            self.cancelled.set()

        def wait_until_ready(self, *, timeout_seconds: float) -> bool:
            return self.ready.wait(timeout_seconds) and not self.cancelled.is_set()

    capture = Capture()
    source = live_source.USBLiveProjectionSource(
        store=store,
        session_id=session.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
        runtime_factory=lambda *_: capture,
        match_lock=threading.RLock(),
    )
    monkeypatch.setattr(
        live_source,
        "build_realtime_rule_facts_with_preimage",
        lambda shot, **context: _synthetic_fact_build(
            _facts(),
            required_cue_ball_placement=context["required_cue_ball_placement"],
            event_config_digest=shot.event_config_digest,
        ),
    )
    source.start()
    try:
        source.channel.publish(
            ProductRealtimeEvent(
                "SHOT_STARTED",
                shot_id="shot-1",
                frame_id=10,
                started_ms=100,
            )
        )
        source.channel.publish(
            ProductRealtimeEvent(
                "SHOT_SETTLED",
                shot_id="shot-1",
                frame_id=20,
                settled=_shot(),
            )
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            projection = source.current().frame.payload
            if (
                projection["rule_status"] == "READY"
                and store.load_session(session.session_id).match.revision == 2
            ):
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"profiled rule result did not become ready: {projection}")
        assert projection["guidance_visible"] is True
        assert store.load_session(session.session_id).match.active_role is WristbandRole.BLUE
        receipt = store.get_rule_decision(session.session_id, "shot-1")
        raw = store.get_rule_fact_preimage(session.session_id, "shot-1")
        assert receipt is not None and raw is not None
        normalized = validate_realtime_rule_fact_preimage(
            json.loads(receipt.facts_json), raw, expected_digest=receipt.fact_digest
        )
        assert normalized["required_cue_ball_placement"] == "none"
        assert normalized["event_config_digest"] is None
        with pytest.raises(ValueError, match="rules engine"):
            source.apply_control({"command": "continue_turn"})
    finally:
        source.close()


def test_usb_source_projects_current_stable_runtime_observation_before_first_shot(
    tmp_path,
) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "stable-source.sqlite3",
        session_id_factory=lambda: "session-profiled",
    )
    session = store.open_or_resume_session(
        artifact_id="usb-profiled",
        rule_profile_id=PROFILE_ID,
    )
    store.save_match(session.session_id, _match(), expected_revision=0)

    class Capture:
        current_observation = _shot().observations[-1]
        current_capture_frame = None
        calibration = None
        failure_reason = None

        def run(self) -> None:
            return None

        def close(self) -> None:
            return None

        def wait_until_ready(self, *, timeout_seconds: float) -> bool:
            return True

    source = live_source.USBLiveProjectionSource(
        store=store,
        session_id=session.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
        runtime_factory=lambda *_: Capture(),
        match_lock=threading.RLock(),
    )

    projection = source.current().frame.payload

    assert projection["phase"] == "stable"
    assert projection["rule_status"] == "READY"
    assert projection["guidance_visible"] is True
    assert projection["guidance"] is not None


def test_rule_fact_deadline_persists_pending_abstention_without_stopping_capture(
    tmp_path, monkeypatch
) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "deadline.sqlite3",
        session_id_factory=lambda: "session-profiled",
    )
    session = store.open_or_resume_session(
        artifact_id="usb-profiled",
        rule_profile_id=PROFILE_ID,
    )
    store.save_match(session.session_id, _match(), expected_revision=0)

    class Capture:
        def __init__(self) -> None:
            self.cancelled = threading.Event()
            self.ready = threading.Event()

        def run(self) -> None:
            self.ready.set()
            self.cancelled.wait(3)

        def close(self) -> None:
            self.cancelled.set()

        def wait_until_ready(self, *, timeout_seconds: float) -> bool:
            return self.ready.wait(timeout_seconds) and not self.cancelled.is_set()

    capture = Capture()
    source = live_source.USBLiveProjectionSource(
        store=store,
        session_id=session.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
        runtime_factory=lambda *_: capture,
        match_lock=threading.RLock(),
    )

    def exceed_deadline(*_args, **_kwargs):
        raise RuleFactDeadlineExceeded("deadline during pots")

    monkeypatch.setattr(live_source, "build_realtime_rule_facts_with_preimage", exceed_deadline)
    source.start()
    try:
        source.channel.publish(
            ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-1", frame_id=10, started_ms=100)
        )
        source.channel.publish(
            ProductRealtimeEvent("SHOT_SETTLED", shot_id="shot-1", frame_id=20, settled=_shot())
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            receipt = store.get_rule_decision(session.session_id, "shot-1")
            if receipt is not None:
                break
            time.sleep(0.01)
        else:
            pytest.fail("deadline abstention was not persisted")
        projection = source.current().frame.payload
        assert receipt.status == "abstained"
        raw = store.get_rule_fact_preimage(session.session_id, "shot-1")
        assert raw is not None
        normalized = validate_realtime_rule_fact_preimage(
            json.loads(receipt.facts_json), raw, expected_digest=receipt.fact_digest
        )
        assert normalized["abstention_reasons"] == ["rule_fact_deadline_exceeded"]
        assert projection["rule_status"] == "RULES_PENDING"
        assert projection["degraded_reason"] is None
        assert projection["guidance"] is None
        assert not capture.cancelled.is_set()
        match = store.load_session(session.session_id).match
        assert match.revision == 1 and match.active_role is WristbandRole.RED
    finally:
        source.close()


def test_native_interruption_settlement_database_failure_remains_terminal(
    tmp_path, monkeypatch
) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "interrupted-failure.sqlite3",
        session_id_factory=lambda: "session-profiled",
    )
    session = store.open_or_resume_session(
        artifact_id="usb-profiled",
        rule_profile_id=PROFILE_ID,
    )
    store.save_match(session.session_id, _match(), expected_revision=0)

    class Capture:
        def __init__(self) -> None:
            self.cancelled = threading.Event()
            self.ready = threading.Event()

        def run(self) -> None:
            self.ready.set()
            self.cancelled.wait(3)

        def close(self) -> None:
            self.cancelled.set()

        def wait_until_ready(self, *, timeout_seconds: float) -> bool:
            return self.ready.wait(timeout_seconds) and not self.cancelled.is_set()

    capture = Capture()
    source = live_source.USBLiveProjectionSource(
        store=store,
        session_id=session.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
        runtime_factory=lambda *_: capture,
        match_lock=threading.RLock(),
    )

    def fail_settlement(*_args, **_kwargs):
        raise ValueError("field record database is invalid")

    monkeypatch.setattr(source.coordinator, "settle_profiled", fail_settlement)
    source.start()
    try:
        source.channel.publish(
            ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-1", frame_id=10, started_ms=100)
        )
        source.channel.publish(
            ProductRealtimeEvent(
                "SHOT_INCOMPLETE",
                shot_id="shot-1",
                frame_id=12,
                reason="NativeCaptureProtocolError: native capture stream is truncated",
            )
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            projection = source.current().frame.payload
            if projection["rule_status"] == "DEGRADED":
                break
            time.sleep(0.01)
        else:
            pytest.fail("failed interruption settlement did not revoke authority")
        assert not source.restart_required
        assert projection["guidance"] is None
        assert "database is invalid" in projection["degraded_reason"]
        assert capture.cancelled.is_set()
    finally:
        source.close()


def test_profiled_terminal_waits_for_durable_enqueued_job_before_ending(
    tmp_path, monkeypatch
) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "terminal.sqlite3",
        session_id_factory=lambda: "session-profiled",
    )
    session = store.open_or_resume_session(
        artifact_id="usb-profiled",
        rule_profile_id=PROFILE_ID,
    )
    store.save_match(
        session.session_id,
        replace(_match(), rack_phase=RackPhase.ON_EIGHT),
        expected_revision=0,
    )

    class Capture:
        def __init__(self) -> None:
            self.cancelled = threading.Event()
            self.ready = threading.Event()

        def run(self) -> None:
            self.ready.set()
            self.cancelled.wait(3)

        def close(self) -> None:
            self.cancelled.set()

        def wait_until_ready(self, *, timeout_seconds: float) -> bool:
            return self.ready.wait(timeout_seconds) and not self.cancelled.is_set()

    capture = Capture()
    source = live_source.USBLiveProjectionSource(
        store=store,
        session_id=session.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
        runtime_factory=lambda *_: capture,
        match_lock=threading.RLock(),
    )
    monkeypatch.setattr(
        live_source,
        "build_realtime_rule_facts_with_preimage",
        lambda shot, **context: _synthetic_fact_build(
            _terminal_facts(),
            required_cue_ball_placement=context["required_cue_ball_placement"],
            event_config_digest=shot.event_config_digest,
        ),
    )
    source.start()
    try:
        source.channel.publish(
            ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-1", frame_id=10, started_ms=100)
        )
        source.channel.publish(
            ProductRealtimeEvent("SHOT_SETTLED", shot_id="shot-1", frame_id=20, settled=_shot())
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            pending = source.current().frame.payload
            if (
                pending["rule_status"] == "EVALUATING"
                and store.load_session(session.session_id).match.rack_phase is RackPhase.ENDED
            ):
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"terminal rule result did not enter durable-job wait: {pending}")
        assert store.load_session(session.session_id).status.value == "open"
        assert store.get_postgame_manifest(session.session_id) is None

        source.channel.publish(
            ProductRealtimeEvent(
                "SHOT_ENQUEUED",
                shot_id="shot-1",
                frame_id=20,
                job=ShotAnalysisJob(
                    "job-1",
                    session.session_id,
                    "shot-1",
                    "epoch-1",
                    10,
                    20,
                    tmp_path / "media",
                    "a" * 64,
                    "setup",
                    "cal-1",
                    "c" * 64,
                    "model",
                ),
            )
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if (
                store.load_session(session.session_id).status.value == "ended"
                and capture.cancelled.is_set()
            ):
                break
            time.sleep(0.01)
        else:
            pytest.fail("durable terminal job did not end the session and capture")
        assert store.get_postgame_manifest(session.session_id).expected_jobs == ("job-1",)
        assert capture.cancelled.is_set()
    finally:
        source.close()
