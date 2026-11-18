"""Typed real-store/coordinator controls. Facts are synthetic, never video acceptance."""

import hashlib
import json
import threading
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope, reduce_field_command
from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionBinding,
    ManagedSessionScope,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import LiveShotMediaReference
from billiards_ai.edge.route_assistance.rule_facts import (
    FactTruth,
    RealtimePotFact,
    RealtimeRuleFacts,
    build_abstained_realtime_rule_facts_with_preimage,
)
from billiards_ai.edge.route_assistance.runtime_resources import ManagedRuntimeResourceSpec
from billiards_ai.lab.field_assistance.live_match import LiveMatchCoordinator
from billiards_ai.lab.mainline1_product_realtime_runtime import (
    LiveSettledShot,
    ProductRealtimeRuntime,
)
from billiards_ai.lab.mainline1_realtime_ring import FrozenCaptureInterval
from billiards_ai.lab.mainline1_shot_jobs import (
    SQLiteShotAnalysisJobStore,
    _interval_payload,
)
from billiards_ai.lab.shot_local.models import BallGroup
from scripts.algorithm.offline_fullgame_observer import FullgameObserver, InputOracle
from scripts.algorithm.run_offline_managed_blackbox import _Actor, _fullgame_queue_job
from tests.unit.edge.route_assistance.test_runtime_resources import payload
from tests.unit.scripts.test_offline_fullgame_observer import EPOCH, SID


def real_records(tmp_path, *, capture_setup_id="setup-a"):
    raw = payload()
    raw["capture_setup_id"] = capture_setup_id
    # Relocate only fixture resource paths; these are metadata, not model assets.
    text = json.dumps(raw).replace("/original/\\u53f0\\u7403", str(tmp_path))
    raw = json.loads(text)
    raw["artifact_id"] = (
        "usb-field-" + hashlib.sha256(raw["paths"]["artifact_root"].encode()).hexdigest()
    )
    spec = ManagedRuntimeResourceSpec.from_payload(raw)
    scope = ManagedSessionScope("offline-development", "offline-device", "offline-table")
    request = "22222222-2222-4222-8222-222222222222"
    store = SQLiteFieldRecordStore(
        tmp_path / "state.db", managed_open_request_id_factory=lambda: request
    )
    store.reserve_managed_open(
        scope, artifact_id=spec.artifact_id, rule_profile_id=PROFILE_ID, resource_spec=spec
    )
    binding = ManagedSessionBinding(
        "1.0", SID, SID[8:], "managed_session_birth_v1", scope.table_id, scope.device_id, "open"
    )
    store.record_managed_binding(scope, request, binding)
    store.materialize_managed_session(scope, request)
    store.register_managed_capture_epoch(scope, request, EPOCH)
    for index, (actor, kind, content) in enumerate(
        [
            ("offline-red", "claim_role", {"role": "red"}),
            ("offline-blue", "claim_role", {"role": "blue"}),
            (
                "offline-red",
                "set_breaker",
                dict(
                    role="red",
                    mode="assistance",
                    format="single",
                    race_target=None,
                    ranking_intent=False,
                    settings_revision=1,
                ),
            ),
        ],
        1,
    ):
        envelope = FieldCommandEnvelope(
            f"command-{index}",
            SID,
            actor,
            kind,
            content,
            store.load_session(SID).match.revision,
            index,
        )
        assert store.apply_field_command(envelope, reduce_field_command).outcome == "applied"
    database = tmp_path / "jobs_root" / "analysis-jobs.sqlite3"
    jobs = SQLiteShotAnalysisJobStore(database)
    actor = _Actor(
        NS(),
        {"capture": {"source_sha256": "2" * 64}},
        threading.Event(),
        (60, 45, 10, 300),
        database,
    )
    actor.store, actor.session, actor.epoch = store, SID, EPOCH
    actor.state = NS(match_lock=threading.RLock())
    coordinator = LiveMatchCoordinator(
        store=store, session_id=SID, source_epoch_id=EPOCH, calibration_id="a" * 64
    )
    return store, spec, jobs, actor, coordinator


def actual_shot(
    store,
    spec,
    jobs,
    coordinator,
    sequence,
    counts,
    *,
    no_contact=False,
    use_actual_abstained_builder=False,
):
    group = (
        BallGroup.STRIPE if sequence == 1 else BallGroup.SOLID if sequence < 9 else BallGroup.EIGHT
    )
    before, after = dict(counts), dict(counts)
    after[group] -= 1
    shot_id = f"shot-{sequence:04d}"
    start, settle = sequence * 100, sequence * 100 + 60
    root = Path(spec.paths["jobs_root"]) / "media" / SID
    frozen = FrozenCaptureInterval(
        root,
        EPOCH,
        1920,
        1080,
        start - 2,
        settle + 2,
        65,
        {f"{start - 2:020d}-{settle + 2:020d}.mjpgseg": "f" * 64},
    )
    # Call the real producer's pure metadata method. No runtime constructor,
    # detector/model initialization, capture, or engine is executed.
    runtime = object.__new__(ProductRealtimeRuntime)
    runtime._session_id = SID
    runtime._capture_setup_id = spec.capture_setup_id
    runtime._calibration_artifact_id = spec.calibration_artifact_id
    runtime._config_sha256 = spec.config_sha256
    runtime._job_media_root = root
    runtime._detector = NS(package_identity=NS(package_id=spec.model_package_id))
    runtime._identity_training_package_id = spec.identity_training_package_id
    runtime._identity_training_package_sha256 = spec.identity_training_package_sha256
    job = runtime._analysis_job(NS(shot_id=shot_id), frozen)
    job_id = job.job_id
    job.media_root.mkdir(parents=True)
    (job.media_root / "interval.json").write_text(json.dumps(_interval_payload(frozen)))
    jobs.enqueue(job)
    coordinator.shot_started(shot_id=shot_id, started_ms=start)
    facts = RealtimeRuleFacts(
        SID,
        shot_id,
        EPOCH,
        store.load_session(SID).match.revision,
        WristbandRole.RED,
        start,
        settle,
        "e" * 64,
        group,
        (
            RealtimePotFact(
                f"track-{sequence}",
                group,
                "top-left",
                settle - 2,
                settle - 1,
                settle,
                (settle - 2, settle - 1, settle),
                0.99,
            ),
        ),
        FactTruth.FALSE,
        FactTruth.FALSE,
        FactTruth.TRUE,
        None,
        before,
        after,
        FactTruth.FALSE,
        (start, settle),
        (),
        str(sequence % 10) * 64,
        PROFILE_ID,
    )
    if no_contact:
        facts = replace(
            facts,
            first_contact_group=None,
            pots=(),
            after_counts=before,
            explicit_no_pot=FactTruth.TRUE,
        )
        after = before
    settled = LiveSettledShot(
        shot_id, start, settle, settle * 1_000_000, (), True, None, {}, EPOCH, job_id
    )
    # Explicit UNIT normalization evidence for this hand-constructed fact set;
    # it is not source-frame/perception evidence or a production builder call.
    if use_actual_abstained_builder:
        match = store.load_session(SID).match
        build = build_abstained_realtime_rule_facts_with_preimage(
            settled,
            session_id=SID,
            actor_role=WristbandRole.RED,
            match_revision=match.revision,
            profile_id=PROFILE_ID,
            reason="unit_missing_tracking_frames",
            required_cue_ball_placement=match.cue_ball_placement,
        )
        facts, preimage = build.facts, build.normalization_preimage_json
        after = before
    else:
        normalized = asdict(facts)
        normalized.pop("digest")
        normalized.update(
            required_cue_ball_placement=store.load_session(SID).match.cue_ball_placement.value,
            event_config_digest=None,
        )
        preimage = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        facts = replace(facts, digest=hashlib.sha256(preimage.encode()).hexdigest())
    projection = coordinator.settle_profiled(
        settled, facts, elapsed_ms=1, fact_preimage_json=preimage
    )
    rule = store.get_rule_decision(SID, shot_id)
    assert rule.status == ("abstained" if no_contact or use_actual_abstained_builder else "applied")
    return after, job, projection


def attach(store, job):
    owner = store.get_shot_ownership(SID, job.shot_id)
    store.attach_live_shot_media(
        LiveShotMediaReference(
            SID,
            job.shot_id,
            EPOCH,
            owner.match_revision,
            job.job_id,
            str(job.media_root),
            job.media_sha256,
        )
    )


def test_actual_builder_basis_flows_through_coordinator_store_and_observer(tmp_path):
    store, spec, jobs, actor, coordinator = real_records(tmp_path)
    initial = store.load_session(SID).match
    counts = {BallGroup.CUE: 1, BallGroup.EIGHT: 1, BallGroup.SOLID: 7, BallGroup.STRIPE: 7}
    # The input shot deliberately has no tracked rule frames. The actual builder
    # must retain its UNKNOWN assessment, never invent a successful contact.
    _, job, _ = actual_shot(
        store, spec, jobs, coordinator, 1, counts, use_actual_abstained_builder=True
    )
    receipt = store.get_rule_decision(SID, job.shot_id)
    raw = store.get_rule_fact_preimage(SID, job.shot_id)
    assert hashlib.sha256(raw.encode()).hexdigest() == receipt.fact_digest
    normalized = json.loads(raw)
    assert normalized["abstention_reasons"] == ["unit_missing_tracking_frames"]
    assert normalized["event_config_digest"] is None
    assert normalized["required_cue_ball_placement"] == initial.cue_ball_placement.value
    assert receipt.status == "abstained"
    assert store.load_session(SID).match == initial

    oracle = InputOracle(
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        1,
        "red",
        (("red", "continue_turn", "legal_break_object_pot"), ("red", "end", "legal_eight")),
    )
    observer = FullgameObserver(actor, oracle, match_seconds=10)
    assert observer.poll() is False
    assert observer.report["perception_fact_quality"] == "UNASSESSED"
    assert "FULLGAME_FACT_PREIMAGE_MISSING" not in observer.report["failures"]
    assert "FULLGAME_FACT_PREIMAGE_INVALID" not in observer.report["failures"]
    assert observer.report["failures"]  # A verified basis is not a successful shot/rule.
    assert observer.report["snapshots"][0]["rule"]["fact_digest"] == receipt.fact_digest


def test_typed_real_store_natural_terminal_waits_for_durable_final_job(tmp_path):
    store, spec, jobs, actor, coordinator = real_records(tmp_path)
    expected = (
        ("red", "continue_turn", "legal_break_object_pot"),
        ("red", "continue_turn", "open_table_group_assigned"),
        *(("red", "continue_turn", "own_group_potted"),) * 6,
        ("red", "end", "legal_eight"),
    )
    # 1 break + 1 assignment + 6 remaining own balls + eight = 9 recorded shots.
    oracle = InputOracle("1" * 64, "2" * 64, "3" * 64, "4" * 64, 1, "red", expected)
    observer = FullgameObserver(actor, oracle, match_seconds=10)
    counts = {BallGroup.CUE: 1, BallGroup.EIGHT: 1, BallGroup.SOLID: 7, BallGroup.STRIPE: 7}
    for sequence in range(1, len(expected) + 1):
        counts, job, projection = actual_shot(store, spec, jobs, coordinator, sequence, counts)
        if sequence == len(expected):
            assert projection.rule_status == "EVALUATING"
            assert store.load_session(SID).status.value == "open"
            assert store.get_postgame_manifest(SID) is None
            assert observer.poll() is False
        attach(store, job)
        if sequence < len(expected):
            assert observer.poll() is False
    coordinator.finalize_profiled_terminal(shot_id=job.shot_id, job_id=job.job_id)
    assert observer.poll() is True
    assert observer.report["failures"] == []
    assert len(observer.durable) == len(expected)
    assert store.load_session(SID).match.terminal_reason == "legal_eight"
    assert observer.report["terminal"]["job_id"] == job.job_id


@pytest.mark.parametrize("fault", ["interval", "resource", "rule_span"])
def test_actual_job_metadata_reader_rejects_wrong_resource_and_rule_span(tmp_path, fault):
    import sqlite3

    store, spec, jobs, actor, coordinator = real_records(tmp_path)
    counts = {BallGroup.CUE: 1, BallGroup.EIGHT: 1, BallGroup.SOLID: 7, BallGroup.STRIPE: 7}
    _, job, _ = actual_shot(store, spec, jobs, coordinator, 1, counts)
    attach(store, job)
    shot = store.list_shots(SID)[0]
    media = store.get_live_shot_media(SID, shot.shot_id)
    assert actor.fullgame_job(shot, media)["resource_spec_sha256"] == spec.spec_sha256
    if fault == "interval":
        interval = job.media_root / "interval.json"
        data = json.loads(interval.read_text())
        data["end_frame_id"] -= 1
        interval.write_text(json.dumps(data))
    elif fault == "resource":
        # Structurally valid queue metadata with a different resource cannot
        # inherit this managed session's pinned runtime resource identity.
        with sqlite3.connect(actor.database) as db:
            raw = json.loads(
                db.execute("select payload_json from shot_analysis_jobs").fetchone()[0]
            )
            raw["config_sha256"] = "b" * 64
            encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"))
            db.execute(
                "update shot_analysis_jobs set payload_json=?,payload_sha256=?",
                (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
            )
    else:
        with sqlite3.connect(store.path) as db:
            db.execute("update field_rule_decisions set start_frame_id=200,settled_frame_id=210")
    with pytest.raises(ValueError):
        actor.fullgame_job(shot, media)


@pytest.mark.parametrize("mutate_identity", [False, True])
def test_producer_unicode_job_identity_and_mutation_control(tmp_path, monkeypatch, mutate_identity):
    import sqlite3

    def forbidden_init(*_, **__):
        pytest.fail("metadata control must not initialize the runtime")

    monkeypatch.setattr(ProductRealtimeRuntime, "__init__", forbidden_init)
    store, spec, jobs, actor, coordinator = real_records(tmp_path, capture_setup_id="台球桌")
    counts = {BallGroup.CUE: 1, BallGroup.EIGHT: 1, BallGroup.SOLID: 7, BallGroup.STRIPE: 7}
    _, job, _ = actual_shot(store, spec, jobs, coordinator, 1, counts)
    attach(store, job)
    shot = store.list_shots(SID)[0]
    media = store.get_live_shot_media(SID, shot.shot_id)
    if mutate_identity:
        with sqlite3.connect(actor.database) as db:
            raw = json.loads(
                db.execute("select payload_json from shot_analysis_jobs").fetchone()[0]
            )
            raw["capture_setup_id"] = "台球桌已更改"
            encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"))
            db.execute(
                "update shot_analysis_jobs set payload_json=?,payload_sha256=?",
                (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
            )
        with pytest.raises(ValueError, match="fullgame job lineage mismatch"):
            actor.fullgame_job(shot, media)
    else:
        observed = actor.fullgame_job(shot, media)
        assert observed["job_id"] == job.job_id
        assert observed["capture_setup_id"] == "台球桌"


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "extra",
        "bool_frame",
        "nonfinite",
        "oversize",
        "sha",
        "relative",
        "surrogate",
    ],
)
def test_fullgame_queue_reader_keeps_strict_bounded_producer_contract(tmp_path, fault):
    import sqlite3

    store, spec, jobs, actor, coordinator = real_records(tmp_path)
    counts = {BallGroup.CUE: 1, BallGroup.EIGHT: 1, BallGroup.SOLID: 7, BallGroup.STRIPE: 7}
    _, job, _ = actual_shot(store, spec, jobs, coordinator, 1, counts)
    with sqlite3.connect(actor.database) as db:
        db.row_factory = sqlite3.Row
        raw = db.execute("select payload_json from shot_analysis_jobs").fetchone()[0]
        data = json.loads(raw)
        if fault == "extra":
            data["unexpected"] = "rejected"
        elif fault == "bool_frame":
            data["start_frame_id"] = True
        elif fault == "nonfinite":
            data["start_frame_id"] = float("nan")
        elif fault == "relative":
            data["media_root"] = "relative/path"
        elif fault == "surrogate":
            data["capture_setup_id"] = "\ud800"
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
        if fault == "duplicate":
            raw = raw[:-1] + ',"job_id":' + json.dumps(job.job_id) + "}"
        elif fault == "oversize":
            raw += " " * (1024**2)
        digest = "0" * 64 if fault == "sha" else hashlib.sha256(raw.encode()).hexdigest()
        db.execute("update shot_analysis_jobs set payload_json=?,payload_sha256=?", (raw, digest))
        with pytest.raises(ValueError):
            _fullgame_queue_job(db, job.job_id, SID, job.shot_id)


def test_real_abstained_receipt_is_retained_and_later_shot_is_still_observed(tmp_path):
    store, spec, jobs, actor, coordinator = real_records(tmp_path)
    expected = (("red", "continue_turn", "legal_break_object_pot"), ("red", "end", "legal_eight"))
    observer = FullgameObserver(
        actor,
        InputOracle("1" * 64, "2" * 64, "3" * 64, "4" * 64, 1, "red", expected),
        match_seconds=10,
    )
    counts = {BallGroup.CUE: 1, BallGroup.EIGHT: 1, BallGroup.SOLID: 7, BallGroup.STRIPE: 7}
    counts, job, _ = actual_shot(store, spec, jobs, coordinator, 1, counts, no_contact=True)
    attach(store, job)
    assert not observer.poll()
    assert "FULLGAME_RULE_NOT_APPLIED" in observer.report["failures"]
    counts, job, _ = actual_shot(store, spec, jobs, coordinator, 2, counts)
    attach(store, job)
    assert not observer.poll()
    assert any(r["sequence"] == 2 for r in observer.report["snapshots"])
    assert any(r.get("rule", {}).get("status") == "abstained" for r in observer.report["snapshots"])
    assert store.load_session(SID).status.value == "open"
