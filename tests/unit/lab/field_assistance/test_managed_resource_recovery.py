"""Original resources use actual retained scenes, epochs and immutable queued jobs."""

import hashlib
import importlib
import importlib.util
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import (
    FieldShotRecord,
    LiveShotMediaReference,
    RecordedMediaV1,
    RecordingBoundaryV1,
    ShotMediaRange,
)
from billiards_ai.lab.field_assistance.runtime_resources import inspect_managed_runtime_resources
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
    SessionRecordingEvidenceStore,
)
from billiards_ai.lab.mainline1_shot_jobs import (
    ShotAnalysisJob,
    SQLiteShotAnalysisJobStore,
    materialize_recorded_interval_reference,
)
from tests.unit.lab.field_assistance import test_runtime_resources as resource_fixtures
from tests.unit.lab.field_assistance.test_managed_startup import BINDING, SCOPE
from tests.unit.lab.test_mainline1_shot_worker import _materialized_interval

resource_assets_fixture = resource_fixtures.assets


@pytest.fixture
def scene(resource_assets_fixture, request):
    variant = getattr(request, "param", None)
    owner_composition = variant in {"owner-composition", "owner-interrupted"}
    args = resource_assets_fixture
    from dataclasses import asdict

    import yaml

    from tests.unit.lab.test_mainline1_shot_worker import _worker_config

    args.config.write_text(yaml.safe_dump(asdict(_worker_config())))
    # Real valid calibration JSON; tiny package assets remain model-boundary fixtures.
    import hashlib
    import json

    from billiards_ai.lab.mainline1_calibration_registry import register_calibration
    from billiards_ai.lab.mainline1_capture import load_capture_setup
    from billiards_ai.shared.calibration.homography import calibrate

    directory = args.calibration_root / ("a" * 64)
    data = (
        calibrate(
            "fixture",
            ((0, 120), (160, 120), (160, 0), (0, 0)),
            table_length_mm=160,
            table_width_mm=120,
        )
        .model_dump_json()
        .encode()
    )
    (directory / "calibration.json").write_bytes(data)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["inventory"]["calibration.json"] = {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (args.calibration_root / "registry.json").unlink()
    register_calibration(
        args.calibration_root,
        SimpleNamespace(path=directory, artifact_id="a" * 64),
        load_capture_setup(args.capture_setup),
    )
    spec = inspect_managed_runtime_resources(args, rule_profile_id=None)
    from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import NOW

    clock = {"now_utc": lambda: NOW} if owner_composition else {}
    store = SQLiteFieldRecordStore(args.artifact_root.parent / "state.sqlite", **clock)
    request = store.reserve_managed_open(SCOPE, artifact_id=spec.artifact_id, resource_spec=spec)
    store.record_managed_binding(SCOPE, request.open_request_id, BINDING)
    store.materialize_managed_session(SCOPE, request.open_request_id)
    store.bind_artifact_path(spec.artifact_id, args.artifact_root)
    from billiards_ai.edge.route_assistance.control import MatchControlSnapshot, WristbandRole

    store.save_match(
        BINDING.session_id,
        MatchControlSnapshot(
            red_user_id="user-a", blue_user_id="user-b", active_role=WristbandRole.RED, revision=1
        ),
        expected_revision=0,
    )
    jobs = SQLiteShotAnalysisJobStore(args.jobs_root / "analysis-jobs.sqlite3")
    for index in (1, 2, 3) if variant == "owner-composition" else (1, 2):
        role = WristbandRole.BLUE if owner_composition and index > 1 else WristbandRole.RED
        if owner_composition and index == 2:
            current = store.load_session(BINDING.session_id).match
            store.save_match(
                BINDING.session_id,
                replace(current, active_role=role, revision=current.revision + 1),
                expected_revision=current.revision,
            )
        epoch, jid, shot = f"epoch-{index}", f"job-{index}", f"shot-{index}"
        store.register_managed_capture_epoch(SCOPE, request.open_request_id, epoch)
        materialized = _materialized_interval(
            args.ring_root / epoch,
            epoch=epoch,
            job_id=jid,
            media_root=args.jobs_root / "media" / BINDING.session_id / jid,
        )
        started = FieldShotRecord(
            session_id=BINDING.session_id,
            shot_id=shot,
            sequence=index,
            actor_role=role,
            started_ms=80,
            settled_ms=None,
            recommendation_id=None,
            power_band=None,
            power_percent=None,
            media=ShotMediaRange(
                artifact_id=spec.artifact_id,
                source_locator=f"usb:{epoch}",
                source_kind="usb_live",
                start_ms=0,
                end_ms=320,
                source_epoch_id=epoch,
            ),
        )
        store.record_shot_start(started, source_start_event_id=f"start-{shot}")
        from billiards_ai.edge.route_assistance.records import LiveShotFact

        fact = (
            LiveShotFact.create(
                session_id=BINDING.session_id,
                shot_id=shot,
                sequence=index,
                actor_role=role,
                actor_user_id="user-a" if index == 1 else "user-b",
                start_frame_id=1,
                settle_frame_id=7,
                mode="observation",
                observed_outcome="unknown",
                rule_revision=index,
                recording_reference=f"session-recording:{epoch}:1-7",
            )
            if owner_composition
            else None
        )
        store.record_shot_settle(
            replace(started, settled_ms=240, live_evidence_digest="e" * 64, live_job_id=jid),
            source_epoch_id=epoch,
            live_fact=fact,
        )
        store.attach_live_shot_media(
            LiveShotMediaReference(
                session_id=BINDING.session_id,
                shot_id=shot,
                source_epoch_id=epoch,
                match_revision=store.load_session(BINDING.session_id).match.revision,
                job_id=jid,
                media_path=str(materialized.root),
                media_sha256=materialized.media_sha256,
            )
        )
        jobs.enqueue(
            ShotAnalysisJob(
                jid,
                BINDING.session_id,
                shot,
                epoch,
                materialized.start_frame_id,
                materialized.end_frame_id,
                materialized.root,
                materialized.media_sha256,
                spec.capture_setup_id,
                spec.calibration_artifact_id,
                spec.config_sha256,
                spec.model_package_id,
                spec.identity_training_package_id,
                spec.identity_training_package_sha256,
            )
        )
    if variant in {"interrupted", "owner-interrupted"}:
        append_capture_interruption(store, request.open_request_id, spec)
    store.end_session(BINDING.session_id)
    return SimpleNamespace(
        args=args, spec=spec, store=store, jobs=jobs, request=request, sid=BINDING.session_id
    )


def append_capture_interruption(store, open_request_id, spec):
    """Actual source/coordinator handling; only the unused capture handle is synthetic."""
    import threading

    from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource
    from billiards_ai.lab.mainline1_product_realtime_runtime import ProductRealtimeEvent

    epoch = "epoch-interrupted"
    store.register_managed_capture_epoch(SCOPE, open_request_id, epoch)
    source = USBLiveProjectionSource(
        store=store,
        session_id=BINDING.session_id,
        source_epoch_id=epoch,
        calibration_id=spec.calibration_artifact_id,
        runtime_factory=lambda *args: SimpleNamespace(close=lambda: None),
        match_lock=threading.RLock(),
    )
    source._apply_event(
        ProductRealtimeEvent(kind="SHOT_STARTED", shot_id="shot-3", frame_id=10, started_ms=250)
    )
    source._apply_event(
        ProductRealtimeEvent(
            kind="CAPTURE_DEGRADED",
            shot_id="shot-3",
            frame_id=11,
            reason="NativeCaptureProtocolError: synthetic source loss",
        )
    )
    assert source.restart_required
    assert store.list_shots(BINDING.session_id)[-1].live_job_id == (
        "abstained:capture_interrupted:epoch-interrupted:shot-3"
    )


def guard_child_pre_admission(monkeypatch, module):
    """Guard both nested and child readers, plus model/provider/queue effects."""
    import builtins

    from billiards_ai.lab.field_assistance import runtime_resources

    def forbidden(*args, **kwargs):
        pytest.fail("unadmitted child reached resource inspection/config/model/provider/queue")

    monkeypatch.setattr(runtime_resources, "_inspect", forbidden)
    monkeypatch.setattr(runtime_resources, "load_mainline1_shot_local_config", forbidden)
    for name in (
        "load_mainline1_shot_local_config",
        "load_development_vision_backend",
        "load_development_group_classifier",
        "Dinov2IdentityFeatureBackend",
        "SQLiteShotAnalysisJobStore",
        "_development_analysis_gateway",
        "WindowsCredentialManagerProvider",
        "UrllibSyncHttpTransport",
    ):
        monkeypatch.setattr(module, name, forbidden)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "onnxruntime", "rfdetr"}:
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def recovery():
    name = "billiards_ai.lab.field_assistance.managed_resource_recovery"
    assert importlib.util.find_spec(name) is not None, "original resource resolver is missing"
    return importlib.import_module(name)


def resolve(scene, purpose="analysis"):
    return recovery().resolve_managed_session_resources(
        SQLiteFieldRecordStore(scene.store.path), SCOPE, scene.sid, purpose=purpose
    )


def test_media_recovery_binds_ordinary_reference_to_original_managed_roots(tmp_path):
    from tests.unit.edge.route_assistance.test_recording_evidence import _store
    from tests.unit.lab.test_mainline1_recording_media_worker import _jpeg

    module = recovery()
    store, session_id = _store(tmp_path)
    jobs_root = (tmp_path / "jobs").resolve()
    ring_root = (tmp_path / "ring").resolve()
    boundary = RecordingBoundaryV1("shot-1", "epoch", 1, 2, 16, 33, True)
    store.record_recording_start(session_id, "shot-1", "epoch", 1, 16)
    store.record_recording_boundary(session_id, boundary)
    raw_root = ring_root / "sessions" / session_id / "epoch" / "session-recording"
    recorder = ContinuousSessionRecorder(
        root=raw_root,
        session_id=session_id,
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )
    evidence_store = SessionRecordingEvidenceStore(
        recorder=recorder,
        session_id=session_id,
        pre_roll_frames=1,
        post_roll_frames=3,
        maximum_shot_frames=60,
    )
    for frame_id in range(6):
        recorder.append(
            CaptureFrame("epoch", frame_id, frame_id * 16_666_667, 32, 24, _jpeg(frame_id))
        )
    interval = evidence_store.freeze_window(
        shot_id="shot-1", source_epoch_id="epoch", start_frame_id=1, end_frame_id=5
    )
    reference_id = "recording-ref-1"
    reference_root = jobs_root / "media" / session_id / reference_id
    materialize_recorded_interval_reference(interval, reference_root, recording_identity=True)
    reference = reference_root / "interval.json"
    store.record_recording_media(
        session_id,
        RecordedMediaV1(
            boundary,
            "RECORDED",
            reference_id,
            str(reference),
            hashlib.sha256(reference.read_bytes()).hexdigest(),
            interval.media_sha256,
            interval.start_frame_id,
            interval.end_frame_id,
        ),
    )
    recorder.close()
    store.prepare_recording_end(
        session_id,
        "epoch",
        {"frame_id": 5, "captured_at_ns": 5 * 16_666_667},
        recorder.finalized_manifest_path,
        recorder.finalized_segment_receipts,
    )
    store.end_session(session_id)
    manifest = store.get_postgame_manifest(session_id)
    shots = store.list_shots(session_id)

    module._verify_recording_media_sources(store, session_id, jobs_root, ring_root, manifest, shots)
    with pytest.raises(ValueError, match="reference path"):
        module._verify_recording_media_sources(
            store, session_id, tmp_path / "foreign-jobs", ring_root, manifest, shots
        )


def test_reconstructed_store_resolves_multiple_original_epochs_with_superseded_calibration(scene):
    before = (scene.args.jobs_root / "analysis-jobs.sqlite3").read_bytes()
    scene.args.calibration("b")
    context = resolve(scene)
    assert context.spec_record.spec == scene.spec
    assert context.jobs_db == scene.args.jobs_root / "analysis-jobs.sqlite3"
    assert context.artifact_root == scene.args.artifact_root
    assert context.jobs_root == scene.args.jobs_root
    assert context.ring_root == scene.args.ring_root
    assert (scene.args.jobs_root / "analysis-jobs.sqlite3").read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_db",
        "wrong_epoch",
        "job_hash",
        "missing_model",
        "media_root",
        "interval",
        "noncanonical",
    ],
)
def test_invalid_original_context_is_private_and_readonly(scene, mutation):
    db = scene.args.jobs_root / "analysis-jobs.sqlite3"
    if mutation == "missing_db":
        db.unlink()
    elif mutation == "interval":
        import json

        path = scene.args.jobs_root / "media" / scene.sid / "job-2" / "interval.json"
        payload = json.loads(path.read_bytes())
        payload["source_epoch_id"] = "foreign-epoch"
        path.write_text(json.dumps(payload))
    elif mutation == "missing_model":
        (scene.args.detector_training_run / "checkpoint_best.pth").unlink()
    else:
        import hashlib
        import json

        with sqlite3.connect(db) as connection:
            raw = connection.execute(
                "SELECT payload_json FROM shot_analysis_jobs WHERE job_id='job-2'"
            ).fetchone()[0]
            payload = json.loads(raw)
            if mutation == "wrong_epoch":
                payload["source_epoch_id"] = "unregistered-epoch"
            if mutation == "media_root":
                payload["media_root"] = str(scene.args.jobs_root / "foreign")
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            if mutation == "noncanonical":
                raw += " "
            digest = (
                "0" * 64 if mutation == "job_hash" else hashlib.sha256(raw.encode()).hexdigest()
            )
            connection.execute(
                "UPDATE shot_analysis_jobs SET payload_json=?,payload_sha256=? "
                "WHERE job_id='job-2'",
                (raw, digest),
            )
    with pytest.raises(recovery().ManagedRuntimeResourceUnavailable) as error:
        resolve(scene)
    assert str(error.value) == "POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE"
    if mutation == "missing_db":
        assert not db.exists()


def test_cleanup_requires_no_models_and_diagnostics_clear_only_recovered_purpose(scene):
    module = recovery()
    diagnostics = {
        scene.sid: "POSTGAME_REVIEW_ORIGINAL_CONTEXT_UNAVAILABLE",
        "saved": "saved-notice",
    }
    resolver = module.ManagedRuntimeResourceResolver(scene.store, SCOPE, diagnostics)
    model = scene.args.detector_training_run / "checkpoint_best.pth"
    original = model.read_bytes()
    model.unlink()
    with pytest.raises(module.ManagedRuntimeResourceUnavailable):
        resolver.resolve(scene.sid, purpose="analysis")
    assert resolver.resolve(scene.sid, purpose="media").spec_record.spec == scene.spec
    assert resolver.resolve(scene.sid, purpose="cleanup").spec_record.spec == scene.spec
    assert diagnostics[scene.sid] == "POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE"
    model.write_bytes(original)
    resolver.resolve(scene.sid, purpose="analysis")
    assert diagnostics[scene.sid] == "POSTGAME_REVIEW_ORIGINAL_CONTEXT_UNAVAILABLE"
    assert diagnostics["saved"] == "saved-notice"
    with pytest.raises(module.ManagedRuntimeResourceUnavailable):
        resolver.resolve("managed-99999999-9999-4999-8999-999999999999", purpose="analysis")
    assert len(diagnostics) == 2


def test_supervisor_paces_unavailable_models_and_recovers_original_command(scene):
    from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundPostgameAnalysisWorker
    from tests.unit.scripts.test_serve_mainline1_usb_field import load_script

    args = SimpleNamespace(**vars(scene.args))
    args.state_db = scene.store.path
    args.jobs_root = scene.args.jobs_root.parent / "current-jobs"
    args.artifact_root = scene.args.artifact_root.parent / "current-artifacts"
    args.config = args.config.parent / "current-config"
    args.cloudbase_environment_id, args.device_id, args.table_id = (
        SCOPE.environment_id,
        SCOPE.device_id,
        SCOPE.table_id,
    )
    args.sync_base_url = None
    args.existing_development_control_only = False
    module = load_script()
    launches, resolutions, clock = [], [], [0]
    resolver = recovery().ManagedRuntimeResourceResolver(scene.store, SCOPE, {})
    original_resolve = resolver.resolve

    def counted(sid, **kwargs):
        resolutions.append((sid, kwargs["purpose"]))
        return original_resolve(sid, **kwargs)

    resolver.resolve = counted
    worker = _BackgroundPostgameAnalysisWorker(
        store=scene.store,
        managed_scope=SCOPE,
        resource_resolver=resolver,
        monotonic=lambda: clock[0],
        command_factory=lambda sid, **kw: module._postgame_worker_command(
            args, server_port=1234, session_id=sid, **kw
        ),
        process_factory=lambda command, **kw: (
            launches.append(command) or SimpleNamespace(poll=lambda: 0)
        ),
    )
    model = scene.args.detector_training_run / "checkpoint_best.pth"
    original = model.read_bytes()
    model.unlink()
    worker.run_once()
    worker.run_once()
    worker.run_once()
    assert len(launches) == 1
    assert resolutions == [(scene.sid, "media"), (scene.sid, "analysis")]
    media_command = launches[0]
    assert media_command[media_command.index("--purpose") + 1] == "media"
    assert "--config" not in media_command
    assert "--detector-training-run" not in media_command
    assert not args.jobs_root.exists() and not args.artifact_root.exists()
    model.write_bytes(original)
    scene.args.calibration("b")
    clock[0] = 31
    worker.run_once()
    command = launches[-1]

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--jobs-db") == str(scene.args.jobs_root / "analysis-jobs.sqlite3")
    assert value("--artifact-root") == str(scene.args.artifact_root / "postgame" / scene.sid)
    assert value("--purpose") == "analysis"
    assert value("--config") == str(scene.args.config)
    assert value("--managed-resource-spec-sha256") == scene.spec.spec_sha256
    assert value("--managed-open-request-id") == scene.request.open_request_id
    assert not args.jobs_root.exists() and not args.artifact_root.exists()


def test_scope_discovery_ignores_malformed_foreign_scene_before_decoding(scene):
    # Copy only a foreign retained candidate row; malformed match data would abort
    # the old eager discovery before the valid same-scope scene was selected.
    with sqlite3.connect(scene.store.path) as connection:
        connection.execute(
            "INSERT INTO field_sessions SELECT 'foreign-scene',artifact_id,status,"
            "created_at_utc,ended_at_utc FROM field_sessions WHERE session_id=?",
            (scene.sid,),
        )
    with pytest.raises(ValueError, match="match state is missing"):
        scene.store.list_ended_sessions_requiring_postgame()
    ids = scene.store.list_ended_session_ids_requiring_postgame(managed_scope=SCOPE)
    assert ids == (scene.sid,)


@pytest.mark.parametrize("flags", [[], ["--managed-environment-id", "dev"]])
def test_managed_child_refuses_missing_context_before_config_or_provider(scene, monkeypatch, flags):
    from tests.unit.scripts.test_run_mainline1_shot_worker import _module

    module = _module()
    guard_child_pre_admission(monkeypatch, module)
    argv = [
        "--state-db",
        str(scene.store.path),
        "--jobs-db",
        str(scene.args.jobs_root / "analysis-jobs.sqlite3"),
        "--artifact-root",
        str(scene.args.artifact_root / "postgame" / scene.sid),
        "--session-id",
        scene.sid,
        "--worker-id",
        "worker",
        "--fps",
        "60",
    ]
    for name in (
        "capture_setup",
        "calibration_root",
        "config",
        "detector_training_run",
        "identity_training_run",
        "dinov2_assets",
    ):
        argv.extend(["--" + name.replace("_", "-"), str(getattr(scene.args, name))])
    assert module.main(argv + flags) == 2
    with sqlite3.connect(scene.args.jobs_root / "analysis-jobs.sqlite3") as connection:
        assert connection.execute(
            "SELECT DISTINCT attempt_count FROM shot_analysis_jobs"
        ).fetchall() == [(0,)]


def test_bad_same_scope_candidate_does_not_starve_valid_scene_or_leak_foreign_ids(scene):
    from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundPostgameAnalysisWorker

    bad_birth = "00000000-0000-4000-8000-000000000001"
    bad_sid = "managed-" + bad_birth
    with sqlite3.connect(scene.store.path) as connection:
        connection.row_factory = sqlite3.Row
        row = dict(connection.execute("SELECT * FROM field_managed_open_requests").fetchone())
        # Adversarial malformed retained authority; it must never authorize work.
        row.update(
            open_request_id="00000000-0000-4000-8000-000000000002",
            session_id=bad_sid,
            birth_id=bad_birth,
            lifecycle="retired",
            remote_status="ended",
            retired_at_utc=row["created_at_utc"],
            sealed_close_json="{}",
            reservation_order=2,
            created_at_utc="2020-01-01T00:00:00Z",
        )
        connection.execute(
            "INSERT INTO field_managed_open_requests ("
            + ",".join(row)
            + ") VALUES ("
            + ",".join("?" for _ in row)
            + ")",
            tuple(row.values()),
        )
        connection.execute(
            "INSERT INTO field_sessions SELECT ?,artifact_id,status,?,ended_at_utc "
            "FROM field_sessions WHERE session_id=?",
            (bad_sid, "2020-01-01T00:00:00Z", scene.sid),
        )
    diagnostics, commands = {}, []
    resolver = recovery().ManagedRuntimeResourceResolver(scene.store, SCOPE, diagnostics)
    worker = _BackgroundPostgameAnalysisWorker(
        store=scene.store,
        managed_scope=SCOPE,
        resource_resolver=resolver,
        command_factory=lambda sid, **kw: [sid],
        process_factory=lambda command, **kw: (
            commands.append(command) or SimpleNamespace(poll=lambda: None)
        ),
    )
    worker.run_once()
    assert commands == [[scene.sid]]
    assert set(diagnostics) <= {bad_sid}
    assert scene.jobs.status("job-1").value == "pending"


def test_old_job_without_identity_pair_is_not_adopted_or_backfilled(scene):
    import hashlib
    import json

    db = scene.args.jobs_root / "analysis-jobs.sqlite3"
    with sqlite3.connect(db) as connection:
        raw = connection.execute(
            "SELECT payload_json FROM shot_analysis_jobs WHERE job_id='job-1'"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload["identity_training_package_id"] = None
        payload["identity_training_package_sha256"] = None
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        connection.execute(
            "UPDATE shot_analysis_jobs SET payload_json=?,payload_sha256=? WHERE job_id='job-1'",
            (raw, hashlib.sha256(raw.encode()).hexdigest()),
        )
    before = db.read_bytes()
    with pytest.raises(recovery().ManagedRuntimeResourceUnavailable):
        resolve(scene)
    assert db.read_bytes() == before
    assert scene.jobs.status("job-1").value == "pending"
