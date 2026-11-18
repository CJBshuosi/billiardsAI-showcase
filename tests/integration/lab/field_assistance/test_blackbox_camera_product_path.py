from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope
from billiards_ai.edge.route_assistance.control import (
    MatchFormat,
    MatchMode,
    RackPhase,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID
from billiards_ai.edge.route_assistance.records import FieldSessionStatus
from billiards_ai.edge.route_assistance.sync import FieldSyncClient, HttpResult
from billiards_ai.lab.field_assistance import server as service
from billiards_ai.lab.field_assistance.cloud_runtime import (
    FieldCloudFactories,
    prepare_cloud_runtime,
)
from billiards_ai.lab.field_assistance.server import build_live_field_assistance_server
from billiards_ai.lab.mainline1_field_runtime import build_live_runtime_factory
from billiards_ai.lab.mainline1_product_realtime_runtime import (
    ProductRealtimeEvent,
    ProductRealtimeEventChannel,
    ProductRealtimeRuntime,
)
from billiards_ai.lab.mainline1_realtime_capture import RealtimeCaptureConfig
from billiards_ai.lab.mainline1_scenario_capture import (
    ScenarioCaptureConfig,
    ScenarioVideoCaptureBackend,
)
from billiards_ai.lab.mainline1_session_audit import ShotBoundaryLedger
from billiards_ai.lab.mainline1_shot_jobs import (
    ShotAnalysisJobStatus,
    SQLiteShotAnalysisJobStore,
)
from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config
from billiards_ai.lab.vision.contracts import VisionFrameObservation, VisionPackageIdentity
from billiards_ai.shared.calibration.homography import calibrate
from scripts.algorithm import serve_mainline1_blackbox_field as blackbox_cli

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SHOT_CONFIG_PATH = _PROJECT_ROOT / "configs/algorithm/shot-local.yaml"
_CAPTURE_CONFIG = RealtimeCaptureConfig(width=160, height=120, fps=60)
_MODEL_PACKAGE_ID = "e" * 64


@dataclass(frozen=True, slots=True)
class _ScenarioRender:
    total_frames: int
    motion_starts_after_frame: int = 90
    width: int = 160
    height: int = 120
    fps: int = 60


@dataclass(frozen=True, slots=True)
class _ExpectedTruth:
    shot_count: int
    start_frame_ids: range
    settle_frame_ids: range
    receipt_status: str
    decision_reason: str
    job_count: int
    completion_mode: str


_SUCCESS_TRUTH = _ExpectedTruth(
    shot_count=1,
    start_frame_ids=range(89, 95),
    settle_frame_ids=range(142, 148),
    receipt_status="abstained",
    decision_reason="eight_status_unknown",
    job_count=1,
    completion_mode="explicit_user_end",
)
_EOF_TRUTH = _ExpectedTruth(
    shot_count=1,
    start_frame_ids=range(89, 95),
    settle_frame_ids=range(0),
    receipt_status="absent",
    decision_reason="capture source ended unexpectedly",
    job_count=0,
    completion_mode="degraded_incomplete",
)


class _EmptyDetector:
    package_identity = VisionPackageIdentity(
        package_id=_MODEL_PACKAGE_ID,
        manifest_sha256="f" * 64,
    )

    def infer(self, frame):
        return VisionFrameObservation(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            width_px=frame.width_px,
            height_px=frame.height_px,
            pixel_sha256=frame.pixel_sha256,
            package=self.package_identity,
            threshold=0.25,
            detections=(),
            runtime_ref="blackbox-empty-observer/1",
        )


class _EmptyClassifier:
    def classify(self, *, image, detections):
        del image
        assert detections == ()
        return {}


class _CommandGateway:
    def __init__(self, commands: list[FieldCommandEnvelope]) -> None:
        self._commands = list(commands)
        self.results = []

    def claim_field_command(self):
        return None if not self._commands else self._commands.pop(0)

    def complete_field_command(self, result) -> None:
        self.results.append(result)


def _until[T](probe: Callable[[], T | None], *, timeout_seconds: float, label: str) -> T:
    deadline = time.monotonic() + timeout_seconds
    last = None
    while time.monotonic() < deadline:
        last = probe()
        if last is not None:
            return last
        time.sleep(0.02)
    pytest.fail(f"timed out waiting for {label}; last={last!r}")


def _write_scenario(path: Path, render: _ScenarioRender) -> str:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        render.fps,
        (render.width, render.height),
    )
    if not writer.isOpened():
        raise RuntimeError("test MJPG writer is unavailable")
    try:
        for frame_id in range(render.total_frames):
            x = (
                40
                if frame_id <= render.motion_starts_after_frame
                else min(100, 40 + 2 * (frame_id - render.motion_starts_after_frame))
            )
            image = np.zeros((render.height, render.width, 3), dtype=np.uint8)
            cv2.rectangle(image, (x - 10, 50), (x + 10, 70), (180, 180, 180), -1)
            writer.write(image)
    finally:
        writer.release()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_server(
    root: Path,
    *,
    video: Path,
    epoch: str,
    setup_gate: threading.Event,
    acceptance_policy=None,
):
    def await_setup(_source) -> None:
        if not setup_gate.wait(3.0):
            raise RuntimeError("player setup gate timed out")

    def capture_diagnostics(source, source_epoch_id: str) -> dict[str, object]:
        return blackbox_cli._capture_diagnostics(
            source,
            source_epoch_id,
            capture_config=_CAPTURE_CONFIG,
            acceptance_policy=acceptance_policy,
        )

    runtime_factory = build_live_runtime_factory(
        capture_factory=lambda source_epoch_id: ScenarioVideoCaptureBackend(
            ScenarioCaptureConfig(video, _CAPTURE_CONFIG),
            source_epoch_id=source_epoch_id,
        ),
        capture_config=_CAPTURE_CONFIG,
        calibration=calibrate(
            "blackbox-vertical-slice",
            [(0, 119), (159, 119), (159, 0), (0, 0)],
            table_length_mm=2540,
            table_width_mm=1270,
        ),
        detector=_EmptyDetector(),
        identity_classifier=_EmptyClassifier(),
        shot_config=load_mainline1_shot_local_config(_SHOT_CONFIG_PATH),
        capture_setup_id="blackbox-test-file-160x120-60",
        calibration_artifact_id="blackbox-test-calibration",
        config_path=_SHOT_CONFIG_PATH,
        ring_root=root / "rings",
        jobs_root=root / "jobs",
        capture_starter=await_setup,
        capture_diagnostics_factory=capture_diagnostics,
    )
    return build_live_field_assistance_server(
        artifact_path=root / "artifacts",
        fresh_session_on_start=True,
        state_db_path=root / "field.sqlite3",
        host="127.0.0.1",
        port=0,
        control_token="blackbox-test-control-token",
        source_epoch_factory=lambda: epoch,
        calibration_id="blackbox-test-calibration",
        runtime_factory=runtime_factory,
        rule_profile_id=PROFILE_ID,
    )


def _command_worker(state, commands: list[FieldCommandEnvelope]):
    gateway = _CommandGateway(commands)
    worker = service._BackgroundFieldCommandWorker(
        gateway=gateway,
        store=state.record_store,
        match_lock=state.match_lock,
        before_application=state.before_field_command,
        match_changed=state.set_command_match,
        application_changed=state.handle_command_application,
        command_reducer=state.reduce_live_command,
    )
    return gateway, worker


def _setup_players(state) -> None:
    commands = [
        FieldCommandEnvelope(
            f"{state.session_id}-join-red",
            state.session_id,
            "alice",
            "claim_role",
            {"role": "red"},
            0,
            1,
        ),
        FieldCommandEnvelope(
            f"{state.session_id}-join-blue",
            state.session_id,
            "bob",
            "claim_role",
            {"role": "blue"},
            1,
            2,
        ),
        FieldCommandEnvelope(
            f"{state.session_id}-freeze-settings",
            state.session_id,
            "alice",
            "set_breaker",
            {
                "role": "red",
                "mode": "assistance",
                "format": "single",
                "race_target": None,
                "ranking_intent": False,
                "settings_revision": 1,
            },
            2,
            3,
        ),
    ]
    gateway, worker = _command_worker(state, commands)
    assert [worker.claim_and_apply_once() for _command in commands] == [True, True, True]
    assert [(result.outcome, result.result_code) for result in gateway.results] == [
        ("applied", "APPLIED"),
        ("applied", "APPLIED"),
        ("applied", "APPLIED"),
    ]
    match = state.record_store.load_session(state.session_id).match
    assert match.revision == 3
    assert match.red_user_id == "alice"
    assert match.blue_user_id == "bob"
    assert match.breaker is WristbandRole.RED
    assert match.active_role is WristbandRole.RED
    assert match.mode is MatchMode.ASSISTANCE
    assert match.match_format is MatchFormat.SINGLE
    assert match.settings_revision == 1


@pytest.mark.parametrize("failed_worker", [None, "facts"])
@pytest.mark.parametrize("development_acceptance", [False, True])
def test_blackbox_cli_uses_shared_authenticated_commands_and_media_lifecycle(
    tmp_path, monkeypatch, capsys, failed_worker, development_acceptance
):
    """Controlled device transport; real gateway, command worker, reducer and capture."""
    video = tmp_path / "static-lead-in.avi"
    _write_scenario(video, _ScenarioRender(total_frames=600, motion_starts_after_frame=9999))
    events, unauthorized_callers = _observe_runtime_publication(monkeypatch)
    calls, workers, clients, server_box = [], {}, [], []
    secret = b"offline-development-device-secret"

    class Credentials:
        def device_id(self):
            return "blackbox-01"

        def secret(self):
            return secret

    credentials = Credentials()

    class Transport:
        def __init__(self):
            self.commands = []
            self.completed = []
            self.lock = threading.Lock()

        def post_json(self, path, headers, body):
            canonical = (
                f"POST\n{path}\n{headers['x-timestamp']}\n{headers['x-nonce']}\n"
                f"{hashlib.sha256(body).hexdigest()}"
            ).encode()
            assert headers["x-device-id"] == "blackbox-01"
            assert headers["x-body-sha256"] == hashlib.sha256(body).hexdigest()
            assert hmac.compare_digest(
                headers["x-signature"], hmac.new(secret, canonical, hashlib.sha256).hexdigest()
            )
            payload = json.loads(body)
            with self.lock:
                if path == "/v1/runtime-attestation":
                    data = dict(
                        schema_version="1.0",
                        runtime_environment="development",
                        cloudbase_environment_id="cloud-development",
                    )
                    calls.append("attest")
                elif path.endswith("/open"):
                    data = dict(
                        session_id=path.split("/")[-2], table_id="blackbox-01", status="open"
                    )
                elif path.endswith("/join-requests/query"):
                    data = {"requests": []}
                elif path == "/v1/field-commands/claim":
                    data = self.commands.pop(0) if self.commands else None
                elif path.endswith("/complete"):
                    data = {"command_id": path.split("/")[-2], **payload}
                    self.completed.append(data)
                else:
                    raise AssertionError(f"unexpected offline transport route: {path}")
            return HttpResult(200, json.dumps({"ok": True, "data": data}).encode())

    transport = Transport()
    defaults = FieldCloudFactories()

    def observed_worker(kind, factory):
        def construct(**kwargs):
            worker = factory(**kwargs)
            workers[kind] = (worker, kwargs)
            for operation in ("start", "stop"):
                original = getattr(worker, operation)

                def observed(original=original, operation=operation):
                    calls.append((kind, operation))
                    result = original()
                    if kind == failed_worker and operation == "start":
                        raise RuntimeError("worker failed after native thread start")
                    return result

                monkeypatch.setattr(worker, operation, observed)
            return worker

        return construct

    def credential_factory(**kwargs):
        assert kwargs == {
            "device_id": "blackbox-01",
            "target_name": "billiardsAI/development/blackbox-01",
        }
        return credentials

    def transport_factory(url, *, allow_redirects):
        assert url == "https://cloud-development.example/sync-api"
        assert allow_redirects is False
        return transport

    def client_factory(**kwargs):
        clients.append(kwargs)
        return FieldSyncClient(**kwargs)

    factories = FieldCloudFactories(
        credentials=credential_factory,
        transport=transport_factory,
        sync_client=client_factory,
        command_worker=observed_worker("commands", defaults.command_worker),
        postgame_worker=observed_worker("postgame", defaults.postgame_worker),
        saved_media_worker=observed_worker("saved", defaults.saved_media_worker),
        live_fact_worker=observed_worker("facts", defaults.live_fact_worker),
    )

    def prepare_cloud(args):
        return prepare_cloud_runtime(args, factories=factories)

    monkeypatch.setattr(blackbox_cli, "prepare_cloud_runtime", prepare_cloud, raising=False)

    def prepare_capture(args, *, initial_source_epoch_id):
        assert calls == ["attest"]
        calls.append("capture")
        prepared = ScenarioVideoCaptureBackend(
            ScenarioCaptureConfig(video, _CAPTURE_CONFIG), source_epoch_id=initial_source_epoch_id
        )
        factory = build_live_runtime_factory(
            capture_factory=lambda epoch: prepared,
            capture_config=_CAPTURE_CONFIG,
            calibration=calibrate(
                "offline-blackbox",
                [(0, 119), (159, 119), (159, 0), (0, 0)],
                table_length_mm=2540,
                table_width_mm=1270,
            ),
            detector=_EmptyDetector(),
            identity_classifier=_EmptyClassifier(),
            shot_config=load_mainline1_shot_local_config(_SHOT_CONFIG_PATH),
            capture_setup_id="offline-file-160x120-60",
            calibration_artifact_id="offline-calibration",
            config_path=_SHOT_CONFIG_PATH,
            ring_root=args.ring_root,
            jobs_root=args.jobs_root,
            capture_diagnostics_factory=lambda source, epoch: blackbox_cli._capture_diagnostics(
                source,
                epoch,
                capture_config=_CAPTURE_CONFIG,
                acceptance_policy=getattr(args, "_development_acceptance_policy", None),
            ),
        )
        return "offline-calibration", factory, prepared

    monkeypatch.setattr(blackbox_cli, "prepare_runtime", prepare_capture)

    def build_server(**kwargs):
        server = build_live_field_assistance_server(**kwargs)
        server_box.append(server)
        state = server.field_state
        assert kwargs["fresh_session_on_start"] is True
        assert state.join_coordinator.gateway is kwargs["cloud_gateway"]
        assert state.sync_wakeup is kwargs["sync_wakeup"]
        transport.commands = [
            dict(
                command_id=f"command-{sequence}",
                session_id=state.session_id,
                actor_user_id=actor,
                command_type=kind,
                validated_payload=payload,
                expected_revision=sequence - 1,
                server_sequence=sequence,
            )
            for sequence, actor, kind, payload in (
                (1, "alice", "claim_role", {"role": "red"}),
                (2, "bob", "claim_role", {"role": "blue"}),
                (
                    3,
                    "alice",
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
            )
        ]
        wait = state.wait_until_capture_ready

        def ready(**kwargs):
            epoch = wait(**kwargs)
            calls.append("ready")
            assert not workers
            return epoch

        monkeypatch.setattr(state, "wait_until_capture_ready", ready)

        def serve():
            _until(
                lambda: True if len(transport.completed) == 3 else None,
                timeout_seconds=5,
                label="authenticated command completions",
            )
            with state.match_lock:
                match = state.record_store.load_session(state.session_id).match
                assert (match.red_user_id, match.blue_user_id, match.revision) == (
                    "alice",
                    "bob",
                    3,
                )
                assert match.settings_frozen and match.breaker is WristbandRole.RED
                assert match.mode is MatchMode.ASSISTANCE
                assert match.match_format is MatchFormat.SINGLE
                assert state.projection_source._match_ready_for_shots()
                assert state.record_store.list_shots(state.session_id) == ()
                diagnostic = state.projection_source.runtime.capture_diagnostics
                assert diagnostic["source_kind"] == "blackbox_video"
                if development_acceptance:
                    assert diagnostic["acceptance"] == expected_acceptance
                else:
                    assert "acceptance" not in diagnostic
            assert all(item["public_result_code"] == "APPLIED" for item in transport.completed)

        monkeypatch.setattr(server, "serve_forever", serve)
        close = server.server_close
        monkeypatch.setattr(server, "server_close", lambda: calls.append("close") or close())
        return server

    monkeypatch.setattr(blackbox_cli, "build_live_field_assistance_server", build_server)
    argv = []
    for name in (
        "capture-setup",
        "detector-training-run",
        "identity-training-run",
        "dinov2-assets",
        "calibration-root",
        "ring-root",
        "jobs-root",
        "artifact-root",
    ):
        path = tmp_path / name
        if name == "capture-setup":
            path.write_text("offline capture setup")
        else:
            path.mkdir()
        argv.extend(("--" + name, str(path)))
    argv.extend(
        (
            "--config",
            str(_SHOT_CONFIG_PATH),
            "--scenario-video",
            str(video),
            "--state-db",
            str(tmp_path / "field.sqlite3"),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--runtime-environment",
            "development",
            "--device-id",
            "blackbox-01",
            "--credential-target",
            "billiardsAI/development/blackbox-01",
            "--sync-base-url",
            "https://cloud-development.example/sync-api",
            "--cloudbase-environment-id",
            "cloud-development",
            "--expected-development-environment-id",
            "cloud-development",
            "--identity-gate-status",
            "PASS",
            "--rule-profile",
            PROFILE_ID,
        )
    )
    expected_acceptance = None
    if development_acceptance:
        from tests.support.development_acceptance import write_identity_assets

        expected_acceptance = write_identity_assets(tmp_path / "identity-training-run")
        argv[argv.index("--identity-gate-status") + 1] = "FAIL"
        argv.append("--development-model-acceptance")
    assert blackbox_cli.main(argv) == (0 if failed_worker is None else 2)
    ready = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    if failed_worker is None:
        assert len(ready) == 1 and ready[0]["status"] == "BLACKBOX_FIELD_READY"
        if development_acceptance:
            assert ready[0]["acceptance"] == expected_acceptance
        else:
            assert "acceptance" not in ready[0]
    else:
        assert ready == []
    assert calls.index("ready") < calls.index(("postgame", "start"))
    shutdown = [call for call in calls if call == "close" or isinstance(call, tuple)]
    assert shutdown == [
        ("postgame", "start"),
        ("saved", "start"),
        ("commands", "start"),
        ("facts", "start"),
        ("commands", "stop"),
        ("facts", "stop"),
        ("postgame", "stop"),
        ("saved", "stop"),
        "close",
    ]
    state = server_box[0].field_state
    _, saved = workers["saved"]
    assert saved["store"] is state.record_store
    assert saved["trusted_postgame_root"] == tmp_path / "artifact-root/postgame"
    assert saved["wake_event"] is state.sync_wakeup.__self__
    assert len(clients) == 2
    assert all(
        client["credentials"] is credentials and client["transport"] is transport
        for client in clients
    )
    assert clients[0]["before_mutation"].__self__ is state.join_coordinator.gateway
    assert all(not worker._thread.is_alive() for worker, _ in workers.values())
    assert events == [] and unauthorized_callers == []


def _end_session_as_player(state) -> None:
    match = state.record_store.load_session(state.session_id).match
    command = FieldCommandEnvelope(
        f"{state.session_id}-end-after-reviewable-shot",
        state.session_id,
        "bob",
        "end_session",
        {},
        match.revision,
        4,
    )
    gateway, worker = _command_worker(state, [command])
    assert worker.claim_and_apply_once()
    assert [(result.outcome, result.result_code) for result in gateway.results] == [
        ("applied", "APPLIED")
    ]


def _observe_runtime_publication(monkeypatch):
    events: list[ProductRealtimeEvent] = []
    unauthorized_callers: list[str] = []
    lock = threading.Lock()
    original_publish = ProductRealtimeEventChannel.publish
    runtime_emit_code = ProductRealtimeRuntime._emit_event.__code__

    def observed_publish(channel, event: ProductRealtimeEvent) -> None:
        frame = inspect.currentframe()
        caller = None if frame is None else frame.f_back
        caller_name = "unknown" if caller is None else caller.f_code.co_qualname
        if caller is None or caller.f_code is not runtime_emit_code:
            with lock:
                unauthorized_callers.append(caller_name)
            raise AssertionError(
                f"event publication bypassed ProductRealtimeRuntime: {caller_name}"
            )
        with lock:
            events.append(event)
        original_publish(channel, event)

    monkeypatch.setattr(ProductRealtimeEventChannel, "publish", observed_publish)
    return events, unauthorized_callers


@pytest.mark.parametrize("development_acceptance", [False, True])
def test_real_video_pixels_drive_the_product_path_without_event_or_fact_injection(
    tmp_path: Path,
    monkeypatch,
    development_acceptance,
) -> None:
    render = _ScenarioRender(total_frames=480)
    video = tmp_path / "one-motion-shot.avi"
    source_sha256 = _write_scenario(video, render)
    events, unauthorized_callers = _observe_runtime_publication(monkeypatch)
    setup_gate = threading.Event()
    epoch = "blackbox-epoch-success"
    policy = None
    acceptance = {}
    if development_acceptance:
        from types import SimpleNamespace

        from billiards_ai.lab.field_assistance.development_acceptance import (
            resolve_acceptance_policy,
        )
        from tests.support.development_acceptance import write_identity_assets

        identity_run = tmp_path / "synthetic-identity"
        acceptance = {"acceptance": write_identity_assets(identity_run)}
        policy = resolve_acceptance_policy(
            SimpleNamespace(
                development_model_acceptance=True,
                runtime_environment="development",
                identity_gate_status="FAIL",
                rule_profile=PROFILE_ID,
                observation_only=False,
                identity_training_run=identity_run,
            )
        )
    server = _build_server(
        tmp_path,
        video=video,
        epoch=epoch,
        setup_gate=setup_gate,
        acceptance_policy=policy,
    )
    state = server.field_state
    session_id = state.session_id
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs/analysis-jobs.sqlite3")
    try:
        _setup_players(state)
        before = state.record_store.load_session(session_id).match
        setup_gate.set()
        assert state.wait_until_capture_ready(timeout_seconds=3.0) == epoch

        def completed_evidence():
            shots = state.record_store.list_shots(session_id)
            if len(shots) != _SUCCESS_TRUTH.shot_count or shots[0].settled_ms is None:
                return None
            shot = shots[0]
            receipt = state.record_store.get_rule_decision(session_id, shot.shot_id)
            media = state.record_store.get_live_shot_media(session_id, shot.shot_id)
            session_jobs = jobs.list_session_jobs(session_id)
            if receipt is None or media is None or len(session_jobs) != _SUCCESS_TRUTH.job_count:
                return None
            return shot, receipt, media, session_jobs[0]

        shot, receipt, media, job = _until(
            completed_evidence,
            timeout_seconds=8.0,
            label="settled shot, rule receipt, media attachment and queued analysis job",
        )
        event_kinds = [event.kind for event in events]
        assert event_kinds == ["SHOT_STARTED", "SHOT_SETTLED", "SHOT_ENQUEUED"]
        assert unauthorized_callers == []

        ownership = state.record_store.get_shot_ownership(session_id, shot.shot_id)
        live_fact = state.record_store.get_live_shot_fact(session_id, shot.shot_id)
        assert ownership is not None
        assert live_fact is not None
        assert (
            ownership.session_id,
            ownership.shot_id,
            ownership.source_epoch_id,
            ownership.match_revision,
        ) == (session_id, shot.shot_id, epoch, before.revision)
        assert ownership.actor_user_id == "alice"
        assert ownership.actor_role is WristbandRole.RED
        assert (
            live_fact.session_id,
            live_fact.shot_id,
            live_fact.actor_user_id,
            live_fact.actor_role,
            live_fact.rule_revision,
        ) == (session_id, shot.shot_id, "alice", WristbandRole.RED, before.revision)
        assert live_fact.start_frame_id in _SUCCESS_TRUTH.start_frame_ids
        assert live_fact.settle_frame_id in _SUCCESS_TRUTH.settle_frame_ids

        assert receipt.status == _SUCCESS_TRUTH.receipt_status
        assert receipt.applied_command_id is None
        assert receipt.source_epoch_id == epoch
        assert receipt.start_frame_id == live_fact.start_frame_id
        assert receipt.settled_frame_id == live_fact.settle_frame_id
        assert receipt.fact_digest and receipt.decision_digest
        assert json.loads(receipt.decision_json)["reason_code"] == _SUCCESS_TRUTH.decision_reason
        assert json.loads(receipt.facts_json)["match_revision_at_start"] == before.revision

        after_rule = state.record_store.load_session(session_id).match
        assert after_rule == before
        assert after_rule.rack_phase is RackPhase.BREAK
        assert after_rule.red_group is None and after_rule.blue_group is None
        assert after_rule.winner_role is None

        assert shot.live_job_id == job.job_id == media.job_id
        assert job.session_id == media.session_id == session_id
        assert job.shot_id == media.shot_id == shot.shot_id
        assert job.source_epoch_id == media.source_epoch_id == epoch
        assert media.match_revision == before.revision
        assert job.capture_setup_id == "blackbox-test-file-160x120-60"
        assert job.calibration_artifact_id == "blackbox-test-calibration"
        assert job.model_package_id == _MODEL_PACKAGE_ID
        assert job.start_frame_id < live_fact.start_frame_id
        assert job.end_frame_id > live_fact.settle_frame_id
        assert job.media_root == Path(media.media_path)
        assert job.media_sha256 == media.media_sha256
        assert (job.media_root / "interval.json").is_file()
        assert list(job.media_root.glob("*.mjpgseg"))
        assert jobs.snapshot(job.job_id).status is ShotAnalysisJobStatus.PENDING

        boundaries = ShotBoundaryLedger(
            tmp_path / "rings" / "sessions" / session_id / epoch / "shot-boundaries.sqlite3",
            session_id=session_id,
            source_epoch_id=epoch,
        ).records()
        assert [(item.kind, item.shot_id) for item in boundaries] == [
            ("SHOT_STARTED", shot.shot_id),
            ("SHOT_SETTLED", shot.shot_id),
        ]
        assert boundaries[0].start_frame_id == live_fact.start_frame_id
        assert boundaries[1].end_frame_id == live_fact.settle_frame_id

        screen = state.projection_source.current().frame.payload
        assert screen["source_kind"] == "usb_live"
        assert screen["session_id"] == session_id
        assert screen["source_epoch_id"] == epoch
        assert screen["shot_sequence"] == 1
        assert screen["phase"] == "stable"
        assert screen["rule_status"] == "RULES_PENDING"
        assert screen["guidance"] is None
        assert screen["rule_pending_reason"] == "RULE_DECISION_ABSTAINED"
        capture = state.projection_source.subsystem_health()["capture_diagnostics"]
        assert capture == {
            "source_kind": "blackbox_video",
            "runtime_environment": "development",
            "source_sha256": source_sha256,
            "source_epoch_id": epoch,
            "width": 160,
            "height": 120,
            "fps": 60,
            **acceptance,
        }
        assert str(video) not in json.dumps(capture)

        assert _SUCCESS_TRUTH.completion_mode == "explicit_user_end"
        _end_session_as_player(state)
        ended = state.record_store.load_session(session_id)
        assert ended.status is FieldSessionStatus.ENDED
        assert session_id in {
            record.session_id
            for record in state.record_store.list_ended_sessions_requiring_postgame()
        }
    finally:
        setup_gate.set()
        server.server_close()


def test_active_motion_eof_stays_incomplete_and_fresh_process_isolates_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "active-motion-eof.avi"
    _write_scenario(video, _ScenarioRender(total_frames=105))
    events, unauthorized_callers = _observe_runtime_publication(monkeypatch)
    old_epoch = "blackbox-epoch-interrupted"
    first_gate = threading.Event()
    first = _build_server(tmp_path, video=video, epoch=old_epoch, setup_gate=first_gate)
    first_state = first.field_state
    old_session_id = first_state.session_id
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs/analysis-jobs.sqlite3")
    original_emit = ProductRealtimeRuntime._emit_event

    def consumer_revokes_on_degradation(runtime, output, event, **kwargs):
        result = original_emit(runtime, output, event, **kwargs)
        if event.kind == "CAPTURE_DEGRADED":
            # Let the real field consumer react before the producer proceeds.
            # A terminal notification must not overtake the original shot end.
            assert first_state.projection_source._stopped.wait(
                runtime._config.background_shutdown_timeout_seconds
            ), "field consumer did not revoke the degraded source"
        return result

    monkeypatch.setattr(ProductRealtimeRuntime, "_emit_event", consumer_revokes_on_degradation)
    try:
        _setup_players(first_state)
        first_gate.set()
        assert first_state.wait_until_capture_ready(timeout_seconds=3.0) == old_epoch

        def degraded_old_shot():
            shots = first_state.record_store.list_shots(old_session_id)
            screen = first_state.projection_source.current().frame.payload
            if len(shots) != _EOF_TRUTH.shot_count or screen["rule_status"] != "DEGRADED":
                return None
            return shots[0], screen

        old_shot, old_screen = _until(
            degraded_old_shot,
            timeout_seconds=5.0,
            label="active-shot EOF degradation",
        )
        assert old_shot.settled_ms is None
        assert old_shot.live_job_id is None
        assert first_state.record_store.get_rule_decision(old_session_id, old_shot.shot_id) is None
        assert first_state.record_store.get_live_shot_fact(old_session_id, old_shot.shot_id) is None
        assert (
            first_state.record_store.get_live_shot_media(old_session_id, old_shot.shot_id) is None
        )
        assert len(jobs.list_session_jobs(old_session_id)) == _EOF_TRUTH.job_count
        assert _EOF_TRUTH.decision_reason in old_screen["degraded_reason"]
        assert first_state.projection_source.restart_required is False
        first_event_kinds = [event.kind for event in events]
        assert first_event_kinds == ["SHOT_STARTED", "SHOT_INCOMPLETE", "CAPTURE_DEGRADED"]
        assert events[0].frame_id in _EOF_TRUTH.start_frame_ids
        assert all(event.kind not in {"SHOT_SETTLED", "SHOT_ENQUEUED"} for event in events)
        assert unauthorized_callers == []
        old_ownership = first_state.record_store.get_shot_ownership(
            old_session_id, old_shot.shot_id
        )
        assert old_ownership is not None
        assert old_ownership.source_epoch_id == old_epoch
        assert (
            first_state.record_store.load_session(old_session_id).status is FieldSessionStatus.OPEN
        )
    finally:
        first_gate.set()
        first.server_close()
        monkeypatch.setattr(ProductRealtimeRuntime, "_emit_event", original_emit)

    old_store = first_state.record_store
    old_shot_snapshot = old_store.list_shots(old_session_id)
    old_ownership_snapshot = old_store.get_shot_ownership(old_session_id, old_shot.shot_id)
    old_event_count = len(events)

    new_epoch = "blackbox-epoch-restarted"
    second_gate = threading.Event()
    second = _build_server(tmp_path, video=video, epoch=new_epoch, setup_gate=second_gate)
    second_state = second.field_state
    new_session_id = second_state.session_id
    try:
        assert new_session_id != old_session_id
        assert old_store.load_session(old_session_id).status is FieldSessionStatus.ENDED
        _setup_players(second_state)
        second_gate.set()
        assert second_state.wait_until_capture_ready(timeout_seconds=3.0) == new_epoch

        def degraded_new_shot():
            shots = second_state.record_store.list_shots(new_session_id)
            screen = second_state.projection_source.current().frame.payload
            if len(shots) != _EOF_TRUTH.shot_count or screen["rule_status"] != "DEGRADED":
                return None
            return shots[0]

        new_shot = _until(
            degraded_new_shot,
            timeout_seconds=5.0,
            label="fresh-process active-shot EOF degradation",
        )
        assert new_shot.shot_id == "shot-0001"
        assert new_shot.settled_ms is None
        new_ownership = second_state.record_store.get_shot_ownership(
            new_session_id, new_shot.shot_id
        )
        assert new_ownership is not None
        assert new_ownership.source_epoch_id == new_epoch
        assert (new_ownership.session_id, new_ownership.source_epoch_id) != (
            old_ownership_snapshot.session_id,
            old_ownership_snapshot.source_epoch_id,
        )
        assert second_state.record_store.get_rule_decision(new_session_id, new_shot.shot_id) is None
        assert (
            second_state.record_store.get_live_shot_media(new_session_id, new_shot.shot_id) is None
        )
        assert jobs.list_session_jobs(new_session_id) == ()
        assert (
            second_state.record_store.load_session(new_session_id).status is FieldSessionStatus.OPEN
        )

        assert old_store.list_shots(old_session_id) == old_shot_snapshot
        assert (
            old_store.get_shot_ownership(old_session_id, old_shot.shot_id) == old_ownership_snapshot
        )
        assert old_store.get_rule_decision(old_session_id, old_shot.shot_id) is None
        assert old_store.get_live_shot_media(old_session_id, old_shot.shot_id) is None
        assert jobs.list_session_jobs(old_session_id) == ()
        assert [event.kind for event in events[old_event_count:]] == [
            "SHOT_STARTED",
            "SHOT_INCOMPLETE",
            "CAPTURE_DEGRADED",
        ]
        assert unauthorized_callers == []
    finally:
        second_gate.set()
        second.server_close()


def test_settled_open_blackbox_restart_isolates_real_video_and_preserves_evidence(
    tmp_path, monkeypatch
):
    # Canonical temp paths are required by the capture source's no-symlink contract.
    tmp_path = tmp_path.resolve()
    video = tmp_path / "settled-restart.avi"
    _write_scenario(video, _ScenarioRender(total_frames=480))
    events, unauthorized_callers = _observe_runtime_publication(monkeypatch)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs/analysis-jobs.sqlite3")

    def evidence(state):
        shots = state.record_store.list_shots(state.session_id)
        if len(shots) != 1 or shots[0].settled_ms is None:
            return None
        shot = shots[0]
        store = state.record_store
        result = (
            shots,
            store.get_shot_ownership(state.session_id, shot.shot_id),
            store.get_live_shot_fact(state.session_id, shot.shot_id),
            store.get_rule_decision(state.session_id, shot.shot_id),
            store.get_live_shot_media(state.session_id, shot.shot_id),
            jobs.list_session_jobs(state.session_id),
        )
        return result if all(result) else None

    gate = threading.Event()
    first = _build_server(tmp_path, video=video, epoch="settled-first", setup_gate=gate)
    old = first.field_state
    try:
        _setup_players(old)
        gate.set()
        assert old.wait_until_capture_ready(timeout_seconds=3) == "settled-first"
        before = _until(lambda: evidence(old), timeout_seconds=8, label="old settled evidence")
    finally:
        gate.set()
        first.server_close()
    assert old.record_store.load_session(old.session_id).status is FieldSessionStatus.OPEN
    old_job_bytes = {
        path: path.read_bytes() for path in before[5][0].media_root.rglob("*") if path.is_file()
    }
    first_events = len(events)
    gate = threading.Event()
    second = _build_server(tmp_path, video=video, epoch="settled-second", setup_gate=gate)
    new = second.field_state
    try:
        assert new.session_id != old.session_id
        assert old.record_store.load_session(old.session_id).status is FieldSessionStatus.ENDED
        assert new.match.red_user_id is None and new.match.blue_user_id is None
        assert new.match.winner_role is None and new.match.revision == 0
        _setup_players(new)
        gate.set()
        assert new.wait_until_capture_ready(timeout_seconds=3) == "settled-second"
        after = _until(lambda: evidence(new), timeout_seconds=8, label="new settled evidence")
        assert after[0][0].shot_id == before[0][0].shot_id == "shot-0001"
        assert after[1].source_epoch_id == "settled-second"
        assert before[1].source_epoch_id == "settled-first"
        assert after[5][0].job_id != before[5][0].job_id
        assert evidence(old) == before
        assert {path: path.read_bytes() for path in old_job_bytes} == old_job_bytes
        assert [event.kind for event in events[first_events:]] == [
            "SHOT_STARTED",
            "SHOT_SETTLED",
            "SHOT_ENQUEUED",
        ]
        assert unauthorized_callers == []
    finally:
        gate.set()
        second.server_close()
