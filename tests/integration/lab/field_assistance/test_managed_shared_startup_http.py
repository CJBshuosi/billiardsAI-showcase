"""Actual Node auth/routes/client + Python startup/CLI/raw pixels; platform is synthetic."""

import hashlib
import json
import shutil
import sqlite3
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.runtime_resources import RESOURCE_PATH_NAMES
from billiards_ai.edge.route_assistance.sync import UrllibSyncHttpTransport
from billiards_ai.lab import mainline1_scenario_capture as scenario
from billiards_ai.lab.field_assistance.cloud_runtime import (
    FieldCloudFactories,
    FieldCloudRuntime,
    _BackgroundPostgameAnalysisWorker,
    prepare_cloud_runtime,
)
from billiards_ai.lab.mainline1_calibration_registry import register_calibration
from billiards_ai.lab.mainline1_capture import CaptureSetup, capture_setup_fingerprint
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.shared.calibration.homography import calibrate
from scripts.algorithm import serve_mainline1_blackbox_field as cli
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _CAPTURE_CONFIG,
    _SHOT_CONFIG_PATH,
    _EmptyClassifier,
    _EmptyDetector,
    _observe_runtime_publication,
    _ScenarioRender,
    _until,
    _write_scenario,
)
from tests.integration.lab.field_assistance.test_managed_command_http import (
    SCOPE,
    Credentials,
)
from tests.integration.lab.field_assistance.test_managed_command_http import (
    bridge as http_bridge,
)
from tests.integration.lab.field_assistance.test_managed_scenario_playback import (
    SETTINGS,
    _later_frame,
)
from tests.unit.lab.field_assistance import test_runtime_resources as resource_fixtures
from tests.unit.scripts.test_serve_mainline1_blackbox_field import arguments

bridge = http_bridge
resource_assets = resource_fixtures.assets


def test_actual_client_login_join_and_command_authorization(bridge):
    _, control = bridge
    # No direct identity/binding/role seeds: go through the production entrypoint.
    anonymous = control("phone", action="auth.me", openid=None)
    assert anonymous["ok"] is False
    alice = control("phone", action="auth.me", openid="synthetic-phone-a")
    bob = control("phone", action="auth.me", openid="synthetic-phone-b")
    assert alice["ok"] and bob["ok"]
    assert alice["data"]["user_id"] != bob["data"]["user_id"]
    again = control("phone", action="auth.me", openid="synthetic-phone-a")
    assert again == alice
    assert control("phone", action="join.request", payload={"table_code": "x"})["ok"] is False


TABLE_CODE = "12345678901234567890123456789012"
HTTPS_ENDPOINT = "https://synthetic.example/sync-api"


class IdlePlatformWorker:
    """Disable unrelated provider/postgame/erase/fact-upload I/O, never command application."""

    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def factories_for(base):
    def transport(url, *, timeout_seconds=20, allow_redirects):
        # Production validates the synthetic HTTPS scope. Only the destination
        # changes here: signing, deadline and no-redirect transport stay real.
        assert url == HTTPS_ENDPOINT
        assert allow_redirects is False
        assert 0 < timeout_seconds <= 20
        return UrllibSyncHttpTransport(
            base, timeout_seconds=timeout_seconds, allow_redirects=allow_redirects
        )

    return replace(
        FieldCloudFactories(),
        credentials=lambda **_: Credentials(),
        transport=transport,
        postgame_worker=IdlePlatformWorker,
        saved_media_worker=IdlePlatformWorker,
        local_cleanup_worker=IdlePlatformWorker,
        live_fact_worker=IdlePlatformWorker,
    )


def runtime_for(base):
    return prepare_cloud_runtime(
        SimpleNamespace(
            managed_sessions=True,
            table_id=SCOPE.table_id,
            runtime_environment="development",
            sync_base_url=HTTPS_ENDPOINT,
            cloudbase_environment_id=SCOPE.environment_id,
            expected_development_environment_id=SCOPE.environment_id,
            device_id=SCOPE.device_id,
            credential_target=f"billiardsAI/development/{SCOPE.device_id}",
        ),
        factories=factories_for(base),
    )


def phone(control, action, *, who="a", **payload):
    result = control("phone", action=action, openid=f"synthetic-phone-{who}", payload=payload)
    assert result["ok"], result
    return result["data"]


def submit(control, state, kind, payload, *, who="a"):
    revision = state.record_store.load_session(state.session_id).match.revision
    issued = phone(
        control,
        "field.command.submit",
        who=who,
        session_id=state.session_id,
        command_type=kind,
        payload=payload,
        expected_revision=revision,
        idempotency_key=f"synthetic-intent-{state.session_id}-{revision:016d}",
    )
    result = _until(
        lambda: (
            result
            if (
                result := phone(
                    control, "field.command.status", who=who, command_id=issued["command_id"]
                )
            )["public_result_code"]
            is not None
            else None
        ),
        timeout_seconds=5,
        label="actual signed command completion",
    )
    assert result["public_result_code"] == "APPLIED", result
    return issued


def test_actual_blackbox_cli_two_births_phone_setup_raw_motion_and_retained_history(
    bridge, tmp_path, monkeypatch, resource_assets
):
    base, control = bridge
    argv = arguments(tmp_path)
    for name in RESOURCE_PATH_NAMES:
        argv[argv.index("--" + name.replace("_", "-")) + 1] = str(getattr(resource_assets, name))
    argv[argv.index("--scenario-video") + 1] = str(tmp_path / "scenario.avi")
    args = cli._parser().parse_args(argv)
    _write_scenario(args.scenario_video, _ScenarioRender(total_frames=480))
    args.config.write_bytes(_SHOT_CONFIG_PATH.read_bytes())
    argv += [
        "--managed-sessions",
        "--table-id",
        SCOPE.table_id,
        "--sync-base-url",
        HTTPS_ENDPOINT,
        "--device-id",
        SCOPE.device_id,
        "--credential-target",
        f"billiardsAI/development/{SCOPE.device_id}",
        "--cloudbase-environment-id",
        SCOPE.environment_id,
        "--expected-development-environment-id",
        SCOPE.environment_id,
        "--rule-profile",
        PROFILE_ID,
        "--identity-gate-status",
        "PASS",
        "--control-token",
        "synthetic-only-control",
    ]
    # Metadata inspection, setup, calibration inventory, model package identities
    # and config digest remain real; only heavy model execution is synthetic.
    setup = CaptureSetup.model_validate(
        dict(
            capture_setup_id="synthetic-160x120",
            table_id=SCOPE.table_id,
            camera_device_id="synthetic",
            lens_id="lens",
            mount_revision="1",
            distortion_revision="1",
            source_kind="file",
            requested_fps=60,
            geometry=dict(width=1920, height=1080, crop=[0, 0, 1920, 1080]),
        )
    )
    args.capture_setup.write_text(setup.model_dump_json())
    calibration = calibrate(
        "synthetic",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=2540,
        table_width_mm=1270,
    )
    directory = args.calibration_root / ("b" * 64)
    directory.mkdir()
    calibration_bytes = calibration.model_dump_json().encode()
    (directory / "calibration.json").write_bytes(calibration_bytes)
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                artifact_id="b" * 64,
                capture_setup_id=setup.capture_setup_id,
                capture_setup_fingerprint=capture_setup_fingerprint(setup),
                source_video_sha256="c" * 64,
                inventory={
                    "calibration.json": {
                        "bytes": len(calibration_bytes),
                        "sha256": hashlib.sha256(calibration_bytes).hexdigest(),
                    }
                },
            )
        )
    )
    register_calibration(
        args.calibration_root, SimpleNamespace(path=directory, artifact_id="b" * 64), setup
    )
    monkeypatch.setattr(cli.realtime, "RealtimeCaptureConfig", lambda: _CAPTURE_CONFIG)
    monkeypatch.setattr(cli.realtime, "Dinov2IdentityFeatureBackend", lambda **_: object())

    def metadata_bound_detector(root, **_):
        from billiards_ai.lab.mainline1_runtime import inspect_development_vision_package

        detector = _EmptyDetector()
        detector.package_identity = inspect_development_vision_package(root).identity
        return detector

    monkeypatch.setattr(cli.realtime, "load_development_vision_backend", metadata_bound_detector)
    monkeypatch.setattr(
        cli.realtime, "load_development_group_classifier", lambda *a, **k: _EmptyClassifier()
    )
    commands = []

    class ModelProcessBoundary:
        # The actual supervisor builds and launches the original command. Child
        # SDK execution is covered independently by the focused recovery tests.
        def __init__(self, purpose):
            self.purpose = purpose

        def poll(self):
            # Media precedes optional analysis; model execution remains controlled.
            return 0 if self.purpose == "media" else None

        def terminate(self):
            pass

        def wait(self, **_):
            return 0

    def postgame(**kwargs):
        def process(command, **_):
            commands.append(command)
            return ModelProcessBoundary(command[command.index("--purpose") + 1])

        return _BackgroundPostgameAnalysisWorker(**kwargs, process_factory=process)

    real_prepare_cloud = prepare_cloud_runtime
    monkeypatch.setattr(
        cli,
        "prepare_cloud_runtime",
        lambda args: real_prepare_cloud(
            args, factories=replace(factories_for(base), postgame_worker=postgame)
        ),
    )
    events, unauthorized = _observe_runtime_publication(monkeypatch)
    decoded_sources = []
    open_video = scenario.cv2.VideoCapture

    class CountedVideo:
        def __init__(self, path):
            self.capture = open_video(path)
            self.reads = 0
            self.closed = False
            decoded_sources.append(self)

        def read(self):
            self.reads += 1
            return self.capture.read()

        def release(self):
            self.closed = True
            return self.capture.release()

        def __getattr__(self, name):
            return getattr(self.capture, name)

    monkeypatch.setattr(scenario.cv2, "VideoCapture", CountedVideo)
    seen = []
    retained = {}
    real_builder = cli.build_live_field_assistance_server

    def build(**kwargs):
        gate = kwargs["scenario_playback_gate"]
        assert not gate.is_released()
        actual_preparer = kwargs["runtime_preparer"]

        def before_capture(*, resource_spec):
            # This is before constructing even the prepared raw backend/frame0.
            rows = control("snapshot")["rows"]
            store = SQLiteFieldRecordStore(args.state_db)
            current = store.get_active_managed_open_request(SCOPE)
            record = store.get_managed_resource_spec(SCOPE, current.open_request_id)
            assert record.spec == resource_spec
            if seen:
                assert decoded_sources[0].closed
                old = seen[0]
                retired = store.get_latest_retired_managed_open_request(SCOPE)
                assert retired.binding.session_id == old["session"]
                assert retired.sealed_close.coverage == "sealed"
                head = rows["session_command_authority"][old["session"]]
                assert head["coverage"] == "sealed"
                assert head["contiguous_ack_sequence"] == head["closed_target"] == 4
                assert store.load_session(old["session"]).status.value == "ended"
                assert old["session"] in store.list_pending_cloud_close_session_ids()
                assert old["session"] in {
                    item.session_id for item in store.list_ended_sessions_requiring_postgame()
                }
                assert (
                    rows["field_session_commands"][old["pending"]]["public_result_code"]
                    == "SESSION_ENDED"
                )
                retained["manifest"] = store.get_postgame_manifest(old["session"])
                assert retained["manifest"] is not None
            return actual_preparer(resource_spec=resource_spec)

        kwargs["runtime_preparer"] = before_capture
        server = real_builder(**kwargs)
        state = server.field_state
        epoch = state.record_store.get_managed_capture_epoch(
            SCOPE, state.session_id, state.projection_source.source_epoch_id
        )
        assert epoch.spec_sha256 == state.managed_resource_spec_record.spec_sha256

        def serve():
            first = _later_frame(state, 2)
            later = _later_frame(state, first.capture_frame_id + 125)
            assert first.mjpeg == later.mjpeg and later.captured_at_ns > first.captured_at_ns
            assert state.record_store.list_shots(state.session_id) == ()
            assert not gate.is_released()
            assert decoded_sources[-1].reads == 1
            assert later.capture_frame_id > 120
            if seen:
                old = seen[0]
                assert state.session_id != old["session"]
                assert first.source_epoch_id != old["epoch"]
                assert state.resource_session_ids == (state.session_id,)
                # Ordinary non-ranked maintenance may complete the close marker
                # after true sealing; the retained marker and pending analysis
                # business must survive that independent transition.
                with sqlite3.connect(args.state_db) as connection:
                    marker = connection.execute(
                        "SELECT status FROM field_cloud_close_intents WHERE session_id=?",
                        (old["session"],),
                    ).fetchone()
                assert marker is not None and marker[0] in {"pending", "complete"}
                command = _until(
                    lambda: next(
                        (c for c in commands if c[c.index("--purpose") + 1] == "analysis"),
                        None,
                    ),
                    timeout_seconds=3,
                    label="real supervisor original-context launch",
                )

                def command_value(flag):
                    return command[command.index(flag) + 1]

                spec = state.record_store.get_managed_resource_spec_for_session(
                    SCOPE, old["session"]
                )
                assert command_value("--managed-resource-spec-sha256") == spec.spec_sha256
                assert command_value("--artifact-root") == str(
                    args.artifact_root / "postgame" / old["session"]
                )
                assert command_value("--jobs-db") == str(args.jobs_root / "analysis-jobs.sqlite3")
                assert command_value("--config") == str(args.config)
                assert old["session"] not in state.recovery_diagnostics
                assert (
                    state.record_store.get_rule_decision(old["session"], old["shot"])
                    == retained["receipt"]
                )
                assert (
                    state.record_store.get_shot_ownership(old["session"], old["shot"])
                    == retained["owner"]
                )
                assert (
                    state.record_store.get_live_shot_media(old["session"], old["shot"])
                    == retained["media"]
                )
                assert state.record_store.list_shots(old["session"]) == retained["shots"]
                assert retained["media_path"].read_bytes() == retained["media_bytes"]
                assert retained["job_path"].read_bytes() == retained["job_bytes"]
                assert {
                    path.relative_to(retained["job"].media_root).as_posix(): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in retained["job"].media_root.rglob("*")
                    if path.is_file()
                } == retained["media_hashes"]
                historical_rows = control("snapshot")["rows"]
                for collection, records in retained["cloud_history"].items():
                    assert all(
                        historical_rows[collection][key] == row for key, row in records.items()
                    )
                with sqlite3.connect(args.jobs_root / "analysis-jobs.sqlite3") as connection:
                    descriptor = connection.execute(
                        "SELECT payload_json,payload_sha256,status FROM shot_analysis_jobs "
                        "WHERE job_id=?",
                        (retained["job"].job_id,),
                    ).fetchone()
                assert descriptor == retained["descriptor"]
                assert (
                    state.record_store.get_postgame_manifest(old["session"]) == retained["manifest"]
                )
                review = state.postgame_review_payload()
                assert review["review_session_id"] == old["session"]
                assert state.record_store.get_postgame_manifest(old["session"]) is not None
                seen.append(dict(session=state.session_id, epoch=first.source_epoch_id))
                return
            # Rejected unjoined actor cannot acquire role or release motion.
            denied = control(
                "phone",
                action="field.command.submit",
                openid="synthetic-phone-c",
                payload=dict(
                    session_id=state.session_id,
                    command_type="claim_role",
                    payload={"role": "red"},
                    expected_revision=0,
                    idempotency_key="synthetic-unjoined-000000000001",
                ),
            )
            assert denied["ok"] is False
            alice = phone(control, "auth.me")["user_id"]
            bob = phone(control, "auth.me", who="b")["user_id"]
            assert alice != bob
            for who in ("a", "b"):
                joined = phone(control, "join.request", who=who, table_code=TABLE_CODE)
                assert joined["session_id"] == state.session_id and joined["role"] is None
            submit(control, state, "claim_role", {"role": "red"})
            assert not gate.is_released()
            submit(control, state, "claim_role", {"role": "blue"}, who="b")
            assert not gate.is_released()
            proposed = phone(
                control,
                "ranking.match.settings.propose",
                session_id=state.session_id,
                mode=SETTINGS["mode"],
                format=SETTINGS["format"],
                race_target=None,
                first_breaker="red",
                ranking_intent=False,
            )
            assert proposed["status"] == "awaiting_settings_confirmations"
            for who in ("a", "b"):
                phone(
                    control, "ranking.match.settings.confirm", who=who, session_id=state.session_id
                )
            submit(control, state, "set_breaker", {"role": "red"})
            assert gate.is_released()
            _until(
                lambda: (
                    state.projection_source.runtime.current_capture_frame.mjpeg != first.mjpeg
                    or None
                ),
                timeout_seconds=3,
                label="source pixels move after real two-user settings",
            )
            assert decoded_sources[-1].reads > 90
            jobs = SQLiteShotAnalysisJobStore(args.jobs_root / "analysis-jobs.sqlite3")

            def evidence():
                shots = state.record_store.list_shots(state.session_id)
                if len(shots) != 1 or shots[0].settled_ms is None:
                    return None
                shot = shots[0]
                media = state.record_store.get_live_shot_media(state.session_id, shot.shot_id)
                queued = jobs.list_session_jobs(state.session_id)
                if media is None or len(queued) != 1:
                    return None
                return shot, media, queued[0]

            shot, media, job = _until(evidence, timeout_seconds=8, label="raw shot and queued job")
            owner = state.record_store.get_shot_ownership(state.session_id, shot.shot_id)
            fact = state.record_store.get_live_shot_fact(state.session_id, shot.shot_id)
            receipt = state.record_store.get_rule_decision(state.session_id, shot.shot_id)
            assert owner.actor_user_id == fact.actor_user_id == alice
            assert receipt.status == "abstained" and receipt.applied_command_id is None
            assert state.match.winner_role is None
            assert {event.kind for event in events} >= {
                "SHOT_STARTED",
                "SHOT_SETTLED",
                "SHOT_ENQUEUED",
            }
            assert unauthorized == []
            media_path = next(job.media_root.glob("*.mjpgseg"))
            job_path = job.media_root / "interval.json"
            with sqlite3.connect(args.jobs_root / "analysis-jobs.sqlite3") as connection:
                descriptor = connection.execute(
                    "SELECT payload_json,payload_sha256,status FROM shot_analysis_jobs "
                    "WHERE job_id=?",
                    (job.job_id,),
                ).fetchone()
            assert descriptor[2] == "pending"
            spec = state.managed_resource_spec_record.spec
            assert job.identity_training_package_id == spec.identity_training_package_id
            assert job.identity_training_package_sha256 == spec.identity_training_package_sha256
            cloud_rows = control("snapshot")["rows"]
            retained.update(
                cloud_history={
                    name: cloud_rows[name]
                    for name in (
                        "managed_session_births",
                        "managed_session_open_requests",
                        "session_role_binding_history",
                        "field_session_commands",
                    )
                },
                media_hashes={
                    path.relative_to(job.media_root).as_posix(): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in job.media_root.rglob("*")
                    if path.is_file()
                },
                receipt=receipt,
                owner=owner,
                media=media,
                job=job,
                descriptor=descriptor,
                shots=state.record_store.list_shots(state.session_id),
                media_path=media_path,
                media_bytes=media_path.read_bytes(),
                job_path=job_path,
                job_bytes=job_path.read_bytes(),
            )
            # Stop only the worker, then leave a real phone-issued outstanding
            # command so next birth must execute closing/drain, not empty close.
            state.cloud_runtime_for_test._workers["commands"].stop()
            pending = phone(
                control,
                "field.command.submit",
                session_id=state.session_id,
                command_type="switch_turn",
                payload={},
                expected_revision=state.match.revision,
                idempotency_key="synthetic-pending-next-birth-000001",
            )
            seen.append(
                dict(
                    session=state.session_id,
                    epoch=first.source_epoch_id,
                    shot=shot.shot_id,
                    pending=pending["command_id"],
                )
            )

        server.serve_forever = serve
        return server

    # Expose the actual owned runtime to stop its worker at the deliberate crash boundary.
    original_start = FieldCloudRuntime.start

    def start(runtime, server, **kwargs):
        server.field_state.cloud_runtime_for_test = runtime
        return original_start(runtime, server, **kwargs)

    monkeypatch.setattr(FieldCloudRuntime, "start", start)
    monkeypatch.setattr(cli, "build_live_field_assistance_server", build)
    assert cli.main(argv) == 0
    # A second artifact/profile keeps historical lookup tied to its original context.
    next_artifact = tmp_path / "artifact-next"
    next_artifact.mkdir()
    index = argv.index("--artifact-root") + 1
    argv[index] = str(next_artifact)
    # Restart with different invocation roots/config path and active revision.
    # Historical work must still use the original retained context.
    for name in ("jobs_root", "ring_root"):
        directory = tmp_path / ("next-" + name)
        directory.mkdir()
        argv[argv.index("--" + name.replace("_", "-")) + 1] = str(directory)
    config = tmp_path / "next-shot.yaml"
    config.write_bytes(args.config.read_bytes() + b"\n# next invocation\n")
    argv[argv.index("--config") + 1] = str(config)
    new_calibration = args.calibration_root / ("c" * 64)
    shutil.copytree(args.calibration_root / ("b" * 64), new_calibration)
    manifest = json.loads((new_calibration / "manifest.json").read_bytes())
    manifest["artifact_id"] = "c" * 64
    (new_calibration / "manifest.json").write_text(json.dumps(manifest))
    register_calibration(
        args.calibration_root, SimpleNamespace(path=new_calibration, artifact_id="c" * 64), setup
    )
    argv.remove("--rule-profile")
    argv.remove(PROFILE_ID)
    assert cli.main(argv) == 0
    assert len(seen) == 2
    assert all(attempt["count"] <= 100 for attempt in control("snapshot")["attempts"])

    # Execute the actual reconstructed child/worker against the actual raw-source
    # frozen job. Only SDK/inference is synthetic; metadata and job fields above
    # came through CLI -> factory -> runtime -> immutable producer, without edits.
    import numpy as np

    from billiards_ai.lab.vision.contracts import VisionFrameObservation
    from tests.unit.lab.test_mainline1_shot_worker import _detection, _FakeClassifier
    from tests.unit.scripts.test_run_mainline1_shot_worker import _module

    child = _module()
    detector = metadata_bound_detector(args.detector_training_run)

    class PixelDetector:
        package_identity = detector.package_identity

        def infer(self, frame):
            ys, xs = np.where(frame.image.mean(axis=2) > 100)
            x, y = int(round(xs.mean())), int(round(ys.mean()))
            return VisionFrameObservation(
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                width_px=frame.width_px,
                height_px=frame.height_px,
                pixel_sha256=frame.pixel_sha256,
                package=self.package_identity,
                threshold=0.25,
                detections=(_detection("cue", x, y), _detection("target", 140, 90)),
                runtime_ref="synthetic-pixel-model/1",
            )

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(from_checkpoint=lambda _: None))
    monkeypatch.setattr(child, "load_development_vision_backend", lambda *a, **k: PixelDetector())
    monkeypatch.setattr(
        child, "load_development_group_classifier", lambda *a, **k: _FakeClassifier()
    )
    monkeypatch.setattr(child, "Dinov2IdentityFeatureBackend", lambda **_: object())
    store = SQLiteFieldRecordStore(args.state_db)
    store.end_session(seen[1]["session"])
    child_args = next(c for c in commands if c[c.index("--purpose") + 1] == "analysis")[3:]
    for flag in (
        "--live-resource-url",
        "--live-resource-audit",
        "--live-server-pid",
        "--sync-base-url",
        "--device-id",
        "--credential-target",
        "--development-cloudbase-environment-id",
    ):
        index = child_args.index(flag)
        del child_args[index : index + 2]
    child_args.remove("--development-analysis-sync")
    index = child_args.index("--max-empty-polls")
    child_args[index + 1] = "1"
    assert child.main(child_args) == 0
    original_jobs = SQLiteShotAnalysisJobStore(args.jobs_root / "analysis-jobs.sqlite3")
    assert original_jobs.status(retained["job"].job_id).value == "available"
    assert store.get_rule_decision(seen[0]["session"], seen[0]["shot"]) == retained["receipt"]
    assert store.get_shot_ownership(seen[0]["session"], seen[0]["shot"]) == retained["owner"]
    with sqlite3.connect(args.jobs_root / "analysis-jobs.sqlite3") as connection:
        assert (
            connection.execute(
                "SELECT payload_json,payload_sha256 FROM shot_analysis_jobs WHERE job_id=?",
                (retained["job"].job_id,),
            ).fetchone()
            == retained["descriptor"][:2]
        )


@pytest.mark.parametrize("loss", ["request-loss", "response-loss"])
@pytest.mark.parametrize("route", ["/open", "/close"])
def test_signed_startup_http_loss_replays_original_request(bridge, tmp_path, loss, route):
    base, control = bridge
    runtime = runtime_for(base)
    startup = runtime.make_managed_startup()
    store = SQLiteFieldRecordStore(tmp_path / "state.sqlite")
    if route == "/close":
        old = startup.prepare(
            store, artifact_id="old-artifact", rule_profile_id=None, policy="usb_resume"
        )
    control("fault", kind=dict(path=route, kind=loss))
    with pytest.raises(OSError):
        startup.prepare(
            store, artifact_id="old-artifact", rule_profile_id=None, policy="blackbox_fresh"
        )
    retained = store.get_active_managed_open_request(SCOPE)
    assert retained is not None
    request = retained.open_request_id
    if route == "/open":
        assert retained.phase == "pending"
        assert len(control("snapshot")["rows"].get("managed_session_births", {})) == (
            1 if loss == "response-loss" else 0
        )
    else:
        assert retained.binding.session_id == old.session_id
        assert retained.lifecycle == "active"
        assert len(control("snapshot")["rows"]["managed_session_births"]) == 1
    recovered = startup.prepare(
        SQLiteFieldRecordStore(store.path),
        artifact_id="old-artifact",
        rule_profile_id=None,
        policy="blackbox_fresh",
    )
    rows = control("snapshot")
    allocations = [
        entry["body"]["open_request_id"]
        for entry in rows["requests"]
        if entry["path"].endswith("/open")
    ]
    if route == "/open":
        assert allocations == [request, request]
        assert store.get_active_managed_open_request(SCOPE).open_request_id == request
        assert len(rows["rows"]["managed_session_births"]) == 1
    else:
        assert recovered.session_id != old.session_id
        assert store.get_latest_retired_managed_open_request(SCOPE).open_request_id == request
        assert len(rows["rows"]["managed_session_births"]) == 2
    runtime.stop()


@pytest.mark.parametrize(
    "boundary",
    [
        "reserve_managed_open",
        "record_managed_binding",
        "materialize_managed_session",
        "end_session",
        "retire_managed_open",
    ],
)
@pytest.mark.parametrize("after", [False, True], ids=["before-commit", "after-commit"])
def test_sqlite_crash_boundaries_preserve_uuid_and_permanent_bindings(
    bridge, tmp_path, monkeypatch, boundary, after
):
    base, control = bridge
    runtime = runtime_for(base)
    startup = runtime.make_managed_startup()
    store = SQLiteFieldRecordStore(tmp_path / "state.sqlite")
    old = None
    if boundary in {"end_session", "retire_managed_open"}:
        old = startup.prepare(
            store, artifact_id="artifact-1", rule_profile_id=None, policy="usb_resume"
        )
    original = getattr(store, boundary)

    def crash(*args, **kwargs):
        if after:
            original(*args, **kwargs)
        raise OSError("synthetic process crash boundary")

    monkeypatch.setattr(store, boundary, crash)
    with pytest.raises(OSError, match="synthetic process crash"):
        startup.prepare(
            store, artifact_id="artifact-1", rule_profile_id=None, policy="blackbox_fresh"
        )
    retained = store.get_active_managed_open_request(SCOPE)
    retired = store.get_latest_retired_managed_open_request(SCOPE)
    before = control("snapshot")["rows"].get("managed_session_open_requests", {})
    # A fresh store stands in for restart. USB resumes a committed materialized
    # birth; a pending/bound attempt must use its exact original request UUID.
    policy = "blackbox_fresh" if old is not None else "usb_resume"
    recovered = startup.prepare(
        SQLiteFieldRecordStore(store.path),
        artifact_id="artifact-1",
        rule_profile_id=None,
        policy=policy,
    )
    current = store.get_active_managed_open_request(SCOPE)
    if old is None and retained is not None:
        assert current.open_request_id == retained.open_request_id
        if retained.binding is not None:
            assert recovered.session_id == retained.binding.session_id
    if retired is not None:
        assert retired.lifecycle == "retired" and retired.binding.status == "ended"
    after_rows = control("snapshot")["rows"]["managed_session_open_requests"]
    assert all(after_rows[key] == value for key, value in before.items())
    assert len(after_rows) == (2 if old is not None else 1)
    if old is not None:
        assert recovered.session_id != old.session_id
        assert (
            store.get_latest_retired_managed_open_request(SCOPE).binding.session_id
            == old.session_id
        )
    runtime.stop()


def test_arbitrary_local_retirement_never_authorizes_node_successor(bridge, tmp_path):
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionCloseState

    base, control = bridge
    runtime = runtime_for(base)
    startup = runtime.make_managed_startup()
    store = SQLiteFieldRecordStore(tmp_path / "state.sqlite")
    old = startup.prepare(
        store, artifact_id="artifact-1", rule_profile_id=None, policy="usb_resume"
    )
    request = store.get_active_managed_open_request(SCOPE)
    store.end_session(old.session_id)
    binding = request.binding
    # Well-typed and identity-matching local audit bytes have no server authority.
    forged = ManagedSessionCloseState(
        binding.schema_version,
        binding.session_id,
        binding.birth_id,
        binding.birth_contract,
        binding.table_id,
        binding.device_id,
        "sealed",
        0,
        0,
    )
    store.retire_managed_open(SCOPE, request.open_request_id, sealed_close=forged)
    pending = store.reserve_managed_open(SCOPE, artifact_id="new-artifact")
    with pytest.raises(OSError, match="SYNC_CONFLICT"):
        runtime.gateway.open_managed_session(
            open_request_id=pending.open_request_id, expected_table_id=SCOPE.table_id
        )
    rows = control("snapshot")["rows"]
    assert len(rows["managed_session_births"]) == 1
    assert rows["session_command_authority"][old.session_id]["coverage"] == "open"
    assert rows["field_live_sessions"][old.session_id]["status"] == "open"
    assert store.get_active_managed_open_request(SCOPE).open_request_id == pending.open_request_id
    runtime.stop()


def test_actual_signed_usb_resume_keeps_birth_and_opens_new_capture_epoch(bridge, tmp_path):
    from billiards_ai.lab.field_assistance.server import build_live_field_assistance_server
    from tests.integration.lab.field_assistance.test_managed_live_startup import options_at

    base, control = bridge
    runtime = runtime_for(base)
    options, captures = options_at(tmp_path)
    epochs = iter(["usb-epoch-first", "usb-epoch-resumed"])
    options["source_epoch_factory"] = lambda: next(epochs)
    sessions = []
    for _ in range(2):
        server = build_live_field_assistance_server(
            **options, managed_startup=runtime.make_managed_startup()
        )
        try:
            sessions.append(server.field_state.session_id)
        finally:
            server.server_close()
    assert sessions[0] == sessions[1]
    assert [capture.source_epoch_id for capture in captures] == [
        "usb-epoch-first",
        "usb-epoch-resumed",
    ]
    snapshot = control("snapshot")
    assert len(snapshot["rows"]["managed_session_births"]) == 1
    allocations = [
        entry["body"]["open_request_id"]
        for entry in snapshot["requests"]
        if entry["path"].endswith("/open")
    ]
    assert len(allocations) == 2 and allocations[0] == allocations[1]
    assert not any(entry["path"].endswith("/close") for entry in snapshot["requests"])
    runtime.stop()
