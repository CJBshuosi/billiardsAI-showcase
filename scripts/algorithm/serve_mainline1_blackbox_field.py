#!/usr/bin/env python3
"""Serve immutable scenario pixels through the live field path, in local development only.

Requires a FILE capture setup and its verified calibration at 1920x1080/60.
Finite EOF retains capture degradation/incomplete-shot behavior; it does not end a match.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import traceback
from functools import partial
from pathlib import Path
from uuid import uuid4

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID  # noqa: E402
from billiards_ai.lab import mainline1_runtime_support as realtime  # noqa: E402
from billiards_ai.lab.field_assistance.cloud_runtime import (  # noqa: E402
    prepare_cloud_runtime,
)
from billiards_ai.lab.field_assistance.development_acceptance import (  # noqa: E402
    resolve_acceptance_policy,
)
from billiards_ai.lab.field_assistance.recording_recovery import (  # noqa: E402
    prepare_recording_recovery_factory,
)
from billiards_ai.lab.field_assistance.runtime_resources import (  # noqa: E402
    inspect_managed_runtime_resources,
)
from billiards_ai.lab.field_assistance.server import (  # noqa: E402
    _cleanup_failed_startup,
    _retry_failed_startup_cleanup,
    build_live_field_assistance_server,
)
from billiards_ai.lab.mainline1_field_runtime import build_live_runtime_factory  # noqa: E402
from billiards_ai.lab.mainline1_scenario_capture import (  # noqa: E402
    ScenarioCaptureConfig,
    ScenarioPlaybackGate,
    ScenarioVideoCaptureBackend,
)
from scripts.algorithm.serve_mainline1_usb_field import (  # noqa: E402
    _cloud_cleanup_pending,
    _postgame_worker_command,
    _resolve_control_token,
    _SessionPolicyParser,
    _shutdown_on_stdin,
)


def _parser():
    parser = _SessionPolicyParser(description=__doc__, allow_abbrev=False)
    for name in (
        "scenario-video",
        "capture-setup",
        "calibration-root",
        "config",
        "state-db",
        "ring-root",
        "jobs-root",
        "artifact-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("detector-training-run", "identity-training-run", "dinov2-assets"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--session-policy", choices=("legacy", "recording-v1"), default="legacy")
    parser.add_argument(
        "--s1-guidance",
        action="store_true",
        help="Enable optional stable-snapshot S1 guidance; requires model paths",
    )
    parser.add_argument("--runtime-environment", choices=("development",), required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--control-token")
    parser.add_argument("--control-token-file", type=Path)
    parser.add_argument("--join-qr-asset", type=Path)
    parser.add_argument("--identity-gate-status", default="UNDECLARED")
    parser.add_argument("--observation-only", action="store_true")
    parser.add_argument("--development-model-acceptance", action="store_true")
    parser.add_argument("--shutdown-stdin", action="store_true")
    parser.add_argument("--rule-profile", choices=(PROFILE_ID,))
    parser.add_argument("--sync-base-url")
    parser.add_argument("--device-id")
    parser.add_argument("--managed-sessions", action="store_true")
    parser.add_argument("--table-id")
    parser.add_argument("--credential-target")
    parser.add_argument("--cloudbase-environment-id")
    parser.add_argument("--expected-development-environment-id")
    parser.set_defaults(existing_development_control_only=False)
    return parser


def _snapshot_factory(args):
    if not getattr(args, "s1_guidance", False):
        return None
    if args.session_policy != "recording-v1" or not all(
        (args.detector_training_run, args.identity_training_run, args.dinov2_assets)
    ):
        raise ValueError("S1 guidance requires recording-v1 and all three model paths")
    from billiards_ai.lab.field_assistance.snapshot_guidance import SnapshotGuidanceFactory
    from billiards_ai.lab.field_assistance.snapshot_models import ModelSnapshotObserverFactory
    from billiards_ai.lab.planning.s1_validation import prepare_s1_physics

    return SnapshotGuidanceFactory(
        ModelSnapshotObserverFactory(
            args.detector_training_run,
            args.identity_training_run,
            args.dinov2_assets,
            device=args.device,
            batch_size=args.batch_size,
        ),
        resource_preparer=prepare_s1_physics,
    )


def _validate_paths(args):
    recording = getattr(args, "session_policy", "legacy") == "recording-v1"
    if recording and args.rule_profile is not None:
        raise ValueError("recording policy does not request automatic rules")
    if args.runtime_environment != "development":
        raise ValueError("black-box capture requires development runtime")
    if not 0 <= args.port <= 65535 or args.batch_size <= 0:
        raise ValueError("invalid port or batch size")
    for name in ("capture_setup", "config", "join_qr_asset", "control_token_file"):
        path = getattr(args, name)
        if path is not None and (path.is_symlink() or not path.is_file()):
            raise ValueError(f"required input file is missing or unsafe: {name}")
    for name in (
        "calibration_root",
        "detector_training_run",
        "identity_training_run",
        "dinov2_assets",
        "ring_root",
        "jobs_root",
        "artifact_root",
    ):
        if recording and name in {
            "detector_training_run",
            "identity_training_run",
            "dinov2_assets",
        }:
            continue
        path = getattr(args, name)
        if path is None or path.is_symlink() or not path.is_dir():
            raise ValueError(f"required directory is missing or unsafe: {name}")
    if (
        args.state_db.is_symlink()
        or not args.state_db.parent.is_dir()
        or (args.state_db.exists() and not args.state_db.is_file())
    ):
        raise ValueError("state DB path or parent is missing or unsafe")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            # Keep original symlink evidence for the backend's whole-path check.
            setattr(args, name, value.absolute() if name == "scenario_video" else value.resolve())


def _capture_diagnostics(source, source_epoch_id, *, capture_config, acceptance_policy=None):
    """Public provenance from the actual backend; never expose its local locator."""
    return {
        "source_kind": "blackbox_video",
        "runtime_environment": "development",
        "source_sha256": source.identity.durable_fingerprint,
        "source_epoch_id": source_epoch_id,
        "width": capture_config.width,
        "height": capture_config.height,
        "fps": capture_config.fps,
        **({} if acceptance_policy is None else {"acceptance": acceptance_policy.as_payload()}),
    }


def prepare_runtime(args, *, initial_source_epoch_id, playback_gate=None, resource_spec=None):
    recording = getattr(args, "session_policy", "legacy") == "recording-v1"
    if recording and resource_spec is not None:
        raise ValueError("recording does not request semantic runtime resources")
    acceptance_policy = None if recording else getattr(args, "_development_acceptance_policy", None)
    if acceptance_policy is None and not recording:
        acceptance_policy = resolve_acceptance_policy(args)
    if acceptance_policy is not None:
        acceptance_policy.verify_identity(args.identity_training_run)
    if (
        resource_spec is not None
        and inspect_managed_runtime_resources(args, rule_profile_id=resource_spec.rule_profile_id)
        != resource_spec
    ):
        raise ValueError("managed runtime preparation resources changed")
    if args.runtime_environment != "development":
        raise ValueError("black-box capture requires development runtime")
    setup = realtime.load_capture_setup(args.capture_setup)
    if setup.source_kind is not realtime.CaptureSourceKind.FILE or (
        setup.geometry.width,
        setup.geometry.height,
        setup.requested_fps,
    ) != (1920, 1080, 60):
        raise ValueError("black-box capture requires a FILE setup at 1920x1080/60")
    record = realtime.resolve_active_calibration(
        args.calibration_root, realtime.capture_setup_fingerprint(setup)
    )
    if record is None:
        raise ValueError("capture setup has no active immutable calibration")
    calibration = realtime.TableCalibration.model_validate_json(
        (args.calibration_root / record.artifact_path / "calibration.json").read_bytes()
    )
    if calibration.device_id != setup.camera_device_id:
        raise ValueError("active calibration belongs to another camera")
    capture_config = realtime.RealtimeCaptureConfig()
    scenario_config = ScenarioCaptureConfig(args.scenario_video, capture_config)
    prepared_source = ScenarioVideoCaptureBackend(
        scenario_config,
        source_epoch_id=initial_source_epoch_id,
        playback_gate=playback_gate,
    )
    pinned_sha256 = prepared_source.identity.durable_fingerprint
    initial_source_available = True

    def capture_factory(epoch):
        nonlocal initial_source_available
        if initial_source_available and epoch == initial_source_epoch_id:
            initial_source_available = False
            # The prepared backend revalidates stat and SHA before its first frame.
            return prepared_source
        source = ScenarioVideoCaptureBackend(
            scenario_config,
            source_epoch_id=epoch,
            playback_gate=playback_gate,
        )
        if source.identity.durable_fingerprint != pinned_sha256:
            source.cancel()
            raise ValueError("scenario source identity changed since preparation")
        return source

    try:
        config = realtime.load_mainline1_shot_local_config(
            args.config,
            **({} if resource_spec is None else {"expected_sha256": resource_spec.config_sha256}),
        )
        vision = identity = None
        if not recording:
            features = realtime.Dinov2IdentityFeatureBackend(
                asset_root=args.dinov2_assets, device=args.device, batch_size=args.batch_size
            )
            vision = realtime.load_development_vision_backend(
                args.detector_training_run,
                model_factory=lambda path: realtime._detector(path, device=args.device),
            )
            identity = realtime.load_development_group_classifier(
                args.identity_training_run,
                feature_backend=features,
                evidence_backend=realtime.OpenCvIdentityVisualEvidenceBackend(),
                session_factory=realtime._session,
            )
        snapshot_factory = _snapshot_factory(args)
        factory = build_live_runtime_factory(
            capture_factory=capture_factory,
            capture_config=capture_config,
            calibration=calibration,
            detector=vision,
            **({"snapshot_guidance_factory": snapshot_factory} if snapshot_factory else {}),
            **({"session_policy": "recording-v1"} if recording else {}),
            identity_classifier=identity,
            shot_config=config,
            capture_setup_id=setup.capture_setup_id,
            calibration_artifact_id=record.artifact_id,
            config_path=args.config,
            **(
                {}
                if resource_spec is None
                else {
                    "config_sha256": resource_spec.config_sha256,
                    "identity_training_package_id": resource_spec.identity_training_package_id,
                    "identity_training_package_sha256": (
                        resource_spec.identity_training_package_sha256
                    ),
                }
            ),
            ring_root=args.ring_root,
            jobs_root=args.jobs_root,
            capture_diagnostics_factory=partial(
                _capture_diagnostics,
                capture_config=capture_config,
                acceptance_policy=acceptance_policy,
            ),
        )
        if acceptance_policy is not None:
            acceptance_policy.verify_identity(args.identity_training_run)
        if resource_spec is not None:
            if (
                inspect_managed_runtime_resources(
                    args, rule_profile_id=resource_spec.rule_profile_id
                )
                != resource_spec
            ):
                raise ValueError("managed runtime preparation resources changed")
            factory.managed_resource_spec_sha256 = resource_spec.spec_sha256
    except BaseException as primary:
        _cleanup_failed_startup(primary, (("blackbox prepared capture", prepared_source.cancel),))
        raise
    return record.artifact_id, factory, prepared_source


def main(argv=None):
    args = _parser().parse_args(argv)
    server = cloud_runtime = prepared_source = previous_term = None
    exit_code = 0
    primary_failure = None
    try:
        _validate_paths(args)
        control_token = _resolve_control_token(args)
        recording = getattr(args, "session_policy", "legacy") == "recording-v1"
        acceptance_policy = None if recording else resolve_acceptance_policy(args)
        if acceptance_policy is not None:
            args._development_acceptance_policy = acceptance_policy
        if (
            args.rule_profile
            and args.identity_gate_status != "PASS"
            and not args.observation_only
            and acceptance_policy is None
        ):
            raise ValueError("identity gate is not PASS; automatic rule authority is blocked")
        cloud_runtime = prepare_cloud_runtime(args)
        initial_epoch = uuid4().hex

        def prepare_bound_runtime(*, resource_spec=None):
            nonlocal prepared_source
            if acceptance_policy is not None:
                acceptance_policy.verify_identity(args.identity_training_run)
            calibration_id, runtime_factory, prepared_source = prepare_runtime(
                args,
                initial_source_epoch_id=initial_epoch,
                **({} if resource_spec is None else {"resource_spec": resource_spec}),
                **({} if playback_gate is None else {"playback_gate": playback_gate}),
            )
            if acceptance_policy is not None:
                acceptance_policy.verify_identity(args.identity_training_run)
            return calibration_id, runtime_factory, prepared_source

        managed_startup = (
            cloud_runtime.make_managed_startup()
            if getattr(args, "managed_sessions", False)
            else None
        )
        # Diagnostic/local legacy playback remains ungated; it supplies no
        # authenticated two-player evidence. Each managed invocation owns a latch.
        playback_gate = None if managed_startup is None else ScenarioPlaybackGate()
        if managed_startup is None:
            calibration_id, runtime_factory, prepared_source = prepare_bound_runtime()
        else:
            calibration_id = runtime_factory = None
        source_epochs = [initial_epoch]

        def next_source_epoch():
            return source_epochs.pop() if source_epochs else uuid4().hex

        server_arguments = dict(
            artifact_path=args.artifact_root,
            fresh_session_on_start=True,
            state_db_path=args.state_db,
            host=args.host,
            port=args.port,
            control_token=control_token,
            source_epoch_factory=next_source_epoch,
            calibration_id=calibration_id,
            runtime_factory=runtime_factory,
            cloud_gateway=cloud_runtime.gateway,
            rule_profile_id=None if args.observation_only else args.rule_profile,
            **({"session_policy": "recording-v1"} if recording else {}),
        )
        if managed_startup is not None:
            server_arguments["managed_startup"] = managed_startup
            server_arguments["runtime_preparer"] = prepare_bound_runtime
            if recording:
                server_arguments["recording_recovery_preparer"] = lambda: (
                    prepare_recording_recovery_factory(
                        capture_setup_path=args.capture_setup,
                        calibration_root=args.calibration_root,
                        config_path=args.config,
                        ring_root=args.ring_root,
                        jobs_root=args.jobs_root,
                        source_kind=realtime.CaptureSourceKind.FILE,
                    )
                )
            if not recording:
                server_arguments["runtime_resource_inspector"] = lambda: (
                    inspect_managed_runtime_resources(
                        args, rule_profile_id=None if args.observation_only else args.rule_profile
                    )
                )
            server_arguments["scenario_playback_gate"] = playback_gate
        if cloud_runtime.sync_wakeup is not None:
            server_arguments["sync_wakeup"] = cloud_runtime.sync_wakeup
        if args.join_qr_asset is not None:
            server_arguments["join_qr_asset_path"] = args.join_qr_asset
        server = build_live_field_assistance_server(**server_arguments)
        validated_epoch = server.field_state.wait_until_capture_ready(timeout_seconds=30.0)
        cloud_runtime.start(
            server,
            postgame_command_factory=lambda session_id, **context: _postgame_worker_command(
                args, server_port=server.server_port, session_id=session_id, **context
            ),
        )
        if args.shutdown_stdin:
            threading.Thread(
                target=_shutdown_on_stdin, args=(server,), daemon=True, name="field-service-stop"
            ).start()

        def terminate(_signum, _frame):
            raise KeyboardInterrupt

        previous_term = signal.signal(signal.SIGTERM, terminate)
        print(
            json.dumps(
                {
                    "status": "BLACKBOX_FIELD_READY",
                    "host": args.host,
                    "port": server.server_port,
                    "source_epoch_id": validated_epoch,
                    **(
                        {}
                        if acceptance_policy is None
                        else {"acceptance": acceptance_policy.as_payload()}
                    ),
                }
            ),
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt as exc:
        primary_failure = exc
        exit_code = 130
    except (OSError, ValueError, RuntimeError) as exc:
        primary_failure = exc
        print(f"Black-box field service failed: {exc}", file=sys.stderr, flush=True)
        exit_code = 2
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        cleanup_failures = []
        if previous_term is not None:
            try:
                signal.signal(signal.SIGTERM, previous_term)
            except BaseException as exc:
                cleanup_failures.append(("signal restore", exc))
        workers_terminated = True
        try:
            if cloud_runtime is not None:
                cloud_runtime.stop()
        except BaseException as exc:
            workers_terminated = not _cloud_cleanup_pending(exc)
            cleanup_failures.append(("cloud workers", exc))
        if workers_terminated:
            if primary_failure is not None:
                try:
                    _retry_failed_startup_cleanup(primary_failure)
                except BaseException as exc:
                    cleanup_failures.append(("startup owners", exc))
            try:
                if server is not None:
                    server.server_close()
                elif prepared_source is not None and not (
                    primary_failure is not None
                    and hasattr(primary_failure, "_field_pending_cleanup")
                ):
                    prepared_source.cancel()
            except BaseException as exc:
                cleanup_failures.append(("capture/service", exc))
        for owner, exc in cleanup_failures:
            print(f"Black-box field cleanup failed ({owner}):", file=sys.stderr, flush=True)
            traceback.print_exception(exc, file=sys.stderr, chain=False)
        if cleanup_failures and exit_code == 0:
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
