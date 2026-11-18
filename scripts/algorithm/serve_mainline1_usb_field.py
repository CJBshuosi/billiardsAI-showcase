#!/usr/bin/env python3
"""Serve native USB capture and the existing field UI in one process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import sys
import threading
import traceback
from pathlib import Path
from uuid import uuid4

# An absolute entrypoint works outside the checkout's working directory. Both
# roots are derived from this shipped file, never from implicit CWD discovery.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID  # noqa: E402
from billiards_ai.lab import mainline1_runtime_support as realtime  # noqa: E402
from billiards_ai.lab.field_assistance.cloud_runtime import (  # noqa: E402
    FieldCloudShutdownError,
    prepare_cloud_runtime,
    validate_development_cloud_scope,
)
from billiards_ai.lab.field_assistance.cloud_runtime import (  # noqa: E402
    _BackgroundFieldCommandWorker as _BackgroundFieldCommandWorker,
)
from billiards_ai.lab.field_assistance.cloud_runtime import (  # noqa: E402
    _BackgroundLiveShotFactSyncWorker as _BackgroundLiveShotFactSyncWorker,
)
from billiards_ai.lab.field_assistance.cloud_runtime import (  # noqa: E402
    _BackgroundPostgameAnalysisWorker as _BackgroundPostgameAnalysisWorker,
)
from billiards_ai.lab.field_assistance.cloud_runtime import (  # noqa: E402
    _BackgroundSavedMediaSyncWorker as _BackgroundSavedMediaSyncWorker,
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
    _report_startup_cleanup_failure,
    _retry_failed_startup_cleanup,
    build_live_field_assistance_server,
)
from billiards_ai.lab.mainline1_calibration_gate import (  # noqa: E402
    CalibrationGateStatus,
    evaluate_usb_calibration_gate,
)
from billiards_ai.lab.mainline1_capture import UsbSourceMetadata  # noqa: E402
from billiards_ai.lab.mainline1_field_runtime import build_live_runtime_factory  # noqa: E402


def _write_startup_progress(stage: str) -> None:
    """Emit one compact marker that the Windows installer can surface."""

    print(
        json.dumps({"status": "USB_FIELD_STARTING", "stage": stage}),
        flush=True,
    )


class _SessionPolicyParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        result = super().parse_args(args, namespace)
        if result.session_policy == "legacy":
            for name in ("detector_training_run", "identity_training_run", "dinov2_assets"):
                if getattr(result, name) is None:
                    self.error(f"--{name.replace('_', '-')} is required in legacy policy")
        return result


def _parser():
    parser = _SessionPolicyParser(description=__doc__)
    for name in (
        "native-helper",
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
    parser.add_argument("--join-qr-asset", type=Path)
    parser.add_argument("--device-locator", required=True)
    parser.add_argument("--device-fingerprint", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--control-token")
    parser.add_argument("--control-token-file", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sync-base-url")
    parser.add_argument("--device-id")
    parser.add_argument("--managed-sessions", action="store_true")
    parser.add_argument("--table-id")
    parser.add_argument("--credential-target")
    parser.add_argument("--runtime-environment")
    parser.add_argument("--cloudbase-environment-id")
    parser.add_argument("--expected-development-environment-id")
    parser.add_argument("--identity-gate-status", default="UNDECLARED")
    parser.add_argument("--observation-only", action="store_true")
    parser.add_argument("--development-model-acceptance", action="store_true")
    parser.add_argument("--existing-development-control-only", action="store_true")
    parser.add_argument("--shutdown-stdin", action="store_true")
    parser.add_argument("--rule-profile", choices=(PROFILE_ID,))
    return parser


def _validate_paths(args):
    recording = getattr(args, "session_policy", "legacy") == "recording-v1"
    if recording and args.rule_profile is not None:
        raise ValueError("recording policy does not request automatic rules")
    if not 0 <= args.port <= 65535 or args.batch_size <= 0:
        raise ValueError("invalid port or batch size")
    if not args.device_locator.strip() or args.device_locator.strip().isdigit():
        raise ValueError("a durable device locator, not a numeric camera index, is required")
    if len(args.device_fingerprint) != 64 or any(
        c not in "0123456789abcdef" for c in args.device_fingerprint
    ):
        raise ValueError("device fingerprint must be SHA-256")
    for name in ("native_helper", "capture_setup", "config"):
        path = getattr(args, name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required input file is missing or unsafe: {name}")
    if args.join_qr_asset is not None and (
        args.join_qr_asset.is_symlink() or not args.join_qr_asset.is_file()
    ):
        raise ValueError("join QR asset is missing or unsafe")
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
    if args.state_db.is_symlink() or not args.state_db.parent.is_dir():
        raise ValueError("state DB parent is missing or unsafe")
    if args.control_token_file is not None and (
        args.control_token_file.is_symlink() or not args.control_token_file.is_file()
    ):
        raise ValueError("control token file is missing or unsafe")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())


def _resolve_control_token(args):
    if args.control_token and args.control_token_file is not None:
        raise ValueError("control token must use exactly one transport")
    if args.control_token_file is not None:
        try:
            value = args.control_token_file.read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise ValueError("control token file is invalid") from exc
        if not value or value.strip() != value or len(value) < 32:
            raise ValueError("control token file is invalid")
        return value
    return args.control_token or secrets.token_urlsafe(32)


def _started_source_frame_batches(
    source,
    *,
    sample_count=12,
    warmup_count=120,
    maximum_batches=1,
):
    if sample_count < 2:
        raise ValueError("USB calibration gate requires at least two current frames")
    if warmup_count < 0:
        raise ValueError("USB calibration warmup count cannot be negative")
    if maximum_batches < 1:
        raise ValueError("USB calibration gate requires at least one evidence batch")
    stop = threading.Event()
    frames = source.frames(stop)
    try:
        try:
            for _ in range(warmup_count):
                next(frames)
            for _ in range(maximum_batches):
                yield tuple(next(frames) for _ in range(sample_count))
        except StopIteration as exc:
            raise ValueError("USB calibration gate did not receive enough current frames") from exc
    finally:
        stop.set()
        close = getattr(frames, "close", None)
        if close is not None:
            close()


def _sample_started_source(source, *, sample_count=12, warmup_count=120):
    batches = _started_source_frame_batches(
        source,
        sample_count=sample_count,
        warmup_count=warmup_count,
    )
    try:
        return next(batches)
    finally:
        batches.close()


def _validate_started_usb_source(
    source,
    *,
    setup,
    calibration,
    calibration_root,
):
    """Validate the fixed camera contract and a bounded live-frame window."""

    del calibration
    # This field installation has an immutable camera, mount, capture mode,
    # image-control profile, and registered calibration.  Startup therefore
    # verifies fresh frames from that exact source without redetecting table
    # corners; runtime capture health still detects stalls and reconnects.
    frames = _sample_started_source(source)
    last = frames[-1]
    metadata = UsbSourceMetadata(
        device_fingerprint=source.identity.durable_fingerprint,
        geometry=setup.geometry,
        fps=realtime.RealtimeCaptureConfig().fps,
        source_epoch_id=last.source_epoch_id,
        evidence_frame_sha256=hashlib.sha256(last.mjpeg).hexdigest(),
        validation_corners_px=None,
        image_controls=setup.image_controls,
    )
    gate = evaluate_usb_calibration_gate(
        source=metadata,
        setup=setup,
        calibration_root=calibration_root,
        max_corner_drift_px=12.0,
        require_current_corner_evidence=False,
    )
    if gate.status is not CalibrationGateStatus.READY:
        reasons = ",".join(gate.reasons) or gate.status.value
        raise ValueError(f"canonical USB calibration gate blocked: {reasons}")
    return gate, frames


def prepare_runtime(args, *, initial_source_epoch_id, resource_spec=None):
    """Reuse the accepted native mode/calibration/table preflight and model loaders."""
    del initial_source_epoch_id
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
    _write_startup_progress("capture_configuration_loading")
    setup = realtime.load_capture_setup(args.capture_setup)
    if (
        setup.source_kind is not realtime.CaptureSourceKind.USB
        or (setup.geometry.width, setup.geometry.height, setup.requested_fps) != (1920, 1080, 60)
        or setup.device_locator_fingerprint != args.device_fingerprint
    ):
        raise ValueError("capture setup does not authorize this exact native camera mode")
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
    preflight = realtime._capture(
        args, epoch=f"preflight-{uuid4().hex}", controls=setup.image_controls
    )
    preflight_cleanup = preflight.cancel
    try:
        _write_startup_progress("calibration_preflight_started")
        preflight.start()
        gate, frames = _validate_started_usb_source(
            preflight,
            setup=setup,
            calibration=calibration,
            calibration_root=args.calibration_root,
        )
        if gate.calibration_artifact_id != record.artifact_id:
            raise ValueError("canonical USB calibration gate selected another artifact")
    except BaseException as primary:
        _cleanup_failed_startup(primary, (("USB preflight", preflight_cleanup),))
        raise
    else:
        try:
            preflight_cleanup()
        except BaseException as primary:
            # Cancellation is the sole failure here. Retain that exact failed
            # operation without attempting it again before main owns cleanup.
            pending = list(getattr(primary, "_field_pending_cleanup", ()))
            if not any(action == preflight_cleanup for _, action in pending):
                pending.append(("USB preflight", preflight_cleanup))
            primary._field_pending_cleanup = tuple(pending)
            _report_startup_cleanup_failure(primary, "USB preflight", primary)
            raise
    _write_startup_progress("calibration_preflight_ready")
    config = realtime.load_mainline1_shot_local_config(
        args.config,
        **({} if resource_spec is None else {"expected_sha256": resource_spec.config_sha256}),
    )
    vision = identity = None
    if not recording:
        features = realtime.Dinov2IdentityFeatureBackend(
            asset_root=args.dinov2_assets, device=args.device, batch_size=args.batch_size
        )
        _write_startup_progress("detector_model_loading")
        vision = realtime.load_development_vision_backend(
            args.detector_training_run,
            model_factory=lambda path: realtime._detector(path, device=args.device),
        )
        _write_startup_progress("detector_model_ready")
        _write_startup_progress("identity_model_loading")
        identity = realtime.load_development_group_classifier(
            args.identity_training_run,
            feature_backend=features,
            evidence_backend=realtime.OpenCvIdentityVisualEvidenceBackend(),
            session_factory=realtime._session,
        )
        _write_startup_progress("identity_model_ready")
        # Warm detector and identity while the preflight camera is closed.
        image = frames[-1].decode_bgr()
        _write_startup_progress("detector_warmup_started")
        observed = vision.infer(
            realtime.make_vision_frame_input(
                image,
                frame_id=frames[-1].capture_frame_id,
                timestamp_ms=frames[-1].timestamp_ms,
            )
        )
        _write_startup_progress("detector_warmup_ready")
        _write_startup_progress("identity_warmup_started")
        identity.classify(image=image, detections=observed.detections)
        _write_startup_progress("identity_warmup_ready")
    factory = build_live_runtime_factory(
        capture_factory=lambda epoch: realtime._capture(
            args, epoch=epoch, controls=setup.image_controls
        ),
        capture_config=realtime.RealtimeCaptureConfig(),
        calibration=calibration,
        detector=vision,
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
                "identity_training_package_sha256": resource_spec.identity_training_package_sha256,
            }
        ),
        ring_root=args.ring_root,
        jobs_root=args.jobs_root,
        capture_starter=lambda source: source.start(),
        startup_validator=lambda source: _validate_started_usb_source(
            source,
            setup=setup,
            calibration=calibration,
            calibration_root=args.calibration_root,
        ),
        startup_progress=_write_startup_progress,
        **(
            {}
            if acceptance_policy is None
            else {
                "capture_diagnostics_factory": lambda source, epoch: {
                    "acceptance": acceptance_policy.as_payload(),
                },
            }
        ),
    )

    if (
        resource_spec is not None
        and inspect_managed_runtime_resources(args, rule_profile_id=resource_spec.rule_profile_id)
        != resource_spec
    ):
        raise ValueError("managed runtime preparation resources changed")
    if acceptance_policy is not None:
        acceptance_policy.verify_identity(args.identity_training_run)
    if resource_spec is not None:
        factory.managed_resource_spec_sha256 = resource_spec.spec_sha256
    _write_startup_progress("runtime_prepared")
    return record.artifact_id, factory, None


def _shutdown_on_stdin(server):
    for line in sys.stdin:
        if line.strip() == "STOP":
            server.shutdown()
            return
    server.shutdown()


def _postgame_worker_command(
    args, *, server_port, session_id, resource_context=None, purpose="analysis"
):
    if purpose not in {"analysis", "media"}:
        raise ValueError("postgame worker purpose is invalid")
    original = args
    if resource_context is not None:
        from types import SimpleNamespace

        record = resource_context.spec_record
        scope = record.scope
        if (
            getattr(original, "cloudbase_environment_id", None),
            getattr(original, "device_id", None),
            getattr(original, "table_id", None),
        ) != (scope.environment_id, scope.device_id, scope.table_id):
            raise ValueError("managed postgame invocation scope mismatch")
        if original.sync_base_url:
            validate_development_cloud_scope(original)
        spec = record.spec
        args = SimpleNamespace(
            **{
                **vars(original),
                **{k: Path(v) for k, v in spec.paths.items()},
                "device": spec.device,
                "batch_size": spec.batch_size,
            }
        )
    audit_root = args.jobs_root / "live-resource-audits"
    artifact_root = args.artifact_root / "postgame" / session_id
    audit_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(_PROJECT_ROOT / "scripts" / "algorithm" / "run_mainline1_shot_worker.py"),
        "--state-db",
        str(args.state_db),
        "--jobs-db",
        str(args.jobs_root / "analysis-jobs.sqlite3"),
        "--artifact-root",
        str(artifact_root),
        "--session-id",
        session_id,
        "--worker-id",
        f"commercial-v19-{purpose}-{session_id}",
        "--purpose",
        purpose,
        "--max-empty-polls",
        "480",
        "--poll-seconds",
        "0.25",
        "--live-resource-url",
        f"http://127.0.0.1:{server_port}/api/screen",
        "--live-resource-audit",
        str(audit_root / f"live-resource-{session_id}.jsonl"),
        "--live-server-pid",
        str(os.getpid()),
    ]
    if purpose == "analysis":
        command.extend(
            [
                "--capture-setup",
                str(args.capture_setup),
                "--calibration-root",
                str(args.calibration_root),
                "--config",
                str(args.config),
                "--detector-training-run",
                str(args.detector_training_run),
                "--identity-training-run",
                str(args.identity_training_run),
                "--dinov2-assets",
                str(args.dinov2_assets),
                "--fps",
                "60",
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
            ]
        )
    if resource_context is not None:
        command.extend(
            [
                "--managed-environment-id",
                scope.environment_id,
                "--managed-device-id",
                scope.device_id,
                "--managed-table-id",
                scope.table_id,
                "--managed-open-request-id",
                record.open_request_id,
                "--managed-resource-spec-sha256",
                record.spec_sha256,
            ]
        )
    if purpose == "analysis" and args.sync_base_url and not args.existing_development_control_only:
        command.extend(
            [
                "--development-analysis-sync",
                "--sync-base-url",
                args.sync_base_url,
                "--device-id",
                args.device_id,
                "--credential-target",
                args.credential_target,
                "--development-cloudbase-environment-id",
                args.expected_development_environment_id,
            ]
        )
    return command


def _cloud_cleanup_pending(exc):
    """A nested worker shutdown failure still owns its live dependencies."""
    if isinstance(exc, BaseExceptionGroup):
        return any(_cloud_cleanup_pending(child) for child in exc.exceptions)
    return isinstance(exc, FieldCloudShutdownError)


def main(argv=None):
    args = _parser().parse_args(argv)
    server = None
    cloud_runtime = None
    prepared_source = None
    previous_term = None
    exit_code = 0
    primary_failure = None
    try:
        _validate_paths(args)
        _write_startup_progress("inputs_verified")
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
        cloud_runtime = prepare_cloud_runtime(
            args,
            allow_existing_development_control_only=True,
            allow_local_development_environment=False,
        )
        _write_startup_progress("cloud_control_ready")
        initial_source_epoch_id = uuid4().hex

        def prepare_bound_runtime(*, resource_spec=None):
            nonlocal prepared_source
            if acceptance_policy is not None:
                acceptance_policy.verify_identity(args.identity_training_run)
            calibration_id, runtime_factory, prepared_source = prepare_runtime(
                args,
                initial_source_epoch_id=initial_source_epoch_id,
                **({} if resource_spec is None else {"resource_spec": resource_spec}),
            )
            if acceptance_policy is not None:
                acceptance_policy.verify_identity(args.identity_training_run)
            return calibration_id, runtime_factory, prepared_source

        managed_startup = (
            cloud_runtime.make_managed_startup()
            if getattr(args, "managed_sessions", False)
            else None
        )
        if managed_startup is None:
            calibration_id, runtime_factory, prepared_source = prepare_bound_runtime()
        else:
            calibration_id = runtime_factory = None
        source_epochs: list[str] = [initial_source_epoch_id]

        def next_source_epoch() -> str:
            if source_epochs:
                source_epoch_id = source_epochs.pop(0)
            else:
                source_epoch_id = uuid4().hex
            return source_epoch_id

        server_arguments = dict(
            artifact_path=args.artifact_root,
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
                        source_kind=realtime.CaptureSourceKind.USB,
                        device_fingerprint=args.device_fingerprint,
                    )
                )
            if not recording:
                server_arguments["runtime_resource_inspector"] = lambda: (
                    inspect_managed_runtime_resources(
                        args, rule_profile_id=None if args.observation_only else args.rule_profile
                    )
                )
        if cloud_runtime.sync_wakeup is not None:
            server_arguments["sync_wakeup"] = cloud_runtime.sync_wakeup
        if args.join_qr_asset is not None:
            server_arguments["join_qr_asset_path"] = args.join_qr_asset
        _write_startup_progress("service_bind_started")
        server = build_live_field_assistance_server(**server_arguments)
        _write_startup_progress("service_bound")
        validated_source_epoch_id = server.field_state.wait_until_capture_ready(
            timeout_seconds=30.0
        )
        _write_startup_progress("live_capture_ready")
        cloud_runtime.start(
            server,
            postgame_command_factory=lambda session_id, **context: _postgame_worker_command(
                args,
                server_port=server.server_port,
                session_id=session_id,
                **context,
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
                    "status": "USB_FIELD_READY",
                    "host": args.host,
                    "port": server.server_port,
                    "source_epoch_id": validated_source_epoch_id,
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
        print(f"USB field service failed: {exc}", file=sys.stderr, flush=True)
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
            print(f"USB field cleanup failed ({owner}):", file=sys.stderr, flush=True)
            traceback.print_exception(exc, file=sys.stderr, chain=False)
        if cleanup_failures and exit_code == 0:
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
