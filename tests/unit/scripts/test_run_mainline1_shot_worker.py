from __future__ import annotations

import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = Path(__file__).parents[3] / "scripts" / "algorithm" / "run_mainline1_shot_worker.py"
    spec = importlib.util.spec_from_file_location("shot_worker_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_cli_requires_an_explicit_ended_session_and_immutable_inputs() -> None:
    """Catches broad directory scanning or an implicit live-session worker entrypoint."""
    with pytest.raises(SystemExit) as exc:
        _module().main([])
    assert exc.value.code == 2


def test_worker_waits_for_live_idle_gate_before_loading_heavy_dependencies() -> None:
    module = _module()
    decisions = iter((False, False, True))
    sleeps = []
    gate = SimpleNamespace(may_run_heavy_analysis=lambda: next(decisions))

    assert (
        module._wait_for_initial_resource_gate(
            gate,
            once=False,
            max_empty_polls=3,
            poll_seconds=0.25,
            sleep=lambda seconds: sleeps.append(seconds),
        )
        is True
    )
    assert sleeps == [0.25, 0.25]


def test_worker_can_exit_without_model_loading_when_live_never_becomes_idle() -> None:
    module = _module()
    gate = SimpleNamespace(may_run_heavy_analysis=lambda: False)

    assert (
        module._wait_for_initial_resource_gate(
            gate,
            once=True,
            max_empty_polls=10,
            poll_seconds=0.25,
            sleep=lambda _seconds: None,
        )
        is False
    )


@pytest.mark.parametrize("activity", ["ready", "shot", "health_missing"])
def test_media_gate_allows_empty_rollover_but_rejects_live_activity(tmp_path, activity):
    from dataclasses import replace

    from billiards_ai.edge.route_assistance.control import WristbandRole
    from tests.unit.lab.test_mainline1_recording_media_worker import _ordinary_scene

    module = _module()
    records, old_id, _ = _ordinary_scene(tmp_path)
    current = records.open_or_resume_session(artifact_id="recording", session_policy="recording-v1")
    payload = dict(
        session=dict(session_id=current.session_id, status="open"),
        playback=dict(playing=True),
        media_encoding_idle=True,
        frame=dict(
            session_id=current.session_id,
            source_kind="usb_live",
            phase="stable",
            rule_status="WAITING_FOR_PLAYERS",
            guidance_visible=False,
        ),
        display=dict(phase="WAITING_FOR_PLAYERS"),
    )
    gate = module.AuditedMediaPriorityGate(
        records=records,
        target_session_id=old_id,
        live_url="http://127.0.0.1:8765/api/screen",
        server_pid=41,
        audit_path=tmp_path / f"live-resource-{old_id}.jsonl",
        probe=lambda _: payload,
        pid_is_alive=lambda _: True,
    )
    assert gate.may_run_heavy_analysis()
    if activity == "ready":
        records.save_match(
            current.session_id,
            replace(
                current.match,
                red_user_id="alice",
                blue_user_id="bob",
                breaker=WristbandRole.RED,
                active_role=WristbandRole.RED,
                revision=1,
            ),
            expected_revision=0,
        )
    elif activity == "shot":
        records.record_recording_start(current.session_id, "shot-next", "next-epoch", 1, 16)
    else:
        payload.pop("media_encoding_idle")
    # A stale healthy HTTP response cannot override the persisted activity fence.
    assert not gate.may_run_heavy_analysis()


def test_media_purpose_runs_without_any_semantic_or_model_dependency(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _module()
    records = object()
    monkeypatch.setattr(module, "_managed_context", lambda _args: (records, None))
    monkeypatch.setattr(
        module,
        "_resource_gate",
        lambda _args, observed: SimpleNamespace(may_run_heavy_analysis=lambda: observed is records),
    )
    jobs = SimpleNamespace(
        snapshot=lambda _job_id: SimpleNamespace(
            status=module.ShotAnalysisJobStatus.AVAILABLE,
            terminal_failure=False,
        )
    )
    monkeypatch.setattr(module, "SQLiteShotAnalysisJobStore", lambda _path: jobs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("media purpose must not construct semantic dependencies")

    for name in (
        "load_mainline1_shot_local_config",
        "load_capture_setup",
        "resolve_active_calibration",
        "resolve_calibration_record",
        "load_development_vision_backend",
        "load_development_group_classifier",
        "DevelopmentSessionAnalysisPublisher",
    ):
        monkeypatch.setattr(module, name, forbidden)

    class MediaWorker:
        def __init__(self, **kwargs):
            assert kwargs["records"] is records
            assert kwargs["jobs"] is jobs
            assert kwargs["artifact_root"] == tmp_path / "postgame" / "session-1"

        def process_one(self, worker_id, session_id, _now):
            assert (worker_id, session_id) == ("media-worker", "session-1")
            return SimpleNamespace(job_id="media-job", artifact_id="recording-playback-id")

        def schedule(self, session_id):
            assert session_id == "session-1"
            return (SimpleNamespace(job_id="media-job"),)

    monkeypatch.setattr(module, "Mainline1RecordingMediaWorker", MediaWorker)

    result = module.main(
        [
            "--state-db",
            str(tmp_path / "field.sqlite3"),
            "--jobs-db",
            str(tmp_path / "jobs.sqlite3"),
            "--artifact-root",
            str(tmp_path / "postgame" / "session-1"),
            "--session-id",
            "session-1",
            "--worker-id",
            "media-worker",
            "--purpose",
            "media",
            "--once",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == [
        {"artifact_id": "recording-playback-id", "job_id": "media-job"}
    ]


def test_media_purpose_fails_closed_when_scheduled_work_is_not_terminal(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_managed_context", lambda _args: (object(), None))
    monkeypatch.setattr(
        module,
        "_resource_gate",
        lambda _args, _records: SimpleNamespace(may_run_heavy_analysis=lambda: True),
    )
    pending = SimpleNamespace(
        status=module.ShotAnalysisJobStatus.FAILED_RETRYING,
        terminal_failure=False,
    )
    jobs = SimpleNamespace(snapshot=lambda job_id: pending if job_id == "media-job" else None)
    monkeypatch.setattr(module, "SQLiteShotAnalysisJobStore", lambda _path: jobs)

    class MediaWorker:
        def __init__(self, **_kwargs):
            pass

        def process_one(self, _worker_id, _session_id, _now):
            return None

        def schedule(self, session_id):
            assert session_id == "session-1"
            return (SimpleNamespace(job_id="media-job"),)

    monkeypatch.setattr(module, "Mainline1RecordingMediaWorker", MediaWorker)

    result = module.main(
        [
            "--state-db",
            str(tmp_path / "field.sqlite3"),
            "--jobs-db",
            str(tmp_path / "jobs.sqlite3"),
            "--artifact-root",
            str(tmp_path / "postgame" / "session-1"),
            "--session-id",
            "session-1",
            "--worker-id",
            "media-worker",
            "--purpose",
            "media",
            "--once",
        ]
    )

    assert result == 3
    assert json.loads(capsys.readouterr().out) == []


def test_analysis_gateway_requires_explicit_development_scope_and_complete_config() -> None:
    module = _module()
    disabled = SimpleNamespace(
        development_analysis_sync=False,
        sync_base_url=None,
        device_id=None,
        credential_target=None,
        development_cloudbase_environment_id=None,
    )
    assert module._development_analysis_gateway(disabled) is None

    for arguments in (
        SimpleNamespace(
            development_analysis_sync=True,
            sync_base_url="https://development.invalid",
            device_id=None,
            credential_target="credential-1",
            development_cloudbase_environment_id="cloud-development",
        ),
        SimpleNamespace(
            development_analysis_sync=False,
            sync_base_url="https://development.invalid",
            device_id="device-1",
            credential_target="credential-1",
            development_cloudbase_environment_id="cloud-development",
        ),
        SimpleNamespace(
            development_analysis_sync=True,
            sync_base_url="https://development.invalid",
            device_id="device-1",
            credential_target="credential-1",
            development_cloudbase_environment_id=None,
        ),
    ):
        with pytest.raises(ValueError, match="development|opt-in|complete"):
            module._development_analysis_gateway(arguments)


def test_explicit_development_analysis_gateway_requires_exact_attestation(monkeypatch) -> None:
    module = _module()

    class FakeSignedGateway:
        def __init__(self, **_kwargs) -> None:
            self.attestations = 0

        def runtime_attestation(self):
            self.attestations += 1
            return SimpleNamespace(
                runtime_environment="development",
                cloudbase_environment_id="cloud-development",
            )

    monkeypatch.setattr(module, "SignedFieldCloudGateway", FakeSignedGateway)
    arguments = SimpleNamespace(
        development_analysis_sync=True,
        sync_base_url="https://development.invalid",
        device_id="device-1",
        credential_target="credential-1",
        development_cloudbase_environment_id="cloud-development",
    )

    gateway = module._development_analysis_gateway(arguments)

    assert isinstance(gateway, module.DevelopmentAttestedFieldCloudGateway)
    assert gateway.runtime_attestation().cloudbase_environment_id == "cloud-development"


@pytest.mark.parametrize("kind", ["gateway", "sync"])
def test_development_child_transports_reject_redirects_offline(monkeypatch, kind):
    module = _module()
    arguments = SimpleNamespace(
        development_analysis_sync=True,
        sync_base_url="https://cloud-development.invalid",
        device_id="device-1",
        credential_target="billiardsAI/development/device-1",
        development_cloudbase_environment_id="cloud-development",
    )
    transports = []

    def transport(url, **kwargs):
        assert url == "https://cloud-development.invalid"
        assert kwargs == {"allow_redirects": False}
        result = object()
        transports.append(result)
        return result

    gateway = SimpleNamespace(
        runtime_attestation=lambda: SimpleNamespace(
            runtime_environment="development", cloudbase_environment_id="cloud-development"
        )
    )
    monkeypatch.setattr(module, "UrllibSyncHttpTransport", transport)
    monkeypatch.setattr(module, "WindowsCredentialManagerProvider", lambda **kw: object())
    monkeypatch.setattr(module, "SignedFieldCloudGateway", lambda **kw: gateway)
    if kind == "gateway":
        assert module._development_analysis_gateway(arguments) is not None
    else:
        assert module._development_field_sync_client(arguments, gateway) is not None
    assert len(transports) == 1


def _offline_receipt(tmp_path: Path, *, server_pid: int = 99_999_999):
    result = (tmp_path / "result").resolve()
    work = result / "work"
    work.mkdir(parents=True)
    state_db = work / "field-state.sqlite3"
    state_db.write_bytes(b"fixture")
    (work / "run-state.json").write_text(
        json.dumps(
            {
                "result_root": str(result),
                "state_db": str(state_db),
                "server_pid": server_pid,
                "started_at_utc": "2026-09-06T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    receipt = work / "server-exit.json"
    receipt.write_text(
        json.dumps(
            {
                "result_root": str(result),
                "state_db": str(state_db),
                "server_pid": server_pid,
                "ready_observed": True,
                "exit_code": 0,
                "stopped_at_utc": "2026-09-06T00:01:00Z",
            }
        ),
        encoding="utf-8",
    )
    return state_db, receipt


def test_verified_development_receipt_allows_ended_session_with_rollover_open(
    tmp_path: Path,
) -> None:
    module = _module()
    state_db, receipt = _offline_receipt(tmp_path)
    records = SimpleNamespace(
        load_session=lambda session_id: SimpleNamespace(
            session_id=session_id,
            status=module.FieldSessionStatus.ENDED,
        )
    )

    gate = module._development_offline_gate(records, state_db, "ended-1", receipt)

    assert gate.may_run_heavy_analysis() is True


@pytest.mark.parametrize("mutation", ["state_db", "live_pid", "not_ended"])
def test_development_receipt_fails_closed_on_conflict(tmp_path: Path, mutation: str) -> None:
    module = _module()
    state_db, receipt = _offline_receipt(
        tmp_path, server_pid=os.getpid() if mutation == "live_pid" else 99_999_999
    )
    if mutation == "state_db":
        state_db = tmp_path / "different.sqlite3"
        state_db.write_bytes(b"fixture")
    status = (
        module.FieldSessionStatus.OPEN
        if mutation == "not_ended"
        else module.FieldSessionStatus.ENDED
    )
    records = SimpleNamespace(
        load_session=lambda session_id: SimpleNamespace(
            session_id=session_id,
            status=status,
        )
    )

    with pytest.raises(ValueError):
        module._development_offline_gate(records, state_db, "ended-1", receipt)


def test_windows_pid_probe_is_read_only(monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module, "_windows_pid_is_alive", lambda pid: pid == 41)
    monkeypatch.setattr(
        module.os,
        "kill",
        lambda pid, signal: (_ for _ in ()).throw(AssertionError("must stay read-only")),
    )

    assert module._pid_is_alive(41) is True
    assert module._pid_is_alive(42) is False


def test_live_priority_gate_requires_ended_target_and_audited_idle_capture(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = {
        "session": {"session_id": "session-live", "status": "open"},
        "playback": {"playing": True},
        "frame": {
            "session_id": "session-live",
            "source_kind": "usb_live",
            "phase": "stable",
            "rule_status": "READY",
            "guidance_visible": False,
        },
        "display": {"phase": "WAITING_FOR_SHOT"},
    }
    target = SimpleNamespace(
        session_id="session-ended",
        status=module.FieldSessionStatus.ENDED,
    )
    receipt = tmp_path / "live-resource-session-ended.jsonl"
    gate = module.AuditedLivePriorityGate(
        records=SimpleNamespace(load_session=lambda _session_id: target),
        target_session_id="session-ended",
        live_url="http://127.0.0.1:8765/api/screen",
        server_pid=41,
        audit_path=receipt,
        probe=lambda _url: payload,
        pid_is_alive=lambda pid: pid == 41,
        now_utc=lambda: datetime(2026, 9, 7, tzinfo=UTC),
    )

    assert gate.may_run_heavy_analysis() is True
    audit = json.loads(receipt.read_text(encoding="utf-8").splitlines()[-1])
    assert audit["allowed"] is True
    assert audit["target_session_id"] == "session-ended"
    assert audit["live_session_id"] == "session-live"
    assert audit["reason"] == "live_capture_healthy_and_idle"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("moving", "live_route_or_capture_busy"),
        ("guidance", "live_route_or_capture_busy"),
        ("degraded", "live_capture_not_healthy"),
        ("same_session", "live_session_identity_invalid"),
        ("target_open", "target_session_not_ended"),
        ("dead_server", "live_server_not_running"),
        ("probe_failure", "live_resource_probe_failed"),
    ],
)
def test_live_priority_gate_fails_closed_when_realtime_priority_is_uncertain(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    module = _module()
    payload = {
        "session": {"session_id": "session-live", "status": "open"},
        "playback": {"playing": True},
        "frame": {
            "session_id": "session-live",
            "source_kind": "usb_live",
            "phase": "stable",
            "rule_status": "READY",
            "guidance_visible": False,
        },
        "display": {"phase": "WAITING_FOR_SHOT"},
    }
    if mutation == "moving":
        payload["frame"]["phase"] = "moving"
    elif mutation == "guidance":
        payload["frame"]["guidance_visible"] = True
    elif mutation == "degraded":
        payload["frame"]["rule_status"] = "DEGRADED"
    elif mutation == "same_session":
        payload["session"]["session_id"] = "session-ended"
        payload["frame"]["session_id"] = "session-ended"
    target_status = (
        module.FieldSessionStatus.OPEN
        if mutation == "target_open"
        else module.FieldSessionStatus.ENDED
    )

    def probe(_url):
        if mutation == "probe_failure":
            raise OSError("offline")
        return payload

    receipt = tmp_path / "live-resource-session-ended.jsonl"
    gate = module.AuditedLivePriorityGate(
        records=SimpleNamespace(
            load_session=lambda _session_id: SimpleNamespace(
                session_id="session-ended", status=target_status
            )
        ),
        target_session_id="session-ended",
        live_url="http://127.0.0.1:8765/api/screen",
        server_pid=41,
        audit_path=receipt,
        probe=probe,
        pid_is_alive=lambda _pid: mutation != "dead_server",
        now_utc=lambda: datetime(2026, 9, 7, tzinfo=UTC),
    )

    assert gate.may_run_heavy_analysis() is False
    audit = json.loads(receipt.read_text(encoding="utf-8").splitlines()[-1])
    assert audit["allowed"] is False
    assert audit["reason"] == reason


def test_live_priority_gate_rejects_non_loopback_or_unsafe_audit_path(
    tmp_path: Path,
) -> None:
    module = _module()
    arguments = {
        "records": SimpleNamespace(),
        "target_session_id": "session-ended",
        "server_pid": 41,
        "probe": lambda _url: {},
        "pid_is_alive": lambda _pid: True,
        "now_utc": lambda: datetime(2026, 9, 7, tzinfo=UTC),
    }

    with pytest.raises(ValueError, match="loopback"):
        module.AuditedLivePriorityGate(
            **arguments,
            live_url="http://192.0.2.1:8765/api/screen",
            audit_path=tmp_path / "audit.jsonl",
        )
    linked = tmp_path / "linked.jsonl"
    linked.symlink_to(tmp_path / "real.jsonl")
    with pytest.raises(ValueError, match="audit"):
        module.AuditedLivePriorityGate(
            **arguments,
            live_url="http://127.0.0.1:8765/api/screen",
            audit_path=linked,
        )


@pytest.fixture
def ended_live_gate(tmp_path):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore

    module = _module()
    ids = iter(("session-ended", "session-next"))
    records = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: next(ids)
    )
    records.open_or_resume_session(artifact_id="artifact-ended")
    records.end_session("session-ended")
    payload = {
        "session": {"session_id": "session-ended", "status": "ended"},
        "playback": {"playing": False},
        "frame": {
            "session_id": "session-ended",
            "source_kind": "usb_live",
            "phase": "stable",
            "rule_status": "SESSION_ENDED",
            "guidance_visible": False,
        },
        "display": {"phase": "SESSION_ENDED"},
    }
    receipt = tmp_path / "live-resource-session-ended.jsonl"
    gate = module.AuditedLivePriorityGate(
        records=records,
        target_session_id="session-ended",
        live_url="http://127.0.0.1:8765/api/screen",
        server_pid=41,
        audit_path=receipt,
        probe=lambda _url: payload,
        pid_is_alive=lambda _pid: True,
    )
    return module, records, payload, gate, receipt


def test_original_ended_session_can_generate_clips_before_another_match(ended_live_gate):
    module, records, payload, gate, receipt = ended_live_gate
    assert records.has_open_session() is False
    assert (
        module._wait_for_initial_resource_gate(
            gate,
            once=True,
            max_empty_polls=1,
            poll_seconds=0,
        )
        is True
    )
    audit = json.loads(receipt.read_text().splitlines()[-1])
    assert audit["allowed"] is True and audit["reason"] == "ended_session_idle"
    # A stale ended screen cannot hide a newly persisted live match.
    records.open_or_resume_session(artifact_id="artifact-next")
    assert gate.may_run_heavy_analysis() is False
    assert json.loads(receipt.read_text().splitlines()[-1])["reason"] == "ended_session_not_idle"


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("session", "status", "open"),
        ("session", "status", None),
        ("frame", "session_id", "foreign"),
        ("frame", "source_kind", "replay"),
        ("frame", "phase", "moving"),
        ("frame", "phase", "degraded"),
        ("frame", "rule_status", "DEGRADED"),
        ("frame", "rule_status", "READY"),
        ("display", "phase", "WAITING_FOR_SHOT"),
        ("display", "phase", "DEGRADED"),
        ("playback", "playing", True),
        ("playback", "playing", 0),
        ("frame", "guidance_visible", True),
        ("frame", "guidance_visible", 0),
        ("frame", "guidance_visible", None),
    ],
)
def test_ended_screen_cannot_bypass_live_resource_protection(
    ended_live_gate,
    section,
    key,
    value,
):
    _module_obj, _records, payload, gate, receipt = ended_live_gate
    if value is None:
        payload[section].pop(key)
    else:
        payload[section][key] = value
    assert gate.may_run_heavy_analysis() is False
    assert json.loads(receipt.read_text().splitlines()[-1])["allowed"] is False


@pytest.mark.parametrize("failure", ["runtime", "sqlite", "uncertain", "audit"])
def test_ended_session_gate_fails_closed_on_record_or_audit_failure(
    ended_live_gate,
    monkeypatch,
    failure,
):
    import sqlite3

    _module_obj, records, _payload, gate, receipt = ended_live_gate
    if failure == "audit":
        receipt.mkdir()
    else:

        def unavailable():
            if failure == "uncertain":
                return 0
            raise (
                sqlite3.OperationalError("busy")
                if failure == "sqlite"
                else RuntimeError("unavailable")
            )

        monkeypatch.setattr(records, "has_open_session", unavailable)
    assert gate.may_run_heavy_analysis() is False
    if failure != "audit":
        assert json.loads(receipt.read_text().splitlines()[-1])["allowed"] is False
