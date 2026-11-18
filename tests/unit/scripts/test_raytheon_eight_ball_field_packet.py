from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import zipfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[3]
PACKAGER = PROJECT / "scripts/diagnostics/package_raytheon_eight_ball_field.py"
EXPORTER = PROJECT / "scripts/diagnostics/raytheon_eight_ball_field/export_evidence.py"
RUNNER = PROJECT / "scripts/diagnostics/raytheon_eight_ball_field/run.ps1"
PROFILE_ID = "cbsa-chinese-pool-2017-observable-v1"
DEVELOPMENT_ENVIRONMENT_ID = "cloud1-public-example"


def _load(path: Path, name: str):
    assert path.is_file(), f"required script is missing: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(root: Path, *, include_qr: bool = True) -> Path:
    candidate = root / "candidate"
    (candidate / "calibrations/active").mkdir(parents=True)
    (candidate / "capture-setup.json").write_text('{"fixture": true}\n', encoding="utf-8")
    (candidate / "calibrations/registry.json").write_text("{}\n", encoding="utf-8")
    (candidate / "calibrations/active/calibration.json").write_text(
        '{"fixture": "calibration"}\n', encoding="utf-8"
    )
    if include_qr:
        (candidate / "join-qr.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    return candidate


def test_runner_targets_windows_powershell_51_without_shell_fallback() -> None:
    source = RUNNER.read_text(encoding="utf-8-sig")

    assert source.startswith("#requires -Version 5.1")
    assert ".ArgumentList" not in source
    assert "RandomNumberGenerator]::GetBytes" not in source
    assert ".Kill($true)" not in source
    assert "Invoke-Expression" not in source
    assert "Start-Process" not in source
    assert "ConvertTo-WindowsProcessArgument" in source
    assert "$StartInfo.Arguments = Join-WindowsProcessArguments $Arguments" in source
    assert "Protect-PrivateFile" in source
    assert '"--control-token-file", $SecretPath' in source
    assert '"--control-token", $ControlToken' not in source
    assert "undeclared file" in source
    assert "[switch]$ObservationOnly" in source
    assert '[ValidateSet("development")]' in source
    assert "CloudbaseEnvironmentId" in source
    assert "Assert-DevelopmentCloudScope" in source
    assert "credential_target -notmatch '^billiardsAI/development/'" in source
    assert "identity gate is not PASS" in source
    assert "operator-gate-evidence.json" in source
    assert "field-gates.json" in source
    assert "Invoke-FieldGateEvaluation" in source
    assert "Assert-FrozenRunState" in source
    assert "Assert-FrozenRunState $State $Manifest" in source
    assert "[switch]$LocalOnly" in source
    assert "if ($LocalOnly -and -not $ObservationOnly)" in source
    assert "LocalOnly rejects all sync and cloud arguments" in source
    assert "$ExistingRuntime = if ($LocalOnly)" in source
    assert "[switch]$ExistingDevelopmentControlOnly" in source
    assert "Assert-ExistingFieldServiceStopped" in source
    assert '"--existing-development-control-only"' in source
    assert "if ($State.analysis_sync_enabled -eq $true)" in source
    assert "ExistingDevelopmentControlOnly rejects explicit sync and cloud arguments" in source


def test_local_only_runner_treats_missing_join_qr_as_explicitly_optional() -> None:
    source = RUNNER.read_text(encoding="utf-8-sig")

    resolver = source.index("function Resolve-JoinQrAsset")
    local_optional = source.index("if ($LocalOnly) { return $null }", resolver)
    nonlocal_failure = source.index(
        "No explicit, packet, or existing field-runtime join QR asset is available.",
        resolver,
    )
    assert local_optional < nonlocal_failure
    assert "if (-not [string]::IsNullOrWhiteSpace([string]$ResolvedJoinQr)) {" in source
    assert '$Arguments += @("--join-qr-asset", $ResolvedJoinQr)' in source
    assert "if (-not [string]::IsNullOrWhiteSpace([string]$State.join_qr_asset) -and" in source
    assert "if ($null -ne $State.join_qr_sha256) {" in source


def test_frozen_control_only_state_is_not_misclassified_as_local_only() -> None:
    source = RUNNER.read_text(encoding="utf-8-sig")
    frozen = source[source.index("function Assert-FrozenRunState") :]

    local_branch = frozen.index("if ($State.local_only -eq $true)")
    control_branch = frozen.index("elseif ($State.existing_development_control_only -eq $true)")
    cloud_branch = frozen.index("} else {", control_branch)
    assert local_branch < control_branch < cloud_branch
    assert "Frozen existing development control-only run is not exactly packet pinned" in frozen


def test_worker_receives_frozen_development_analysis_sync_scope() -> None:
    source = RUNNER.read_text(encoding="utf-8-sig")

    assert '"--development-analysis-sync"' in source
    assert '"--sync-base-url", $State.sync_base_url' in source
    assert '"--device-id", $State.device_id' in source
    assert '"--credential-target", $State.credential_target' in source
    assert '"--development-cloudbase-environment-id", $State.cloudbase_environment_id' in source


def test_runner_waits_for_clean_worker_exit_and_durable_cloud_publication() -> None:
    source = RUNNER.read_text(encoding="utf-8-sig")

    assert "if ($Available -or $Process.HasExited)" not in source
    assert "if ($Process.HasExited) { break }" in source
    assert "Invoke-WorkerPublicationInspection" in source
    assert "$ExitCode -eq 0" in source
    assert "$Publication.verified -eq $true" in source
    assert "publication_verified = $Publication.verified" in source
    publication = source[source.index("function Invoke-WorkerPublicationInspection") :]
    assert "if ($State.analysis_sync_enabled -ne $true)" in publication


def test_keep_running_rollover_starts_one_audited_background_worker() -> None:
    source = RUNNER.read_text(encoding="utf-8-sig")

    keep_running = source.index("if ($KeepRunning)")
    rollover = source.index("Wait-ForSessionRollover", keep_running)
    state_write = source.index("Update-RunStateLocked $StatePath", rollover)
    worker_start = source.index("Start-SessionWorkerBackground", state_write)
    branch_return = source.index("return", worker_start)
    assert rollover < state_write < worker_start < branch_return
    assert "Update-RunStateLocked" in source
    assert "worker_identity" in source
    assert "launcher_pid" in source
    assert "command_identity_sha256" in source
    assert "stdout_path" in source
    assert "stderr_path" in source
    assert "live_resource_audit_path" in source
    assert 'resource_isolation = "development_best_effort_not_proven"' in source
    assert "deadline_at_utc" in source


def test_final_end_reaps_matching_worker_and_never_relaunches_it() -> None:
    source = RUNNER.read_text(encoding="utf-8-sig")

    assert "Assert-WorkerProcessIdentity" in source
    assert "Wait-SessionWorkerResult" in source
    assert "Record-WorkerHarvest" in source
    assert "worker result identity does not match the frozen launch" in source
    assert "if ($null -eq $SessionReceipt.worker_launch)" in source
    assert "Start-SessionWorkerBackground" in source
    assert "worker_diagnostics" in source
    assert "deadline_missed" in source
    assert (
        'if ($ExpiredBeforeWorker) { throw "Worker manifest deadline already expired." }' in source
    )
    assert "$HardStopAt = [DateTimeOffset]::UtcNow.AddSeconds(30)" not in source


def test_runner_reads_session_identity_from_the_real_screen_payload_shape() -> None:
    source = RUNNER.read_text(encoding="utf-8-sig")

    assert "$InitialScreen.session.session_id" in source
    assert "$Live.session.session_id" in source
    assert "$InitialScreen.session_id" not in source


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", '""'),
        ("plain", '"plain"'),
        ("two words", '"two words"'),
        ('say "hi"', '"say \\"hi\\""'),
        ("ends\\", '"ends\\\\"'),
        ("C:\\Program Files\\Python\\", '"C:\\Program Files\\Python\\\\"'),
    ],
)
def test_windows_process_argument_encoder(value: str, expected: str) -> None:
    powershell = os.environ.get("RAYTHEON_TEST_PWSH") or shutil.which("pwsh")
    if not powershell:
        pytest.skip("PowerShell is unavailable; Windows argument encoding remains a field gate")
    quoted_runner = str(RUNNER).replace("'", "''")
    environment = os.environ.copy()
    environment["RAYTHEON_ARGUMENT_VALUE"] = value
    encode_command = (
        ". '"
        + quoted_runner
        + "'; [Console]::Out.Write((ConvertTo-WindowsProcessArgument "
        + "$env:RAYTHEON_ARGUMENT_VALUE))"
    )
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            encode_command,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == expected


def test_packet_contains_independent_development_field_workflow(tmp_path: Path) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_packager")
    archive = module.build_packet(
        PROJECT,
        tmp_path / "out",
        candidate_root=_candidate(tmp_path),
    )

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        manifest_name = next(name for name in names if name.endswith("packet-manifest.json"))
        root = manifest_name.removesuffix("packet-manifest.json")
        manifest = json.loads(bundle.read(manifest_name))
        launcher = bundle.read(root + "run.ps1").decode("utf-8-sig")

        assert manifest["kind"] == "mainline1_eight_ball_field_development_acceptance"
        assert manifest["development_evidence_only"] is True
        assert manifest["production_ready"] is False
        assert manifest["packet_version"].endswith("20260907-v14")
        assert manifest["baseline_commit"] == ("9a144955377b289ce47de71769d1305dd8585de4")
        assert manifest["source_identity"] == "packet_manifest_per_file_sha256"
        assert manifest["production_gates_pending"] == [
            "trusted_pnp_to_capture_handle_binding",
            "product_controls_lock",
            "low_level_camera_read_timeout_and_cancel",
            "actual_table_corner_detector",
            "signed_native_release",
            "non_windows_product_hardware_base",
            "cross_process_revocable_gpu_preemption",
        ]
        assert manifest["writes_installed_service_or_state_db"] is True
        assert manifest["installed_state_write_scope"] == (
            "retire_one_existing_development_session_only"
        )
        assert manifest["writes_production_state"] is False
        assert manifest["writes_development_cloud_state"] is True
        assert manifest["native_helper"]["trust"] == "unsigned_source_compiled_at_runtime"
        assert manifest["run"]["rule_profile"] == PROFILE_ID
        assert manifest["run"]["development_environment_id"] == DEVELOPMENT_ENVIRONMENT_ID
        assert manifest["run"]["existing_development_control_base_url"] == (
            "https://cloud1-public-example-1461180277.ap-shanghai.app.tcloudbase.com"
        )
        assert manifest["run"]["identity_gate_required_for_rule_authority"] == "PASS"
        assert manifest["run"]["failed_identity_behavior"] == "observation_only_or_block"
        assert manifest["run"]["phases"] == ["Start", "EndSession"]
        assert manifest["run"]["postgame_deadline_seconds"] == 120
        assert manifest["run"]["qr_source"] == (
            "packet_if_present_else_current_json_in_existing_development_control_only"
        )
        assert manifest["run"]["sync_source"] == (
            "current_json_in_existing_development_control_only"
        )
        assert manifest["run"]["local_only_supported"] is True
        assert manifest["run"]["existing_development_control_only_supported"] is True
        assert manifest["run"]["recommended_field_mode"] == (
            "observation_only_existing_development_control_only"
        )
        assert manifest["run"]["analysis_sync_enabled_in_recommended_mode"] is False
        assert manifest["run"]["live_priority_isolation"] == ("development_best_effort_not_proven")
        assert manifest["run"]["result_evidence"] == [
            "launch-audit.json",
            "native-helper.json",
            "server-exit.json",
            "state-audit.json",
            "field-gates.json",
            "summary.json",
            "existing-session-retirement.json",
            "session-worker-result-<session_id>.json",
            "live-resource-<session_id>.jsonl",
        ]
        assert manifest["run"]["launch_parameters_redacted"] is True
        for relative in (
            "run.ps1",
            "README.txt",
            "export_evidence.py",
            "retire_existing_session.py",
            "field_gate_evaluator.py",
            "scripts/algorithm/serve_mainline1_usb_field.py",
            "scripts/algorithm/run_mainline1_product_realtime.py",
            "scripts/algorithm/run_mainline1_shot_worker.py",
            "scripts/algorithm/windows/serve-mainline1-usb-field.ps1",
            "configs/algorithm/shot-local.yaml",
            "native/windows/mainline1_uvc_capture_csharp/main.cs",
            "native/windows/mainline1_uvc_capture/protocol.md",
            "candidate/capture-setup.json",
            "candidate/calibrations/registry.json",
            "candidate/join-qr.png",
        ):
            assert root + relative in names
        assert root + "src/billiards_ai/edge/route_assistance/eight_ball_rules.py" in names
        # Availability and byte identity of both dormant entrypoints, their
        # startup/playback helpers, and the exact schema21 migration module.
        for relative in (
            "scripts/algorithm/serve_mainline1_usb_field.py",
            "scripts/algorithm/serve_mainline1_blackbox_field.py",
            "src/billiards_ai/lab/field_assistance/managed_startup.py",
            "src/billiards_ai/lab/field_assistance/managed_lifecycle.py",
            "src/billiards_ai/lab/field_assistance/scenario_playback.py",
            "src/billiards_ai/lab/field_assistance/cloud_runtime.py",
            "src/billiards_ai/lab/field_assistance/managed_resource_recovery.py",
            "src/billiards_ai/lab/field_assistance/runtime_resources.py",
            "src/billiards_ai/edge/route_assistance/runtime_resources.py",
            "src/billiards_ai/lab/mainline1_field_runtime.py",
            "src/billiards_ai/lab/mainline1_product_realtime_runtime.py",
            "src/billiards_ai/lab/mainline1_shot_jobs.py",
            "scripts/algorithm/run_mainline1_shot_worker.py",
            "src/billiards_ai/lab/mainline1_scenario_capture.py",
            "src/billiards_ai/edge/route_assistance/managed_sessions.py",
            "src/billiards_ai/edge/route_assistance/persistence.py",
        ):
            archived = bundle.read(root + relative)
            expected = (PROJECT / relative).read_bytes()
            assert archived == expected
            assert manifest["files"][relative] == {
                "bytes": len(expected),
                "sha256": hashlib.sha256(expected).hexdigest(),
            }
        persistence = bundle.read(root + "src/billiards_ai/edge/route_assistance/persistence.py")
        assert b"_SCHEMA_VERSION = 30" in persistence
        assert not any(".project-archive" in name for name in names)
        assert "$PSScriptRoot" in launcher
        assert PROFILE_ID in launcher
        assert "[switch]$KeepRunning" in launcher
        assert "$JoinQrAsset" in launcher
        assert '"billiardsAI\\field-runtime\\current.json"' in launcher
        assert "current_session_id" in launcher
        assert "session_receipts" in launcher
        assert "initial_session_id" in launcher
        assert "join_qr_sha256" in launcher
        assert "Resolve-JoinQrAsset" in launcher
        assert "Resolve-SyncConfiguration" in launcher
        assert "Stop-PreviousUsbFieldPackets" in launcher
        assert "Invoke-ExistingSessionRetirement" in launcher
        assert '"billiardsAI\\field-runtime\\field-state.sqlite3"' in launcher
        assert "existing-session-retirement.json" in launcher
        assert '"eight-ball-field-result-*"' in launcher
        assert "join_qr_source = if ($ExistingDevelopmentControlOnly -and" in launcher
        assert '"packet_private_asset"' in launcher
        assert "Assert-DevelopmentCloudScope" in launcher
        assert "--runtime-environment" in launcher
        assert "--cloudbase-environment-id" in launcher
        assert "--expected-development-environment-id" in launcher
        assert "--identity-gate-status" in launcher
        assert "--observation-only" in launcher
        assert ".sync_base_url" in launcher
        assert ".device_id" in launcher
        assert ".credential_target" in launcher
        assert '"--sync-base-url", $ResolvedSync.sync_base_url' in launcher
        assert "Complete-FrozenSession" in launcher
        assert "Wait-ForSessionRollover" in launcher
        assert "Invoke-SessionWorker" in launcher
        assert "launch-audit.json" in launcher
        assert "server-exit.json" in launcher
        assert "state-audit.json" in launcher
        assert '"<redacted>"' in launcher
        assert '"--control-token-file", $SecretPath' in launcher
        assert '"--control-token", $ControlToken' not in launcher
        assert "if ($KeepRunning)" in launcher
        assert 'status -eq "ended"' in launcher
        assert 'status -ne "open"' in launcher
        assert "Set-Content -LiteralPath $CurrentPointer" not in launcher
        assert '"/screen"' in launcher
        assert '"/api/table-code.png"' in launcher
        assert "set_participant" not in launcher
        assert "set_breaker" not in launcher
        assert "emergency_" not in launcher


def test_readme_has_self_locating_single_line_command_and_no_placeholder() -> None:
    readme = (RUNNER.parent / "README.txt").read_text(encoding="utf-8")

    assert "完整路径" not in readme
    command_lines = [
        line.strip() for line in readme.splitlines() if line.strip().startswith("$Run=")
    ]
    assert len(command_lines) == 1
    command = command_lines[0]
    assert "billiardsAI-raytheon-eight-ball-field-20260907-v14" in command
    assert "powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run" in command
    assert "-RuntimeEnvironment" not in command
    assert "-CloudbaseEnvironmentId" not in command
    assert "-ObservationOnly" in command
    assert "-ExistingDevelopmentControlOnly" in command
    assert "runtime-attestation" in readme
    assert "session-analysis" in readme
    assert "大屏只复用 settled schematic" in readme
    assert "不得用本轮本地结果补写 PASS" in readme
    for instruction in (
        "30 分钟",
        "2 秒",
        "缺失事实",
        "空闲拔线",
        "杆中拔线",
        "10 杆",
        "-KeepRunning",
        "operator-gate-evidence.json",
        "field-gates.json",
    ):
        assert instruction in readme


def _field_gate_database(root: Path) -> Path:
    state = root / "field-gates.sqlite3"
    with sqlite3.connect(state) as connection:
        connection.executescript(
            """
            CREATE TABLE field_shots (
              session_id TEXT, shot_id TEXT, sequence INTEGER, payload_json TEXT
            );
            CREATE TABLE field_rule_decisions (
              session_id TEXT, shot_id TEXT, source_epoch_id TEXT,
              status TEXT, elapsed_ms REAL,
              facts_json TEXT, decision_json TEXT
            );
            """
        )
        for session_index, session_id in enumerate(("session-a", "session-b")):
            for offset in range(5):
                sequence = session_index * 5 + offset + 1
                shot_id = f"shot-{sequence:04d}"
                connection.execute(
                    "INSERT INTO field_shots VALUES (?,?,?,?)",
                    (
                        session_id,
                        shot_id,
                        sequence,
                        json.dumps(
                            {
                                "settled_ms": sequence * 1000 + 500,
                                "media": {"source_epoch_id": f"epoch-{session_index}"},
                            }
                        ),
                    ),
                )
                abstained = sequence == 1
                connection.execute(
                    "INSERT INTO field_rule_decisions VALUES (?,?,?,?,?,?,?)",
                    (
                        session_id,
                        shot_id,
                        f"epoch-{session_index}",
                        "abstained" if abstained else "applied",
                        1250.0,
                        json.dumps(
                            {"abstention_reasons": (["first_contact_unknown"] if abstained else [])}
                        ),
                        json.dumps(
                            {
                                "reason_code": (
                                    "fact_adapter_abstained" if abstained else "legal_miss"
                                )
                            }
                        ),
                    ),
                )
    return state


def _complete_gate_context(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    qr_hash = "a" * 64
    session_results = []
    for index, session_id in enumerate(("session-a", "session-b"), start=1):
        artifact_root = tmp_path / "postgame" / session_id
        artifact_root.mkdir(parents=True)
        session_results.append(
            {
                "session_id": session_id,
                "report_session_id": session_id,
                "artifact_root": str(artifact_root),
                "job_ids": [f"job-{index}"],
                "artifact_ids": [f"artifact-{index}"],
                "worker_exit_code": 0,
                "worker_timed_out": False,
                "analysis_status": "AVAILABLE",
                "field_sync_status": "complete",
                "field_sync_noncomplete_media_count": 0,
                "session_analysis_pending_count": 0,
                "cloud_analysis_status": "AVAILABLE",
                "publication_verified": True,
                "local_accepted": True,
                "ended_at_utc": ("2026-09-06T23:59:00Z" if index == 1 else "2026-09-07T00:00:00Z"),
                "deadline_at_utc": f"2026-09-07T00:0{index}:00Z",
                "completed_at_utc": f"2026-09-07T00:0{index - 1}:50Z",
                "accepted": True,
            }
        )
    context = {
        "phase": "FINAL",
        "state": {
            "server_pid": 31415,
            "local_only": False,
            "observation_only": False,
            "identity_gate_status": "PASS",
            "join_qr_sha256": qr_hash,
            "session_receipts": [
                {"session_id": "session-a", "join_qr_sha256": qr_hash},
                {"session_id": "session-b", "join_qr_sha256": qr_hash},
            ],
        },
        "launch_audit": {
            "server_pid": 31415,
            "started_at_utc": "2026-09-06T23:30:00Z",
        },
        "server_exit": {
            "exit_code": 0,
            "ready_observed": True,
            "stopped_at_utc": "2026-09-07T00:00:01Z",
        },
        "session_results": session_results,
        "current_join_qr_sha256": qr_hash,
    }
    operator = {
        "schema_version": "1.0",
        "capture_30m": {
            "status": "PASS",
            "observed_at_utc": "2026-09-07T00:00:00Z",
            "continuous_healthy_capture": True,
            "evidence": ["operator capture-health log reviewed"],
            "reason": None,
        },
        "disconnect_recovery": {
            "status": "PASS",
            "observed_at_utc": "2026-09-06T23:50:00Z",
            "idle_disconnect_abstained": True,
            "in_shot_disconnect_abstained": True,
            "reconnect_ready": True,
            "synthesized_shot_end": False,
            "evidence": ["operator photo and runtime log reviewed"],
            "reason": None,
        },
    }
    return context, operator


def test_field_gate_evaluator_accepts_only_complete_two_session_evidence(
    tmp_path: Path,
) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_gate_evaluator")
    state = _field_gate_database(tmp_path)
    context, operator = _complete_gate_context(tmp_path)

    receipt = module.evaluate_field_gates(
        state_db=state,
        context=context,
        operator_evidence=operator,
    )

    expected_gate_ids = {
        "healthy_capture_30m",
        "rule_decision_latency_2s",
        "missing_fact_abstain",
        "disconnect_recovery",
        "settled_shots_10",
        "same_process_two_session_rollover",
        "permanent_qr_hash",
        "session_artifact_isolation",
        "postgame_120s",
        "development_cloud_publication",
    }
    assert {gate["gate_id"] for gate in receipt["gates"]} == expected_gate_ids
    assert {gate["status"] for gate in receipt["gates"]} == {"PASS"}
    assert receipt["all_required_pass"] is True
    assert receipt["accepted"] is True


def test_local_only_keeps_local_postgame_evidence_but_cloud_gate_not_run(
    tmp_path: Path,
) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_gate_local_only")
    state = _field_gate_database(tmp_path)
    context, operator = _complete_gate_context(tmp_path)
    context["state"]["local_only"] = True
    context["state"]["observation_only"] = True
    context["state"]["identity_gate_status"] = "FAIL"
    for result in context["session_results"]:
        result["field_sync_status"] = None
        result["session_analysis_pending_count"] = None
        result["cloud_analysis_status"] = None
        result["publication_verified"] = False
        result["accepted"] = False

    receipt = module.evaluate_field_gates(
        state_db=state,
        context=context,
        operator_evidence=operator,
    )
    by_id = {gate["gate_id"]: gate for gate in receipt["gates"]}

    assert by_id["postgame_120s"]["status"] == "PASS"
    assert by_id["development_cloud_publication"]["status"] == "NOT_RUN"
    assert receipt["all_required_pass"] is False
    assert receipt["accepted"] is False


def test_existing_control_only_never_treats_local_available_as_cloud_available(
    tmp_path: Path,
) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_gate_control_only")
    state = _field_gate_database(tmp_path)
    context, operator = _complete_gate_context(tmp_path)
    context["state"]["existing_development_control_only"] = True
    context["state"]["analysis_sync_enabled"] = False
    context["state"]["observation_only"] = True
    context["state"]["identity_gate_status"] = "FAIL"
    for result in context["session_results"]:
        result["field_sync_status"] = None
        result["session_analysis_pending_count"] = None
        result["cloud_analysis_status"] = None
        result["publication_verified"] = False
        result["accepted"] = False

    receipt = module.evaluate_field_gates(
        state_db=state,
        context=context,
        operator_evidence=operator,
    )
    by_id = {gate["gate_id"]: gate for gate in receipt["gates"]}

    assert by_id["postgame_120s"]["status"] == "PASS"
    assert by_id["development_cloud_publication"]["status"] == "NOT_RUN"
    assert receipt["accepted"] is False


def test_local_only_without_join_qr_records_permanent_qr_gate_not_run(
    tmp_path: Path,
) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_gate_local_no_qr")
    state = _field_gate_database(tmp_path)
    context, operator = _complete_gate_context(tmp_path)
    context["state"]["local_only"] = True
    context["state"]["observation_only"] = True
    context["state"]["identity_gate_status"] = "FAIL"
    context["state"]["join_qr_sha256"] = None
    context["current_join_qr_sha256"] = None
    for session in context["state"]["session_receipts"]:
        session["join_qr_sha256"] = None

    receipt = module.evaluate_field_gates(
        state_db=state,
        context=context,
        operator_evidence=operator,
    )
    gate = next(item for item in receipt["gates"] if item["gate_id"] == "permanent_qr_hash")

    assert gate["status"] == "NOT_RUN"
    assert "LocalOnly" in gate["reason"]
    assert receipt["all_required_pass"] is False
    assert receipt["accepted"] is False


def test_nonlocal_final_without_join_qr_fails_permanent_qr_gate(tmp_path: Path) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_gate_nonlocal_no_qr")
    state = _field_gate_database(tmp_path)
    context, operator = _complete_gate_context(tmp_path)
    context["state"]["join_qr_sha256"] = None
    context["current_join_qr_sha256"] = None
    for session in context["state"]["session_receipts"]:
        session["join_qr_sha256"] = None

    receipt = module.evaluate_field_gates(
        state_db=state,
        context=context,
        operator_evidence=operator,
    )
    gate = next(item for item in receipt["gates"] if item["gate_id"] == "permanent_qr_hash")

    assert gate["status"] == "FAIL"
    assert receipt["accepted"] is False


@pytest.mark.parametrize(
    ("mutation", "expected_gate"),
    [
        ("operator_not_run", "disconnect_recovery"),
        ("capture_operator_not_run", "healthy_capture_30m"),
        ("capture_receipt_outside_run", "healthy_capture_30m"),
        ("short_capture", "healthy_capture_30m"),
        ("slow_rule", "rule_decision_latency_2s"),
        ("missing_rule_receipt", "rule_decision_latency_2s"),
        ("wrong_abstain_reason", "missing_fact_abstain"),
        ("one_session", "same_process_two_session_rollover"),
        ("reused_source_epoch", "same_process_two_session_rollover"),
        ("worker_nonzero", "postgame_120s"),
        ("publication_pending", "development_cloud_publication"),
        ("sync_not_complete", "development_cloud_publication"),
        ("cloud_not_available", "development_cloud_publication"),
        ("extended_postgame", "postgame_120s"),
    ],
)
def test_field_gate_evaluator_never_infers_pass_from_missing_or_failed_evidence(
    tmp_path: Path,
    mutation: str,
    expected_gate: str,
) -> None:
    module = _load(PACKAGER, f"raytheon_eight_ball_field_gate_{mutation}")
    state = _field_gate_database(tmp_path)
    context, operator = _complete_gate_context(tmp_path)
    if mutation == "operator_not_run":
        operator["disconnect_recovery"] = {"status": "NOT_RUN"}
    elif mutation == "capture_operator_not_run":
        operator["capture_30m"] = {"status": "NOT_RUN"}
    elif mutation == "capture_receipt_outside_run":
        operator["capture_30m"]["observed_at_utc"] = "2026-09-06T22:00:00Z"
    elif mutation == "short_capture":
        context["server_exit"]["stopped_at_utc"] = "2026-09-06T23:59:59Z"
    elif mutation == "slow_rule":
        with sqlite3.connect(state) as connection:
            connection.execute(
                "UPDATE field_rule_decisions SET elapsed_ms=2000.001 WHERE shot_id='shot-0002'"
            )
    elif mutation == "missing_rule_receipt":
        with sqlite3.connect(state) as connection:
            connection.execute("DELETE FROM field_rule_decisions WHERE shot_id='shot-0002'")
    elif mutation == "wrong_abstain_reason":
        with sqlite3.connect(state) as connection:
            connection.execute(
                "UPDATE field_rule_decisions SET decision_json=? WHERE shot_id='shot-0001'",
                (json.dumps({"reason_code": "unrelated_abstention"}),),
            )
    elif mutation == "one_session":
        context["state"]["session_receipts"] = [context["state"]["session_receipts"][0]]
        context["session_results"] = [context["session_results"][0]]
    elif mutation == "reused_source_epoch":
        with sqlite3.connect(state) as connection:
            connection.execute(
                "UPDATE field_rule_decisions SET source_epoch_id='epoch-0' "
                "WHERE session_id='session-b'"
            )
    elif mutation == "worker_nonzero":
        context["session_results"][0]["worker_exit_code"] = 2
    elif mutation == "publication_pending":
        context["session_results"][0]["publication_verified"] = False
    elif mutation == "sync_not_complete":
        context["session_results"][0]["field_sync_status"] = "partial"
    elif mutation == "cloud_not_available":
        context["session_results"][0]["cloud_analysis_status"] = "PROCESSING"
    else:
        context["session_results"][0]["ended_at_utc"] = "2026-09-06T23:00:00Z"

    receipt = module.evaluate_field_gates(
        state_db=state,
        context=context,
        operator_evidence=operator,
    )
    by_id = {gate["gate_id"]: gate for gate in receipt["gates"]}

    assert by_id[expected_gate]["status"] in {"FAIL", "NOT_RUN"}
    assert receipt["all_required_pass"] is False
    assert receipt["accepted"] is False


def test_disconnect_tombstone_cannot_also_pass_missing_fact_gate(
    tmp_path: Path,
) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_gate_disconnect_tombstone")
    state = _field_gate_database(tmp_path)
    with sqlite3.connect(state) as connection:
        connection.execute(
            "UPDATE field_rule_decisions SET facts_json=? WHERE shot_id='shot-0001'",
            (json.dumps({"abstention_reasons": ["capture_interrupted"]}),),
        )
    context, operator = _complete_gate_context(tmp_path)

    receipt = module.evaluate_field_gates(
        state_db=state,
        context=context,
        operator_evidence=operator,
    )
    by_id = {gate["gate_id"]: gate for gate in receipt["gates"]}

    assert by_id["disconnect_recovery"]["status"] == "PASS"
    assert by_id["missing_fact_abstain"]["status"] == "NOT_RUN"
    assert receipt["accepted"] is False


def _worker_publication_databases(
    root: Path,
    *,
    sync_status: str = "complete",
    media_status: str = "complete",
    publication_status: str = "published",
    cloud_status: str = "AVAILABLE",
    payload_hash_valid: bool = True,
) -> tuple[Path, Path]:
    state = root / "publication-state.sqlite3"
    jobs = root / "publication-jobs.sqlite3"
    with sqlite3.connect(state) as connection:
        connection.executescript(
            """
            CREATE TABLE field_sync_jobs (
              sync_job_id TEXT, session_id TEXT, status TEXT, last_error_code TEXT
            );
            CREATE TABLE field_sync_media (
              sync_job_id TEXT, clip_id TEXT, status TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO field_sync_jobs VALUES (?,?,?,?)",
            ("sync-1", "session-a", sync_status, None),
        )
        connection.execute(
            "INSERT INTO field_sync_media VALUES (?,?,?)",
            ("sync-1", "clip-1", media_status),
        )
    with sqlite3.connect(jobs) as connection:
        connection.execute(
            """CREATE TABLE session_analysis_publications (
                session_id TEXT, revision INTEGER, payload_sha256 TEXT, payload_json TEXT,
                status TEXT, attempt_count INTEGER, last_error TEXT
            )"""
        )
        payload_json = json.dumps({"session_id": "session-a", "status": cloud_status})
        payload_sha256 = hashlib.sha256(payload_json.encode()).hexdigest()
        if not payload_hash_valid:
            payload_sha256 = "0" * 64
        connection.execute(
            "INSERT INTO session_analysis_publications VALUES (?,?,?,?,?,?,?)",
            (
                "session-a",
                1,
                payload_sha256,
                payload_json,
                publication_status,
                1,
                None,
            ),
        )
    return state, jobs


def test_worker_publication_inspection_requires_complete_sync_and_available_cloud(
    tmp_path: Path,
) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_worker_publication")
    state, jobs = _worker_publication_databases(tmp_path)

    receipt = module.inspect_worker_publication(
        state_db=state,
        jobs_db=jobs,
        session_id="session-a",
    )

    assert receipt["verified"] is True
    assert receipt["field_sync_status"] == "complete"
    assert receipt["session_analysis_pending_count"] == 0
    assert receipt["cloud_analysis_status"] == "AVAILABLE"


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"sync_status": "pending"}, "field sync"),
        ({"sync_status": "blocked"}, "field sync"),
        ({"publication_status": "pending"}, "outbox"),
        ({"cloud_status": "PROCESSING"}, "AVAILABLE"),
        ({"payload_hash_valid": False}, "hash"),
    ],
)
def test_worker_publication_inspection_fails_closed(
    tmp_path: Path,
    overrides: dict[str, object],
    reason_fragment: str,
) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_worker_publication_failed")
    state, jobs = _worker_publication_databases(tmp_path, **overrides)

    receipt = module.inspect_worker_publication(
        state_db=state,
        jobs_db=jobs,
        session_id="session-a",
    )

    assert receipt["verified"] is False
    assert reason_fragment in receipt["reason"]


def test_field_gate_evaluator_keeps_unfinished_ten_shot_gate_not_run(
    tmp_path: Path,
) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_gate_interim")
    state = _field_gate_database(tmp_path)
    with sqlite3.connect(state) as connection:
        connection.execute("DELETE FROM field_shots WHERE shot_id='shot-0010'")
    context, operator = _complete_gate_context(tmp_path)
    context["phase"] = "INTERIM"

    receipt = module.evaluate_field_gates(
        state_db=state,
        context=context,
        operator_evidence=operator,
    )
    by_id = {gate["gate_id"]: gate for gate in receipt["gates"]}

    assert by_id["settled_shots_10"]["status"] == "NOT_RUN"
    assert receipt["accepted"] is False


@pytest.mark.parametrize(
    ("observation_only", "identity_gate_status"),
    [(True, "PASS"), (False, "FAIL")],
)
def test_field_gate_evaluator_requires_rule_authority_for_acceptance(
    tmp_path: Path,
    observation_only: bool,
    identity_gate_status: str,
) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_gate_authority")
    state = _field_gate_database(tmp_path)
    context, operator = _complete_gate_context(tmp_path)
    context["state"]["observation_only"] = observation_only
    context["state"]["identity_gate_status"] = identity_gate_status

    receipt = module.evaluate_field_gates(
        state_db=state,
        context=context,
        operator_evidence=operator,
    )

    assert receipt["all_required_pass"] is True
    assert receipt["rule_authority_accepted"] is False
    assert receipt["accepted"] is False


def test_packet_without_qr_defers_to_existing_field_runtime(tmp_path: Path) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_packager_runtime_qr")
    archive = module.build_packet(
        PROJECT,
        tmp_path / "out",
        candidate_root=_candidate(tmp_path, include_qr=False),
    )

    with zipfile.ZipFile(archive) as bundle:
        manifest_name = next(
            name for name in bundle.namelist() if name.endswith("packet-manifest.json")
        )
        manifest = json.loads(bundle.read(manifest_name))

        assert manifest["run"]["join_qr_asset"] is None
        assert manifest["run"]["qr_source"] == (
            "packet_if_present_else_current_json_in_existing_development_control_only"
        )


def test_packet_can_reuse_candidate_from_v13_archive(tmp_path: Path) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_packager_archive")
    candidate = _candidate(tmp_path)
    source_archive = tmp_path / "v13.zip"
    with zipfile.ZipFile(source_archive, "w") as bundle:
        for path in candidate.rglob("*"):
            if path.is_file():
                bundle.write(path, f"v13-root/candidate/{path.relative_to(candidate).as_posix()}")

    archive = module.build_packet(
        PROJECT,
        tmp_path / "out",
        candidate_archive=source_archive,
    )

    with zipfile.ZipFile(archive) as bundle:
        capture = next(
            name for name in bundle.namelist() if name.endswith("candidate/capture-setup.json")
        )
        assert json.loads(bundle.read(capture))["fixture"] is True


def test_candidate_archive_rejects_ambiguous_or_unsafe_inventory(tmp_path: Path) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_packager_unsafe")
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../candidate/capture-setup.json", "{}")
        bundle.writestr("root/candidate/calibrations/registry.json", "{}")

    with pytest.raises(ValueError, match="unsafe|missing"):
        module.build_packet(PROJECT, tmp_path / "out", candidate_archive=archive)


def test_packet_rejects_candidate_without_real_calibration(tmp_path: Path) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_packager_calibration")
    candidate = _candidate(tmp_path)
    (candidate / "calibrations/active/calibration.json").unlink()

    with pytest.raises(ValueError, match="calibration"):
        module.build_packet(PROJECT, tmp_path / "out", candidate_root=candidate)


def test_packet_rejects_join_qr_with_false_image_suffix(tmp_path: Path) -> None:
    module = _load(PACKAGER, "raytheon_eight_ball_field_packager_qr")
    candidate = _candidate(tmp_path)
    (candidate / "join-qr.png").write_bytes(b"not an image")

    with pytest.raises(ValueError, match="join QR"):
        module.build_packet(PROJECT, tmp_path / "out", candidate_root=candidate)


def _state_databases(root: Path, *, finalized: bool, job_status: str) -> tuple[Path, Path]:
    state = root / "field-state.sqlite3"
    jobs = root / "analysis-jobs.sqlite3"
    with sqlite3.connect(state) as connection:
        connection.executescript(
            """
            CREATE TABLE field_sessions (
              session_id TEXT PRIMARY KEY, artifact_id TEXT, status TEXT,
              created_at_utc TEXT, ended_at_utc TEXT
            );
            CREATE TABLE field_match_state (
              session_id TEXT PRIMARY KEY, payload_json TEXT, revision INTEGER
            );
            CREATE TABLE field_shots (
              session_id TEXT, shot_id TEXT, sequence INTEGER, payload_json TEXT
            );
            CREATE TABLE field_postgame_manifests (
              session_id TEXT PRIMARY KEY, ended_at_utc TEXT,
              deadline_at_utc TEXT, expected_jobs_json TEXT
            );
            CREATE TABLE field_postgame_finalizations (
              session_id TEXT PRIMARY KEY, manifest_sha256 TEXT, manifest_json TEXT
            );
            CREATE TABLE local_postgame_outbox (
              session_id TEXT PRIMARY KEY, manifest_sha256 TEXT,
              manifest_json TEXT, status TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO field_sessions VALUES (?,?,?,?,?)",
            ("session-1", "artifact-1", "ended", "2026-09-06T00:00:00Z", "2026-09-06T00:01:00Z"),
        )
        connection.execute(
            "INSERT INTO field_match_state VALUES (?,?,?)",
            ("session-1", json.dumps({"rule_profile_id": PROFILE_ID}), 7),
        )
        connection.execute(
            "INSERT INTO field_postgame_manifests VALUES (?,?,?,?)",
            (
                "session-1",
                "2026-09-06T00:01:00Z",
                "2099-09-06T00:03:00Z",
                json.dumps(["job-1"]),
            ),
        )
        if finalized:
            connection.execute(
                "INSERT INTO field_postgame_finalizations VALUES (?,?,?)",
                ("session-1", "a" * 64, '{"artifact_ids":["artifact-1"]}'),
            )
    with sqlite3.connect(jobs) as connection:
        connection.execute(
            """CREATE TABLE shot_analysis_jobs (
                job_id TEXT PRIMARY KEY, payload_json TEXT, payload_sha256 TEXT,
                status TEXT, claimed_by TEXT, claimed_at_utc TEXT,
                artifact_id TEXT, failure_reason TEXT, attempt_count INTEGER,
                lease_token TEXT, next_attempt_at_utc TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO shot_analysis_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "job-1",
                '{"session_id":"session-1"}',
                "b" * 64,
                job_status,
                None,
                None,
                "artifact-1" if job_status == "available" else None,
                None,
                1,
                None,
                None,
            ),
        )
    return state, jobs


def test_evidence_exporter_requires_jobs_and_finalization_for_success(tmp_path: Path) -> None:
    module = _load(EXPORTER, "raytheon_eight_ball_field_exporter")
    state, jobs = _state_databases(tmp_path, finalized=True, job_status="available")
    before = (state.read_bytes(), jobs.read_bytes())

    report = module.export_evidence(state, jobs, "session-1")

    assert report["analysis"] == {
        "status": "AVAILABLE",
        "expected_count": 1,
        "available_count": 1,
        "finalized": True,
    }
    assert report["match"]["rule_profile_id"] == PROFILE_ID
    assert (state.read_bytes(), jobs.read_bytes()) == before


def test_evidence_exporter_does_not_treat_worker_exit_zero_as_success(tmp_path: Path) -> None:
    module = _load(EXPORTER, "raytheon_eight_ball_field_exporter_pending")
    state, jobs = _state_databases(tmp_path, finalized=False, job_status="available")

    report = module.export_evidence(state, jobs, "session-1")

    assert report["analysis"]["status"] == "PARTIAL"
    assert report["analysis"]["finalized"] is False
