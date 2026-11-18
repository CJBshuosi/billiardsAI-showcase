from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WINDOWS = ROOT / "scripts" / "algorithm" / "windows"


def _read(name: str) -> str:
    return (WINDOWS / name).read_text(encoding="utf-8")


def test_installer_verifies_release_and_keeps_mutable_data_outside_release() -> None:
    script = _read("install-field-route-service.ps1")

    assert "Get-FileHash" in script and "SHA256" in script
    assert "release-manifest.json" in script
    assert "field-releases" in script
    assert "field-runtime" in script
    assert "current.json" in script
    assert "billiardsAI-field-service" in script
    assert "billiardsAI-field-kiosk" in script
    assert "New-ScheduledTaskSettingsSet" in script
    assert "RestartCount" in script and "RestartInterval" in script
    assert "CheckOnly" in script
    assert "CredentialSecret" not in script
    assert "Test-Path -LiteralPath $Artifact -PathType Container" in script


def test_installer_tasks_bypass_machine_execution_policy() -> None:
    script = _read("install-field-route-service.ps1")

    service_parts = re.search(r"\$ServiceParts = @\((.*?)\)\n", script)
    kiosk_parts = re.search(r"\$KioskParts = @\((.*?)\)\n", script)

    assert service_parts is not None
    assert kiosk_parts is not None
    assert '"-ExecutionPolicy", "Bypass"' in service_parts.group(1)
    assert '"-ExecutionPolicy", "Bypass"' in kiosk_parts.group(1)
    assert '"-RuntimeRoot", (ConvertTo-TaskArgument $ResolvedRuntimeRoot)' in service_parts.group(1)
    assert '"-Artifact"' not in service_parts.group(1)


def test_usb_task_replaces_stale_definition_and_pins_the_exact_capture_profile() -> None:
    """Catches an old replay task surviving a USB release switch."""
    installer = _read("install-field-route-service.ps1")
    launcher = _read("serve-field-route-shell.ps1")

    register = installer[
        installer.index("function Register-FieldTasks") : installer.index(
            "function Wait-FieldTaskStopped"
        )
    ]
    unregister = register.index("Unregister-ScheduledTask")
    service_register = register.index("Register-ScheduledTask -TaskName $ServiceTaskName")
    assert unregister < service_register
    assert '"-WindowStyle", "Hidden"' in register
    assert '"-CaptureProfilePath"' in register
    assert "$Configuration.capture_profile_path" in register
    assert '[string]$CaptureProfilePath = ""' in launcher
    assert "Get-Content -LiteralPath $ResolvedCaptureProfilePath" in launcher


def test_usb_install_health_gate_rejects_a_replay_listener_and_starts_once() -> None:
    """Catches a stale replay service on 8787 being accepted as the USB service."""
    script = _read("install-field-route-service.ps1")
    health = script[
        script.index("function Wait-FieldServiceHealthy") : script.index(
            "function Assert-FieldReplayArtifact"
        )
    ]

    assert "[switch]$RequireUsbCapture" in health
    assert 'frame.source_kind -eq "usb_live"' in health
    assert "$null -ne $CurrentMediaFrameId" in health
    assert "[long]$CurrentMediaFrameId -gt [long]$FirstMediaFrameId" in health
    assert '$HealthPayload.frame.phase -notin @("degraded", "recovering")' in health
    assert '$HealthPayload.frame.rule_status -ne "DEGRADED"' in health
    assert "[string]::IsNullOrWhiteSpace($HealthPayload.frame.degraded_reason)" in health
    assert "$StartAttempted" in health
    assert "if (-not $StartAttempted)" in health
    assert "Set-Content -LiteralPath $ServiceErrorLog" in health


def test_usb_install_health_gate_accepts_pre_match_without_a_recorder() -> None:
    """Catches reading capture phase instead of the player-join display phase."""
    script = _read("install-field-route-service.ps1")
    health = script[
        script.index("function Wait-FieldServiceHealthy") : script.index(
            "function Assert-FieldReplayArtifact"
        )
    ]

    assert '$HealthPayload.display.phase -eq "WAITING_FOR_PLAYERS"' in health
    assert '$HealthPayload.frame.phase -eq "waiting_for_players"' not in health
    assert '$RecordingStatus -eq "unavailable"' in health


def test_usb_install_allows_bounded_model_cold_start_and_fails_fast_on_exit() -> None:
    """A live GPU cold start must not be confused with a dead scheduled task."""
    script = _read("install-field-route-service.ps1")
    health = script[
        script.index("function Wait-FieldServiceHealthy") : script.index(
            "function Assert-FieldReplayArtifact"
        )
    ]

    assert "[ValidateRange(60, 900)][int]$StartupTimeoutSeconds = 300" in health
    assert "AddSeconds($StartupTimeoutSeconds)" in health
    assert "$ScheduledTaskExited" in health
    assert "if ($ScheduledTaskExited) { break }" in health


def test_usb_activation_reports_the_current_python_startup_stage() -> None:
    script = _read("activate-mainline1-usb-capture.ps1")
    health_progress = script[
        script.index("function Write-ServiceHealthProgress") : script.index(
            "function ConvertTo-ProcessArgument"
        )
    ]

    assert "function Get-ServiceStartupStage" in script
    assert "field-service-error.log" in script
    assert "startup_stage" in health_progress
    assert "Get-ServiceStartupStage" in health_progress


def test_installer_stops_running_field_tasks_before_switching_release() -> None:
    script = _read("install-field-route-service.ps1")

    assert "function Stop-FieldTasks" in script
    assert "Stop-ScheduledTask" in script
    switch = script.index("$Configuration | ConvertTo-Json -Depth 5 | Set-Content")
    stop = script.rfind("Stop-FieldTasks", 0, switch)
    register = script.index("Register-FieldTasks -Configuration", switch)
    assert 0 <= stop < switch < register


def test_release_upgrade_restores_previous_configuration_when_health_check_fails() -> None:
    script = _read("install-field-route-service.ps1")
    install = script[script.index("New-Item -ItemType Directory -Path $ResolvedRuntimeRoot") :]

    assert "$OriginalJson" in install
    switch = install.index("Write-CurrentConfigurationAtomically -Configuration")
    health = install.index("Wait-FieldServiceHealthy", switch)
    recovery = install.index("Restore-FieldReleaseConfiguration", health)
    assert switch < health < recovery


def test_installer_waits_for_scheduler_and_port_to_finish_stopping() -> None:
    script = _read("install-field-route-service.ps1")

    assert "$ScheduledTaskRunningResult = 267009" in script
    assert "function Wait-FieldTaskStopped" in script
    stopped = script[
        script.index("function Wait-FieldTaskStopped") : script.index(
            "function Stop-FieldServiceProcesses"
        )
    ]
    assert "Get-ScheduledTaskInfo" in stopped
    assert "$TaskInfo.LastTaskResult -ne $ScheduledTaskRunningResult" in stopped
    assert "Scheduled task instance did not finish stopping" in stopped

    stop_tasks = script[
        script.index("function Stop-FieldTasks") : script.index(
            "function Write-CurrentConfigurationAtomically"
        )
    ]
    assert "Wait-FieldTaskStopped -TaskName $TaskName" in stop_tasks
    assert "Get-NetTCPConnection -LocalPort $Port -State Listen" in stop_tasks
    assert "Field service port did not become available" in stop_tasks


def test_health_wait_retries_an_ignored_scheduled_task_start() -> None:
    script = _read("install-field-route-service.ps1")
    health = script[
        script.index("function Wait-FieldServiceHealthy") : script.index(
            "function Assert-FieldReplayArtifact"
        )
    ]

    assert "Get-ScheduledTask -TaskName $ServiceTaskName" in health
    assert '$Task.State -ne "Running"' in health
    assert "Start-ScheduledTask -TaskName $ServiceTaskName" in health
    assert "Get-ScheduledTaskInfo -TaskName $ServiceTaskName" in health
    assert "last_task_result=" in health
    assert "listener_pids=" in health


def test_installer_terminates_only_the_residual_field_service_process() -> None:
    script = _read("install-field-route-service.ps1")

    cleanup = script[
        script.index("function Stop-FieldServiceProcesses") : script.index(
            "function Stop-FieldTasks"
        )
    ]
    assert "Get-CimInstance Win32_Process" in cleanup
    assert 'IndexOf("serve_field_route_shell.py"' in cleanup
    assert "$ExpectedStateDb" in cleanup
    assert "Invoke-CimMethod -InputObject $Process -MethodName Terminate" in cleanup
    assert "Residual field service process did not terminate" in cleanup
    assert script.index("Stop-FieldServiceProcesses", script.index("function Stop-FieldTasks"))


def test_installer_can_atomically_switch_only_a_verified_replay_artifact() -> None:
    script = _read("install-field-route-service.ps1")

    assert "[switch]$SwitchArtifact" in script
    assert "[int]$ExpectedShotCount" in script
    assert "function Assert-FieldReplayArtifact" in script
    assert 'kind -ne "field_route_file_replay"' in script
    assert 'pipeline_version -ne "field-route-file-replay/2"' in script
    assert "Get-FileHash" in script and "inventory" in script
    assert "shot_count -ne $RequiredShotCount" in script
    assert "Write-CurrentConfigurationAtomically" in script
    assert "Restore-FieldConfiguration" in script

    switch = script[script.index("if ($SwitchArtifact)") : script.index("if (-not $ReleaseArchive")]
    verify = switch.index("Assert-FieldReplayArtifact -ArtifactRoot")
    stop = switch.index("Stop-FieldTasks", verify)
    write = switch.index("Write-CurrentConfigurationAtomically", stop)
    start_and_health = switch.index("Wait-FieldServiceHealthy", write)
    assert verify < stop < write < start_and_health


def test_artifact_switch_restarts_the_existing_dynamic_task_without_reregistering() -> None:
    script = _read("install-field-route-service.ps1")
    switch = script[script.index("if ($SwitchArtifact)") : script.index("if (-not $ReleaseArchive")]
    restore = script[
        script.index("function Restore-FieldConfiguration") : script.index("if ($Rollback)")
    ]

    assert "Register-FieldTasks" not in switch
    assert "Register-FieldTasks" not in restore
    assert "Wait-FieldServiceHealthy" in switch
    assert "Wait-FieldServiceHealthy" in restore


def test_service_launcher_loads_the_current_artifact_and_records_stderr() -> None:
    script = _read("serve-field-route-shell.ps1")

    assert '[string]$RuntimeRoot = ""' in script
    assert 'Join-Path $ResolvedRuntimeRoot "current.json"' in script
    assert "$RuntimeConfiguration.artifact" in script
    assert 'Join-Path $StateDbParent "field-service-error.log"' in script
    assert "*>&1 | Tee-Object -FilePath $ErrorLog" in script
    assert "$RunnerExitCode = $LASTEXITCODE" in script
    assert "exit $RunnerExitCode" in script

    installer = _read("install-field-route-service.ps1")
    assert 'Join-Path $ResolvedRuntimeRoot "field-service-error.log"' in installer
    assert "Get-Content -LiteralPath $ServiceErrorLog -Tail 20" in installer


def test_service_launcher_handles_an_empty_optional_sync_configuration() -> None:
    script = _read("serve-field-route-shell.ps1")

    assert "$SyncValues = @(@(" in script


def test_installed_service_launcher_selects_native_usb_from_current_configuration() -> None:
    script = _read("serve-field-route-shell.ps1")

    assert "$RuntimeConfiguration.capture_profile" in script
    assert '"mainline1_usb_native"' in script
    assert (
        'Join-Path $ResolvedProjectRoot "scripts\\algorithm\\serve_mainline1_usb_field.py"'
        in script
    )
    assert '"--state-db", $ResolvedStateDb' in script
    assert '"--join-qr-asset", $ResolvedJoinQrAsset' in script
    assert '"--native-helper", $ResolvedNativeHelper' in script
    assert '"--device-locator", [string]$CaptureProfile.device_locator' in script
    assert '"--device-fingerprint", [string]$CaptureProfile.device_fingerprint' in script
    assert '"--capture-setup", $ResolvedCaptureSetup' in script
    assert '"--calibration-root", $ResolvedCalibrationRoot' in script
    assert '"--detector-training-run", $ResolvedDetectorTrainingRun' in script
    assert '"--identity-training-run", $ResolvedIdentityTrainingRun' in script
    assert '"--dinov2-assets", $ResolvedDinov2Assets' in script
    assert '"--ring-root", $ResolvedRingRoot' in script
    assert '"--jobs-root", $ResolvedJobsRoot' in script
    assert '"--artifact-root", $ResolvedLiveArtifactRoot' in script
    assert '"--port", [string]$Port' in script
    assert '"--rule-profile", "cbsa-chinese-pool-2017-observable-v1"' in script
    assert '[string]$CaptureProfile.identity_gate_status -ne "PASS"' in script
    assert '$RunnerArguments += "--observation-only"' in script
    assert '"--existing-development-control-only"' not in script
    assert "8765" not in script


def test_installed_service_launcher_rejects_unknown_capture_mode() -> None:
    script = _read("serve-field-route-shell.ps1")

    assert 'throw "Unsupported installed capture profile mode:' in script


def test_installer_persists_capture_profile_without_replacing_business_configuration() -> None:
    script = _read("install-field-route-service.ps1")

    assert '[string]$CaptureProfile = ""' in script
    assert "function Read-CaptureProfile" in script
    assert 'mode -ne "mainline1_usb_native"' in script
    assert "native_helper" in script and "device_fingerprint" in script
    assert "$Configuration.capture_profile = $SelectedCaptureProfile" in script
    assert "$ExistingConfiguration.join_qr_asset" in script
    assert "$ExistingConfiguration.sync_base_url" in script
    assert "$ExistingConfiguration.device_id" in script
    assert "$ExistingConfiguration.credential_target" in script
    assert "$ExistingConfiguration.browser_path" in script
    assert "$ExistingConfiguration.python_path" in script
    assert "$ExistingConfiguration.artifact" in script


def test_capture_upgrade_backs_up_state_and_restores_it_before_old_service_restart() -> None:
    script = _read("install-field-route-service.ps1")

    assert '$StateDbPath = Join-Path $ResolvedRuntimeRoot "field-state.sqlite3"' in script
    assert "$StateDbBackup" in script
    assert "Copy-Item -LiteralPath $StatePath -Destination $PendingStateDbBackup" in script
    restore = script[
        script.index("function Restore-FieldReleaseConfiguration") : script.index(
            "if ($SwitchArtifact)"
        )
    ]
    assert "Stop-FieldTasks" in restore
    assert "Copy-Item -LiteralPath $BackupPath -Destination $StatePath -Force" in restore
    assert restore.index("Copy-Item -LiteralPath $BackupPath") < restore.index(
        "Wait-FieldServiceHealthy"
    )


def test_release_upgrade_archives_state_bundle_and_starts_with_clean_database() -> None:
    script = _read("install-field-route-service.ps1")
    configuration = script.index("$Configuration = [ordered]@")
    activation = script[script.index("  try {\n    Stop-FieldTasks", configuration) :]

    assert "$StateDbCompanionPaths" in script
    assert '"field-state.sqlite3-wal"' in script
    assert '"field-state.sqlite3-shm"' in script
    assert "Copy-Item -LiteralPath $StatePath -Destination $PendingStateDbBackup" in activation
    assert "Remove-Item -LiteralPath $StatePath -Force" in activation
    assert activation.index("Copy-Item -LiteralPath $StatePath") < activation.index(
        "Remove-Item -LiteralPath $StatePath"
    )
    assert activation.index("Remove-Item -LiteralPath $StatePath") < activation.index(
        "Write-CurrentConfigurationAtomically"
    )
    assert activation.index("$StateDbBackup = $PendingStateDbBackup") > activation.index(
        "Field state archive verification failed"
    )
    assert activation.index("$StateDbBackup = $PendingStateDbBackup") < activation.index(
        "Remove-Item -LiteralPath $StatePath"
    )
    restore = script[
        script.index("function Restore-FieldReleaseConfiguration") : script.index(
            "if ($SwitchArtifact)"
        )
    ]
    assert restore.index("Remove-Item -LiteralPath $StatePath") < restore.index(
        "Copy-Item -LiteralPath $BackupPath"
    )


def test_release_upgrade_retires_cloud_session_before_archiving_local_state() -> None:
    """A clean local DB must not leave the permanent table QR on the old cloud match."""
    script = _read("install-field-route-service.ps1")
    activation = script[script.index("  try {\n    Stop-FieldTasks") :]

    assert "function Retire-ExistingFieldSession" in script
    assert "scripts\\algorithm\\retire_existing_field_session.py" in script
    assert '"--sync-base-url", $SyncBaseUrl' in script
    assert '"--device-id", $DeviceId' in script
    assert '"--credential-target", $CredentialTarget' in script
    retire = activation.index("Retire-ExistingFieldSession")
    archive = activation.index("Copy-Item -LiteralPath $StatePath")
    remove = activation.index("Remove-Item -LiteralPath $StatePath")
    assert retire < archive < remove


def test_installer_stops_replay_or_usb_process_for_the_same_installed_state_db() -> None:
    script = _read("install-field-route-service.ps1")
    cleanup = script[
        script.index("function Stop-FieldServiceProcesses") : script.index(
            "function Stop-FieldTasks"
        )
    ]

    assert 'IndexOf("serve_field_route_shell.py"' in cleanup
    assert 'IndexOf("serve_mainline1_usb_field.py"' in cleanup
    assert "$ExpectedStateDb" in cleanup


def test_usb_activation_reuses_the_installed_service_and_business_configuration() -> None:
    script = _read("activate-mainline1-usb-capture.ps1")

    assert '"billiardsAI\\field-runtime"' in script
    assert '"current.json"' in script
    assert '"install-field-route-service.ps1"' in script
    assert '"-CaptureProfile", $CaptureProfilePath' in script
    assert '"-RuntimeRoot", $ResolvedRuntimeRoot' in script
    assert "billiardsAI-field-service" in script
    assert "billiardsAI-field-kiosk" in script
    assert "8787" in script
    assert "8765" not in script
    assert "ExpectedJoinQrSha256" in script
    assert "JoinQrSha256After" in script
    assert '"-JoinQrAsset", $InstalledJoinQr' in script
    assert "join_qr_launch_gate_status" in script
    assert "Manifest-pinned table QR was not activated" in script
    assert '"-SyncBaseUrl"' not in script
    assert '"-DeviceId"' not in script
    assert '"-CredentialTarget"' not in script


def test_usb_activation_never_claims_a_rollback_that_the_inner_installer_rejected() -> None:
    """Catches the outer wrapper reporting a restored service after rollback failed."""
    script = _read("activate-mainline1-usb-capture.ps1")

    assert "failed and the previous service was restored" not in script
    assert "see the installer error above for rollback status" in script


def test_usb_activation_leaves_task_replacement_to_transactional_installer() -> None:
    """Catches deleting installed tasks before rollback state has been captured."""
    script = _read("activate-mainline1-usb-capture.ps1")

    preflight_call = (
        "$PreflightExitCode = Invoke-InstallerProcess -Arguments "
        '@($InstallArguments + "-CheckOnly")'
    )
    preflight = script.index(preflight_call)
    install = script.index(
        "$InstallExitCode = Invoke-InstallerProcess -Arguments $InstallArguments",
        preflight,
    )
    assert preflight < install
    assert "Remove-InstalledFieldTaskDefinitions\n" not in script
    installer = _read("install-field-route-service.ps1")
    assert "Export-FieldTaskDefinitions" in installer
    assert "Restore-FieldTaskDefinitions" in installer


def test_usb_activation_does_not_pipe_a_native_installer_process() -> None:
    """Catches an inherited stdout handle making the outer installer look frozen."""
    script = _read("activate-mainline1-usb-capture.ps1")

    assert "function Invoke-InstallerProcess" in script
    assert 'Start-Process -FilePath "powershell.exe"' in script
    assert "-RedirectStandardOutput $Stdout" in script
    assert "-RedirectStandardError $Stderr" in script
    assert "$Process.WaitForExit(1000)" in script
    assert 'Message "Still running: $Description"' in script
    assert "& powershell.exe @InstallArguments" not in script


def test_usb_activation_can_recover_when_an_interrupted_upgrade_removed_the_tasks() -> None:
    """Catches a half-installed V4 making the recovery packet impossible to start."""
    script = _read("activate-mainline1-usb-capture.ps1")

    assert "The original installed task is missing" not in script
    assert "if (-not (Test-Path -LiteralPath $CurrentPointer -PathType Leaf))" in script


def test_usb_activation_reuses_only_the_hash_pinned_approved_native_helper() -> None:
    script = _read("activate-mainline1-usb-capture.ps1")

    assert "function Resolve-ApprovedNativeCaptureHelper" in script
    assert "$CurrentConfiguration.capture_profile.native_helper" in script
    assert "approved_native_helper_sha256" in script
    assert "Assert-FileHash -Path $Candidate -Expected $ExpectedSha256" in script
    assert "Compile-NativeCaptureHelper" not in script
    assert "csc.exe" not in script
    assert "mainline1_uvc_capture_csharp\\main.cs" not in script


def test_usb_activation_writes_the_installed_capture_profile() -> None:
    script = _read("activate-mainline1-usb-capture.ps1")

    assert 'mode = "mainline1_usb_native"' in script
    assert "native_helper = $NativeHelper" in script
    assert "device_locator = [string]$Manifest.capture.device_locator" in script
    assert "device_fingerprint = [string]$Manifest.capture.device_fingerprint" in script
    assert "capture_setup = $CaptureSetup" in script
    assert "calibration_root = $CalibrationRoot" in script
    assert "detector_training_run = $DetectorRoot" in script
    assert "identity_training_run = $IdentityRoot" in script
    assert "dinov2_assets = $Dinov2Root" in script


def test_usb_activation_stops_only_the_obsolete_sidecar_before_using_the_camera() -> None:
    script = _read("activate-mainline1-usb-capture.ps1")

    assert "function Stop-PreviousSidecarProcesses" in script
    cleanup = script[
        script.index("function Stop-PreviousSidecarProcesses") : script.index(
            "if (-not (Test-Path -LiteralPath $CurrentPointer"
        )
    ]
    assert 'IndexOf("serve_mainline1_usb_field.py"' in cleanup
    assert 'IndexOf("billiardsAI-raytheon-eight-ball-field-"' in cleanup
    assert "Invoke-CimMethod" in cleanup
    assert "Stop-PreviousSidecarProcesses" in script[script.index("$Manifest =") :]


def test_usb_activation_requires_admin_before_creating_any_capture_files() -> None:
    script = _read("activate-mainline1-usb-capture.ps1")

    assert "#requires -RunAsAdministrator" in script
    assert script.index("#requires -RunAsAdministrator") < script.index(
        "New-Item -ItemType Directory -Path $CaptureRoot"
    )


def test_usb_activation_replaces_only_an_inactive_partial_capture_release() -> None:
    script = _read("activate-mainline1-usb-capture.ps1")

    assert "function Remove-InactiveCaptureRelease" in script
    cleanup = script[
        script.index("function Remove-InactiveCaptureRelease") : script.index(
            "if (-not (Test-Path -LiteralPath $CurrentPointer"
        )
    ]
    assert "$CurrentConfiguration.capture_profile.capture_setup" in cleanup
    assert "Split-Path -Parent $ActiveCaptureSetup" in cleanup
    assert "Capture release is already active" in cleanup
    assert "Remove-Item -LiteralPath $CaptureRoot -Recurse -Force" in cleanup
    assert "Remove-InactiveCaptureRelease -CaptureRoot $CaptureRoot" in script


def test_atomic_pointer_replace_uses_a_real_windows_backup_path() -> None:
    script = _read("install-field-route-service.ps1")
    atomic_write = script[
        script.index("function Write-CurrentConfigurationAtomically") : script.index(
            "function Wait-FieldServiceHealthy"
        )
    ]

    assert "$BackupPointer = Join-Path $ResolvedRuntimeRoot" in atomic_write
    assert (
        "[System.IO.File]::Replace($TemporaryPointer, $CurrentPointer, "
        "$BackupPointer, $true)" in atomic_write
    )
    assert "Replace($TemporaryPointer, $CurrentPointer, $null" not in atomic_write
    assert "Remove-Item -LiteralPath $BackupPointer -Force" in atomic_write


def test_video43_refresh_generates_validates_and_switches_the_new_replay() -> None:
    script = _read("refresh-video43-field-replay.ps1")

    assert "current.json" in script
    assert "mainline1-shot-local-video-runs" in script
    assert "field-route-replay-runs" in script
    assert "run_field_route_replay.py" in script
    assert "--mainline1-artifact" in script
    assert "--context-group" in script
    assert "video_sha256" in script and "Get-FileHash" in script
    assert 'pipeline_version -ne "field-route-file-replay/2"' in script
    assert "shot_count -ne $ExpectedShotCount" in script
    assert '"-SwitchArtifact"' in script
    assert '"-ExpectedShotCount"' in script
    assert 'status = "PASS"' in script


def test_video43_refresh_grants_the_interactive_user_access_before_switching() -> None:
    script = _read("refresh-video43-field-replay.ps1")

    grant = script.index("billiards_ai.deployment.windows_acl")
    switch = script.index("$InstallerArguments")
    assert "$env:USERDOMAIN" in script and "$env:USERNAME" in script
    assert '"--artifact" $ArtifactRoot' in script
    assert '"--identity" $TaskIdentity' in script
    assert "artifact permission grant failed" in script
    assert grant < switch


def test_bundle_bootstrap_validates_receipt_assets_and_credentials_before_install() -> None:
    script = _read("install-field-deployment.ps1")

    assert "deployment-receipt.json" in script
    assert 'schema_version -ne "1.0"' in script
    assert "Assert-UnderBundleRoot" in script
    assert "Get-FileHash" in script and "SHA256" in script
    assert 'kind -ne "field_route_file_replay"' in script
    assert 'mode -ne "development_replay"' in script
    assert "cmdkey.exe" in script and "/list:" in script
    assert "CredentialSecret" not in script
    assert "install-field-route-service.ps1" in script
    assert "-CheckOnly" in script
    assert "ProgramFiles" in script and "msedge.exe" in script
    assert ".venv-rfdetr\\Scripts\\python.exe" in script
    assert "http://" not in script


def test_kiosk_is_fixed_to_localhost_public_screen_and_recovers_browser() -> None:
    script = _read("start-field-route-kiosk.ps1")

    assert "http://127.0.0.1:8787/screen" in script
    assert "--kiosk" in script
    assert "kiosk-profile" in script
    assert "Start-Process" in script
    assert "/tablet" not in script


def test_kiosk_terminates_a_residual_browser_for_its_private_profile_first() -> None:
    script = _read("start-field-route-kiosk.ps1")

    cleanup = script.index("Get-CimInstance Win32_Process")
    launch = script.index("Start-Process")
    assert 'CommandLine -like "*--user-data-dir=$ProfileRoot*"' in script
    assert "Invoke-CimMethod -InputObject $Process -MethodName Terminate" in script
    assert cleanup < launch


def test_uninstall_only_targets_named_tasks_and_preserves_runtime() -> None:
    script = _read("uninstall-field-route-service.ps1")

    assert '"billiardsAI-field-service"' in script
    assert '"billiardsAI-field-kiosk"' in script
    assert "Unregister-ScheduledTask" in script
    assert "field-runtime" in script
    assert "Remove-Item" not in script


def test_runbook_powershell_fences_are_each_one_physical_line() -> None:
    runbook = (ROOT / "docs/operations/display-phone-field-deployment.md").read_text(
        encoding="utf-8"
    )
    blocks = re.findall(r"```powershell\n(.*?)\n```", runbook, flags=re.DOTALL)

    assert len(blocks) >= 6
    assert all(len(block.splitlines()) == 1 for block in blocks)
    assert "1080p" in runbook
    assert "100%" in runbook
    assert "PASS — file-replay display-and-phone development loop" in runbook


def test_runbook_starts_from_one_mac_preparation_and_one_bundle_bootstrap() -> None:
    runbook = (ROOT / "docs/operations/display-phone-field-deployment.md").read_text(
        encoding="utf-8"
    )

    assert "prepare_display_phone_deployment.py" in runbook
    assert "install-field-deployment.ps1" in runbook
    assert "私下传输完整目录" in runbook
    assert "-ExpectedSha256" not in runbook
    assert "MINIPROGRAM_WXACODE_ENV_VERSION" not in runbook
    assert "雷神安装微信开发者工具" not in runbook


def test_two_user_acceptance_requires_two_sessions_with_one_permanent_qr() -> None:
    runbook = (ROOT / "docs/operations/display-phone-two-user-acceptance.md").read_text(
        encoding="utf-8"
    )
    powershell_blocks = re.findall(r"```powershell\n(.*?)\n```", runbook, flags=re.DOTALL)

    assert "连续完成两场" in runbook
    assert "3 秒" in runbook
    assert "两个不同的场次 ID" in runbook
    assert "同一个永久球桌码 SHA-256" in runbook
    assert "两场之间不得重启服务" in runbook
    assert "每场恰好 1 杆事实和 1 个击球片段" in runbook
    assert "旧场次迟到命令不能修改新场次" in runbook
    assert powershell_blocks
    assert all(len(block.splitlines()) == 1 for block in powershell_blocks)
