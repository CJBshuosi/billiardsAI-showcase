from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from billiards_ai.edge.benchmark import (
    BenchmarkCounter,
    BenchmarkCounterName,
    BenchmarkEvidenceScope,
    ClockTimebase,
    CollectorResourceBudgetV1,
    EnvironmentAuthorityAttestationV1,
    FaultCommandV1,
    FaultEffectCaptureV1,
    FaultEffectOutcome,
    FaultExecutionStatus,
    FaultKind,
    LatencyBoundary,
    LatencyStage,
    RuntimeBenchmarkCollector,
    ThermalAvailability,
    ThermalReadingV1,
    ThermalSampleResultV1,
    canonical_benchmark_evidence_bytes,
    canonical_environment_manifest_bytes,
    decode_benchmark_evidence,
    execute_fault_plan,
    issue_environment_manifest,
    issue_fault_plan,
    issue_fault_safety_policy,
)

DEVICE_LINEAGE = "a" * 64


def _manifest(*, scope: BenchmarkEvidenceScope = BenchmarkEvidenceScope.SYNTHETIC, authority=None):
    return issue_environment_manifest(
        evidence_scope=scope,
        captured_at=datetime(2026, 8, 21, 8, tzinfo=UTC),
        device_family="synthetic-runner" if scope is BenchmarkEvidenceScope.SYNTHETIC else "phone",
        device_model="fixture-v1",
        os_name="test-os",
        os_version="1",
        runtime_version="coach-runtime-v1",
        app_version="not-connected",
        git_commit="15dc3c96d339ee211ea7639073d1e1a0fb7a6c56",
        scenario_revision="synthetic-matrix-v1",
        fault_matrix_revision="fault-matrix-v1",
        thermal_availability=ThermalAvailability.UNAVAILABLE,
        device_lineage_sha256=DEVICE_LINEAGE,
        clock_id="clock-1",
        clock_timebase=ClockTimebase.MONOTONIC,
        max_clock_sync_error_ns=0,
        authority=authority,
    )


def _budget(**updates: int) -> CollectorResourceBudgetV1:
    values = dict(
        max_latency_samples=32,
        max_counters=8,
        max_fault_receipts=8,
        max_thermal_attempts=8,
        max_wire_bytes=256_000,
    )
    values.update(updates)
    return CollectorResourceBudgetV1(**values)


def _collector(**kwargs) -> RuntimeBenchmarkCollector:
    return RuntimeBenchmarkCollector(
        run_id="run-1",
        environment=_manifest(),
        run_started_monotonic_ns=1_000,
        expected_duration_ns=10_000,
        resource_budget=_budget(),
        **kwargs,
    )


def _record_complete_shot(collector: RuntimeBenchmarkCollector, shot: int = 1) -> None:
    values = (
        (
            LatencyStage.PERCEPTION_OBSERVATION,
            LatencyBoundary.VIDEO_CAPTURED,
            LatencyBoundary.OBSERVATION_EMITTED,
            2_000,
            3_000,
        ),
        (
            LatencyStage.POLICY_DECISION,
            LatencyBoundary.OBSERVATION_EMITTED,
            LatencyBoundary.POLICY_DECISION_READY,
            3_000,
            4_000,
        ),
        (
            LatencyStage.FEEDBACK_BUILD,
            LatencyBoundary.POLICY_DECISION_READY,
            LatencyBoundary.FEEDBACK_READY,
            4_000,
            5_000,
        ),
        (
            LatencyStage.EVENT_TRANSPORT_TO_APP,
            LatencyBoundary.FEEDBACK_READY,
            LatencyBoundary.APP_EVENT_RECEIVED,
            5_000,
            6_000,
        ),
        (
            LatencyStage.END_TO_END,
            LatencyBoundary.VIDEO_CAPTURED,
            LatencyBoundary.APP_EVENT_RECEIVED,
            2_000,
            6_000,
        ),
    )
    for stage, start_boundary, end_boundary, started, completed in values:
        collector.record_latency(
            sample_id=f"{shot}-{stage.value}",
            shot_sequence=shot,
            stage=stage,
            start_boundary=start_boundary,
            end_boundary=end_boundary,
            started_monotonic_ns=started,
            completed_monotonic_ns=completed,
            clock_id="clock-1",
            clock_timebase=ClockTimebase.MONOTONIC,
            device_lineage_sha256=DEVICE_LINEAGE,
            clock_sync_error_ns=0,
        )


def _policy():
    return issue_fault_safety_policy(
        policy_id="lab-policy",
        policy_revision="1",
        max_network_disconnect_ms=1_000,
        max_runtime_restart_ms=1_000,
        max_high_load_duration_ms=2_000,
        max_high_load_units=8,
        realized_duration_tolerance_ms=10,
    )


def _plan():
    return issue_fault_plan(
        plan_id="plan-1",
        fault_matrix_revision="fault-matrix-v1",
        commands=(
            FaultCommandV1(
                fault_id="disconnect",
                ordinal=1,
                kind=FaultKind.NETWORK_DISCONNECT,
                duration_ms=100,
            ),
            FaultCommandV1(
                fault_id="load",
                ordinal=2,
                kind=FaultKind.HIGH_LOAD,
                duration_ms=200,
                load_units=4,
            ),
        ),
    )


class _FaultPort:
    safety_policy = _policy()

    def __init__(self, *, fail_second: bool = False) -> None:
        self.fail_second = fail_second

    def disconnect_network(self, *, duration_ms: int) -> FaultEffectCaptureV1:
        return FaultEffectCaptureV1(
            started_monotonic_ns=2_000,
            completed_monotonic_ns=2_000 + duration_ms * 1_000_000,
            outcome=FaultEffectOutcome.RECOVERED,
            evidence_reference_id="network-trace",
            evidence_content=b"network evidence",
        )

    def restart_runtime(self) -> FaultEffectCaptureV1:
        raise AssertionError("not used")

    def apply_high_load(self, *, duration_ms: int, load_units: int) -> FaultEffectCaptureV1:
        return FaultEffectCaptureV1(
            started_monotonic_ns=200_002_000,
            completed_monotonic_ns=200_002_000 + duration_ms * 1_000_000,
            outcome=(
                FaultEffectOutcome.FAILED if self.fail_second else FaultEffectOutcome.COMPLETED
            ),
            failure_code="LOAD_GENERATOR_FAILED" if self.fail_second else None,
            evidence_reference_id="load-trace",
            evidence_content=b"load evidence",
        )


def test_strict_revalidation_rejects_constructed_and_copied_models() -> None:
    manifest = _manifest()
    bypass = manifest.model_copy(update={"evidence_scope": "not-an-enum"})
    with pytest.raises(ValueError):
        canonical_environment_manifest_bytes(bypass)

    collector = _collector()
    malformed = BenchmarkCounter.model_construct(name="bad-name", value=-1)
    with pytest.raises(ValueError):
        collector.record_counter(malformed)

    malformed_command = FaultCommandV1(
        fault_id="bad-copy",
        ordinal=1,
        kind=FaultKind.NETWORK_DISCONNECT,
        duration_ms=1,
    ).model_copy(update={"duration_ms": 0})
    with pytest.raises(ValueError):
        issue_fault_plan(
            plan_id="bad-plan",
            fault_matrix_revision="fault-matrix-v1",
            commands=(malformed_command,),
        )

    bad_hash_plan = _plan().model_copy(update={"plan_sha256": "0" * 64})
    with pytest.raises(ValueError, match="plan hash"):
        execute_fault_plan(bad_hash_plan, _FaultPort())


def test_stage_boundaries_and_same_shot_chain_are_closed() -> None:
    collector = _collector()
    _record_complete_shot(collector)
    assert collector.finalize(run_completed_monotonic_ns=11_000).latency_samples

    broken = _collector()
    _record_complete_shot(broken)
    sample = broken._samples["1-POLICY_DECISION"]  # noqa: SLF001 - adversarial invariant test
    broken._samples[sample.sample_id] = sample.model_copy(  # noqa: SLF001
        update={
            "started_monotonic_ns": sample.started_monotonic_ns + 1,
            "duration_ns": sample.duration_ns - 1,
        }
    )
    with pytest.raises(ValueError, match="chain"):
        broken.finalize(run_completed_monotonic_ns=11_000)


def test_run_window_exposes_elapsed_and_expected_duration() -> None:
    collector = _collector()
    evidence = collector.finalize(run_completed_monotonic_ns=11_000)
    assert evidence.run_started_monotonic_ns == 1_000
    assert evidence.run_completed_monotonic_ns == 11_000
    assert evidence.run_elapsed_ns == 10_000
    assert evidence.expected_duration_ns == 10_000


def test_fault_plan_hash_matrix_and_receipts_are_fully_bound() -> None:
    plan = _plan()
    result = execute_fault_plan(plan, _FaultPort())
    assert result.status is FaultExecutionStatus.COMPLETED
    assert all(receipt.plan_sha256 == plan.plan_sha256 for receipt in result.completed_receipts)
    assert all(
        receipt.fault_matrix_revision == plan.fault_matrix_revision
        for receipt in result.completed_receipts
    )
    assert result.completed_receipts[0].effect.evidence_reference.content_sha256 != "0" * 64

    first = result.completed_receipts[0]
    tampered = first.model_copy(update={"plan_sha256": "0" * 64})
    bypass = result.model_copy(
        update={"completed_receipts": (tampered, *result.completed_receipts[1:])}
    )
    with pytest.raises(ValueError, match="fault execution"):
        _collector().record_fault_execution(bypass)


def test_environment_owned_fault_policy_blocks_unbounded_commands_before_port_call() -> None:
    unsafe = issue_fault_plan(
        plan_id="unsafe",
        fault_matrix_revision="fault-matrix-v1",
        commands=(
            FaultCommandV1(
                fault_id="disconnect",
                ordinal=1,
                kind=FaultKind.NETWORK_DISCONNECT,
                duration_ms=1_001,
            ),
        ),
    )
    with pytest.raises(ValueError, match="safety policy"):
        execute_fault_plan(unsafe, _FaultPort())


def test_partial_fault_failure_is_typed_and_fail_stop() -> None:
    result = execute_fault_plan(_plan(), _FaultPort(fail_second=True))
    assert result.status is FaultExecutionStatus.FAILED
    assert tuple(receipt.fault_id for receipt in result.completed_receipts) == ("disconnect",)
    assert result.failed_command is not None and result.failed_command.fault_id == "load"
    assert result.failed_receipt is not None
    assert result.failed_receipt.effect.outcome is FaultEffectOutcome.FAILED


class _Authority:
    def attest_environment(self, *, claim_sha256, evidence_scope, device_lineage_sha256):
        return EnvironmentAuthorityAttestationV1(
            authority_id="device-registry",
            attestation_id="attestation-1",
            evidence_scope=evidence_scope,
            device_lineage_sha256=device_lineage_sha256,
            claim_sha256=claim_sha256,
        )


def test_non_synthetic_scope_requires_authority_port_and_cannot_be_self_reported() -> None:
    with pytest.raises(ValueError, match="authority"):
        _manifest(scope=BenchmarkEvidenceScope.IOS_DEVICE)
    issued = _manifest(scope=BenchmarkEvidenceScope.IOS_DEVICE, authority=_Authority())
    assert issued.authority_attestation is not None
    with pytest.raises((TypeError, ValueError)):
        issue_environment_manifest(
            **issued.model_dump(mode="python", exclude={"manifest_sha256"}),
            authority_attestation=issued.authority_attestation,
        )


class _ThermalPort:
    availability = ThermalAvailability.OS_SENSOR

    def __init__(self, results):
        self.results = iter(results)

    def read_thermal_sample(self) -> ThermalSampleResultV1:
        return next(self.results)


def test_thermal_attempts_record_success_and_failure_with_cached_authority() -> None:
    manifest = issue_environment_manifest(
        **{
            **_manifest().model_dump(
                mode="python",
                exclude={
                    "manifest_sha256",
                    "evidence_scope",
                    "thermal_availability",
                    "authority_attestation",
                },
            ),
            "evidence_scope": BenchmarkEvidenceScope.EDGE_DEVICE,
            "thermal_availability": ThermalAvailability.OS_SENSOR,
        },
        authority=_Authority(),
    )
    reading = ThermalReadingV1(
        reading_id="thermal-1",
        captured_monotonic_ns=3_000,
        celsius_milli=42_000,
        source=ThermalAvailability.OS_SENSOR,
        sensor_id="soc",
    )
    port = _ThermalPort(
        (
            ThermalSampleResultV1(reading=reading),
            ThermalSampleResultV1(failure_reason="SENSOR_READ_FAILED"),
        )
    )
    collector = RuntimeBenchmarkCollector(
        run_id="thermal-run",
        environment=manifest,
        run_started_monotonic_ns=1_000,
        expected_duration_ns=10_000,
        resource_budget=_budget(),
        thermal_port=port,
    )
    assert collector.capture_thermal(captured_monotonic_ns=3_000).reading_id == "thermal-1"
    assert (
        collector.capture_thermal(captured_monotonic_ns=4_000).failure_reason
        == "SENSOR_READ_FAILED"
    )
    evidence = collector.finalize(run_completed_monotonic_ns=11_000)
    assert len(evidence.thermal_attempts) == 2
    assert len(evidence.thermal_readings) == 1


def test_chronology_and_thermal_identity_fail_closed() -> None:
    collector = _collector()
    collector.record_latency(
        sample_id="outside-run",
        shot_sequence=0,
        stage=LatencyStage.RECONNECT_RECOVERY,
        start_boundary=LatencyBoundary.NETWORK_DISCONNECTED,
        end_boundary=LatencyBoundary.APP_RECOVERED,
        started_monotonic_ns=500,
        completed_monotonic_ns=600,
        clock_id="clock-1",
        clock_timebase=ClockTimebase.MONOTONIC,
        device_lineage_sha256=DEVICE_LINEAGE,
        clock_sync_error_ns=0,
    )
    with pytest.raises(ValueError, match="run window"):
        collector.finalize(run_completed_monotonic_ns=11_000)


def test_collector_budget_and_finalize_seal_are_deterministic() -> None:
    collector = RuntimeBenchmarkCollector(
        run_id="sealed-run",
        environment=_manifest(),
        run_started_monotonic_ns=1_000,
        expected_duration_ns=10_000,
        resource_budget=_budget(max_counters=1),
    )
    collector.record_counter(BenchmarkCounter(name=BenchmarkCounterName.DROPPED_FRAMES, value=1))
    with pytest.raises(ValueError, match="resource budget"):
        collector.record_counter(BenchmarkCounter(name=BenchmarkCounterName.ABSTENTIONS, value=1))
    first = collector.finalize(run_completed_monotonic_ns=11_000)
    assert collector.finalize(run_completed_monotonic_ns=11_000) is first
    with pytest.raises(ValueError, match="sealed"):
        collector.record_counter(BenchmarkCounter(name=BenchmarkCounterName.INPUT_FRAMES, value=1))
    with pytest.raises(ValueError, match="conflict"):
        collector.finalize(run_completed_monotonic_ns=12_000)


def test_encoder_revalidates_bypassed_evidence_and_decoder_normalizes_depth_errors() -> None:
    evidence = _collector().finalize(run_completed_monotonic_ns=11_000)
    bypass = evidence.model_copy(update={"run_elapsed_ns": -1})
    with pytest.raises(ValueError):
        canonical_benchmark_evidence_bytes(bypass)

    deep = b'{"x":' + b"[" * 1_100 + b"0" + b"]" * 1_100 + b"}"
    with pytest.raises(ValueError, match="invalid benchmark evidence JSON"):
        decode_benchmark_evidence(deep)

    encoded = canonical_benchmark_evidence_bytes(evidence)
    assert decode_benchmark_evidence(encoded) == evidence
    assert json.loads(encoded)["run_elapsed_ns"] == 10_000
