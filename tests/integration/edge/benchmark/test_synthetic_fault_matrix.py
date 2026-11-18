from __future__ import annotations

from datetime import UTC, datetime

import pytest

from billiards_ai.edge.benchmark import (
    BenchmarkCounter,
    BenchmarkCounterName,
    BenchmarkEvidenceScope,
    ClockTimebase,
    CollectorResourceBudgetV1,
    FaultCommandV1,
    FaultEffectCaptureV1,
    FaultEffectOutcome,
    FaultKind,
    LatencyBoundary,
    LatencyStage,
    RuntimeBenchmarkCollector,
    ThermalAvailability,
    canonical_benchmark_evidence_bytes,
    decode_benchmark_evidence,
    execute_fault_plan,
    issue_environment_manifest,
    issue_fault_plan,
    issue_fault_safety_policy,
)

DEVICE_LINEAGE = "a" * 64


def _budget():
    return CollectorResourceBudgetV1(
        max_latency_samples=64,
        max_counters=8,
        max_fault_receipts=8,
        max_thermal_attempts=8,
        max_wire_bytes=256_000,
    )


class _SyntheticMatrixPort:
    safety_policy = issue_fault_safety_policy(
        policy_id="synthetic-matrix",
        policy_revision="1",
        max_network_disconnect_ms=1_000,
        max_runtime_restart_ms=1_000,
        max_high_load_duration_ms=2_000,
        max_high_load_units=16,
        realized_duration_tolerance_ms=0,
    )

    def __init__(self) -> None:
        self.network_available = True
        self.restart_count = 0
        self.load_units = 0

    def disconnect_network(self, *, duration_ms: int) -> FaultEffectCaptureV1:
        self.network_available = False
        self.network_available = True
        return FaultEffectCaptureV1(
            started_monotonic_ns=1_000,
            completed_monotonic_ns=1_000 + duration_ms * 1_000_000,
            outcome=FaultEffectOutcome.RECOVERED,
            evidence_reference_id="synthetic/network/1",
            evidence_content=b"network evidence",
        )

    def restart_runtime(self) -> FaultEffectCaptureV1:
        self.restart_count += 1
        return FaultEffectCaptureV1(
            started_monotonic_ns=2_000_000_000,
            completed_monotonic_ns=2_050_000_000,
            outcome=FaultEffectOutcome.RECOVERED,
            evidence_reference_id="synthetic/restart/1",
            evidence_content=b"restart evidence",
        )

    def apply_high_load(self, *, duration_ms: int, load_units: int) -> FaultEffectCaptureV1:
        self.load_units = load_units
        return FaultEffectCaptureV1(
            started_monotonic_ns=3_000_000_000,
            completed_monotonic_ns=3_000_000_000 + duration_ms * 1_000_000,
            outcome=FaultEffectOutcome.COMPLETED,
            evidence_reference_id="synthetic/load/1",
            evidence_content=b"load evidence",
        )


def _manifest():
    return issue_environment_manifest(
        evidence_scope=BenchmarkEvidenceScope.SYNTHETIC,
        captured_at=datetime(2026, 8, 21, 8, tzinfo=UTC),
        device_family="synthetic-runner",
        device_model="fault-matrix-v1",
        os_name="test-os",
        os_version="1",
        runtime_version="coach-runtime-v1",
        app_version="not-connected",
        git_commit="15dc3c96d339ee211ea7639073d1e1a0fb7a6c56",
        scenario_revision="synthetic-integration-v1",
        fault_matrix_revision="fault-matrix-v1",
        thermal_availability=ThermalAvailability.UNAVAILABLE,
        device_lineage_sha256=DEVICE_LINEAGE,
        clock_id="clock-1",
        clock_timebase=ClockTimebase.MONOTONIC,
        max_clock_sync_error_ns=0,
    )


def _collector(run_id: str):
    return RuntimeBenchmarkCollector(
        run_id=run_id,
        environment=_manifest(),
        run_started_monotonic_ns=0,
        expected_duration_ns=5_000_000_000,
        resource_budget=_budget(),
    )


def test_synthetic_fault_matrix_produces_round_trip_enabling_evidence() -> None:
    plan = issue_fault_plan(
        plan_id="matrix-v1",
        fault_matrix_revision="fault-matrix-v1",
        commands=(
            FaultCommandV1(
                fault_id="disconnect",
                ordinal=1,
                kind=FaultKind.NETWORK_DISCONNECT,
                duration_ms=500,
            ),
            FaultCommandV1(
                fault_id="restart",
                ordinal=2,
                kind=FaultKind.RUNTIME_RESTART,
            ),
            FaultCommandV1(
                fault_id="load",
                ordinal=3,
                kind=FaultKind.HIGH_LOAD,
                duration_ms=1_000,
                load_units=8,
            ),
        ),
    )
    fault_port = _SyntheticMatrixPort()
    result = execute_fault_plan(plan, fault_port)
    collector = _collector("synthetic-integration-1")
    collector.record_fault_execution(result)
    collector.record_latency(
        sample_id="reconnect-1",
        shot_sequence=0,
        stage=LatencyStage.RECONNECT_RECOVERY,
        start_boundary=LatencyBoundary.NETWORK_DISCONNECTED,
        end_boundary=LatencyBoundary.APP_RECOVERED,
        started_monotonic_ns=1_000,
        completed_monotonic_ns=500_001_000,
        clock_id="clock-1",
        clock_timebase=ClockTimebase.MONOTONIC,
        device_lineage_sha256=DEVICE_LINEAGE,
        clock_sync_error_ns=0,
    )
    collector.record_counter(
        BenchmarkCounter(name=BenchmarkCounterName.RECOVERY_SUCCESSES, value=2)
    )
    collector.record_counter(BenchmarkCounter(name=BenchmarkCounterName.DROPPED_FRAMES, value=7))

    evidence = collector.finalize(run_completed_monotonic_ns=5_000_000_000)
    encoded = canonical_benchmark_evidence_bytes(evidence)

    assert decode_benchmark_evidence(encoded) == evidence
    assert fault_port.network_available is True
    assert fault_port.restart_count == 1
    assert fault_port.load_units == 8
    assert evidence.environment.evidence_scope is BenchmarkEvidenceScope.SYNTHETIC
    assert evidence.thermal_readings == ()


@pytest.mark.parametrize(
    "mutate",
    (
        lambda data: b" " + data,
        lambda data: data.replace(b'"schema_version":"1.0"', b'"unknown":1,', 1),
        lambda data: data.replace(
            b'"schema_version":"1.0"',
            b'"schema_version":"1.0","schema_version":"1.0"',
            1,
        ),
        lambda data: data.replace(b'"run_id":"synthetic-integration-1"', b'"run_id":"x"'),
    ),
)
def test_evidence_decoder_rejects_noncanonical_unknown_duplicate_and_tampered_input(
    mutate,
) -> None:
    collector = _collector("synthetic-integration-1")
    collector.record_latency(
        sample_id="one",
        shot_sequence=1,
        stage=LatencyStage.RECONNECT_RECOVERY,
        start_boundary=LatencyBoundary.NETWORK_DISCONNECTED,
        end_boundary=LatencyBoundary.APP_RECOVERED,
        started_monotonic_ns=1,
        completed_monotonic_ns=2,
        clock_id="clock-1",
        clock_timebase=ClockTimebase.MONOTONIC,
        device_lineage_sha256=DEVICE_LINEAGE,
        clock_sync_error_ns=0,
    )
    encoded = canonical_benchmark_evidence_bytes(collector.finalize(run_completed_monotonic_ns=10))

    with pytest.raises(ValueError):
        decode_benchmark_evidence(mutate(encoded))


def test_unavailable_thermal_port_cannot_be_used_to_create_a_reading() -> None:
    collector = _collector("synthetic-integration-2")

    with pytest.raises(ValueError, match="unavailable"):
        collector.capture_thermal(captured_monotonic_ns=1)
