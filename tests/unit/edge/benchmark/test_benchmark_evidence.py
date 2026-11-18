from __future__ import annotations

from datetime import UTC, datetime

import pytest

from billiards_ai.edge.benchmark import (
    BenchmarkCounter,
    BenchmarkCounterName,
    BenchmarkEvidenceScope,
    BenchmarkEvidenceV1,
    ClockTimebase,
    CollectorResourceBudgetV1,
    FaultCommandV1,
    FaultEffectCaptureV1,
    FaultEffectOutcome,
    FaultKind,
    GateDisposition,
    LatencyBoundary,
    LatencyStage,
    RuntimeBenchmarkCollector,
    ThermalAvailability,
    canonical_benchmark_evidence_bytes,
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


def _environment():
    return issue_environment_manifest(
        evidence_scope=BenchmarkEvidenceScope.SYNTHETIC,
        captured_at=datetime(2026, 8, 21, 8, tzinfo=UTC),
        device_family="synthetic-runner",
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
    )


def _collector(run_id: str):
    return RuntimeBenchmarkCollector(
        run_id=run_id,
        environment=_environment(),
        run_started_monotonic_ns=0,
        expected_duration_ns=1_000_000,
        resource_budget=_budget(),
    )


_BOUNDARIES = {
    LatencyStage.POLICY_DECISION: (
        LatencyBoundary.OBSERVATION_EMITTED,
        LatencyBoundary.POLICY_DECISION_READY,
    ),
    LatencyStage.END_TO_END: (
        LatencyBoundary.VIDEO_CAPTURED,
        LatencyBoundary.APP_EVENT_RECEIVED,
    ),
}


def _record(
    collector, *, sample_id, shot_sequence, stage, started_monotonic_ns, completed_monotonic_ns
):
    start_boundary, end_boundary = _BOUNDARIES[stage]
    collector.record_latency(
        sample_id=sample_id,
        shot_sequence=shot_sequence,
        stage=stage,
        start_boundary=start_boundary,
        end_boundary=end_boundary,
        started_monotonic_ns=started_monotonic_ns,
        completed_monotonic_ns=completed_monotonic_ns,
        clock_id="clock-1",
        clock_timebase=ClockTimebase.MONOTONIC,
        device_lineage_sha256=DEVICE_LINEAGE,
        clock_sync_error_ns=0,
    )


def test_collector_issues_deterministic_not_evaluated_evidence() -> None:
    first = _collector("synthetic-run-001")
    second = _collector("synthetic-run-001")
    samples = (
        ("shot-2-policy", 2, LatencyStage.POLICY_DECISION, 20_000, 24_000),
        ("shot-1-policy", 1, LatencyStage.POLICY_DECISION, 10_000, 12_000),
    )
    for collector, ordered in ((first, samples), (second, tuple(reversed(samples)))):
        for sample_id, sequence, stage, started_ns, completed_ns in ordered:
            _record(
                collector,
                sample_id=sample_id,
                shot_sequence=sequence,
                stage=stage,
                started_monotonic_ns=started_ns,
                completed_monotonic_ns=completed_ns,
            )
        collector.record_counter(
            BenchmarkCounter(name=BenchmarkCounterName.DROPPED_FRAMES, value=3)
        )

    evidence = first.finalize(run_completed_monotonic_ns=50_000)
    assert evidence == second.finalize(run_completed_monotonic_ns=50_000)
    assert canonical_benchmark_evidence_bytes(evidence) == canonical_benchmark_evidence_bytes(
        second.finalize(run_completed_monotonic_ns=50_000)
    )
    assert evidence.gate_disposition is GateDisposition.NOT_EVALUATED
    assert tuple(gate.value for gate in evidence.missing_field_gates) == (
        "REALTIME_OBSERVATION_ADAPTER_10_7",
        "USER_APP_PRODUCTION_INTEGRATION_10_8",
        "IOS_DEVICE_EVIDENCE",
        "ANDROID_DEVICE_EVIDENCE",
        "FIELD_DEVICE_EVIDENCE",
        "FIELD_THERMAL_EVIDENCE",
    )
    assert evidence.thermal_readings == ()
    assert evidence.stage_statistics[0].stage is LatencyStage.POLICY_DECISION
    assert evidence.stage_statistics[0].count == 2
    assert evidence.stage_statistics[0].minimum_ns == 2_000
    assert evidence.stage_statistics[0].p50_ns == 2_000
    assert evidence.stage_statistics[0].p95_ns == 4_000
    assert evidence.stage_statistics[0].p99_ns == 4_000
    assert evidence.stage_statistics[0].maximum_ns == 4_000
    assert evidence.stage_statistics[0].mean_floor_ns == 3_000


def test_exact_duplicate_is_idempotent_but_conflict_and_bad_timeline_fail_closed() -> None:
    collector = _collector("synthetic-run-002")
    kwargs = {
        "sample_id": "policy-1",
        "shot_sequence": 1,
        "stage": LatencyStage.POLICY_DECISION,
        "started_monotonic_ns": 100,
        "completed_monotonic_ns": 120,
    }
    _record(collector, **kwargs)
    _record(collector, **kwargs)

    with pytest.raises(ValueError, match="conflict"):
        _record(collector, **{**kwargs, "completed_monotonic_ns": 121})
    with pytest.raises(ValueError, match="timeline"):
        _record(collector, **{**kwargs, "sample_id": "bad", "completed_monotonic_ns": 99})
    assert len(collector.finalize(run_completed_monotonic_ns=1_000).latency_samples) == 1


def test_runtime_counter_is_an_explicit_versioned_contract() -> None:
    counter = BenchmarkCounter(name=BenchmarkCounterName.DROPPED_FRAMES, value=3)

    assert counter.model_dump(mode="json") == {
        "schema_version": "1.0",
        "name": "DROPPED_FRAMES",
        "value": 3,
    }


def test_nested_tampering_is_rejected_by_canonical_encoder() -> None:
    collector = _collector("synthetic-run-003")
    _record(
        collector,
        sample_id="total-1",
        shot_sequence=1,
        stage=LatencyStage.POLICY_DECISION,
        started_monotonic_ns=1,
        completed_monotonic_ns=2,
    )
    evidence = collector.finalize(run_completed_monotonic_ns=10)

    with pytest.raises(ValueError, match="environment manifest hash"):
        canonical_benchmark_evidence_bytes(
            evidence.model_copy(
                update={
                    "environment": evidence.environment.model_copy(
                        update={"device_model": "tampered"}
                    )
                }
            )
        )
    with pytest.raises(ValueError):
        canonical_benchmark_evidence_bytes(evidence.model_copy(update={"latency_samples": ()}))


def test_evidence_schema_recomputes_statistics_instead_of_trusting_input() -> None:
    collector = _collector("synthetic-run-004")
    _record(
        collector,
        sample_id="policy-1",
        shot_sequence=1,
        stage=LatencyStage.POLICY_DECISION,
        started_monotonic_ns=10,
        completed_monotonic_ns=20,
    )
    values = collector.finalize(run_completed_monotonic_ns=100).model_dump(mode="python")
    values["stage_statistics"][0]["minimum_ns"] = 9

    with pytest.raises(ValueError, match="statistics"):
        BenchmarkEvidenceV1.model_validate(values, strict=True)


class _RecordingFaultPort:
    safety_policy = issue_fault_safety_policy(
        policy_id="synthetic",
        policy_revision="1",
        max_network_disconnect_ms=2_000,
        max_runtime_restart_ms=1_000,
        max_high_load_duration_ms=3_000,
        max_high_load_units=8,
        realized_duration_tolerance_ms=0,
    )

    def __init__(self) -> None:
        self.state: list[tuple[FaultKind, int, int]] = []

    def disconnect_network(self, *, duration_ms: int) -> FaultEffectCaptureV1:
        self.state.append((FaultKind.NETWORK_DISCONNECT, duration_ms, 0))
        return FaultEffectCaptureV1(
            started_monotonic_ns=100,
            completed_monotonic_ns=100 + duration_ms * 1_000_000,
            outcome=FaultEffectOutcome.RECOVERED,
            evidence_reference_id="synthetic-network-trace-1",
            evidence_content=b"network trace",
        )

    def restart_runtime(self) -> FaultEffectCaptureV1:
        self.state.append((FaultKind.RUNTIME_RESTART, 0, 0))
        return FaultEffectCaptureV1(
            started_monotonic_ns=1_000_000_100,
            completed_monotonic_ns=1_050_000_100,
            outcome=FaultEffectOutcome.RECOVERED,
            evidence_reference_id="synthetic-restart-trace-1",
            evidence_content=b"restart trace",
        )

    def apply_high_load(self, *, duration_ms: int, load_units: int) -> FaultEffectCaptureV1:
        self.state.append((FaultKind.HIGH_LOAD, duration_ms, load_units))
        return FaultEffectCaptureV1(
            started_monotonic_ns=1_050_000_100,
            completed_monotonic_ns=1_050_000_100 + duration_ms * 1_000_000,
            outcome=FaultEffectOutcome.COMPLETED,
            evidence_reference_id="synthetic-load-trace-1",
            evidence_content=b"load trace",
        )


def test_fault_plan_executes_explicit_entrypoints_and_hashes_receipts() -> None:
    plan = issue_fault_plan(
        plan_id="synthetic-fault-plan-1",
        fault_matrix_revision="fault-matrix-v1",
        commands=(
            FaultCommandV1(
                fault_id="network-1",
                ordinal=1,
                kind=FaultKind.NETWORK_DISCONNECT,
                duration_ms=1_000,
            ),
            FaultCommandV1(
                fault_id="restart-1",
                ordinal=2,
                kind=FaultKind.RUNTIME_RESTART,
            ),
            FaultCommandV1(
                fault_id="load-1",
                ordinal=3,
                kind=FaultKind.HIGH_LOAD,
                duration_ms=2_000,
                load_units=4,
            ),
        ),
    )
    port = _RecordingFaultPort()

    result = execute_fault_plan(plan, port)
    receipts = result.completed_receipts

    assert port.state == [
        (FaultKind.NETWORK_DISCONNECT, 1_000, 0),
        (FaultKind.RUNTIME_RESTART, 0, 0),
        (FaultKind.HIGH_LOAD, 2_000, 4),
    ]
    assert tuple(receipt.fault_id for receipt in receipts) == (
        "network-1",
        "restart-1",
        "load-1",
    )
    assert all(len(receipt.content_sha256) == 64 for receipt in receipts)


@pytest.mark.parametrize(
    "kwargs",
    (
        dict(
            fault_id="bad-restart-duration",
            ordinal=1,
            kind=FaultKind.RUNTIME_RESTART,
            duration_ms=1,
        ),
        dict(
            fault_id="bad-network-load",
            ordinal=1,
            kind=FaultKind.NETWORK_DISCONNECT,
            duration_ms=1,
            load_units=1,
        ),
    ),
)
def test_fault_command_rejects_parameters_for_the_wrong_entrypoint(kwargs) -> None:
    with pytest.raises(ValueError):
        FaultCommandV1(**kwargs)
