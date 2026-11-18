"""Deterministic, budgeted, seal-on-finalize benchmark collector."""

from __future__ import annotations

from typing import Protocol

from billiards_ai.edge.benchmark.models import (
    BenchmarkCounter,
    BenchmarkCounterName,
    BenchmarkEvidenceV1,
    ClockTimebase,
    CollectorResourceBudgetV1,
    EnvironmentManifestV1,
    FaultExecutionResultV1,
    GateDisposition,
    LatencyBoundary,
    LatencySampleV1,
    LatencyStage,
    MissingFieldGate,
    ThermalAvailability,
    ThermalReadingV1,
    ThermalSampleResultV1,
    ThermalSamplingAttemptV1,
    _calculate_statistics,
    _strict_revalidate,
    canonical_benchmark_evidence_bytes,
    canonical_environment_manifest_bytes,
    issue_benchmark_evidence,
)


class ThermalTelemetryPort(Protocol):
    """Environment-owned real OS sensor or external probe seam."""

    @property
    def availability(self) -> ThermalAvailability: ...

    def read_thermal_sample(self) -> ThermalSampleResultV1: ...


_STAGE_ORDER = {stage: ordinal for ordinal, stage in enumerate(LatencyStage)}


class RuntimeBenchmarkCollector:
    def __init__(
        self,
        *,
        run_id: str,
        environment: EnvironmentManifestV1,
        run_started_monotonic_ns: int,
        expected_duration_ns: int,
        resource_budget: CollectorResourceBudgetV1,
        thermal_port: ThermalTelemetryPort | None = None,
    ) -> None:
        if type(run_id) is not str or not run_id.strip():
            raise ValueError("run_id must be non-empty")
        if type(run_started_monotonic_ns) is not int or run_started_monotonic_ns < 0:
            raise ValueError("run start must be a non-negative strict integer")
        if type(expected_duration_ns) is not int or expected_duration_ns <= 0:
            raise ValueError("expected duration must be a positive strict integer")
        self._environment = _strict_revalidate(
            environment, EnvironmentManifestV1, "environment manifest"
        )
        canonical_environment_manifest_bytes(self._environment)
        self._resource_budget = _strict_revalidate(
            resource_budget, CollectorResourceBudgetV1, "collector resource budget"
        )
        if self._environment.thermal_availability is ThermalAvailability.UNAVAILABLE:
            if thermal_port is not None:
                raise ValueError("thermal port is forbidden when manifest authority is unavailable")
            cached_thermal_authority = ThermalAvailability.UNAVAILABLE
        else:
            if thermal_port is None:
                raise ValueError("available thermal authority requires a telemetry port")
            cached_thermal_authority = thermal_port.availability
            if type(cached_thermal_authority) is not ThermalAvailability:
                raise TypeError("thermal port availability must be a ThermalAvailability")
            if cached_thermal_authority is not self._environment.thermal_availability:
                raise ValueError("thermal port does not match environment authority")
        self._run_id = run_id.strip()
        self._run_started_monotonic_ns = run_started_monotonic_ns
        self._expected_duration_ns = expected_duration_ns
        self._thermal_port = thermal_port
        self._thermal_authority = cached_thermal_authority
        self._samples: dict[str, LatencySampleV1] = {}
        self._counters: dict[BenchmarkCounterName, BenchmarkCounter] = {}
        self._fault_execution: FaultExecutionResultV1 | None = None
        self._thermal_attempts: list[ThermalSamplingAttemptV1] = []
        self._thermal_readings: dict[str, ThermalReadingV1] = {}
        self._thermal_identity: dict[tuple[int, str], str] = {}
        self._sealed_evidence: BenchmarkEvidenceV1 | None = None
        self._sealed_completion_ns: int | None = None

    def _require_open(self) -> None:
        if self._sealed_evidence is not None:
            raise ValueError("benchmark collector is sealed")

    @staticmethod
    def _require_capacity(current: int, maximum: int, label: str) -> None:
        if current >= maximum:
            raise ValueError(f"collector resource budget exceeded for {label}")

    def record_latency(
        self,
        *,
        sample_id: str,
        shot_sequence: int,
        stage: LatencyStage,
        start_boundary: LatencyBoundary,
        end_boundary: LatencyBoundary,
        started_monotonic_ns: int,
        completed_monotonic_ns: int,
        clock_id: str,
        clock_timebase: ClockTimebase,
        device_lineage_sha256: str,
        clock_sync_error_ns: int,
    ) -> None:
        self._require_open()
        if (
            type(started_monotonic_ns) is not int
            or type(completed_monotonic_ns) is not int
            or completed_monotonic_ns <= started_monotonic_ns
        ):
            raise ValueError("latency sample timeline must advance")
        sample = LatencySampleV1.model_validate(
            {
                "sample_id": sample_id,
                "shot_sequence": shot_sequence,
                "stage": stage,
                "start_boundary": start_boundary,
                "end_boundary": end_boundary,
                "started_monotonic_ns": started_monotonic_ns,
                "completed_monotonic_ns": completed_monotonic_ns,
                "duration_ns": completed_monotonic_ns - started_monotonic_ns,
                "clock_id": clock_id,
                "clock_timebase": clock_timebase,
                "device_lineage_sha256": device_lineage_sha256,
                "clock_sync_error_ns": clock_sync_error_ns,
            },
            strict=True,
        )
        previous = self._samples.get(sample.sample_id)
        if previous is not None:
            if previous != sample:
                raise ValueError("latency sample identity conflict")
            return
        self._require_capacity(
            len(self._samples), self._resource_budget.max_latency_samples, "latency samples"
        )
        self._samples[sample.sample_id] = sample

    def record_counter(self, counter: BenchmarkCounter) -> None:
        self._require_open()
        counter = _strict_revalidate(counter, BenchmarkCounter, "benchmark counter")
        previous = self._counters.get(counter.name)
        if previous is not None:
            if previous != counter:
                raise ValueError("benchmark counter identity conflict")
            return
        self._require_capacity(len(self._counters), self._resource_budget.max_counters, "counters")
        self._counters[counter.name] = counter

    def record_fault_execution(self, execution: FaultExecutionResultV1) -> None:
        self._require_open()
        execution = _strict_revalidate(execution, FaultExecutionResultV1, "fault execution result")
        receipt_count = len(execution.completed_receipts) + (
            1 if execution.failed_receipt is not None else 0
        )
        if receipt_count > self._resource_budget.max_fault_receipts:
            raise ValueError("collector resource budget exceeded for fault receipts")
        if execution.plan.fault_matrix_revision != self._environment.fault_matrix_revision:
            raise ValueError("fault execution matrix revision does not match environment")
        if self._fault_execution is not None:
            if self._fault_execution != execution:
                raise ValueError("fault execution identity conflict")
            return
        self._fault_execution = execution

    def capture_thermal(self, *, captured_monotonic_ns: int) -> ThermalSamplingAttemptV1:
        self._require_open()
        if self._thermal_authority is ThermalAvailability.UNAVAILABLE or self._thermal_port is None:
            raise ValueError("thermal telemetry is unavailable for this environment")
        if type(captured_monotonic_ns) is not int or captured_monotonic_ns < 0:
            raise ValueError("thermal capture time must be a non-negative strict integer")
        if captured_monotonic_ns < self._run_started_monotonic_ns:
            raise ValueError("thermal attempt is outside benchmark run window")
        self._require_capacity(
            len(self._thermal_attempts),
            self._resource_budget.max_thermal_attempts,
            "thermal attempts",
        )
        result = _strict_revalidate(
            self._thermal_port.read_thermal_sample(),
            ThermalSampleResultV1,
            "thermal sample result",
        )
        ordinal = len(self._thermal_attempts) + 1
        if result.reading is not None:
            reading = _strict_revalidate(result.reading, ThermalReadingV1, "thermal reading")
            if reading.captured_monotonic_ns != captured_monotonic_ns:
                raise ValueError("thermal reading time does not match sampling attempt")
            if reading.source is not self._thermal_authority:
                raise ValueError("thermal reading source does not match cached authority")
            previous = self._thermal_readings.get(reading.reading_id)
            identity = (reading.captured_monotonic_ns, reading.sensor_id)
            previous_identity = self._thermal_identity.get(identity)
            if previous is not None and previous != reading:
                raise ValueError("thermal reading identity conflict")
            if previous_identity is not None and previous_identity != reading.reading_id:
                raise ValueError("thermal reading time/sensor identity conflict")
            self._thermal_readings[reading.reading_id] = reading
            self._thermal_identity[identity] = reading.reading_id
            attempt = ThermalSamplingAttemptV1(
                attempt_id=f"thermal-attempt-{ordinal}",
                ordinal=ordinal,
                captured_monotonic_ns=captured_monotonic_ns,
                authority=self._thermal_authority,
                succeeded=True,
                reading_id=reading.reading_id,
            )
        else:
            attempt = ThermalSamplingAttemptV1(
                attempt_id=f"thermal-attempt-{ordinal}",
                ordinal=ordinal,
                captured_monotonic_ns=captured_monotonic_ns,
                authority=self._thermal_authority,
                succeeded=False,
                failure_reason=result.failure_reason,
            )
        self._thermal_attempts.append(attempt)
        return attempt

    def finalize(self, *, run_completed_monotonic_ns: int) -> BenchmarkEvidenceV1:
        if self._sealed_evidence is not None:
            if run_completed_monotonic_ns != self._sealed_completion_ns:
                raise ValueError("finalize completion time conflicts with sealed evidence")
            return self._sealed_evidence
        if type(run_completed_monotonic_ns) is not int:
            raise ValueError("run completion must be a strict integer")
        if run_completed_monotonic_ns <= self._run_started_monotonic_ns:
            raise ValueError("benchmark run timeline must advance")
        samples = tuple(
            sorted(
                self._samples.values(),
                key=lambda sample: (
                    _STAGE_ORDER[sample.stage],
                    sample.shot_sequence,
                    sample.sample_id,
                ),
            )
        )
        statistics = tuple(
            _calculate_statistics(
                stage, tuple(sample for sample in samples if sample.stage is stage)
            )
            for stage in LatencyStage
            if any(sample.stage is stage for sample in samples)
        )
        fault_receipts = ()
        if self._fault_execution is not None:
            fault_receipts = self._fault_execution.completed_receipts + (
                (self._fault_execution.failed_receipt,)
                if self._fault_execution.failed_receipt is not None
                else ()
            )
        evidence = issue_benchmark_evidence(
            run_id=self._run_id,
            environment=self._environment,
            run_started_monotonic_ns=self._run_started_monotonic_ns,
            run_completed_monotonic_ns=run_completed_monotonic_ns,
            run_elapsed_ns=run_completed_monotonic_ns - self._run_started_monotonic_ns,
            expected_duration_ns=self._expected_duration_ns,
            latency_samples=samples,
            stage_statistics=statistics,
            counters=tuple(
                self._counters[name] for name in BenchmarkCounterName if name in self._counters
            ),
            fault_execution=self._fault_execution,
            fault_receipts=fault_receipts,
            thermal_attempts=tuple(self._thermal_attempts),
            thermal_readings=tuple(
                sorted(
                    self._thermal_readings.values(),
                    key=lambda reading: (
                        reading.captured_monotonic_ns,
                        reading.sensor_id,
                        reading.reading_id,
                    ),
                )
            ),
            gate_disposition=GateDisposition.NOT_EVALUATED,
            missing_field_gates=tuple(MissingFieldGate),
        )
        if len(canonical_benchmark_evidence_bytes(evidence)) > self._resource_budget.max_wire_bytes:
            raise ValueError("collector resource budget exceeded for evidence wire bytes")
        self._sealed_completion_ns = run_completed_monotonic_ns
        self._sealed_evidence = evidence
        return evidence
