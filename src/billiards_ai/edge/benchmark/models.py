"""Strict local contracts for Story 10.9 benchmark evidence.

The module records evidence only. It cannot declare a product gate pass and it never
manufactures device, clock, fault, or thermal authority.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import Field, StrictBool, StrictInt, field_validator, model_validator
from pydantic_core import to_jsonable_python

from billiards_ai.shared.models.events import (
    CoreContractModel,
    NonEmptyStr,
    Sha256,
    require_utc,
)

MAX_BENCHMARK_EVIDENCE_BYTES = 8 * 1024 * 1024
MAX_BENCHMARK_JSON_DEPTH = 64


class BenchmarkEvidenceScope(StrEnum):
    SYNTHETIC = "SYNTHETIC"
    EDGE_DEVICE = "EDGE_DEVICE"
    IOS_DEVICE = "IOS_DEVICE"
    ANDROID_DEVICE = "ANDROID_DEVICE"


class GateDisposition(StrEnum):
    NOT_EVALUATED = "NOT_EVALUATED"


class MissingFieldGate(StrEnum):
    REALTIME_OBSERVATION_ADAPTER_10_7 = "REALTIME_OBSERVATION_ADAPTER_10_7"
    USER_APP_PRODUCTION_INTEGRATION_10_8 = "USER_APP_PRODUCTION_INTEGRATION_10_8"
    IOS_DEVICE_EVIDENCE = "IOS_DEVICE_EVIDENCE"
    ANDROID_DEVICE_EVIDENCE = "ANDROID_DEVICE_EVIDENCE"
    FIELD_DEVICE_EVIDENCE = "FIELD_DEVICE_EVIDENCE"
    FIELD_THERMAL_EVIDENCE = "FIELD_THERMAL_EVIDENCE"


class ThermalAvailability(StrEnum):
    UNAVAILABLE = "UNAVAILABLE"
    OS_SENSOR = "OS_SENSOR"
    EXTERNAL_PROBE = "EXTERNAL_PROBE"


class ClockTimebase(StrEnum):
    MONOTONIC = "MONOTONIC"


class LatencyStage(StrEnum):
    PERCEPTION_OBSERVATION = "PERCEPTION_OBSERVATION"
    POLICY_DECISION = "POLICY_DECISION"
    FEEDBACK_BUILD = "FEEDBACK_BUILD"
    EVENT_TRANSPORT_TO_APP = "EVENT_TRANSPORT_TO_APP"
    END_TO_END = "END_TO_END"
    VIDEO_FIRST_FRAME = "VIDEO_FIRST_FRAME"
    EVENT_TO_OVERLAY = "EVENT_TO_OVERLAY"
    RECONNECT_RECOVERY = "RECONNECT_RECOVERY"


class LatencyBoundary(StrEnum):
    VIDEO_CAPTURED = "VIDEO_CAPTURED"
    OBSERVATION_EMITTED = "OBSERVATION_EMITTED"
    POLICY_DECISION_READY = "POLICY_DECISION_READY"
    FEEDBACK_READY = "FEEDBACK_READY"
    APP_EVENT_RECEIVED = "APP_EVENT_RECEIVED"
    VIDEO_STREAM_REQUESTED = "VIDEO_STREAM_REQUESTED"
    VIDEO_FIRST_FRAME_RENDERED = "VIDEO_FIRST_FRAME_RENDERED"
    OVERLAY_RENDERED = "OVERLAY_RENDERED"
    NETWORK_DISCONNECTED = "NETWORK_DISCONNECTED"
    APP_RECOVERED = "APP_RECOVERED"


_STAGE_BOUNDARIES: dict[LatencyStage, tuple[LatencyBoundary, LatencyBoundary]] = {
    LatencyStage.PERCEPTION_OBSERVATION: (
        LatencyBoundary.VIDEO_CAPTURED,
        LatencyBoundary.OBSERVATION_EMITTED,
    ),
    LatencyStage.POLICY_DECISION: (
        LatencyBoundary.OBSERVATION_EMITTED,
        LatencyBoundary.POLICY_DECISION_READY,
    ),
    LatencyStage.FEEDBACK_BUILD: (
        LatencyBoundary.POLICY_DECISION_READY,
        LatencyBoundary.FEEDBACK_READY,
    ),
    LatencyStage.EVENT_TRANSPORT_TO_APP: (
        LatencyBoundary.FEEDBACK_READY,
        LatencyBoundary.APP_EVENT_RECEIVED,
    ),
    LatencyStage.END_TO_END: (
        LatencyBoundary.VIDEO_CAPTURED,
        LatencyBoundary.APP_EVENT_RECEIVED,
    ),
    LatencyStage.VIDEO_FIRST_FRAME: (
        LatencyBoundary.VIDEO_STREAM_REQUESTED,
        LatencyBoundary.VIDEO_FIRST_FRAME_RENDERED,
    ),
    LatencyStage.EVENT_TO_OVERLAY: (
        LatencyBoundary.APP_EVENT_RECEIVED,
        LatencyBoundary.OVERLAY_RENDERED,
    ),
    LatencyStage.RECONNECT_RECOVERY: (
        LatencyBoundary.NETWORK_DISCONNECTED,
        LatencyBoundary.APP_RECOVERED,
    ),
}


class BenchmarkCounterName(StrEnum):
    INPUT_FRAMES = "INPUT_FRAMES"
    DROPPED_FRAMES = "DROPPED_FRAMES"
    ABSTENTIONS = "ABSTENTIONS"
    RECOVERY_SUCCESSES = "RECOVERY_SUCCESSES"
    RECOVERY_FAILURES = "RECOVERY_FAILURES"


class FaultKind(StrEnum):
    NETWORK_DISCONNECT = "NETWORK_DISCONNECT"
    RUNTIME_RESTART = "RUNTIME_RESTART"
    HIGH_LOAD = "HIGH_LOAD"


class FaultEffectOutcome(StrEnum):
    RECOVERED = "RECOVERED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FaultExecutionStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EnvironmentAuthorityAttestationV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    authority_id: NonEmptyStr
    attestation_id: NonEmptyStr
    evidence_scope: BenchmarkEvidenceScope
    device_lineage_sha256: Sha256
    claim_sha256: Sha256


class EnvironmentAuthorityPort(Protocol):
    def attest_environment(
        self,
        *,
        claim_sha256: str,
        evidence_scope: BenchmarkEvidenceScope,
        device_lineage_sha256: str,
    ) -> EnvironmentAuthorityAttestationV1: ...


class EnvironmentManifestV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_scope: BenchmarkEvidenceScope
    captured_at: datetime
    device_family: NonEmptyStr
    device_model: NonEmptyStr
    os_name: NonEmptyStr
    os_version: NonEmptyStr
    runtime_version: NonEmptyStr
    app_version: NonEmptyStr
    git_commit: NonEmptyStr = Field(pattern=r"^[0-9a-f]{40}$")
    scenario_revision: NonEmptyStr
    fault_matrix_revision: NonEmptyStr
    thermal_availability: ThermalAvailability
    device_lineage_sha256: Sha256
    clock_id: NonEmptyStr
    clock_timebase: Literal[ClockTimebase.MONOTONIC] = ClockTimebase.MONOTONIC
    max_clock_sync_error_ns: StrictInt = Field(ge=0)
    authority_attestation: EnvironmentAuthorityAttestationV1 | None = None
    manifest_sha256: Sha256

    _captured_at_utc = field_validator("captured_at")(require_utc)

    @model_validator(mode="after")
    def scope_has_matching_authority(self) -> EnvironmentManifestV1:
        if self.evidence_scope is BenchmarkEvidenceScope.SYNTHETIC:
            if self.authority_attestation is not None:
                raise ValueError("synthetic evidence cannot carry device authority")
            if self.thermal_availability is not ThermalAvailability.UNAVAILABLE:
                raise ValueError("synthetic evidence cannot claim a thermal authority")
            return self
        if self.authority_attestation is None:
            raise ValueError("non-synthetic evidence requires environment authority")
        if (
            self.authority_attestation.evidence_scope is not self.evidence_scope
            or self.authority_attestation.device_lineage_sha256 != self.device_lineage_sha256
        ):
            raise ValueError("environment authority attestation does not match claim")
        claim = self.model_dump(mode="json", exclude={"authority_attestation", "manifest_sha256"})
        expected_claim_sha256 = _domain_hash("runtime-benchmark-environment-claim/v1", claim)
        if self.authority_attestation.claim_sha256 != expected_claim_sha256:
            raise ValueError("environment authority attestation claim hash mismatch")
        return self


class LatencySampleV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    sample_id: NonEmptyStr
    shot_sequence: StrictInt = Field(ge=0)
    stage: LatencyStage
    start_boundary: LatencyBoundary
    end_boundary: LatencyBoundary
    started_monotonic_ns: StrictInt = Field(ge=0)
    completed_monotonic_ns: StrictInt = Field(ge=0)
    duration_ns: StrictInt = Field(gt=0)
    clock_id: NonEmptyStr
    clock_timebase: Literal[ClockTimebase.MONOTONIC] = ClockTimebase.MONOTONIC
    device_lineage_sha256: Sha256
    clock_sync_error_ns: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def timeline_and_boundaries_are_exact(self) -> LatencySampleV1:
        if self.completed_monotonic_ns <= self.started_monotonic_ns:
            raise ValueError("latency sample timeline must advance")
        if self.duration_ns != self.completed_monotonic_ns - self.started_monotonic_ns:
            raise ValueError("latency sample duration does not match timeline")
        if (self.start_boundary, self.end_boundary) != _STAGE_BOUNDARIES[self.stage]:
            raise ValueError("latency stage boundaries do not match frozen definition")
        return self


class StageLatencyStatisticsV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    stage: LatencyStage
    count: StrictInt = Field(gt=0)
    minimum_ns: StrictInt = Field(gt=0)
    p50_ns: StrictInt = Field(gt=0)
    p95_ns: StrictInt = Field(gt=0)
    p99_ns: StrictInt = Field(gt=0)
    maximum_ns: StrictInt = Field(gt=0)
    mean_floor_ns: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def quantiles_are_ordered(self) -> StageLatencyStatisticsV1:
        if not (self.minimum_ns <= self.p50_ns <= self.p95_ns <= self.p99_ns <= self.maximum_ns):
            raise ValueError("latency statistics are not ordered")
        return self


class BenchmarkCounter(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    name: BenchmarkCounterName
    value: StrictInt = Field(ge=0)


class ThermalReadingV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    reading_id: NonEmptyStr
    captured_monotonic_ns: StrictInt = Field(ge=0)
    celsius_milli: StrictInt = Field(ge=-100_000, le=200_000)
    source: Literal[ThermalAvailability.OS_SENSOR, ThermalAvailability.EXTERNAL_PROBE]
    sensor_id: NonEmptyStr


class ThermalSampleResultV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    reading: ThermalReadingV1 | None = None
    failure_reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def exactly_one_result_is_present(self) -> ThermalSampleResultV1:
        if (self.reading is None) == (self.failure_reason is None):
            raise ValueError("thermal result must contain reading xor failure reason")
        return self


class ThermalSamplingAttemptV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    attempt_id: NonEmptyStr
    ordinal: StrictInt = Field(gt=0)
    captured_monotonic_ns: StrictInt = Field(ge=0)
    authority: Literal[ThermalAvailability.OS_SENSOR, ThermalAvailability.EXTERNAL_PROBE]
    succeeded: StrictBool
    reading_id: NonEmptyStr | None = None
    failure_reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def outcome_is_complete(self) -> ThermalSamplingAttemptV1:
        if self.succeeded:
            if self.reading_id is None or self.failure_reason is not None:
                raise ValueError("successful thermal attempt requires only reading identity")
        elif self.failure_reason is None or self.reading_id is not None:
            raise ValueError("failed thermal attempt requires only failure reason")
        return self


class CollectorResourceBudgetV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    max_latency_samples: StrictInt = Field(gt=0)
    max_counters: StrictInt = Field(gt=0)
    max_fault_receipts: StrictInt = Field(gt=0)
    max_thermal_attempts: StrictInt = Field(gt=0)
    max_wire_bytes: StrictInt = Field(gt=0, le=MAX_BENCHMARK_EVIDENCE_BYTES)


class FaultCommandV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    fault_id: NonEmptyStr
    ordinal: StrictInt = Field(gt=0)
    kind: FaultKind
    duration_ms: StrictInt = Field(default=0, ge=0)
    load_units: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def parameters_match_entrypoint(self) -> FaultCommandV1:
        if self.kind is FaultKind.RUNTIME_RESTART:
            valid = self.duration_ms == 0 and self.load_units == 0
        elif self.kind is FaultKind.NETWORK_DISCONNECT:
            valid = self.duration_ms > 0 and self.load_units == 0
        else:
            valid = self.duration_ms > 0 and self.load_units > 0
        if not valid:
            raise ValueError("fault command parameters do not match entrypoint")
        return self


class FaultPlanV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    plan_id: NonEmptyStr
    fault_matrix_revision: NonEmptyStr
    commands: tuple[FaultCommandV1, ...] = Field(min_length=1)
    plan_sha256: Sha256

    @model_validator(mode="after")
    def commands_have_canonical_identity_and_order(self) -> FaultPlanV1:
        if tuple(command.ordinal for command in self.commands) != tuple(
            range(1, len(self.commands) + 1)
        ):
            raise ValueError("fault plan ordinals must be canonical and contiguous")
        ids = tuple(command.fault_id for command in self.commands)
        if len(ids) != len(set(ids)):
            raise ValueError("fault plan identities must be unique")
        return self


class FaultSafetyPolicyV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    policy_id: NonEmptyStr
    policy_revision: NonEmptyStr
    max_network_disconnect_ms: StrictInt = Field(gt=0)
    max_runtime_restart_ms: StrictInt = Field(gt=0)
    max_high_load_duration_ms: StrictInt = Field(gt=0)
    max_high_load_units: StrictInt = Field(gt=0)
    realized_duration_tolerance_ms: StrictInt = Field(ge=0)
    policy_sha256: Sha256


class FaultEffectCaptureV1(CoreContractModel):
    """Environment result before evidence bytes are digested by the harness."""

    schema_version: Literal["1.0"] = "1.0"
    started_monotonic_ns: StrictInt = Field(ge=0)
    completed_monotonic_ns: StrictInt = Field(ge=0)
    outcome: FaultEffectOutcome
    failure_code: NonEmptyStr | None = None
    evidence_reference_id: NonEmptyStr
    evidence_content: bytes = Field(min_length=1, max_length=1_048_576)

    @model_validator(mode="after")
    def outcome_and_timeline_are_valid(self) -> FaultEffectCaptureV1:
        if self.completed_monotonic_ns < self.started_monotonic_ns:
            raise ValueError("fault effect timeline is invalid")
        if (self.outcome is FaultEffectOutcome.FAILED) != (self.failure_code is not None):
            raise ValueError("fault failure outcome and code must agree")
        return self


class EvidenceReferenceV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    reference_id: NonEmptyStr
    content_sha256: Sha256


class FaultEffectV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    started_monotonic_ns: StrictInt = Field(ge=0)
    completed_monotonic_ns: StrictInt = Field(ge=0)
    realized_duration_ms: StrictInt = Field(ge=0)
    outcome: FaultEffectOutcome
    failure_code: NonEmptyStr | None = None
    evidence_reference: EvidenceReferenceV1

    @model_validator(mode="after")
    def timeline_and_failure_are_valid(self) -> FaultEffectV1:
        if self.completed_monotonic_ns < self.started_monotonic_ns:
            raise ValueError("fault effect timeline is invalid")
        if (
            self.realized_duration_ms
            != (self.completed_monotonic_ns - self.started_monotonic_ns) // 1_000_000
        ):
            raise ValueError("fault realized duration does not match timeline")
        if (self.outcome is FaultEffectOutcome.FAILED) != (self.failure_code is not None):
            raise ValueError("fault failure outcome and code must agree")
        return self


class FaultReceiptV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    plan_id: NonEmptyStr
    plan_sha256: Sha256
    fault_matrix_revision: NonEmptyStr
    command: FaultCommandV1
    command_sha256: Sha256
    safety_policy_sha256: Sha256
    effect: FaultEffectV1
    content_sha256: Sha256

    @property
    def fault_id(self) -> str:
        return self.command.fault_id

    @property
    def ordinal(self) -> int:
        return self.command.ordinal

    @property
    def kind(self) -> FaultKind:
        return self.command.kind


class FaultExecutionResultV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    status: FaultExecutionStatus
    plan: FaultPlanV1
    safety_policy: FaultSafetyPolicyV1
    completed_receipts: tuple[FaultReceiptV1, ...]
    failed_command: FaultCommandV1 | None = None
    failed_receipt: FaultReceiptV1 | None = None

    @model_validator(mode="after")
    def result_is_complete_and_fail_stop(self) -> FaultExecutionResultV1:
        completed_ordinals = tuple(receipt.ordinal for receipt in self.completed_receipts)
        if completed_ordinals != tuple(range(1, len(completed_ordinals) + 1)):
            raise ValueError("completed fault receipts must be a contiguous prefix")
        if self.status is FaultExecutionStatus.COMPLETED:
            if self.failed_command is not None or self.failed_receipt is not None:
                raise ValueError("completed fault execution cannot contain failure")
            if len(self.completed_receipts) != len(self.plan.commands):
                raise ValueError("completed fault execution must cover the plan")
        else:
            if self.failed_command is None or self.failed_receipt is None:
                raise ValueError("failed fault execution requires command and receipt")
            expected_ordinal = len(self.completed_receipts) + 1
            if (
                self.failed_command.ordinal != expected_ordinal
                or self.failed_receipt.command != self.failed_command
                or self.failed_receipt.effect.outcome is not FaultEffectOutcome.FAILED
            ):
                raise ValueError("failed fault execution violates fail-stop ordering")
        receipts = self.completed_receipts + (
            (self.failed_receipt,) if self.failed_receipt is not None else ()
        )
        for receipt in receipts:
            if receipt.ordinal > len(self.plan.commands):
                raise ValueError("fault receipt ordinal is outside exact plan")
            command = self.plan.commands[receipt.ordinal - 1]
            if (
                receipt.plan_id != self.plan.plan_id
                or receipt.plan_sha256 != self.plan.plan_sha256
                or receipt.fault_matrix_revision != self.plan.fault_matrix_revision
                or receipt.safety_policy_sha256 != self.safety_policy.policy_sha256
                or receipt.command != command
            ):
                raise ValueError("fault receipt is not bound to exact plan and safety policy")
        return self


class BenchmarkEvidenceV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: NonEmptyStr
    environment: EnvironmentManifestV1
    run_started_monotonic_ns: StrictInt = Field(ge=0)
    run_completed_monotonic_ns: StrictInt = Field(ge=0)
    run_elapsed_ns: StrictInt = Field(gt=0)
    expected_duration_ns: StrictInt = Field(gt=0)
    latency_samples: tuple[LatencySampleV1, ...]
    stage_statistics: tuple[StageLatencyStatisticsV1, ...]
    counters: tuple[BenchmarkCounter, ...]
    fault_execution: FaultExecutionResultV1 | None
    fault_receipts: tuple[FaultReceiptV1, ...]
    thermal_attempts: tuple[ThermalSamplingAttemptV1, ...]
    thermal_readings: tuple[ThermalReadingV1, ...]
    gate_disposition: Literal[GateDisposition.NOT_EVALUATED] = GateDisposition.NOT_EVALUATED
    missing_field_gates: tuple[MissingFieldGate, ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def all_evidence_is_consistent(self) -> BenchmarkEvidenceV1:
        if self.run_completed_monotonic_ns <= self.run_started_monotonic_ns:
            raise ValueError("benchmark run timeline must advance")
        if self.run_elapsed_ns != self.run_completed_monotonic_ns - self.run_started_monotonic_ns:
            raise ValueError("benchmark run elapsed does not match timeline")
        if self.missing_field_gates != tuple(MissingFieldGate):
            raise ValueError("field gate gaps must remain explicit")
        _validate_latency_evidence(self)
        _validate_counters(self.counters)
        _validate_fault_evidence(self)
        _validate_thermal_evidence(self)
        return self


def _strict_revalidate[ModelT: CoreContractModel](
    value: ModelT, model_type: type[ModelT], label: str
) -> ModelT:
    if type(value) is not model_type:
        raise TypeError(f"{label} must be an exact {model_type.__name__}")
    try:
        return model_type.model_validate(
            value.model_dump(mode="python", warnings=False), strict=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc


def _validate_latency_evidence(evidence: BenchmarkEvidenceV1) -> None:
    stage_order = {stage: ordinal for ordinal, stage in enumerate(LatencyStage)}
    expected_samples = tuple(
        sorted(
            evidence.latency_samples,
            key=lambda sample: (stage_order[sample.stage], sample.shot_sequence, sample.sample_id),
        )
    )
    if evidence.latency_samples != expected_samples:
        raise ValueError("latency samples must have canonical order")
    if len({sample.sample_id for sample in evidence.latency_samples}) != len(
        evidence.latency_samples
    ):
        raise ValueError("latency sample identities must be unique")
    if len({(sample.shot_sequence, sample.stage) for sample in evidence.latency_samples}) != len(
        evidence.latency_samples
    ):
        raise ValueError("each shot may contain only one sample per latency stage")
    for sample in evidence.latency_samples:
        if not (
            evidence.run_started_monotonic_ns
            <= sample.started_monotonic_ns
            < sample.completed_monotonic_ns
            <= evidence.run_completed_monotonic_ns
        ):
            raise ValueError("latency sample is outside benchmark run window")
        if (
            sample.clock_id != evidence.environment.clock_id
            or sample.clock_timebase is not evidence.environment.clock_timebase
            or sample.device_lineage_sha256 != evidence.environment.device_lineage_sha256
            or sample.clock_sync_error_ns > evidence.environment.max_clock_sync_error_ns
        ):
            raise ValueError("latency sample clock or device lineage does not match environment")
    expected_statistics = tuple(
        _calculate_statistics(
            stage,
            tuple(sample for sample in evidence.latency_samples if sample.stage is stage),
        )
        for stage in LatencyStage
        if any(sample.stage is stage for sample in evidence.latency_samples)
    )
    if evidence.stage_statistics != expected_statistics:
        raise ValueError("stage statistics do not match latency samples")
    for shot in {sample.shot_sequence for sample in evidence.latency_samples}:
        by_stage = {
            sample.stage: sample
            for sample in evidence.latency_samples
            if sample.shot_sequence == shot
        }
        if LatencyStage.END_TO_END not in by_stage:
            continue
        chain_stages = (
            LatencyStage.PERCEPTION_OBSERVATION,
            LatencyStage.POLICY_DECISION,
            LatencyStage.FEEDBACK_BUILD,
            LatencyStage.EVENT_TRANSPORT_TO_APP,
        )
        if any(stage not in by_stage for stage in chain_stages):
            raise ValueError("END_TO_END sample requires a complete same-shot stage chain")
        chain = tuple(by_stage[stage] for stage in chain_stages)
        total = by_stage[LatencyStage.END_TO_END]
        if not _within_sync(
            total.started_monotonic_ns, chain[0].started_monotonic_ns, total, chain[0]
        ):
            raise ValueError("same-shot END_TO_END start does not close stage chain")
        if not _within_sync(
            total.completed_monotonic_ns, chain[-1].completed_monotonic_ns, total, chain[-1]
        ):
            raise ValueError("same-shot END_TO_END completion does not close stage chain")
        for previous, following in zip(chain, chain[1:], strict=False):
            if previous.end_boundary is not following.start_boundary or not _within_sync(
                previous.completed_monotonic_ns,
                following.started_monotonic_ns,
                previous,
                following,
            ):
                raise ValueError("same-shot latency stage chain is not closed")


def _within_sync(left: int, right: int, *samples: LatencySampleV1) -> bool:
    return abs(left - right) <= sum(sample.clock_sync_error_ns for sample in samples)


def _validate_counters(counters: tuple[BenchmarkCounter, ...]) -> None:
    counter_order = {name: ordinal for ordinal, name in enumerate(BenchmarkCounterName)}
    if counters != tuple(sorted(counters, key=lambda counter: counter_order[counter.name])):
        raise ValueError("benchmark counters must have canonical order")
    if len({counter.name for counter in counters}) != len(counters):
        raise ValueError("benchmark counters must be unique")


def _validate_fault_evidence(evidence: BenchmarkEvidenceV1) -> None:
    if evidence.fault_execution is None:
        if evidence.fault_receipts:
            raise ValueError("fault receipts require typed fault execution result")
        return
    execution = evidence.fault_execution
    if execution.plan.fault_matrix_revision != evidence.environment.fault_matrix_revision:
        raise ValueError("fault execution matrix revision does not match environment")
    receipts = execution.completed_receipts + (
        (execution.failed_receipt,) if execution.failed_receipt is not None else ()
    )
    if evidence.fault_receipts != receipts:
        raise ValueError("fault receipts do not match typed execution result")
    previous_completion = evidence.run_started_monotonic_ns
    for receipt in receipts:
        if receipt.effect.started_monotonic_ns < previous_completion:
            raise ValueError("fault receipt chronology is not monotonic")
        if receipt.effect.completed_monotonic_ns > evidence.run_completed_monotonic_ns:
            raise ValueError("fault receipt is outside benchmark run window")
        previous_completion = receipt.effect.completed_monotonic_ns


def _validate_thermal_evidence(evidence: BenchmarkEvidenceV1) -> None:
    if evidence.environment.thermal_availability is ThermalAvailability.UNAVAILABLE:
        if evidence.thermal_attempts or evidence.thermal_readings:
            raise ValueError("unavailable thermal authority cannot produce attempts or readings")
        return
    if tuple(attempt.ordinal for attempt in evidence.thermal_attempts) != tuple(
        range(1, len(evidence.thermal_attempts) + 1)
    ):
        raise ValueError("thermal attempts must have contiguous order")
    if len({attempt.attempt_id for attempt in evidence.thermal_attempts}) != len(
        evidence.thermal_attempts
    ):
        raise ValueError("thermal attempt identities must be unique")
    expected_attempts = tuple(
        sorted(
            evidence.thermal_attempts,
            key=lambda attempt: (attempt.captured_monotonic_ns, attempt.ordinal),
        )
    )
    if evidence.thermal_attempts != expected_attempts:
        raise ValueError("thermal attempt chronology is not canonical")
    expected_readings = tuple(
        sorted(
            evidence.thermal_readings,
            key=lambda reading: (
                reading.captured_monotonic_ns,
                reading.sensor_id,
                reading.reading_id,
            ),
        )
    )
    if evidence.thermal_readings != expected_readings:
        raise ValueError("thermal readings must have canonical chronology")
    if len({reading.reading_id for reading in evidence.thermal_readings}) != len(
        evidence.thermal_readings
    ) or len(
        {
            (reading.captured_monotonic_ns, reading.sensor_id)
            for reading in evidence.thermal_readings
        }
    ) != len(evidence.thermal_readings):
        raise ValueError("thermal reading identity must be unique")
    successful_ids = {
        attempt.reading_id for attempt in evidence.thermal_attempts if attempt.succeeded
    }
    successful_id_sequence = tuple(
        attempt.reading_id for attempt in evidence.thermal_attempts if attempt.succeeded
    )
    if len(successful_id_sequence) != len(set(successful_id_sequence)):
        raise ValueError("thermal successful attempt identities must be unique")
    if successful_ids != {reading.reading_id for reading in evidence.thermal_readings}:
        raise ValueError("thermal attempts and readings do not match")
    for attempt in evidence.thermal_attempts:
        if (
            attempt.authority is not evidence.environment.thermal_availability
            or not evidence.run_started_monotonic_ns
            <= attempt.captured_monotonic_ns
            <= evidence.run_completed_monotonic_ns
        ):
            raise ValueError("thermal attempt authority or chronology is invalid")
    for reading in evidence.thermal_readings:
        if reading.source is not evidence.environment.thermal_availability:
            raise ValueError("thermal reading source does not match manifest authority")


def _nearest_rank(values: tuple[int, ...], percentile: int) -> int:
    return values[(percentile * len(values) + 99) // 100 - 1]


def _calculate_statistics(
    stage: LatencyStage,
    samples: tuple[LatencySampleV1, ...],
) -> StageLatencyStatisticsV1:
    values = tuple(sorted(sample.duration_ns for sample in samples))
    return StageLatencyStatisticsV1(
        stage=stage,
        count=len(values),
        minimum_ns=values[0],
        p50_ns=_nearest_rank(values, 50),
        p95_ns=_nearest_rank(values, 95),
        p99_ns=_nearest_rank(values, 99),
        maximum_ns=values[-1],
        mean_floor_ns=sum(values) // len(values),
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        to_jsonable_python(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _domain_hash(domain: str, value: object) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + _canonical_bytes(value)).hexdigest()


def _without_hash(model: CoreContractModel, field: str) -> dict[str, object]:
    return model.model_dump(mode="json", exclude={field})


def _environment_claim(values: dict[str, object]) -> dict[str, object]:
    claim = {
        key: to_jsonable_python(value)
        for key, value in values.items()
        if key not in {"manifest_sha256", "authority_attestation"}
    }
    claim.setdefault("schema_version", "1.0")
    claim.setdefault("clock_timebase", ClockTimebase.MONOTONIC.value)
    return claim


def issue_environment_manifest(
    *, authority: EnvironmentAuthorityPort | None = None, **values: object
) -> EnvironmentManifestV1:
    if "authority_attestation" in values:
        raise TypeError("authority attestation must come from the environment authority port")
    scope = values.get("evidence_scope")
    if type(scope) is not BenchmarkEvidenceScope:
        raise TypeError("evidence_scope must be a BenchmarkEvidenceScope")
    claim_sha256 = _domain_hash(
        "runtime-benchmark-environment-claim/v1", _environment_claim(values)
    )
    attestation = None
    if scope is not BenchmarkEvidenceScope.SYNTHETIC:
        if authority is None:
            raise ValueError("non-synthetic evidence requires environment authority")
        attestation = _strict_revalidate(
            authority.attest_environment(
                claim_sha256=claim_sha256,
                evidence_scope=scope,
                device_lineage_sha256=str(values.get("device_lineage_sha256", "")),
            ),
            EnvironmentAuthorityAttestationV1,
            "environment authority attestation",
        )
        if attestation.claim_sha256 != claim_sha256:
            raise ValueError("environment authority attestation claim hash mismatch")
    candidate = EnvironmentManifestV1.model_validate(
        {
            **values,
            "authority_attestation": attestation,
            "manifest_sha256": "0" * 64,
        },
        strict=True,
    )
    candidate = _strict_revalidate(candidate, EnvironmentManifestV1, "environment manifest")
    digest = _domain_hash(
        "runtime-benchmark-environment/v1", _without_hash(candidate, "manifest_sha256")
    )
    return EnvironmentManifestV1.model_validate(
        {**candidate.model_dump(mode="python"), "manifest_sha256": digest}, strict=True
    )


def canonical_environment_manifest_bytes(manifest: EnvironmentManifestV1) -> bytes:
    manifest = _strict_revalidate(manifest, EnvironmentManifestV1, "environment manifest")
    expected = _domain_hash(
        "runtime-benchmark-environment/v1", _without_hash(manifest, "manifest_sha256")
    )
    if manifest.manifest_sha256 != expected:
        raise ValueError("environment manifest hash mismatch")
    return _canonical_bytes(manifest.model_dump(mode="json"))


def issue_fault_safety_policy(**values: object) -> FaultSafetyPolicyV1:
    candidate = FaultSafetyPolicyV1.model_validate(
        {**values, "policy_sha256": "0" * 64}, strict=True
    )
    candidate = _strict_revalidate(candidate, FaultSafetyPolicyV1, "fault safety policy")
    digest = _domain_hash(
        "runtime-benchmark-fault-safety-policy/v1",
        _without_hash(candidate, "policy_sha256"),
    )
    return FaultSafetyPolicyV1.model_validate(
        {**candidate.model_dump(mode="python"), "policy_sha256": digest}, strict=True
    )


def canonical_fault_safety_policy_bytes(policy: FaultSafetyPolicyV1) -> bytes:
    policy = _strict_revalidate(policy, FaultSafetyPolicyV1, "fault safety policy")
    if policy.policy_sha256 != _domain_hash(
        "runtime-benchmark-fault-safety-policy/v1",
        _without_hash(policy, "policy_sha256"),
    ):
        raise ValueError("fault safety policy hash mismatch")
    return _canonical_bytes(policy.model_dump(mode="json"))


def issue_fault_plan(**values: object) -> FaultPlanV1:
    candidate = FaultPlanV1.model_validate({**values, "plan_sha256": "0" * 64}, strict=True)
    candidate = _strict_revalidate(candidate, FaultPlanV1, "fault plan")
    digest = _domain_hash(
        "runtime-benchmark-fault-plan/v1", _without_hash(candidate, "plan_sha256")
    )
    return FaultPlanV1.model_validate(
        {**candidate.model_dump(mode="python"), "plan_sha256": digest}, strict=True
    )


def canonical_fault_plan_bytes(plan: FaultPlanV1) -> bytes:
    plan = _strict_revalidate(plan, FaultPlanV1, "fault plan")
    if plan.plan_sha256 != _domain_hash(
        "runtime-benchmark-fault-plan/v1", _without_hash(plan, "plan_sha256")
    ):
        raise ValueError("fault plan hash mismatch")
    return _canonical_bytes(plan.model_dump(mode="json"))


def _command_hash(command: FaultCommandV1) -> str:
    command = _strict_revalidate(command, FaultCommandV1, "fault command")
    return _domain_hash("runtime-benchmark-fault-command/v1", command.model_dump(mode="json"))


def issue_evidence_reference(*, reference_id: str, content: bytes) -> EvidenceReferenceV1:
    if type(content) is not bytes or not content:
        raise ValueError("fault evidence content must be non-empty bytes")
    return EvidenceReferenceV1.model_validate(
        {
            "reference_id": reference_id,
            "content_sha256": hashlib.sha256(content).hexdigest(),
        },
        strict=True,
    )


def _fault_receipt_hash(receipt: FaultReceiptV1) -> str:
    return _domain_hash(
        "runtime-benchmark-fault-receipt/v1", _without_hash(receipt, "content_sha256")
    )


def issue_fault_receipt(**values: object) -> FaultReceiptV1:
    candidate = FaultReceiptV1.model_validate({**values, "content_sha256": "0" * 64}, strict=True)
    candidate = _strict_revalidate(candidate, FaultReceiptV1, "fault receipt")
    return FaultReceiptV1.model_validate(
        {
            **candidate.model_dump(mode="python"),
            "content_sha256": _fault_receipt_hash(candidate),
        },
        strict=True,
    )


def canonical_fault_receipt_bytes(receipt: FaultReceiptV1) -> bytes:
    receipt = _strict_revalidate(receipt, FaultReceiptV1, "fault receipt")
    if receipt.command_sha256 != _command_hash(receipt.command):
        raise ValueError("fault receipt command hash mismatch")
    if receipt.content_sha256 != _fault_receipt_hash(receipt):
        raise ValueError("fault receipt content hash mismatch")
    return _canonical_bytes(receipt.model_dump(mode="json"))


def _evidence_hash(evidence: BenchmarkEvidenceV1) -> str:
    return _domain_hash("runtime-benchmark-evidence/v1", _without_hash(evidence, "content_sha256"))


def issue_benchmark_evidence(**values: object) -> BenchmarkEvidenceV1:
    candidate = BenchmarkEvidenceV1.model_validate(
        {**values, "content_sha256": "0" * 64}, strict=True
    )
    candidate = _strict_revalidate(candidate, BenchmarkEvidenceV1, "benchmark evidence")
    return BenchmarkEvidenceV1.model_validate(
        {**candidate.model_dump(mode="python"), "content_sha256": _evidence_hash(candidate)},
        strict=True,
    )


def canonical_benchmark_evidence_bytes(evidence: BenchmarkEvidenceV1) -> bytes:
    evidence = _strict_revalidate(evidence, BenchmarkEvidenceV1, "benchmark evidence")
    canonical_environment_manifest_bytes(evidence.environment)
    if evidence.fault_execution is not None:
        canonical_fault_plan_bytes(evidence.fault_execution.plan)
        canonical_fault_safety_policy_bytes(evidence.fault_execution.safety_policy)
    for receipt in evidence.fault_receipts:
        canonical_fault_receipt_bytes(receipt)
    if evidence.content_sha256 != _evidence_hash(evidence):
        raise ValueError("evidence content hash mismatch")
    encoded = _canonical_bytes(evidence.model_dump(mode="json"))
    if len(encoded) > MAX_BENCHMARK_EVIDENCE_BYTES:
        raise ValueError("benchmark evidence exceeds maximum wire size")
    return encoded


def _validate_json_depth(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if depth > MAX_BENCHMARK_JSON_DEPTH:
            raise ValueError("benchmark evidence JSON exceeds maximum depth")
        if isinstance(item, dict):
            stack.extend((nested, depth + 1) for nested in item.values())
        elif isinstance(item, list):
            stack.extend((nested, depth + 1) for nested in item)


def decode_benchmark_evidence(data: bytes) -> BenchmarkEvidenceV1:
    if type(data) is not bytes:
        raise TypeError("benchmark evidence input must be bytes")
    if not data or len(data) > MAX_BENCHMARK_EVIDENCE_BYTES:
        raise ValueError("invalid benchmark evidence wire size")
    try:
        parsed = json.loads(data, object_pairs_hook=_reject_duplicate_keys)
        _validate_json_depth(parsed)
        evidence = BenchmarkEvidenceV1.model_validate_json(data, strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("invalid benchmark evidence JSON") from exc
    if canonical_benchmark_evidence_bytes(evidence) != data:
        raise ValueError("benchmark evidence JSON must be canonical")
    return evidence
