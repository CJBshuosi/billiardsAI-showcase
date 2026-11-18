"""Environment-owned entrypoints for bounded deterministic benchmark faults."""

from __future__ import annotations

from typing import Protocol

from billiards_ai.edge.benchmark.models import (
    FaultCommandV1,
    FaultEffectCaptureV1,
    FaultEffectOutcome,
    FaultEffectV1,
    FaultExecutionResultV1,
    FaultExecutionStatus,
    FaultKind,
    FaultPlanV1,
    FaultReceiptV1,
    FaultSafetyPolicyV1,
    _command_hash,
    _strict_revalidate,
    canonical_fault_plan_bytes,
    canonical_fault_safety_policy_bytes,
    issue_evidence_reference,
    issue_fault_receipt,
)


class FaultInjectionPort(Protocol):
    @property
    def safety_policy(self) -> FaultSafetyPolicyV1: ...

    def disconnect_network(self, *, duration_ms: int) -> FaultEffectCaptureV1: ...

    def restart_runtime(self) -> FaultEffectCaptureV1: ...

    def apply_high_load(self, *, duration_ms: int, load_units: int) -> FaultEffectCaptureV1: ...


def _validate_command_safety(command: FaultCommandV1, policy: FaultSafetyPolicyV1) -> None:
    if command.kind is FaultKind.NETWORK_DISCONNECT:
        safe = command.duration_ms <= policy.max_network_disconnect_ms
    elif command.kind is FaultKind.RUNTIME_RESTART:
        safe = True
    else:
        safe = (
            command.duration_ms <= policy.max_high_load_duration_ms
            and command.load_units <= policy.max_high_load_units
        )
    if not safe:
        raise ValueError("fault command exceeds environment safety policy")


def _invoke(command: FaultCommandV1, port: FaultInjectionPort) -> FaultEffectCaptureV1:
    if command.kind is FaultKind.NETWORK_DISCONNECT:
        result = port.disconnect_network(duration_ms=command.duration_ms)
    elif command.kind is FaultKind.RUNTIME_RESTART:
        result = port.restart_runtime()
    else:
        result = port.apply_high_load(
            duration_ms=command.duration_ms, load_units=command.load_units
        )
    return _strict_revalidate(result, FaultEffectCaptureV1, "fault effect capture")


def _validate_effect(
    command: FaultCommandV1,
    capture: FaultEffectCaptureV1,
    policy: FaultSafetyPolicyV1,
) -> None:
    realized_ms = (capture.completed_monotonic_ns - capture.started_monotonic_ns) // 1_000_000
    if command.kind is FaultKind.RUNTIME_RESTART:
        duration_valid = realized_ms <= policy.max_runtime_restart_ms
        outcome_valid = capture.outcome in {
            FaultEffectOutcome.RECOVERED,
            FaultEffectOutcome.FAILED,
        }
    else:
        duration_valid = (
            abs(realized_ms - command.duration_ms) <= policy.realized_duration_tolerance_ms
        )
        expected_outcome = (
            FaultEffectOutcome.RECOVERED
            if command.kind is FaultKind.NETWORK_DISCONNECT
            else FaultEffectOutcome.COMPLETED
        )
        outcome_valid = capture.outcome in {expected_outcome, FaultEffectOutcome.FAILED}
    if not duration_valid:
        raise ValueError("fault realized duration exceeds environment tolerance")
    if not outcome_valid:
        raise ValueError("fault effect outcome does not match command kind")


def _receipt(
    *,
    plan: FaultPlanV1,
    policy: FaultSafetyPolicyV1,
    command: FaultCommandV1,
    capture: FaultEffectCaptureV1,
) -> FaultReceiptV1:
    effect = FaultEffectV1(
        started_monotonic_ns=capture.started_monotonic_ns,
        completed_monotonic_ns=capture.completed_monotonic_ns,
        realized_duration_ms=(capture.completed_monotonic_ns - capture.started_monotonic_ns)
        // 1_000_000,
        outcome=capture.outcome,
        failure_code=capture.failure_code,
        evidence_reference=issue_evidence_reference(
            reference_id=capture.evidence_reference_id,
            content=capture.evidence_content,
        ),
    )
    return issue_fault_receipt(
        plan_id=plan.plan_id,
        plan_sha256=plan.plan_sha256,
        fault_matrix_revision=plan.fault_matrix_revision,
        command=command,
        command_sha256=_command_hash(command),
        safety_policy_sha256=policy.policy_sha256,
        effect=effect,
    )


def execute_fault_plan(
    plan: FaultPlanV1,
    port: FaultInjectionPort,
) -> FaultExecutionResultV1:
    """Execute one exact bounded plan, preserving typed partial failure evidence."""

    plan = _strict_revalidate(plan, FaultPlanV1, "fault plan")
    canonical_fault_plan_bytes(plan)
    policy = _strict_revalidate(port.safety_policy, FaultSafetyPolicyV1, "fault safety policy")
    canonical_fault_safety_policy_bytes(policy)
    # Validate the whole plan before the first real entrypoint can be touched.
    for command in plan.commands:
        _validate_command_safety(command, policy)

    completed: list[FaultReceiptV1] = []
    previous_completion: int | None = None
    for command in plan.commands:
        capture = _invoke(command, port)
        _validate_effect(command, capture, policy)
        if previous_completion is not None and capture.started_monotonic_ns < previous_completion:
            raise ValueError("fault effects violate plan chronology")
        receipt = _receipt(plan=plan, policy=policy, command=command, capture=capture)
        previous_completion = capture.completed_monotonic_ns
        if capture.outcome is FaultEffectOutcome.FAILED:
            return FaultExecutionResultV1(
                status=FaultExecutionStatus.FAILED,
                plan=plan,
                safety_policy=policy,
                completed_receipts=tuple(completed),
                failed_command=command,
                failed_receipt=receipt,
            )
        completed.append(receipt)
    return FaultExecutionResultV1(
        status=FaultExecutionStatus.COMPLETED,
        plan=plan,
        safety_policy=policy,
        completed_receipts=tuple(completed),
    )
