from __future__ import annotations

from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from billiards_ai.edge.coach_runtime import VerifiedPlanAdmission
from billiards_ai.edge.plan_admission import (
    AntiRollbackDecision,
    PlanAdmissionFailure,
    PlanAdmissionFailureCode,
    PlanAdmissionResult,
    UntrustedPlanBundle,
)


def test_admission_result_enforces_exactly_one_nominal_branch() -> None:
    admission = VerifiedPlanAdmission(
        admission_id="bundle:verified",
        session_id=UUID("00000000-0000-4000-8000-000000000002"),
        plan_id=UUID("00000000-0000-4000-8000-00000000000c"),
        plan_version=1,
    )
    failure = PlanAdmissionFailure(
        code=PlanAdmissionFailureCode.INVALID_INPUT,
        stage="input",
    )

    assert PlanAdmissionResult(admission=admission).accepted is True
    assert PlanAdmissionResult(failure=failure).accepted is False
    with pytest.raises(ValueError, match="exactly one"):
        PlanAdmissionResult()
    with pytest.raises(ValueError, match="exactly one"):
        PlanAdmissionResult(admission=admission, failure=failure)
    with pytest.raises(TypeError, match="VerifiedPlanAdmission"):
        PlanAdmissionResult(admission="structural-admission")
    with pytest.raises(TypeError, match="PlanAdmissionFailure"):
        PlanAdmissionResult(failure="structural-failure")


def test_failure_contract_uses_stable_codes_without_sensitive_detail() -> None:
    failure = PlanAdmissionFailure(
        code=PlanAdmissionFailureCode.SIGNATURE_INVALID,
        stage="signature",
    )

    assert failure.code.value == "SIGNATURE_INVALID"
    assert failure.stage == "signature"
    assert tuple(decision.value for decision in AntiRollbackDecision) == (
        "ACCEPTED",
        "REPLAYED",
        "REJECTED",
    )
    assert "signature" not in repr(failure).lower()


def test_untrusted_wire_contract_is_frozen_and_hides_envelope_bytes() -> None:
    bundle = UntrustedPlanBundle(envelope_bytes=b"untrusted-final-v2-envelope")

    with pytest.raises(FrozenInstanceError):
        bundle.envelope_bytes = b"tampered"
    assert "untrusted-final-v2-envelope" not in repr(bundle)
