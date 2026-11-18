"""Story 2.8 RED matrix for the optional local adjudication orchestrator."""

from __future__ import annotations

import socket
from decimal import Decimal
from uuid import uuid4

import pytest


def _segment(before: dict[int, tuple[int, int]], after: dict[int, tuple[int, int]]):
    from billiards_ai.shared.models.geometry import Point
    from billiards_ai.shared.models.segmentation import BallPosition, ShotSegment, TableSnapshot

    def snapshot(timestamp_ms: int, positions: dict[int, tuple[int, int]]):
        return TableSnapshot(
            timestamp_ms=timestamp_ms,
            positions=tuple(
                BallPosition(ball_id=ball_id, position=Point(x_mm=x_mm, y_mm=y_mm))
                for ball_id, (x_mm, y_mm) in sorted(positions.items())
            ),
        )

    return ShotSegment(
        t_start_ms=500,
        t_end_ms=900,
        ball_positions_before=snapshot(400, before),
        ball_positions_after=snapshot(900, after),
    )


def _ambiguous_observation():
    from billiards_ai.cloud.perception.observer import observe_shot

    return observe_shot(
        _segment(
            before={0: (100, 100), 1: (400, 400), 2: (700, 700)},
            after={0: (200, 100), 1: (450, 400), 2: (750, 700)},
        )
    )


def _certain_observation():
    from billiards_ai.shared.models.observation import (
        ObservationConfidence,
        ObservationField,
        ObservationSource,
        RawObservation,
    )

    observation = _ambiguous_observation()
    return observation.model_copy(
        update={
            "struck_ball": ObservationField(
                value=1,
                confidence=ObservationConfidence.HIGH,
                source=ObservationSource.SEGMENT_SNAPSHOT_DELTA,
            ),
            "raw_observation": RawObservation(),
        }
    )


def _single_low_confidence_observation():
    from billiards_ai.cloud.perception.observer import observe_shot

    return observe_shot(
        _segment(
            before={0: (100, 100), 1: (400, 400), 2: (700, 700)},
            after={0: (200, 100), 1: (450, 400), 2: (700, 700)},
        )
    )


class ApprovalReader:
    def __init__(self, approval=None, error: Exception | None = None):
        self.approval = approval
        self.error = error
        self.calls = 0

    def find_active_provider_approval(self, provider_id: str):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.approval is None or self.approval.provider_id != provider_id:
            return None
        return self.approval


class FakeProvider:
    provider_id = "synthetic-provider"

    def __init__(self, output: str = '{"ball_id":1,"confidence":"MEDIUM"}'):
        self.output = output
        self.requests = []
        self.error = None

    def adjudicate(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.output


def _approval(**changes):
    from billiards_ai.shared.models.access import ThirdPartyProcessingApproval
    from billiards_ai.shared.models.privacy import MaterialUsePurpose

    values = {
        "approval_id": uuid4(),
        "provider_id": "synthetic-provider",
        "endpoint_region": "cn-test",
        "data_scope": "KEYFRAME",
        "purpose": MaterialUsePurpose.SERVICE_ANALYSIS,
        "cross_border": False,
        "cross_border_assessment": "synthetic fixture; no transfer",
        "max_cost_cny": Decimal("1.00"),
        "approved_by": "test-admin",
        "approved_at": "2026-07-29T00:00:00Z",
        "approval_evidence_sha256": "d" * 64,
    }
    values.update(changes)
    return ThirdPartyProcessingApproval(**values)


def _enabled_policy(**changes):
    from billiards_ai.cloud.perception.adjudicator import AdjudicationPolicy
    from billiards_ai.shared.models.privacy import MaterialUsePurpose

    values = {
        "enabled": True,
        "provider_id": "synthetic-provider",
        "endpoint_region": "cn-test",
        "purpose": MaterialUsePurpose.SERVICE_ANALYSIS,
        "cross_border": False,
        "estimated_cost_cny": Decimal("0.50"),
        "max_cost_per_shot_cny": Decimal("1.00"),
    }
    values.update(changes)
    return AdjudicationPolicy(**values)


def _service(*, policy=None, provider=None, approvals=None, ledger=None):
    from billiards_ai.cloud.perception.adjudicator import AmbiguityAdjudicator

    return AmbiguityAdjudicator(
        policy=policy,
        provider=provider,
        approval_reader=approvals,
        ledger=ledger,
    )


def test_default_disabled_path_has_no_network_or_approval_access(monkeypatch):
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: pytest.fail("disabled adjudicator must not use network"),
    )
    approvals = ApprovalReader(error=AssertionError("approval store must not be read"))

    result = _service(approvals=approvals).adjudicate(
        shot_id=uuid4(), observation=_ambiguous_observation()
    )

    assert result.status is AdjudicationStatus.DISABLED
    assert result.result is None
    assert approvals.calls == 0


def test_high_confidence_cv_result_never_calls_provider_or_approval_store():
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    provider = FakeProvider()
    approvals = ApprovalReader(error=AssertionError("approval store must not be read"))

    result = _service(policy=_enabled_policy(), provider=provider, approvals=approvals).adjudicate(
        shot_id=uuid4(), observation=_certain_observation()
    )

    assert result.status is AdjudicationStatus.NOT_REQUIRED
    assert provider.requests == []
    assert approvals.calls == 0


@pytest.mark.parametrize(
    ("approval_changes", "missing", "reader_error", "expected_status"),
    [
        ({}, True, None, "APPROVAL_REQUIRED"),
        ({"endpoint_region": "eu-test"}, False, None, "APPROVAL_MISMATCH"),
        ({"purpose": "TRAINING"}, False, None, "APPROVAL_MISMATCH"),
        ({"cross_border": True}, False, None, "APPROVAL_MISMATCH"),
        ({}, True, OSError("store unavailable"), "APPROVAL_REQUIRED"),
    ],
)
def test_approval_validation_fails_closed_before_provider(
    approval_changes, missing, reader_error, expected_status
):
    provider = FakeProvider()
    approval = None if missing else _approval(**approval_changes)

    result = _service(
        policy=_enabled_policy(),
        provider=provider,
        approvals=ApprovalReader(approval, reader_error),
    ).adjudicate(shot_id=uuid4(), observation=_ambiguous_observation())

    assert result.status.value == expected_status
    assert result.result is None
    assert provider.requests == []


@pytest.mark.parametrize("reader_error", [RuntimeError("unexpected"), AssertionError("broken")])
def test_approval_reader_unexpected_error_fails_closed_before_provider(reader_error):
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    provider = FakeProvider()
    result = _service(
        policy=_enabled_policy(),
        provider=provider,
        approvals=ApprovalReader(_approval(), reader_error),
    ).adjudicate(shot_id=uuid4(), observation=_ambiguous_observation())

    assert result.status is AdjudicationStatus.APPROVAL_REQUIRED
    assert provider.requests == []


def test_revoked_approval_fails_closed_before_provider():
    from datetime import UTC, datetime

    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    provider = FakeProvider()
    result = _service(
        policy=_enabled_policy(),
        provider=provider,
        approvals=ApprovalReader(
            _approval(
                revoked_at=datetime(2026, 7, 29, 1, tzinfo=UTC),
                revocation_reason="withdrawn",
            )
        ),
    ).adjudicate(shot_id=uuid4(), observation=_ambiguous_observation())

    assert result.status is AdjudicationStatus.APPROVAL_REQUIRED
    assert provider.requests == []


@pytest.mark.parametrize(
    ("policy_changes", "approval_changes"),
    [
        ({"estimated_cost_cny": Decimal("1.01")}, {}),
        (
            {
                "estimated_cost_cny": Decimal("0.75"),
                "max_cost_per_shot_cny": Decimal("0.50"),
            },
            {},
        ),
        ({"estimated_cost_cny": Decimal("0.75")}, {"max_cost_cny": "0.50"}),
    ],
)
def test_cost_limit_is_enforced_before_provider(policy_changes, approval_changes):
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    provider = FakeProvider()
    result = _service(
        policy=_enabled_policy(**policy_changes),
        provider=provider,
        approvals=ApprovalReader(_approval(**approval_changes)),
    ).adjudicate(shot_id=uuid4(), observation=_ambiguous_observation())

    assert result.status is AdjudicationStatus.COST_LIMIT_EXCEEDED
    assert provider.requests == []


def test_success_passes_only_minimal_evidence_and_keeps_raw_text_out_of_result():
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    provider = FakeProvider(
        '{"ball_id":1,"confidence":"MEDIUM","free_text":"implementation-only prose"}'
    )
    shot_id = uuid4()
    result = _service(
        policy=_enabled_policy(), provider=provider, approvals=ApprovalReader(_approval())
    ).adjudicate(shot_id=shot_id, observation=_ambiguous_observation())

    assert result.status is AdjudicationStatus.SUCCEEDED
    assert result.result.ball_id == 1
    assert "implementation-only prose" not in repr(result.model_dump(exclude={"raw_observation"}))
    assert "implementation-only prose" in result.raw_observation.provider_output
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.shot_id == shot_id
    assert tuple(candidate.ball_id for candidate in request.candidates) == (1, 2)
    assert not hasattr(request, "frame")
    assert not hasattr(request, "ball_positions_before")


def test_one_low_confidence_candidate_triggers_adjudication():
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    provider = FakeProvider()
    result = _service(
        policy=_enabled_policy(), provider=provider, approvals=ApprovalReader(_approval())
    ).adjudicate(shot_id=uuid4(), observation=_single_low_confidence_observation())

    assert result.status is AdjudicationStatus.SUCCEEDED
    assert tuple(candidate.ball_id for candidate in provider.requests[0].candidates) == (1,)


@pytest.mark.parametrize(
    "output",
    [
        "not-json",
        '{"ball_id":9,"confidence":"HIGH"}',
        '{"ball_id":1,"confidence":"UNKNOWN"}',
    ],
)
def test_unparseable_or_unapproved_candidate_output_is_a_non_retryable_failure(output):
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    result = _service(
        policy=_enabled_policy(),
        provider=FakeProvider(output),
        approvals=ApprovalReader(_approval()),
    ).adjudicate(shot_id=uuid4(), observation=_ambiguous_observation())

    assert result.status is AdjudicationStatus.PARSE_FAILED
    assert result.retryable is False
    assert result.result is None
    assert result.raw_observation.provider_output == output


@pytest.mark.parametrize("retryable", [False, True])
def test_provider_failure_preserves_typed_retryability(retryable):
    from billiards_ai.cloud.perception.adjudicator import ProviderAdjudicationError
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    provider = FakeProvider()
    provider.error = ProviderAdjudicationError("synthetic failure", retryable=retryable)
    result = _service(
        policy=_enabled_policy(), provider=provider, approvals=ApprovalReader(_approval())
    ).adjudicate(shot_id=uuid4(), observation=_ambiguous_observation())

    assert result.status is AdjudicationStatus.PROVIDER_FAILED
    assert result.retryable is retryable
    assert result.result is None
    assert "synthetic failure" not in repr(result.model_dump(exclude={"raw_observation"}))
    assert result.raw_observation.failure_detail == "synthetic failure"


def test_retryable_failure_can_retry_with_the_same_key_within_the_shot_cost_limit():
    from billiards_ai.cloud.perception.adjudicator import ProviderAdjudicationError
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    provider = FakeProvider()
    provider.error = ProviderAdjudicationError("temporary", retryable=True)
    service = _service(
        policy=_enabled_policy(), provider=provider, approvals=ApprovalReader(_approval())
    )
    shot_id = uuid4()
    observation = _ambiguous_observation()

    first = service.adjudicate(shot_id=shot_id, observation=observation)
    provider.error = None
    second = service.adjudicate(shot_id=shot_id, observation=observation)
    third = service.adjudicate(shot_id=shot_id, observation=observation)

    assert first.status is AdjudicationStatus.PROVIDER_FAILED
    assert first.retryable is True
    assert second.status is AdjudicationStatus.SUCCEEDED
    assert third == second
    assert len(provider.requests) == 2
    assert provider.requests[0].idempotency_key == provider.requests[1].idempotency_key


def test_untyped_provider_exception_is_contained_as_non_retryable_failure():
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    provider = FakeProvider()
    provider.error = RuntimeError("unexpected synthetic failure")

    result = _service(
        policy=_enabled_policy(), provider=provider, approvals=ApprovalReader(_approval())
    ).adjudicate(shot_id=uuid4(), observation=_ambiguous_observation())

    assert result.status is AdjudicationStatus.PROVIDER_FAILED
    assert result.retryable is False
    assert result.raw_observation.failure_detail == "unexpected synthetic failure"


@pytest.mark.parametrize("output", [None, {"ball_id": 1}, ["not", "text"]])
def test_non_string_provider_output_is_contained_without_entering_raw_observation(output):
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    result = _service(
        policy=_enabled_policy(),
        provider=FakeProvider(output),
        approvals=ApprovalReader(_approval()),
    ).adjudicate(shot_id=uuid4(), observation=_ambiguous_observation())

    assert result.status is AdjudicationStatus.PARSE_FAILED
    assert result.retryable is False
    assert result.raw_observation.provider_output is None
    assert result.raw_observation.failure_detail == "provider returned non-text output"


def test_repeated_shot_is_idempotent_and_conflicting_evidence_fails_closed():
    from billiards_ai.shared.models.adjudication import AdjudicationStatus

    provider = FakeProvider()
    service = _service(
        policy=_enabled_policy(), provider=provider, approvals=ApprovalReader(_approval())
    )
    shot_id = uuid4()
    observation = _ambiguous_observation()

    first = service.adjudicate(shot_id=shot_id, observation=observation)
    second = service.adjudicate(shot_id=shot_id, observation=observation)
    conflict = service.adjudicate(
        shot_id=shot_id,
        observation=observation.model_copy(
            update={
                "raw_observation": observation.raw_observation.model_copy(
                    update={"struck_ball_candidates": observation.struck_ball_candidates[:1]}
                )
            }
        ),
    )

    assert first == second
    assert first.idempotency_key == second.idempotency_key
    assert len(provider.requests) == 1
    assert conflict.status is AdjudicationStatus.IDEMPOTENCY_CONFLICT
    assert len(provider.requests) == 1


def test_json_ledger_preserves_terminal_result_across_adjudicator_instances(tmp_path):
    from billiards_ai.cloud.perception.adjudicator import JsonAdjudicationLedger

    shot_id = uuid4()
    first_provider = FakeProvider()
    first = _service(
        policy=_enabled_policy(),
        provider=first_provider,
        approvals=ApprovalReader(_approval()),
        ledger=JsonAdjudicationLedger(tmp_path / "adjudication-ledger.json"),
    ).adjudicate(shot_id=shot_id, observation=_ambiguous_observation())
    second_provider = FakeProvider()
    second = _service(
        policy=_enabled_policy(),
        provider=second_provider,
        approvals=ApprovalReader(_approval()),
        ledger=JsonAdjudicationLedger(tmp_path / "adjudication-ledger.json"),
    ).adjudicate(shot_id=shot_id, observation=_ambiguous_observation())

    assert second == first
    assert len(first_provider.requests) == 1
    assert second_provider.requests == []


def test_adjudicator_never_mutates_the_input_observation_or_a_shot():
    from datetime import UTC, datetime

    from billiards_ai.shared.models.geometry import Point
    from billiards_ai.shared.models.shot import Confidence, Shot

    observation = _ambiguous_observation()
    shot = Shot(
        id=uuid4(),
        session_id=uuid4(),
        sequence=1,
        t_start_ms=500,
        t_end_ms=900,
        ball_positions_before={0: Point(x_mm=100, y_mm=100)},
        ball_positions_after={0: Point(x_mm=200, y_mm=100)},
        cue_ball_end=Point(x_mm=200, y_mm=100),
        struck_ball=None,
        confidence=Confidence.LOW,
        created_at=datetime.now(UTC),
    )
    before_observation = observation.model_dump()
    before_shot = shot.model_dump()

    _service(
        policy=_enabled_policy(), provider=FakeProvider(), approvals=ApprovalReader(_approval())
    ).adjudicate(shot_id=shot.id, observation=observation)

    assert observation.model_dump() == before_observation
    assert shot.model_dump() == before_shot
