from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.coach_agent.context import (
    CapabilityManifest,
    CoachContextSourceBundle,
    CoachSourceReadFenceV3,
    CoachTaskType,
    build_coach_context,
    coach_history_content_sha256,
)
from billiards_ai.cloud.coach_agent.growth_analysis import (
    GrowthAnalysisCompleted,
    GrowthAnalysisFailure,
    GrowthAnalysisFailureCode,
    GrowthAnalysisResultStatus,
    GrowthAnalysisWorkflow,
)
from billiards_ai.cloud.coach_agent.provider import (
    CoachModelFailure,
    CoachModelFailureCode,
    CoachModelRequest,
    CoachModelResponse,
)
from billiards_ai.shared.models.coaching import (
    AgentRun,
    AgentRunStatus,
    CoachAssessmentStatus,
    FindingSupportStatus,
    FindingType,
)
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.growth_metrics import MetricReliability
from tests.unit.cloud.coach_agent.test_context import _bundle
from tests.unit.cloud.growth.snapshot_test_support import NOW, uid
from tests.unit.cloud.growth.test_snapshot_v3 import _project, _window_batches


class ScriptedProvider:
    def __init__(self, result: CoachModelResponse | CoachModelFailure | BaseException):
        self.result = result
        self.requests: list[CoachModelRequest] = []

    @property
    def audit_provider_name(self) -> str:
        return "scripted"

    @property
    def audit_model_name(self) -> str:
        return "synthetic-coach-v1"

    async def generate_structured(
        self, request: CoachModelRequest
    ) -> CoachModelResponse | CoachModelFailure:
        self.requests.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _manifest(user_id: UUID) -> CapabilityManifest:
    return CapabilityManifest(
        manifest_version="synthetic-v3-capabilities",
        user_id=user_id,
        user_level=2,
    )


def _context(snapshot):
    manifest = _manifest(snapshot.user_id)
    history_hash = coach_history_content_sha256(CoachTaskType.GROWTH_ASSESSMENT, (), (), ())
    fence = CoachSourceReadFenceV3.build_v3(
        snapshot_read_fence=snapshot.read_fence,
        authority_revision="aux-r1",
        history_revision="history-r0",
        history_content_sha256=history_hash,
        catalog_version=None,
        capability_manifest_version=manifest.manifest_version,
    )
    return build_coach_context(
        CoachContextSourceBundle(
            task_type=CoachTaskType.GROWTH_ASSESSMENT,
            user_id=snapshot.user_id,
            as_of=snapshot.created_at,
            read_fence=fence,
            growth_snapshot=snapshot,
            capability_manifest=manifest,
        )
    )


def _response(structured_output: dict[str, object]) -> CoachModelResponse:
    return CoachModelResponse(
        provider="scripted",
        model="synthetic-coach-v1",
        provider_request_id="provider-request-1",
        structured_output=structured_output,
        finish_reason="stop",
        input_tokens=12,
        output_tokens=6,
        latency_ms=7,
    )


def _uuid_values() -> Iterator[UUID]:
    yield uid(970_001)  # run/request
    yield uid(970_002)  # assessment


def _workflow(provider: ScriptedProvider) -> GrowthAnalysisWorkflow:
    values = _uuid_values()
    times = iter(
        (
            datetime(2026, 8, 16, 9, 0, tzinfo=UTC),
            datetime(2026, 8, 16, 9, 0, 1, tzinfo=UTC),
        )
    )
    return GrowthAnalysisWorkflow(
        provider=provider,
        uuid_factory=lambda: next(values),
        clock=lambda: next(times),
    )


@pytest.mark.asyncio
async def test_supported_improvement_uses_exact_v3_facts_and_creates_draft_audit() -> None:
    snapshot = _project(_window_batches(0, 5))
    metric = next(
        item
        for item in snapshot.metric_assessments
        if item.metric.metric_code.value == "potting.success_rate"
    )
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": metric.metric_identity_sha256,
                    "finding_type": "IMPROVEMENT",
                },
                "secondary": None,
            }
        )
    )

    result = await _workflow(provider).analyze(_context(snapshot))

    assert isinstance(result, GrowthAnalysisCompleted)
    assessment = result.assessment
    assert assessment.status is CoachAssessmentStatus.DRAFT
    assert assessment.snapshot_id == snapshot.snapshot_id
    assert assessment.snapshot_content_sha256 == snapshot.content_sha256
    assert assessment.agent_run_id == result.agent_run.run_id
    finding = assessment.findings[0]
    trend_ids = tuple(
        sorted(metric.trend.previous_evidence_ref_ids + metric.trend.current_evidence_ref_ids)
    )
    assert finding.finding_type is FindingType.IMPROVEMENT
    assert finding.support_status is FindingSupportStatus.SUPPORTED
    assert finding.reliability is metric.trend.reliability
    assert finding.evidence_ref_ids == trend_ids
    assert finding.trend_lineage_id is not None
    assert "不证明因果关系" in finding.explanation
    assert result.agent_run.status is AgentRunStatus.SUCCEEDED
    assert result.agent_run.repair_count == 0
    assert result.agent_run.context_sha256 == _context(snapshot).context_sha256
    assert result.agent_run.output_sha256 == assessment.content_sha256
    assert provider.requests[0].task_type is CoachTaskType.GROWTH_ASSESSMENT
    payload = provider.requests[0].input_payload
    assert set(payload) == {"schema_version", "metric_candidates"}
    assert len(payload["metric_candidates"]) <= 64
    assert len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False)) <= 32_768
    forbidden_lineage = {
        "batch_id",
        "session_id",
        "shot_id",
        "shot_ids",
        "source_envelope_id",
        "source_envelope_ids",
        "evidence_ref_id",
        "evidence_ref_ids",
        "shot_trace_bindings",
        "context_sha256",
        "snapshot_id",
        "snapshot_content_sha256",
    }
    assert not forbidden_lineage & set(json.dumps(payload).replace('"', " ").split())
    for candidate in payload["metric_candidates"]:
        assert set(candidate) == {
            "metric_identity_sha256",
            "metric_code",
            "metric_value_field_code",
            "metric_value",
            "support_status",
            "reliability",
            "evidence_ref_count",
            "trend_direction",
            "trend_support_status",
            "trend_reliability",
            "allowed_finding_types",
        }
        assert candidate["evidence_ref_count"] <= 256


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("previous", "current", "finding_type"),
    [
        (0, 5, FindingType.IMPROVEMENT),
        (5, 0, FindingType.DECLINE),
        (3, 3, FindingType.PLATEAU),
    ],
)
async def test_all_substantive_trend_types_bind_the_exact_d1_lineage(
    previous: int,
    current: int,
    finding_type: FindingType,
) -> None:
    snapshot = _project(_window_batches(previous, current))
    metric = next(
        item
        for item in snapshot.metric_assessments
        if item.metric.metric_code.value == "potting.success_rate"
    )
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": metric.metric_identity_sha256,
                    "finding_type": finding_type.value,
                },
                "secondary": None,
            }
        )
    )

    result = await _workflow(provider).analyze(_context(snapshot))

    assert isinstance(result, GrowthAnalysisCompleted)
    finding = result.assessment.findings[0]
    lineage = next(
        item
        for item in result.assessment.snapshot_trend_lineage
        if item.metric_identity_sha256 == metric.metric_identity_sha256
    )
    assert finding.finding_type is finding_type
    assert finding.trend_lineage_id == lineage.lineage_id
    assert finding.reliability is lineage.reliability
    assert finding.evidence_ref_ids == lineage.evidence_ref_ids


@pytest.mark.asyncio
async def test_supported_weakness_inherits_metric_support_reliability_and_all_refs() -> None:
    snapshot = _project(_window_batches(2, 2))
    metric = next(
        item
        for item in snapshot.metric_assessments
        if item.metric.metric_code.value == "potting.success_rate"
    )
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": metric.metric_identity_sha256,
                    "finding_type": "WEAKNESS",
                },
                "secondary": None,
            }
        )
    )

    result = await _workflow(provider).analyze(_context(snapshot))

    assert isinstance(result, GrowthAnalysisCompleted)
    finding = result.assessment.findings[0]
    assert finding.finding_type is FindingType.WEAKNESS
    assert finding.support_status is FindingSupportStatus.SUPPORTED
    assert finding.reliability is metric.metric.reliability
    assert finding.evidence_ref_ids == tuple(
        sorted(item.evidence_ref_id for item in metric.evidence_refs)
    )
    assert finding.trend_lineage_id is None
    assert result.assessment.primary_finding_id == finding.finding_id


@pytest.mark.asyncio
async def test_empty_selection_and_invalid_context_still_return_audited_no_conclusion() -> None:
    snapshot = _project(_window_batches(0, 5))
    empty = await _workflow(
        ScriptedProvider(
            _response(
                {
                    "schema_version": "1.0",
                    "primary": None,
                    "secondary": None,
                }
            )
        )
    ).analyze(_context(snapshot))

    assert isinstance(empty, GrowthAnalysisCompleted)
    assert empty.assessment.primary_finding_id is None
    assert empty.assessment.findings[0].finding_type is FindingType.INSUFFICIENT_EVIDENCE

    invalid = await _workflow(ScriptedProvider(_response({}))).analyze("invalid")  # type: ignore[arg-type]
    assert isinstance(invalid, GrowthAnalysisFailure)
    assert invalid.code is GrowthAnalysisFailureCode.INVALID_CONTEXT
    assert invalid.agent_run.status is AgentRunStatus.FAILED
    assert invalid.agent_run.context_sha256 == canonical_sha256(
        {"failure_identity": "growth-analysis-v1:INVALID_CONTEXT"}
    )

    valid = _context(snapshot)
    unsafe_values = dict(vars(valid))
    unsafe_values["growth_snapshot"] = "invalid"
    unsafe_values["context_sha256"] = "f" * 64
    malicious = type(valid).model_construct(**unsafe_values)
    rejected = await _workflow(ScriptedProvider(_response({}))).analyze(malicious)
    assert isinstance(rejected, GrowthAnalysisFailure)
    assert rejected.code is GrowthAnalysisFailureCode.INVALID_CONTEXT
    assert rejected.agent_run.context_sha256 != malicious.context_sha256


@pytest.mark.asyncio
@pytest.mark.parametrize("attempted_type", ["IMPROVEMENT", "DECLINE", "PLATEAU"])
async def test_unknown_or_direction_mismatch_is_downgraded_to_limited_finding(
    attempted_type: str,
) -> None:
    current_only = tuple(
        batch for batch in _window_batches(0, 5) if batch.approved_at > NOW - timedelta(days=90)
    )
    snapshot = _project(current_only)
    metric = next(
        item
        for item in snapshot.metric_assessments
        if item.metric.metric_code.value == "potting.success_rate"
    )
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": metric.metric_identity_sha256,
                    "finding_type": attempted_type,
                },
                "secondary": None,
            }
        )
    )

    result = await _workflow(provider).analyze(_context(snapshot))

    assert isinstance(result, GrowthAnalysisCompleted)
    finding = result.assessment.findings[0]
    assert finding.finding_type is FindingType.INSUFFICIENT_EVIDENCE
    assert finding.support_status in {
        FindingSupportStatus.LIMITED,
        FindingSupportStatus.UNSUPPORTED,
    }
    assert finding.reliability is MetricReliability.UNKNOWN
    assert finding.trend_lineage_id is None
    assert result.assessment.primary_finding_id is None


@pytest.mark.asyncio
async def test_missing_metric_identity_is_no_conclusion_not_fabricated_evidence() -> None:
    snapshot = _project(_window_batches(0, 5))
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": "f" * 64,
                    "finding_type": "WEAKNESS",
                },
                "secondary": None,
            }
        )
    )

    result = await _workflow(provider).analyze(_context(snapshot))

    assert isinstance(result, GrowthAnalysisCompleted)
    finding = result.assessment.findings[0]
    assert finding.finding_type is FindingType.INSUFFICIENT_EVIDENCE
    assert finding.support_status is FindingSupportStatus.UNSUPPORTED
    assert finding.evidence_ref_ids == ()
    assert result.assessment.primary_finding_id is None


@pytest.mark.asyncio
async def test_two_distinct_missing_metric_candidates_remain_distinct_limited_findings() -> None:
    snapshot = _project(_window_batches(0, 5))
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": "e" * 64,
                    "finding_type": "WEAKNESS",
                },
                "secondary": {
                    "metric_identity_sha256": "f" * 64,
                    "finding_type": "STRENGTH",
                },
            }
        )
    )

    result = await _workflow(provider).analyze(_context(snapshot))

    assert isinstance(result, GrowthAnalysisCompleted)
    assert len(result.assessment.findings) == 2
    assert len({item.finding_id for item in result.assessment.findings}) == 2
    assert all(
        item.finding_type is FindingType.INSUFFICIENT_EVIDENCE
        for item in result.assessment.findings
    )


@pytest.mark.asyncio
async def test_old_snapshot_fails_closed_before_provider_call() -> None:
    old_context = build_coach_context(_bundle(task_type=CoachTaskType.GROWTH_ASSESSMENT))
    provider = ScriptedProvider(_response({}))

    result = await _workflow(provider).analyze(old_context)

    assert isinstance(result, GrowthAnalysisFailure)
    assert result.status is GrowthAnalysisResultStatus.NO_CONCLUSION
    assert result.code is GrowthAnalysisFailureCode.UNSUPPORTED_SNAPSHOT
    assert result.agent_run is not None
    assert result.agent_run.status is AgentRunStatus.FAILED
    assert provider.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_result", "expected_code"),
    [
        (
            CoachModelFailure(
                provider="scripted",
                model="synthetic-coach-v1",
                code=CoachModelFailureCode.TIMEOUT,
                retryable=True,
                latency_ms=9,
            ),
            GrowthAnalysisFailureCode.PROVIDER_FAILURE,
        ),
        (
            _response({"not": "the output contract"}).model_copy(
                update={"provider": "untrusted-result", "model": "untrusted-result"}
            ),
            GrowthAnalysisFailureCode.INVALID_OUTPUT,
        ),
        (RuntimeError("secret prompt and payload"), GrowthAnalysisFailureCode.PROVIDER_FAILURE),
    ],
)
async def test_provider_failures_are_stable_payload_free_and_never_repaired(
    provider_result: CoachModelResponse | CoachModelFailure | BaseException,
    expected_code: GrowthAnalysisFailureCode,
) -> None:
    provider = ScriptedProvider(provider_result)

    result = await _workflow(provider).analyze(_context(_project(_window_batches(0, 5))))

    assert isinstance(result, GrowthAnalysisFailure)
    assert result.code is expected_code
    assert result.agent_run is not None
    assert result.agent_run.status is AgentRunStatus.FAILED
    assert result.agent_run.repair_count == 0
    assert (result.agent_run.provider_name, result.agent_run.provider_model) == (
        "scripted",
        "synthetic-coach-v1",
    )
    assert "secret" not in repr(result)


@pytest.mark.asyncio
async def test_invalid_provider_result_uses_trusted_injected_audit_metadata() -> None:
    provider = ScriptedProvider(object())  # type: ignore[arg-type]

    result = await _workflow(provider).analyze(_context(_project(_window_batches(0, 5))))

    assert isinstance(result, GrowthAnalysisFailure)
    assert result.code is GrowthAnalysisFailureCode.PROVIDER_FAILURE
    assert (result.agent_run.provider_name, result.agent_run.provider_model) == (
        "scripted",
        "synthetic-coach-v1",
    )


@pytest.mark.asyncio
async def test_secondary_is_promoted_or_cleared_when_primary_is_absent_or_downgraded() -> None:
    snapshot = _project(_window_batches(0, 5))
    substantive_metric = next(
        item
        for item in snapshot.metric_assessments
        if item.metric.metric_code.value == "potting.success_rate"
    )
    output = {
        "schema_version": "1.0",
        "primary": {
            "metric_identity_sha256": "f" * 64,
            "finding_type": "WEAKNESS",
        },
        "secondary": {
            "metric_identity_sha256": substantive_metric.metric_identity_sha256,
            "finding_type": "IMPROVEMENT",
        },
    }

    promoted = await _workflow(ScriptedProvider(_response(output))).analyze(_context(snapshot))

    assert isinstance(promoted, GrowthAnalysisCompleted)
    assert promoted.assessment.primary_finding_id is not None
    assert promoted.assessment.secondary_finding_id is None
    selected = next(
        finding
        for finding in promoted.assessment.findings
        if finding.finding_id == promoted.assessment.primary_finding_id
    )
    assert selected.finding_type is FindingType.IMPROVEMENT

    secondary_only = dict(output)
    secondary_only["primary"] = None
    promoted_from_missing = await _workflow(ScriptedProvider(_response(secondary_only))).analyze(
        _context(snapshot)
    )
    assert isinstance(promoted_from_missing, GrowthAnalysisCompleted)
    assert promoted_from_missing.assessment.primary_finding_id is not None
    assert promoted_from_missing.assessment.secondary_finding_id is None


@pytest.mark.asyncio
async def test_completed_result_enforces_audit_closure_and_clamps_utc_rollback() -> None:
    snapshot = _project(_window_batches(0, 5))
    metric = next(
        item
        for item in snapshot.metric_assessments
        if item.metric.metric_code.value == "potting.success_rate"
    )
    provider = ScriptedProvider(
        _response(
            {
                "schema_version": "1.0",
                "primary": {
                    "metric_identity_sha256": metric.metric_identity_sha256,
                    "finding_type": "IMPROVEMENT",
                },
                "secondary": None,
            }
        )
    )
    values = _uuid_values()
    times = iter(
        (
            datetime(2026, 8, 16, 9, 0, 1, tzinfo=UTC),
            datetime(2026, 8, 16, 9, 0, tzinfo=UTC),
        )
    )
    workflow = GrowthAnalysisWorkflow(
        provider=provider,
        uuid_factory=lambda: next(values),
        clock=lambda: next(times),
    )

    result = await workflow.analyze(_context(snapshot))

    assert isinstance(result, GrowthAnalysisCompleted)
    assert result.agent_run.completed_at >= result.agent_run.started_at
    assert result.assessment.created_at == result.agent_run.completed_at

    failed_audit = AgentRun.model_validate(
        result.agent_run.model_dump() | {"status": AgentRunStatus.FAILED, "output_sha256": None},
        strict=True,
    )
    wrong_output_audit = AgentRun.model_validate(
        result.agent_run.model_dump() | {"output_sha256": "f" * 64},
        strict=True,
    )
    for invalid_audit in (failed_audit, wrong_output_audit):
        with pytest.raises(ValidationError):
            GrowthAnalysisCompleted(
                assessment=result.assessment,
                agent_run=invalid_audit,
            )

    wrong_run_payload = result.assessment.model_dump(
        mode="python",
        exclude={"content_sha256"},
    )
    wrong_run_payload["agent_run_id"] = uid(970_099)
    wrong_run_assessment = type(result.assessment)(
        **wrong_run_payload,
        content_sha256=canonical_sha256(wrong_run_payload),
    )
    wrong_run_audit = AgentRun.model_validate(
        result.agent_run.model_dump() | {"output_sha256": wrong_run_assessment.content_sha256},
        strict=True,
    )
    with pytest.raises(ValidationError):
        GrowthAnalysisCompleted(
            assessment=wrong_run_assessment,
            agent_run=wrong_run_audit,
        )

    secondary_only_payload = result.assessment.model_dump(
        mode="python",
        exclude={"content_sha256"},
    )
    secondary_only_payload["primary_finding_id"] = None
    secondary_only_payload["secondary_finding_id"] = result.assessment.primary_finding_id
    secondary_only = type(result.assessment)(
        **secondary_only_payload,
        content_sha256=canonical_sha256(secondary_only_payload),
    )
    secondary_only_audit = AgentRun.model_validate(
        result.agent_run.model_dump() | {"output_sha256": secondary_only.content_sha256},
        strict=True,
    )
    with pytest.raises(ValidationError):
        GrowthAnalysisCompleted(
            assessment=secondary_only,
            agent_run=secondary_only_audit,
        )


@pytest.mark.asyncio
async def test_provider_cancellation_propagates() -> None:
    provider = ScriptedProvider(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await _workflow(provider).analyze(_context(_project(_window_batches(0, 5))))


@pytest.mark.asyncio
async def test_same_frozen_inputs_ids_and_times_replay_byte_equivalent_artifacts() -> None:
    snapshot = _project(_window_batches(0, 5))
    metric = next(
        item
        for item in snapshot.metric_assessments
        if item.metric.metric_code.value == "potting.success_rate"
    )
    output = _response(
        {
            "schema_version": "1.0",
            "primary": {
                "metric_identity_sha256": metric.metric_identity_sha256,
                "finding_type": "IMPROVEMENT",
            },
            "secondary": None,
        }
    )

    first = await _workflow(ScriptedProvider(output)).analyze(_context(snapshot))
    second = await _workflow(ScriptedProvider(output)).analyze(_context(snapshot))

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
