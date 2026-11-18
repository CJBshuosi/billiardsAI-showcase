"""Shared synthetic composition support for Story 11.4 integration tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import UUID4, BaseModel, ConfigDict, Field, StrictInt
from sqlalchemy import text

from billiards_ai.cloud.review.application import ReviewApplicationService
from billiards_ai.cloud.review.application_models import (
    ClaimReviewCommand,
    CorrectionIntent,
    FinalizeReviewCommand,
    RejectReviewCommand,
    ShotDecisionIntent,
    TaskRef,
)
from billiards_ai.cloud.review.identity import TableBindingSessionSubjectAuthority
from billiards_ai.cloud.review.metric_authority import (
    HumanReviewMetricAuthorityAssembler,
    MetricFieldReviewStates,
    MetricPointInput,
    MetricReviewState,
    MetricShotReviewInput,
)
from billiards_ai.cloud.review.persistence import (
    PostgresApprovedDataRepository,
    PostgresApprovedEvidencePublisher,
    PostgresOutboxRepository,
    PostgresQualityGateRepository,
)
from billiards_ai.shared.models import (
    AccessActorRole,
    Confidence,
    ReportDocument,
    ReviewableField,
    ReviewerIdentity,
    ReviewStatus,
    ReviewTask,
    ReviewTaskStatus,
    SessionRejectionReason,
    Shot,
    ShotAnalysisStatus,
    ShotReviewOutcome,
    TargetZoneSource,
)
from billiards_ai.shared.models.calibration import CalibrationLineage, CalibrationStatus
from billiards_ai.shared.models.metric_evidence import (
    FoulResult,
    PocketCode,
    PotAttemptStatus,
    PotResult,
    TargetZoneEvaluationStatus,
)
from billiards_ai.shared.models.session_capture import SessionMediaBindingRecord


class FrozenFixtureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SyntheticIdentifiers(FrozenFixtureModel):
    session_id: UUID4
    source_report_id: UUID4
    task_id: UUID4
    user_id: UUID4
    attempt_id: UUID4
    asset_id: UUID4
    calibration_id: UUID4
    calibration_revision_id: UUID4


class SyntheticReviewer(FrozenFixtureModel):
    principal_id: str = Field(min_length=1)
    credential_id: str = Field(min_length=1)


class SyntheticShot(FrozenFixtureModel):
    shot_id: UUID4
    source_envelope_id: UUID4
    sequence: StrictInt = Field(ge=1)
    struck_ball: StrictInt = Field(ge=1, le=15)
    potted_balls: tuple[StrictInt, ...]
    cushion_count: StrictInt = Field(ge=0)
    travel_distance_mm: StrictInt = Field(gt=0)


class SyntheticQualityGateFixture(FrozenFixtureModel):
    fixture_schema_version: str
    report_fixture: str
    now: str
    reviewer: SyntheticReviewer
    identifiers: SyntheticIdentifiers
    shots: tuple[SyntheticShot, ...]


class ReviewScenario(StrEnum):
    APPROVE = "approve"
    CORRECT = "correct"
    APPROVE_WITH_REJECTED_SHOT = "approve_with_rejected_shot"
    CORRECT_WITH_REJECTED_SHOT = "correct_with_rejected_shot"
    ALL_REJECTED = "all_rejected"


@dataclass(frozen=True, slots=True)
class SyntheticGraph:
    report: ReportDocument
    task: ReviewTask
    shots: tuple[Shot, ...]
    source_envelope_ids: dict[UUID, UUID]


@dataclass(frozen=True, slots=True)
class QualityGateRuntime:
    service: ReviewApplicationService
    repository: PostgresQualityGateRepository
    publisher: PostgresApprovedEvidencePublisher
    approved: PostgresApprovedDataRepository
    outbox: PostgresOutboxRepository
    factories: CountingAuthorities


class CountingAuthorities:
    def __init__(self, fixture: SyntheticQualityGateFixture) -> None:
        self.fixture = fixture
        self.clock_calls = 0
        self.uuid_calls = 0
        self.subject_calls = 0
        self.calibration_calls = 0

    def now_utc(self) -> datetime:
        self.clock_calls += 1
        return datetime.fromisoformat(self.fixture.now.replace("Z", "+00:00"))

    def uuid4(self) -> UUID:
        self.uuid_calls += 1
        return UUID(f"a0000000-0000-4000-8000-{self.uuid_calls:012d}")

    def get_media_binding(self, *, session_id: UUID) -> SessionMediaBindingRecord:
        self.subject_calls += 1
        ids = self.fixture.identifiers
        return SessionMediaBindingRecord(
            session_id=session_id,
            attempt_id=ids.attempt_id,
            asset_id=ids.asset_id,
            content_sha256="a" * 64,
            storage_locator="controlled://story-11-4/synthetic",
            subject_user_ids=(str(ids.user_id),),
            interval_revision=1,
            accepted_at=self.now_value,
        )

    def calibration_for(self, *, session_id: UUID) -> CalibrationLineage:
        self.calibration_calls += 1
        ids = self.fixture.identifiers
        if session_id != ids.session_id:
            raise ValueError("synthetic calibration session mismatch")
        return CalibrationLineage(
            calibration_id=ids.calibration_id,
            revision_id=ids.calibration_revision_id,
            revision_number=1,
            status=CalibrationStatus.ACTIVE,
            table_profile_id="cn-8-ball",
            table_length_mm=2540,
            table_width_mm=1270,
            table_configuration_fingerprint="b" * 64,
        )

    @property
    def now_value(self) -> datetime:
        return datetime.fromisoformat(self.fixture.now.replace("Z", "+00:00")).astimezone(UTC)


class _IdentityProvider:
    def __init__(self, fixture: SyntheticQualityGateFixture) -> None:
        self._identity = ReviewerIdentity(
            principal_id=fixture.reviewer.principal_id,
            credential_id=fixture.reviewer.credential_id,
            roles=(AccessActorRole.REVIEWER,),
        )

    def current_identity(self) -> ReviewerIdentity:
        return self._identity


def load_synthetic_fixture(path: Path) -> SyntheticQualityGateFixture:
    return SyntheticQualityGateFixture.model_validate_json(
        json.dumps(
            json.loads(path.read_text(encoding="utf-8")),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        strict=True,
    )


def compose_runtime(session_factory, fixture: SyntheticQualityGateFixture) -> QualityGateRuntime:
    factories = CountingAuthorities(fixture)
    repository = PostgresQualityGateRepository(session_factory)
    publisher = PostgresApprovedEvidencePublisher(session_factory)
    service = ReviewApplicationService(
        repository=repository,
        publisher=publisher,
        identity_provider=_IdentityProvider(fixture),
        subject_provider=TableBindingSessionSubjectAuthority(factories),
        metric_assembler=HumanReviewMetricAuthorityAssembler(factories),
        now=factories.now_utc,
        uuid4_factory=factories.uuid4,
    )
    return QualityGateRuntime(
        service=service,
        repository=repository,
        publisher=publisher,
        approved=PostgresApprovedDataRepository(session_factory),
        outbox=PostgresOutboxRepository(session_factory),
        factories=factories,
    )


async def stage_synthetic_graph(
    runtime: QualityGateRuntime,
    fixture: SyntheticQualityGateFixture,
    *,
    shot_overrides: Mapping[str, object] | None = None,
) -> SyntheticGraph:
    project_root = Path(__file__).resolve().parents[4]
    base_report = ReportDocument.model_validate_json(
        (project_root / fixture.report_fixture).read_text(encoding="utf-8")
    )
    ids = fixture.identifiers
    report = ReportDocument.model_validate(
        base_report.model_dump(mode="python")
        | {
            "report_id": ids.source_report_id,
            "session_id": ids.session_id,
            "generated_at": runtime.factories.now_value,
        },
        strict=True,
    )
    task = ReviewTask(
        id=ids.task_id,
        session_id=ids.session_id,
        report_id=ids.source_report_id,
        report_revision=report.revision,
        status=ReviewTaskStatus.PENDING_REVIEW,
    )
    overrides = dict(shot_overrides or {})
    unsupported = set(overrides) - {"analysis_status", "confidence"}
    if unsupported:
        raise ValueError(f"unsupported synthetic shot overrides: {sorted(unsupported)}")
    analysis_status = ShotAnalysisStatus(
        overrides.get("analysis_status", ShotAnalysisStatus.ANALYZED)
    )
    confidence = Confidence(overrides.get("confidence", Confidence.HIGH))
    shots = tuple(
        Shot(
            id=item.shot_id,
            session_id=ids.session_id,
            sequence=item.sequence,
            t_start_ms=(item.sequence - 1) * 1000,
            t_end_ms=item.sequence * 1000,
            ball_positions_before={0: {"x_mm": 700, "y_mm": 500}},
            ball_positions_after={0: {"x_mm": 800, "y_mm": 500}},
            cue_ball_end={"x_mm": 800, "y_mm": 500},
            struck_ball=item.struck_ball,
            potted_balls=item.potted_balls,
            cushion_count=item.cushion_count,
            travel_distance_mm=item.travel_distance_mm,
            is_foul=False,
            confidence=confidence,
            analysis_status=analysis_status,
            review_status=ReviewStatus.PENDING,
            target_zone_source=TargetZoneSource.NONE,
            raw_observation={},
            created_at=runtime.factories.now_value,
        )
        for item in fixture.shots
    )
    envelopes = {item.shot_id: item.source_envelope_id for item in fixture.shots}
    await runtime.repository.stage_review_draft(
        report=report,
        task=task,
        shots=shots,
        source_envelope_ids=envelopes,
        created_at=runtime.factories.now_value,
    )
    return SyntheticGraph(report=report, task=task, shots=shots, source_envelope_ids=envelopes)


def _task_ref(graph: SyntheticGraph, lock_version: int) -> TaskRef:
    return TaskRef(
        task_id=graph.task.id,
        expected_lock_version=lock_version,
        source_report_id=graph.report.report_id,
        source_report_revision=graph.report.revision,
    )


def claim_command(graph: SyntheticGraph) -> ClaimReviewCommand:
    return ClaimReviewCommand(
        task=_task_ref(graph, 0),
        idempotency_key="story-11-4-claim-0123456789abcdef",
    )


def _metric_input(shot: Shot) -> MetricShotReviewInput:
    confirmed = MetricReviewState.CONFIRMED
    return MetricShotReviewInput(
        shot_id=shot.id,
        cue_ball_start_mm=MetricPointInput(x_mm=700.0, y_mm=500.0),
        object_ball_start_mm=MetricPointInput(x_mm=1000.0, y_mm=500.0),
        intended_object_ball=shot.struck_ball,
        intended_target_pocket=PocketCode.BOTTOM_RIGHT,
        pot_attempt_status=PotAttemptStatus.ATTEMPTED,
        pot_result=(PotResult.LEGAL_SUCCESS if shot.potted_balls else PotResult.MISS),
        foul_result=FoulResult.LEGAL,
        target_zone_source=TargetZoneSource.NONE,
        target_zone=None,
        cue_ball_end_mm=None,
        target_zone_evaluation_status=TargetZoneEvaluationStatus.NOT_APPLICABLE,
        review_states=MetricFieldReviewStates(
            geometry=confirmed,
            intent=confirmed,
            pot_attempt=confirmed,
            pot_result=confirmed,
            foul_result=confirmed,
            target_zone=confirmed,
            cue_ball_end=confirmed,
        ),
    )


def finalize_command(
    graph: SyntheticGraph, *, lock_version: int, scenario: ReviewScenario
) -> FinalizeReviewCommand:
    corrected = scenario in {
        ReviewScenario.CORRECT,
        ReviewScenario.CORRECT_WITH_REJECTED_SHOT,
    }
    rejected_sequences = (
        {1, 2, 3, 4}
        if scenario is ReviewScenario.ALL_REJECTED
        else ({2} if "rejected_shot" in scenario.value else set())
    )
    decisions = []
    for shot in graph.shots:
        if shot.sequence in rejected_sequences:
            outcome = ShotReviewOutcome.REJECTED
            corrections = ()
        elif corrected and shot.sequence == 1:
            outcome = ShotReviewOutcome.CORRECTED
            corrections = (
                CorrectionIntent(
                    field=ReviewableField.CUSHION_COUNT,
                    correct_value=1,
                    note="synthetic visible rail contact",
                ),
            )
        else:
            outcome = ShotReviewOutcome.APPROVED
            corrections = ()
        decisions.append(
            ShotDecisionIntent(
                shot_id=shot.id,
                outcome=outcome,
                trajectory_reviewed=True,
                corrections=corrections,
            )
        )
    promotable = tuple(shot for shot in graph.shots if shot.sequence not in rejected_sequences)
    metric_shots = tuple(_metric_input(shot) for shot in promotable)
    if not metric_shots:
        metric_shots = (_metric_input(graph.shots[0]),)
    return FinalizeReviewCommand(
        task=_task_ref(graph, lock_version),
        decisions=tuple(decisions),
        metric_shots=metric_shots,
        idempotency_key=f"story-11-4-finalize-{scenario.value}-0123456789abcdef",
    )


def reject_command(graph: SyntheticGraph, *, lock_version: int) -> RejectReviewCommand:
    return RejectReviewCommand(
        task=_task_ref(graph, lock_version),
        reason=SessionRejectionReason.OCCLUDED,
        idempotency_key="story-11-4-reject-0123456789abcdef",
    )


async def authority_counts(session_factory) -> dict[str, int]:
    tables = (
        "review_reports",
        "review_result_reports",
        "review_tasks",
        "review_shot_sources",
        "review_shot_decisions",
        "review_corrections",
        "review_shot_results",
        "review_rejections",
        "review_session_ownership",
        "review_publication_lineage",
        "review_metric_authority",
        "approved_evidence_batches",
        "approved_batch_sources",
        "quality_gate_idempotency_receipts",
        "quality_gate_outbox",
    )
    async with session_factory() as session:
        return {
            table_name: (
                await session.execute(text(f"SELECT count(*) FROM {table_name}"))  # noqa: S608
            ).scalar_one()
            for table_name in tables
        }
