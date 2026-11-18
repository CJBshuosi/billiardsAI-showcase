"""Fail-closed approved evidence ingestion service."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError

from billiards_ai.cloud.growth.errors import EvidenceIngestError, EvidenceIngestErrorCode
from billiards_ai.cloud.growth.projector import project_approved_shots
from billiards_ai.cloud.growth.repository import (
    ApprovedEvidenceLineage,
    ApprovedSourceEnvelopeLineage,
    EvidenceIngestResult,
    GrowthEvidenceRepository,
    revalidate_approved_batch,
    stored_model_values,
)
from billiards_ai.cloud.review.errors import ReportNotDeliverableError
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.evidence import (
    ApprovedEvidenceBatchAny,
    ApprovedEvidenceBatchV2,
)
from billiards_ai.shared.models.metric_evidence import ApprovedMetricEvidenceAuthorityRecord
from billiards_ai.shared.models.review import ApprovedSessionData
from billiards_ai.shared.models.shot import ReviewStatus


class AsyncGrowthEvidenceRepository(Protocol):
    async def resolve_approved_batch(
        self,
        batch: ApprovedEvidenceBatchAny,
        *,
        authenticated_user_id: UUID,
    ) -> EvidenceIngestResult | None: ...

    async def append_approved_batch(
        self,
        batch: ApprovedEvidenceBatchAny,
        *,
        metric_authority: ApprovedMetricEvidenceAuthorityRecord | None,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> EvidenceIngestResult: ...


class _CaptureGrowthRepository:
    def __init__(self) -> None:
        self.batch: ApprovedEvidenceBatchAny | None = None
        self.metric_authority: ApprovedMetricEvidenceAuthorityRecord | None = None

    def resolve_approved_batch(self, batch, *, authenticated_user_id):
        return None

    def append_approved_batch(
        self, batch, *, metric_authority=None, clock, uuid_factory
    ) -> EvidenceIngestResult:
        self.batch = batch
        self.metric_authority = metric_authority
        return None  # type: ignore[return-value]

    def list_batches(self, user_id):
        return ()

    def list_accepted_metric_inputs(self, user_id):
        return ()


class ApprovedEvidenceAuthority(Protocol):
    def get_approved_session_data(
        self, session_id: UUID, report_id: UUID, revision: int
    ) -> ApprovedSessionData: ...


class ApprovedEvidenceLineageAuthority(Protocol):
    def get_approved_evidence_lineage(
        self, session_id: UUID, report_id: UUID, revision: int
    ) -> ApprovedEvidenceLineage: ...


class ApprovedMetricEvidenceAuthority(Protocol):
    def get_approved_metric_evidence(
        self, session_id: UUID, report_id: UUID, revision: int
    ) -> ApprovedMetricEvidenceAuthorityRecord: ...


class EvidenceIngestService:
    def __init__(
        self,
        *,
        approved_repository: ApprovedEvidenceAuthority,
        lineage_repository: ApprovedEvidenceLineageAuthority,
        metric_repository: ApprovedMetricEvidenceAuthority | None = None,
        growth_repository: GrowthEvidenceRepository,
        clock: Callable[[], datetime] | None = None,
        uuid_factory: Callable[[], UUID] | None = None,
    ):
        self._approved_repository = approved_repository
        self._lineage_repository = lineage_repository
        self._metric_repository = metric_repository
        self._growth_repository = growth_repository
        self._clock = clock or (lambda: datetime.now(UTC))
        self._uuid_factory = uuid_factory or uuid4

    def ingest(
        self,
        batch: ApprovedEvidenceBatchAny,
        *,
        authenticated_user_id: UUID,
    ) -> EvidenceIngestResult:
        try:
            validated = revalidate_approved_batch(batch)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError) as exc:
            raise EvidenceIngestError(EvidenceIngestErrorCode.INVALID_BATCH) from exc

        if validated.user_id != authenticated_user_id:
            raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)
        if existing := self._growth_repository.resolve_approved_batch(
            validated,
            authenticated_user_id=authenticated_user_id,
        ):
            return existing
        if validated.report_revision != validated.review_revision:
            raise EvidenceIngestError(EvidenceIngestErrorCode.REVISION_MISMATCH)
        expected_digest = canonical_sha256(
            validated.model_dump(mode="json", exclude={"content_sha256"})
        )
        if validated.content_sha256 != expected_digest:
            raise EvidenceIngestError(EvidenceIngestErrorCode.DIGEST_MISMATCH)

        try:
            approved = self._approved_repository.get_approved_session_data(
                validated.session_id,
                validated.report_id,
                validated.review_revision,
            )
        except ReportNotDeliverableError as exc:
            raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED) from exc
        try:
            approved = ApprovedSessionData.model_validate(
                stored_model_values(approved), strict=True
            )
        except (AttributeError, TypeError, ValidationError) as exc:
            raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH) from exc

        lineage = self._lineage_repository.get_approved_evidence_lineage(
            validated.session_id,
            validated.report_id,
            validated.review_revision,
        )
        try:
            lineage = ApprovedEvidenceLineage.model_validate(
                stored_model_values(lineage), strict=True
            )
        except (AttributeError, TypeError, ValidationError) as exc:
            raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH) from exc

        if (
            approved.session_id != validated.session_id
            or approved.report.session_id != validated.session_id
            or approved.report.report_id != validated.report_id
            or approved.report.revision != validated.review_revision
        ):
            raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)

        metric_authority: ApprovedMetricEvidenceAuthorityRecord | None = None
        if isinstance(validated, ApprovedEvidenceBatchV2):
            if self._metric_repository is None:
                raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED)
            try:
                metric_authority = ApprovedMetricEvidenceAuthorityRecord.model_validate(
                    stored_model_values(
                        self._metric_repository.get_approved_metric_evidence(
                            validated.session_id,
                            validated.report_id,
                            validated.review_revision,
                        )
                    ),
                    strict=True,
                )
            except (ReportNotDeliverableError, LookupError) as exc:
                raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED) from exc
            except (AttributeError, TypeError, ValidationError, ValueError) as exc:
                raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH) from exc
            authority_metric_shots = tuple(
                sorted(
                    metric_authority.metric_shots,
                    key=lambda item: (item.sequence, str(item.shot_id)),
                )
            )
            submitted_metric_shots = tuple(
                sorted(
                    validated.metric_shots,
                    key=lambda item: (item.sequence, str(item.shot_id)),
                )
            )
            if (
                authority_metric_shots != submitted_metric_shots
                or metric_authority.approval_kind is not validated.approval_kind
                or metric_authority.approval_policy_version != validated.approval_policy_version
            ):
                raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)
        authority_shots = tuple(
            sorted(
                (
                    shot
                    for shot in approved.shots
                    if shot.review_status in {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}
                ),
                key=lambda shot: shot.sequence,
            )
        )
        if authority_shots != validated.shots:
            raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)
        if len(validated.source_envelope_ids) != len(validated.shots):
            raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)
        expected_sources = tuple(
            ApprovedSourceEnvelopeLineage(envelope_id=envelope_id, shot_id=shot.shot_id)
            for envelope_id, shot in zip(
                validated.source_envelope_ids, validated.shots, strict=True
            )
        )
        if (
            lineage.user_id != validated.user_id
            or lineage.session_id != validated.session_id
            or lineage.report_id != validated.report_id
            or lineage.review_revision != validated.review_revision
            or lineage.approved_at != validated.approved_at
            or {(item.envelope_id, item.shot_id) for item in lineage.source_envelopes}
            != {(item.envelope_id, item.shot_id) for item in expected_sources}
        ):
            raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)

        try:
            statistics, scores = project_approved_shots(validated.shots)
        except (OverflowError, ValueError) as exc:
            raise EvidenceIngestError(EvidenceIngestErrorCode.PROJECTION_UNAVAILABLE) from exc
        if validated.statistics != statistics:
            raise EvidenceIngestError(EvidenceIngestErrorCode.STATISTICS_MISMATCH)
        if validated.scores != scores:
            raise EvidenceIngestError(EvidenceIngestErrorCode.SCORES_MISMATCH)
        return self._growth_repository.append_approved_batch(
            validated,
            metric_authority=metric_authority,
            clock=self._clock,
            uuid_factory=self._uuid_factory,
        )


class AsyncEvidenceIngestService:
    """Async application boundary backed by a durable growth repository port."""

    def __init__(
        self,
        *,
        approved_repository: ApprovedEvidenceAuthority,
        lineage_repository: ApprovedEvidenceLineageAuthority,
        growth_repository: AsyncGrowthEvidenceRepository,
        metric_repository: ApprovedMetricEvidenceAuthority | None = None,
        clock: Callable[[], datetime] | None = None,
        uuid_factory: Callable[[], UUID] | None = None,
    ) -> None:
        self._approved = approved_repository
        self._lineage = lineage_repository
        self._metric = metric_repository
        self._growth = growth_repository
        self._clock = clock or (lambda: datetime.now(UTC))
        self._uuid_factory = uuid_factory or uuid4

    async def ingest(
        self,
        batch: ApprovedEvidenceBatchAny,
        *,
        authenticated_user_id: UUID,
    ) -> EvidenceIngestResult:
        try:
            validated = revalidate_approved_batch(batch)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError) as exc:
            raise EvidenceIngestError(EvidenceIngestErrorCode.INVALID_BATCH) from exc
        if validated.user_id != authenticated_user_id:
            raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)
        existing = await self._growth.resolve_approved_batch(
            validated, authenticated_user_id=authenticated_user_id
        )
        if existing is not None:
            return existing

        capture = _CaptureGrowthRepository()
        EvidenceIngestService(
            approved_repository=self._approved,
            lineage_repository=self._lineage,
            metric_repository=self._metric,
            growth_repository=capture,
            clock=self._clock,
            uuid_factory=self._uuid_factory,
        ).ingest(validated, authenticated_user_id=authenticated_user_id)
        if capture.batch is None:
            raise EvidenceIngestError(EvidenceIngestErrorCode.INVALID_BATCH)
        return await self._growth.append_approved_batch(
            capture.batch,
            metric_authority=capture.metric_authority,
            clock=self._clock,
            uuid_factory=self._uuid_factory,
        )
