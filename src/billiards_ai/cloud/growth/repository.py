"""Atomic repository contract and concurrent in-memory implementation for Story 8.2."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from enum import StrEnum
from threading import RLock
from typing import Protocol
from uuid import UUID

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from billiards_ai.cloud.growth.errors import EvidenceIngestError, EvidenceIngestErrorCode
from billiards_ai.shared.models.events import DomainEvent, Sha256, canonical_sha256, require_utc
from billiards_ai.shared.models.evidence import (
    AcceptedMetricEvidenceInput,
    ApprovedEvidenceBatch,
    ApprovedEvidenceBatchAny,
    ApprovedEvidenceBatchV2,
    _seal_accepted_metric_evidence_input,
)
from billiards_ai.shared.models.metric_evidence import ApprovedMetricEvidenceAuthorityRecord

_UUID4 = TypeAdapter(UUID4)


class GrowthRepositoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvidenceIngestStatus(StrEnum):
    CREATED = "CREATED"
    REPLAYED = "REPLAYED"


class EvidenceInboxReceipt(GrowthRepositoryModel):
    receipt_id: UUID4
    batch_id: UUID4
    user_id: UUID4
    session_id: UUID4
    report_id: UUID4
    review_revision: int = Field(ge=1, strict=True)
    content_sha256: Sha256
    received_at: datetime

    _received_at_utc = field_validator("received_at")(require_utc)


class EvidenceSourceSummary(GrowthRepositoryModel):
    batch_id: UUID4
    user_id: UUID4
    session_id: UUID4
    report_id: UUID4
    report_revision: int = Field(ge=1, strict=True)
    review_revision: int = Field(ge=1, strict=True)
    shot_ids: tuple[UUID4, ...]
    source_envelope_ids: tuple[UUID4, ...]
    content_sha256: Sha256


class ApprovedSourceEnvelopeLineage(GrowthRepositoryModel):
    envelope_id: UUID4
    shot_id: UUID4


class ApprovedEvidenceLineage(GrowthRepositoryModel):
    """Trusted owner and source mapping resolved outside the submitted batch."""

    user_id: UUID4
    session_id: UUID4
    report_id: UUID4
    review_revision: int = Field(ge=1, strict=True)
    approved_at: datetime
    source_envelopes: tuple[ApprovedSourceEnvelopeLineage, ...] = Field(min_length=1)

    _approved_at_utc = field_validator("approved_at")(require_utc)

    @model_validator(mode="after")
    def source_mappings_are_unique(self) -> ApprovedEvidenceLineage:
        envelope_ids = tuple(item.envelope_id for item in self.source_envelopes)
        shot_ids = tuple(item.shot_id for item in self.source_envelopes)
        if len(envelope_ids) != len(set(envelope_ids)):
            raise ValueError("source envelope lineage must be unique")
        if len(shot_ids) != len(set(shot_ids)):
            raise ValueError("source shot lineage must be unique")
        return self


class EvidenceIngestResult(GrowthRepositoryModel):
    status: EvidenceIngestStatus
    receipt: EvidenceInboxReceipt
    accepted_metric_input: AcceptedMetricEvidenceInput | None = None


class GrowthEvidenceRepository(Protocol):
    def resolve_approved_batch(
        self,
        batch: ApprovedEvidenceBatchAny,
        *,
        authenticated_user_id: UUID,
    ) -> EvidenceIngestResult | None: ...

    def append_approved_batch(
        self,
        batch: ApprovedEvidenceBatchAny,
        *,
        metric_authority: ApprovedMetricEvidenceAuthorityRecord | None = None,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> EvidenceIngestResult: ...

    def list_batches(self, user_id: UUID) -> tuple[ApprovedEvidenceBatchAny, ...]: ...

    def list_accepted_metric_inputs(
        self, user_id: UUID
    ) -> tuple[AcceptedMetricEvidenceInput, ...]: ...


def stored_model_values(value: object) -> object:
    if isinstance(value, BaseModel):
        values = dict(vars(value))
        if value.__pydantic_extra__:
            values.update(value.__pydantic_extra__)
        return {key: stored_model_values(item) for key, item in values.items()}
    if isinstance(value, tuple):
        return tuple(stored_model_values(item) for item in value)
    if isinstance(value, list):
        return [stored_model_values(item) for item in value]
    if isinstance(value, dict):
        return {key: stored_model_values(item) for key, item in value.items()}
    return value


def revalidate_approved_batch(batch: ApprovedEvidenceBatchAny) -> ApprovedEvidenceBatchAny:
    try:
        values = stored_model_values(batch)
        if not isinstance(values, dict):
            raise TypeError("approved batch must be an object")
        schema_version = values.get("schema_version")
        if schema_version == "1.0":
            return ApprovedEvidenceBatch.model_validate(values, strict=True)
        if schema_version == "2.0":
            return ApprovedEvidenceBatchV2.model_validate(values, strict=True)
        raise ValueError("unsupported approved batch schema version")
    except RecursionError as exc:
        raise ValueError("approved batch contains recursive values") from exc


class InMemoryGrowthEvidenceRepository:
    """Single-process atomic reference implementation; not production persistence."""

    def __init__(self, *, failure_injector: Callable[[str], None] | None = None):
        self._lock = RLock()
        self._failure_injector = failure_injector
        self._batches_by_identity: dict[tuple[str, UUID, int], ApprovedEvidenceBatchAny] = {}
        self._identities_by_batch_id: dict[tuple[str, UUID], tuple[str, UUID, int]] = {}
        self._receipts: dict[tuple[str, UUID, int], EvidenceInboxReceipt] = {}
        self._source_summaries: dict[tuple[str, UUID, int], EvidenceSourceSummary] = {}
        self._outbox_events: dict[tuple[str, UUID, int], DomainEvent] = {}
        self._accepted_metric_inputs: dict[tuple[str, UUID, int], AcceptedMetricEvidenceInput] = {}
        self._metric_authorities: dict[
            tuple[str, UUID, int], ApprovedMetricEvidenceAuthorityRecord
        ] = {}

    def _inject(self, stage: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(stage)

    def _resolve_locked(
        self,
        batch: ApprovedEvidenceBatchAny,
        identity: tuple[str, UUID, int],
        *,
        authenticated_user_id: UUID | None = None,
    ) -> EvidenceIngestResult | None:
        existing = self._batches_by_identity.get(identity)
        if existing is not None:
            if authenticated_user_id is not None and existing.user_id != authenticated_user_id:
                return None
            if existing != batch:
                raise EvidenceIngestError(EvidenceIngestErrorCode.IDEMPOTENCY_CONFLICT)
            return EvidenceIngestResult(
                status=EvidenceIngestStatus.REPLAYED,
                receipt=self._receipts[identity],
                accepted_metric_input=self._accepted_metric_inputs.get(identity),
            )
        reused_identity = self._identities_by_batch_id.get((batch.schema_version, batch.batch_id))
        if reused_identity is not None and reused_identity != identity:
            if (
                authenticated_user_id is not None
                and self._batches_by_identity[reused_identity].user_id != authenticated_user_id
            ):
                return None
            raise EvidenceIngestError(EvidenceIngestErrorCode.IDEMPOTENCY_CONFLICT)
        return None

    def resolve_approved_batch(
        self,
        batch: ApprovedEvidenceBatchAny,
        *,
        authenticated_user_id: UUID,
    ) -> EvidenceIngestResult | None:
        validated = revalidate_approved_batch(batch)
        identity = (
            validated.schema_version,
            validated.session_id,
            validated.review_revision,
        )
        with self._lock:
            return self._resolve_locked(
                validated,
                identity,
                authenticated_user_id=authenticated_user_id,
            )

    def append_approved_batch(
        self,
        batch: ApprovedEvidenceBatchAny,
        *,
        metric_authority: ApprovedMetricEvidenceAuthorityRecord | None = None,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> EvidenceIngestResult:
        validated = revalidate_approved_batch(batch)
        identity = (
            validated.schema_version,
            validated.session_id,
            validated.review_revision,
        )
        with self._lock:
            if existing := self._resolve_locked(validated, identity):
                return existing

            authority: ApprovedMetricEvidenceAuthorityRecord | None = None
            if isinstance(validated, ApprovedEvidenceBatchV2):
                if metric_authority is None:
                    raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED)
                try:
                    authority = ApprovedMetricEvidenceAuthorityRecord.model_validate(
                        stored_model_values(metric_authority), strict=True
                    )
                except (AttributeError, TypeError, ValidationError, ValueError) as exc:
                    raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED) from exc
                authority_shots = tuple(
                    sorted(
                        authority.metric_shots,
                        key=lambda item: (item.sequence, str(item.shot_id)),
                    )
                )
                if (
                    authority.approval_kind is not validated.approval_kind
                    or authority.approval_policy_version != validated.approval_policy_version
                    or authority_shots != validated.metric_shots
                ):
                    raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)
                authority = ApprovedMetricEvidenceAuthorityRecord(
                    approval_kind=validated.approval_kind,
                    approval_policy_version=validated.approval_policy_version,
                    metric_shots=validated.metric_shots,
                )

            received_at = clock()
            receipt_id = _UUID4.validate_python(uuid_factory(), strict=True)
            event_id = _UUID4.validate_python(uuid_factory(), strict=True)
            receipt = EvidenceInboxReceipt(
                receipt_id=receipt_id,
                batch_id=validated.batch_id,
                user_id=validated.user_id,
                session_id=validated.session_id,
                report_id=validated.report_id,
                review_revision=validated.review_revision,
                content_sha256=validated.content_sha256,
                received_at=received_at,
            )
            accepted_metric_input = None
            if isinstance(validated, ApprovedEvidenceBatchV2):
                accepted_metric_input = _seal_accepted_metric_evidence_input(
                    AcceptedMetricEvidenceInput(
                        admission_id=receipt.receipt_id,
                        admitted_at=receipt.received_at,
                        batch_schema_version=validated.schema_version,
                        batch_id=validated.batch_id,
                        session_id=validated.session_id,
                        report_id=validated.report_id,
                        review_revision=validated.review_revision,
                        batch_content_sha256=validated.content_sha256,
                        approval_kind=validated.approval_kind,
                        approval_policy_version=validated.approval_policy_version,
                        batch=validated,
                    )
                )
            summary = EvidenceSourceSummary(
                batch_id=validated.batch_id,
                user_id=validated.user_id,
                session_id=validated.session_id,
                report_id=validated.report_id,
                report_revision=validated.report_revision,
                review_revision=validated.review_revision,
                shot_ids=tuple(shot.shot_id for shot in validated.shots),
                source_envelope_ids=validated.source_envelope_ids,
                content_sha256=validated.content_sha256,
            )
            event = DomainEvent(
                event_id=event_id,
                event_type="growth.evidence.batch.accepted.v1",
                aggregate_type="approved_evidence_batch",
                aggregate_id=validated.batch_id,
                aggregate_version=validated.review_revision,
                occurred_at=received_at,
                causation_id=validated.report_id,
                correlation_id=validated.batch_id,
                idempotency_key=(
                    f"{validated.schema_version}:{validated.session_id}:{validated.review_revision}"
                ),
                payload={
                    "batch_id": str(validated.batch_id),
                    "user_id": str(validated.user_id),
                    "session_id": str(validated.session_id),
                    "report_id": str(validated.report_id),
                    "review_revision": validated.review_revision,
                    "content_sha256": validated.content_sha256,
                },
            )

            original = (
                self._batches_by_identity,
                self._identities_by_batch_id,
                self._receipts,
                self._source_summaries,
                self._outbox_events,
                self._accepted_metric_inputs,
                self._metric_authorities,
            )
            try:
                self._batches_by_identity = self._batches_by_identity | {identity: validated}
                self._identities_by_batch_id = self._identities_by_batch_id | {
                    (validated.schema_version, validated.batch_id): identity
                }
                self._inject("batch")
                self._receipts = self._receipts | {identity: receipt}
                self._inject("inbox")
                self._source_summaries = self._source_summaries | {identity: summary}
                self._inject("source_summary")
                self._outbox_events = self._outbox_events | {identity: event}
                if accepted_metric_input is not None:
                    self._accepted_metric_inputs = self._accepted_metric_inputs | {
                        identity: accepted_metric_input
                    }
                if authority is not None:
                    self._metric_authorities = self._metric_authorities | {identity: authority}
                self._inject("metric_authority")
                self._inject("outbox")
            except BaseException:
                (
                    self._batches_by_identity,
                    self._identities_by_batch_id,
                    self._receipts,
                    self._source_summaries,
                    self._outbox_events,
                    self._accepted_metric_inputs,
                    self._metric_authorities,
                ) = original
                raise
            return EvidenceIngestResult(
                status=EvidenceIngestStatus.CREATED,
                receipt=receipt,
                accepted_metric_input=accepted_metric_input,
            )

    def list_batches(self, user_id: UUID) -> tuple[ApprovedEvidenceBatchAny, ...]:
        with self._lock:
            values = (
                revalidate_approved_batch(batch)
                for batch in self._batches_by_identity.values()
                if batch.user_id == user_id
            )
            return tuple(
                sorted(
                    values,
                    key=lambda batch: (
                        batch.approved_at,
                        str(batch.session_id),
                        batch.review_revision,
                        str(batch.batch_id),
                        batch.schema_version,
                    ),
                )
            )

    def list_accepted_metric_inputs(self, user_id: UUID) -> tuple[AcceptedMetricEvidenceInput, ...]:
        """Re-admit trusted stored facts and issue fresh process-local seals.

        A durable Story 8.6 adapter must reconstruct and verify the inbox, batch,
        source-lineage, and approval-authority records before using the private
        sealing capability.  A serialized admission object is never authority.
        """

        with self._lock:
            admitted: list[AcceptedMetricEvidenceInput] = []
            for identity, stored_batch in self._batches_by_identity.items():
                if not isinstance(stored_batch, ApprovedEvidenceBatchV2):
                    continue
                batch = revalidate_approved_batch(stored_batch)
                if not isinstance(batch, ApprovedEvidenceBatchV2) or batch.user_id != user_id:
                    continue
                receipt = self._receipts.get(identity)
                summary = self._source_summaries.get(identity)
                stored_admission = self._accepted_metric_inputs.get(identity)
                stored_authority = self._metric_authorities.get(identity)
                if (
                    receipt is None
                    or summary is None
                    or stored_admission is None
                    or stored_authority is None
                ):
                    raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED)
                try:
                    receipt = EvidenceInboxReceipt.model_validate(
                        stored_model_values(receipt), strict=True
                    )
                    summary = EvidenceSourceSummary.model_validate(
                        stored_model_values(summary), strict=True
                    )
                    authority = ApprovedMetricEvidenceAuthorityRecord.model_validate(
                        stored_model_values(stored_authority), strict=True
                    )
                    expected_receipt = EvidenceInboxReceipt(
                        receipt_id=receipt.receipt_id,
                        batch_id=batch.batch_id,
                        user_id=batch.user_id,
                        session_id=batch.session_id,
                        report_id=batch.report_id,
                        review_revision=batch.review_revision,
                        content_sha256=batch.content_sha256,
                        received_at=receipt.received_at,
                    )
                    expected_summary = EvidenceSourceSummary(
                        batch_id=batch.batch_id,
                        user_id=batch.user_id,
                        session_id=batch.session_id,
                        report_id=batch.report_id,
                        report_revision=batch.report_revision,
                        review_revision=batch.review_revision,
                        shot_ids=tuple(shot.shot_id for shot in batch.shots),
                        source_envelope_ids=batch.source_envelope_ids,
                        content_sha256=batch.content_sha256,
                    )
                    expected_authority = ApprovedMetricEvidenceAuthorityRecord(
                        approval_kind=batch.approval_kind,
                        approval_policy_version=batch.approval_policy_version,
                        metric_shots=batch.metric_shots,
                    )
                    expected = AcceptedMetricEvidenceInput(
                        admission_id=receipt.receipt_id,
                        admitted_at=receipt.received_at,
                        batch_schema_version=batch.schema_version,
                        batch_id=batch.batch_id,
                        session_id=batch.session_id,
                        report_id=batch.report_id,
                        review_revision=batch.review_revision,
                        batch_content_sha256=batch.content_sha256,
                        approval_kind=batch.approval_kind,
                        approval_policy_version=batch.approval_policy_version,
                        batch=batch,
                    )
                except (AttributeError, TypeError, ValidationError, ValueError) as exc:
                    raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED) from exc
                if (
                    batch.content_sha256
                    != canonical_sha256(batch.model_dump(mode="json", exclude={"content_sha256"}))
                    or receipt != expected_receipt
                    or summary != expected_summary
                    or authority != expected_authority
                    or stored_model_values(stored_admission) != stored_model_values(expected)
                ):
                    raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED)
                admitted.append(_seal_accepted_metric_evidence_input(expected))
            return tuple(
                sorted(
                    admitted,
                    key=lambda value: (
                        value.batch.approved_at,
                        str(value.session_id),
                        value.review_revision,
                        str(value.batch_id),
                        str(value.admission_id),
                    ),
                )
            )

    def list_receipts(self) -> tuple[EvidenceInboxReceipt, ...]:
        with self._lock:
            return tuple(self._receipts[key] for key in sorted(self._receipts, key=str))

    def list_source_summaries(self) -> tuple[EvidenceSourceSummary, ...]:
        with self._lock:
            return tuple(
                self._source_summaries[key] for key in sorted(self._source_summaries, key=str)
            )

    def list_outbox_events(self) -> tuple[DomainEvent, ...]:
        with self._lock:
            return deepcopy(
                tuple(self._outbox_events[key] for key in sorted(self._outbox_events, key=str))
            )

    def snapshot(self) -> tuple[object, ...]:
        """Frozen authority-state view used for equality checks and diagnostics."""

        with self._lock:
            return deepcopy(
                (
                    tuple(self._batches_by_identity.items()),
                    tuple(self._identities_by_batch_id.items()),
                    tuple(self._receipts.items()),
                    tuple(self._source_summaries.items()),
                    tuple(self._metric_authorities.items()),
                    tuple(
                        (key, stored_model_values(value))
                        for key, value in self._accepted_metric_inputs.items()
                    ),
                    tuple(self._outbox_events.items()),
                )
            )
