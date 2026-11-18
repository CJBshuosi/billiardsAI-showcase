"""Pure, fail-closed ApprovedEvidenceBatch v2 publication builder."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import UUID4, BaseModel, TypeAdapter, ValidationError

from billiards_ai.cloud.engine.scoring import score_session_performance
from billiards_ai.shared.models.events import DomainEvent, canonical_sha256
from billiards_ai.shared.models.evidence import ApprovedEvidenceBatchV2
from billiards_ai.shared.models.metric_evidence import (
    ApprovedMetricEvidenceAuthorityRecord,
    EvidenceApprovalKind,
)
from billiards_ai.shared.models.quality import (
    SessionQualityStatus,
    SessionQualitySummary,
    ShotAnalysisStatus,
)
from billiards_ai.shared.models.report import ReportDocument, ReportStatus
from billiards_ai.shared.models.review import ApprovedShotData, ReviewTaskStatus
from billiards_ai.shared.models.scoring import SessionPerformanceScores
from billiards_ai.shared.models.shot import Confidence, ReviewStatus, Shot, TargetZoneSource
from billiards_ai.shared.models.statistics import (
    CushionCountFrequency,
    SessionStatistics,
    TargetZoneDeviationSummary,
)

if TYPE_CHECKING:
    from billiards_ai.cloud.review.persistence.publication import (
        ReviewPublicationAuthority,
        ReviewSessionOwnership,
    )

_UUID4_ADAPTER = TypeAdapter(UUID4)


class BatchBuilderErrorCode(StrEnum):
    NO_PROJECTABLE_APPROVED_EVIDENCE = "NO_PROJECTABLE_APPROVED_EVIDENCE"
    INVALID_PUBLICATION_AUTHORITY = "INVALID_PUBLICATION_AUTHORITY"
    INVALID_METRIC_AUTHORITY = "INVALID_METRIC_AUTHORITY"


class ApprovedEvidenceBatchBuilderError(ValueError):
    """Stable pure-domain failure which does not expose validation payloads."""

    def __init__(self, code: BatchBuilderErrorCode):
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True, slots=True)
class ApprovedEvidencePublication:
    batch: ApprovedEvidenceBatchV2
    event: DomainEvent


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


def _invalid_publication() -> ApprovedEvidenceBatchBuilderError:
    return ApprovedEvidenceBatchBuilderError(BatchBuilderErrorCode.INVALID_PUBLICATION_AUTHORITY)


def _invalid_metric() -> ApprovedEvidenceBatchBuilderError:
    return ApprovedEvidenceBatchBuilderError(BatchBuilderErrorCode.INVALID_METRIC_AUTHORITY)


def _project_shot(shot: Shot) -> ApprovedShotData:
    """Mirror the existing approved-only narrow projection without importing growth."""

    return ApprovedShotData(
        shot_id=shot.id,
        session_id=shot.session_id,
        sequence=shot.sequence,
        struck_ball=shot.struck_ball,
        potted_balls=tuple(shot.potted_balls or ()),
        cushion_count=shot.cushion_count,
        travel_distance_mm=shot.travel_distance_mm,
        is_foul=shot.is_foul,
        deviation_mm=shot.deviation_mm,
        target_zone_source=shot.target_zone_source,
        confidence=shot.confidence,
        analysis_status=shot.analysis_status,
        review_status=shot.review_status,
    )


def _project_statistics(
    shots: tuple[ApprovedShotData, ...],
) -> tuple[SessionStatistics, SessionPerformanceScores]:
    """Rebuild only facts supported by the deliberately lossy approved projection."""

    valid = tuple(
        shot
        for shot in shots
        if shot.analysis_status is ShotAnalysisStatus.ANALYZED
        and shot.confidence is not Confidence.ABSTAIN
    )
    if not valid:
        raise ApprovedEvidenceBatchBuilderError(
            BatchBuilderErrorCode.NO_PROJECTABLE_APPROVED_EVIDENCE
        )

    abstained_count = len(shots) - len(valid)
    abstention_rate = abstained_count / len(shots)
    quality = SessionQualitySummary(
        shot_count=len(shots),
        abstained_shot_count=abstained_count,
        abstention_rate=abstention_rate,
        session_quality_status=(
            SessionQualityStatus.LOW_COVERAGE
            if abstention_rate > 0.5
            else SessionQualityStatus.SUFFICIENT_COVERAGE
        ),
    )
    travel_distances = tuple(
        shot.travel_distance_mm for shot in valid if shot.travel_distance_mm is not None
    )
    cushion_counts = Counter(shot.cushion_count for shot in valid if shot.cushion_count is not None)
    deviations: dict[TargetZoneSource, list[int]] = defaultdict(list)
    for shot in valid:
        if shot.deviation_mm is not None and shot.target_zone_source is not TargetZoneSource.NONE:
            deviations[shot.target_zone_source].append(shot.deviation_mm)

    statistics = SessionStatistics(
        total_shot_count=len(shots),
        valid_shot_count=len(valid),
        session_quality=quality,
        potted_ball_count=sum(len(shot.potted_balls) for shot in valid),
        successful_potting_shot_count=0,
        potting_shot_count=0,
        pot_rate=None,
        average_cue_ball_travel_distance_mm=(
            sum(travel_distances) / len(travel_distances) if travel_distances else None
        ),
        cue_ball_travel_shot_count=len(travel_distances),
        cushion_distribution=tuple(
            CushionCountFrequency(cushion_count=count, shot_count=frequency)
            for count, frequency in sorted(cushion_counts.items())
        ),
        cushion_count_shot_count=sum(cushion_counts.values()),
        cue_ball_end_distribution=(),
        cue_ball_end_shot_count=0,
        target_zone_deviation_by_source=tuple(
            TargetZoneDeviationSummary(
                target_zone_source=source,
                shot_count=len(values),
                average_deviation_mm=sum(values) / len(values),
            )
            for source in (TargetZoneSource.SCENE_PRESET, TargetZoneSource.POST_HOC_INFERRED)
            if (values := deviations.get(source))
        ),
    )
    return statistics, score_session_performance(statistics)


class ApprovedEvidenceBatchV2Builder:
    """Build one deterministic Batch/Event pair exclusively from authority facts."""

    @staticmethod
    def build(
        *,
        result_report: ReportDocument,
        result_shots: tuple[Shot, ...],
        source_envelope_ids: Mapping[UUID, UUID],
        ownership: ReviewSessionOwnership,
        authority: ReviewPublicationAuthority,
    ) -> ApprovedEvidencePublication:
        from billiards_ai.cloud.review.persistence.publication import (
            ReviewPublicationAuthority,
            ReviewSessionOwnership,
        )

        try:
            if type(result_shots) is not tuple or not isinstance(source_envelope_ids, Mapping):
                raise TypeError("authority containers must use their frozen shapes")
            ownership_record = ReviewSessionOwnership.model_validate(
                _stored_model_values(ownership), strict=True
            )
            authority_values = _stored_model_values(authority)
            if (
                type(authority_values.get("task_status")) is not ReviewTaskStatus
                or type(authority_values.get("approved_at")) is not datetime
                or type(authority_values.get("review_revision")) is not int
                or authority_values["review_revision"] < 1
                or authority_values["approved_at"].tzinfo is None
                or authority_values["approved_at"].utcoffset()
                != UTC.utcoffset(authority_values["approved_at"])
            ):
                raise TypeError("terminal status and approval time require exact types")
            for field in (
                "task_id",
                "session_id",
                "result_report_id",
                "batch_id",
                "event_id",
            ):
                _UUID4_ADAPTER.validate_python(authority_values[field], strict=True)
            publication_authority = ReviewPublicationAuthority.model_construct(**authority_values)
            report = ReportDocument.model_validate(_stored_model_values(result_report), strict=True)
            shots = tuple(
                Shot.model_validate(_stored_model_values(shot), strict=True)
                for shot in result_shots
            )
            for shot in shots:
                _UUID4_ADAPTER.validate_python(shot.id, strict=True)
                _UUID4_ADAPTER.validate_python(shot.session_id, strict=True)
            for shot_id, envelope_id in source_envelope_ids.items():
                _UUID4_ADAPTER.validate_python(shot_id, strict=True)
                _UUID4_ADAPTER.validate_python(envelope_id, strict=True)
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise _invalid_publication() from exc

        task_status = publication_authority.task_status
        approved_at = publication_authority.approved_at
        review_revision = publication_authority.review_revision
        expected_report_status = {
            ReviewTaskStatus.APPROVED: ReportStatus.APPROVED,
            ReviewTaskStatus.CORRECTED: ReportStatus.CORRECTED,
        }.get(task_status)
        if (
            type(task_status) is not ReviewTaskStatus
            or expected_report_status is None
            or ownership_record.task_id != publication_authority.task_id
            or ownership_record.session_id != publication_authority.session_id
            or publication_authority.session_id != report.session_id
            or publication_authority.result_report_id != report.report_id
            or report.report_status is not expected_report_status
            or review_revision != report.revision
            or not shots
            or len(shots) != report.session_statistics.total_shot_count
            or ownership_record.bound_at > approved_at
            or approved_at < report.generated_at
            or any(
                shot.created_at.tzinfo is None or shot.created_at > approved_at for shot in shots
            )
            or any(shot.session_id != report.session_id for shot in shots)
        ):
            raise _invalid_publication()

        ordered = tuple(sorted(shots, key=lambda shot: (shot.sequence, shot.id)))
        shot_keys = tuple((shot.sequence, shot.id) for shot in shots)
        allowed_statuses = {
            ReviewStatus.APPROVED,
            ReviewStatus.CORRECTED,
            ReviewStatus.REJECTED,
        }
        if (
            shots != ordered
            or len(shot_keys) != len(set(shot_keys))
            or {shot.sequence for shot in shots} != set(range(1, len(shots) + 1))
            or len({shot.sequence for shot in shots}) != len(shots)
            or len({shot.id for shot in shots}) != len(shots)
            or any(shot.review_status not in allowed_statuses for shot in shots)
            or (
                task_status is ReviewTaskStatus.APPROVED
                and any(shot.review_status is ReviewStatus.CORRECTED for shot in shots)
            )
            or (
                task_status is ReviewTaskStatus.CORRECTED
                and not any(shot.review_status is ReviewStatus.CORRECTED for shot in shots)
            )
        ):
            raise _invalid_publication()

        source_keys = set(source_envelope_ids)
        result_ids = {shot.id for shot in shots}
        envelope_values = tuple(source_envelope_ids.get(shot.id) for shot in shots)
        if (
            source_keys != result_ids
            or any(value is None for value in envelope_values)
            or len(set(envelope_values)) != len(envelope_values)
        ):
            raise _invalid_publication()

        retained_shots = tuple(
            shot for shot in shots if shot.review_status is not ReviewStatus.REJECTED
        )
        if not retained_shots:
            raise ApprovedEvidenceBatchBuilderError(
                BatchBuilderErrorCode.NO_PROJECTABLE_APPROVED_EVIDENCE
            )

        try:
            metric = ApprovedMetricEvidenceAuthorityRecord.model_validate(
                _stored_model_values(authority_values["metric_authority"]), strict=True
            )
        except (AttributeError, KeyError, TypeError, ValueError, ValidationError) as exc:
            raise _invalid_metric() from exc
        if (
            metric.approval_kind is not EvidenceApprovalKind.HUMAN_REVIEW
            or metric.approval_policy_version != "human-review-v1"
        ):
            raise _invalid_metric()
        try:
            publication_authority = ReviewPublicationAuthority.model_validate(
                authority_values, strict=True
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise _invalid_publication() from exc

        approved_shots = tuple(_project_shot(shot) for shot in retained_shots)
        statistics, scores = _project_statistics(approved_shots)

        expected_metric_keys = tuple(
            (shot.shot_id, shot.session_id, shot.sequence) for shot in approved_shots
        )
        actual_metric_keys = tuple(
            (shot.shot_id, shot.session_id, shot.sequence) for shot in metric.metric_shots
        )
        calibration_keys = {shot.table_geometry.calibration_key for shot in metric.metric_shots}
        if (
            metric.approval_kind is not EvidenceApprovalKind.HUMAN_REVIEW
            or metric.approval_policy_version != "human-review-v1"
            or actual_metric_keys != expected_metric_keys
            or len(actual_metric_keys) != len(set(actual_metric_keys))
            or any(shot.review_revision != review_revision for shot in metric.metric_shots)
            or len(calibration_keys) != 1
        ):
            raise _invalid_metric()

        retained_envelopes = tuple(source_envelope_ids[shot.id] for shot in retained_shots)
        payload = {
            "schema_version": "2.0",
            "batch_id": publication_authority.batch_id,
            "user_id": ownership_record.user_id,
            "session_id": report.session_id,
            "report_id": report.report_id,
            "report_revision": report.revision,
            "review_revision": review_revision,
            "approved_at": approved_at,
            "approval_kind": EvidenceApprovalKind.HUMAN_REVIEW,
            "approval_policy_version": "human-review-v1",
            "shots": approved_shots,
            "metric_shots": metric.metric_shots,
            "statistics": statistics,
            "scores": scores,
            "source_envelope_ids": retained_envelopes,
        }
        try:
            batch = ApprovedEvidenceBatchV2.model_validate(
                payload | {"content_sha256": canonical_sha256(payload)}, strict=True
            )
            event = DomainEvent(
                event_id=publication_authority.event_id,
                event_type="evidence.session.approved.v1",
                aggregate_type="review_session",
                aggregate_id=batch.session_id,
                aggregate_version=batch.review_revision,
                occurred_at=approved_at,
                causation_id=publication_authority.task_id,
                correlation_id=batch.session_id,
                idempotency_key=(f"evidence-approved:{batch.session_id}:{batch.review_revision}"),
                payload={"approvedEvidenceBatch": batch.model_dump(mode="json")},
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise _invalid_publication() from exc
        return ApprovedEvidencePublication(batch=batch, event=event)


__all__ = [
    "ApprovedEvidenceBatchBuilderError",
    "ApprovedEvidenceBatchV2Builder",
    "ApprovedEvidencePublication",
    "BatchBuilderErrorCode",
]
