"""Strict review authority, audit facts, and safe downstream projections."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from billiards_ai.shared.constants import CONFIDENCE_ABSTAIN_BELOW
from billiards_ai.shared.models.access import AccessActorRole, ReviewerIdentity
from billiards_ai.shared.models.calibration import CalibrationLineage, CalibrationStatus
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.quality import (
    FinalFieldSource,
    SessionQualitySummary,
    ShotAnalysisStatus,
)
from billiards_ai.shared.models.report import (
    CalibratedReportSessionStatistics,
    CoachingRecommendation,
    ObjectiveStatisticsSection,
    ReportDocument,
    ReportHeadline,
    ReportQualityNotice,
    ReportSessionStatistics,
    ReportTrustScope,
    TrainingRecommendationSection,
    TypicalShotAnalysisSection,
    TypicalShotSummary,
)
from billiards_ai.shared.models.scoring import SessionPerformanceScores
from billiards_ai.shared.models.segmentation import BallPosition
from billiards_ai.shared.models.shot import Confidence, ReviewStatus, TargetZoneSource

ReviewNote = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ReviewValue = StrictInt | StrictBool | tuple[StrictInt, ...] | None


class ReviewStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


class ReviewTaskStatus(StrEnum):
    PENDING_REVIEW = "pending_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class SessionRejectionDetail(StrEnum):
    """Controlled audit explanations; arbitrary reviewer text is never public."""

    CAMERA_VIEW_OBSTRUCTED = "camera_view_obstructed"
    RECORDING_WINDOW_MISMATCH = "recording_window_does_not_cover_session"
    RECORDING_QUALITY_INSUFFICIENT = "recording_quality_insufficient_for_reliable_analysis"


class UserNotificationMessage(StrEnum):
    """Fixed, safe user copy for a rejected session."""

    CAMERA_VIEW_OBSTRUCTED = "这次录像因画面被遮挡，没能分析成功。"
    RECORDING_WINDOW_MISMATCH = "这次录像未覆盖正确的时间段，没能分析成功。"
    RECORDING_QUALITY_INSUFFICIENT = "这次录像质量不足，没能可靠分析。"


class SessionRejectionReason(StrEnum):
    OCCLUDED = "occluded"
    WRONG_TIME_WINDOW = "wrong_time_window"
    INSUFFICIENT_MATERIAL_QUALITY = "insufficient_material_quality"

    @property
    def detail(self) -> SessionRejectionDetail:
        return {
            SessionRejectionReason.OCCLUDED: SessionRejectionDetail.CAMERA_VIEW_OBSTRUCTED,
            SessionRejectionReason.WRONG_TIME_WINDOW: (
                SessionRejectionDetail.RECORDING_WINDOW_MISMATCH
            ),
            SessionRejectionReason.INSUFFICIENT_MATERIAL_QUALITY: (
                SessionRejectionDetail.RECORDING_QUALITY_INSUFFICIENT
            ),
        }[self]

    @property
    def notification_message(self) -> UserNotificationMessage:
        return {
            SessionRejectionReason.OCCLUDED: UserNotificationMessage.CAMERA_VIEW_OBSTRUCTED,
            SessionRejectionReason.WRONG_TIME_WINDOW: (
                UserNotificationMessage.RECORDING_WINDOW_MISMATCH
            ),
            SessionRejectionReason.INSUFFICIENT_MATERIAL_QUALITY: (
                UserNotificationMessage.RECORDING_QUALITY_INSUFFICIENT
            ),
        }[self]


class UserNotificationStatus(StrEnum):
    PENDING = "pending"


class ReviewableField(StrEnum):
    """Safe business fields that a reviewer may correct in Phase 0."""

    STRUCK_BALL = "struck_ball"
    POTTED_BALLS = "potted_balls"
    CUSHION_COUNT = "cushion_count"
    TRAVEL_DISTANCE_MM = "travel_distance_mm"
    IS_FOUL = "is_foul"
    DEVIATION_MM = "deviation_mm"


class ShotReviewOutcome(StrEnum):
    APPROVED = "approved"
    CORRECTED = "corrected"
    REJECTED = "rejected"


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("datetime must use UTC")
    return value.astimezone(UTC)


def _validate_field_value(field: ReviewableField, value: ReviewValue) -> None:
    if field is ReviewableField.POTTED_BALLS:
        if not isinstance(value, tuple) or any(type(item) is not int or item < 0 for item in value):
            raise ValueError("potted_balls values must be a tuple of non-negative integers")
        return
    if field is ReviewableField.IS_FOUL:
        if value is not None and type(value) is not bool:
            raise ValueError("is_foul values must be booleans or null")
        return
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{field.value} values must be non-negative integers or null")


class ReviewCorrection(ReviewStrictFrozenModel):
    """Immutable AI/correct-value pair suitable for audit and learning."""

    shot_id: UUID4
    field: ReviewableField
    ai_value: ReviewValue
    correct_value: ReviewValue
    note: ReviewNote
    reviewer: ReviewerIdentity
    reviewed_at: datetime

    _reviewed_at_utc = field_validator("reviewed_at")(_require_utc)

    @model_validator(mode="after")
    def values_match_field_and_represent_a_change(self) -> ReviewCorrection:
        _validate_field_value(self.field, self.ai_value)
        _validate_field_value(self.field, self.correct_value)
        if type(self.ai_value) is type(self.correct_value) and self.ai_value == self.correct_value:
            raise ValueError("a correction must change the AI value")
        return self


class ShotReviewDecision(ReviewStrictFrozenModel):
    shot_id: UUID4
    outcome: ShotReviewOutcome
    reviewer: ReviewerIdentity
    reviewed_at: datetime
    corrections: tuple[ReviewCorrection, ...] = ()
    trajectory_reviewed: StrictBool = False

    _reviewed_at_utc = field_validator("reviewed_at")(_require_utc)

    @model_validator(mode="after")
    def correction_pairing_is_consistent(self) -> ShotReviewDecision:
        if AccessActorRole.REVIEWER not in self.reviewer.roles:
            raise ValueError("shot decisions require the REVIEWER role")
        if (self.outcome is ShotReviewOutcome.CORRECTED) != bool(self.corrections):
            raise ValueError("corrected decisions require correction records exclusively")
        fields: set[ReviewableField] = set()
        for correction in self.corrections:
            if correction.shot_id != self.shot_id:
                raise ValueError("correction shot must match its decision shot")
            if correction.reviewer != self.reviewer:
                raise ValueError("correction reviewer must match its decision reviewer")
            if correction.reviewed_at != self.reviewed_at:
                raise ValueError("correction time must match its decision time")
            if correction.field in fields:
                raise ValueError("a field may be corrected only once per shot")
            fields.add(correction.field)
        return self


class ReviewShotValues(ReviewStrictFrozenModel):
    struck_ball: StrictInt | None
    potted_balls: tuple[StrictInt, ...]
    cushion_count: StrictInt | None
    travel_distance_mm: StrictInt | None
    is_foul: StrictBool | None
    deviation_mm: StrictInt | None
    target_zone_source: TargetZoneSource
    confidence: Confidence


class ReviewActualPathUnavailableReason(StrEnum):
    NOT_OBSERVED = "NOT_OBSERVED"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class ReviewActualPath(ReviewStrictFrozenModel):
    points: tuple[Point, ...]
    source: FinalFieldSource
    confidence: float = Field(ge=0.0, le=1.0)
    calibration: CalibrationLineage | None = None
    unavailable_reason: ReviewActualPathUnavailableReason | None = None

    @model_validator(mode="after")
    def path_is_canonical_table_mm(self) -> ReviewActualPath:
        if self.points:
            if len(set(self.points)) < 2:
                raise ValueError("reviewed trajectory requires at least two distinct points")
            if self.source is FinalFieldSource.UNKNOWN or self.confidence <= 0.0:
                raise ValueError("reviewed trajectory requires source and confidence")
            if self.calibration is None:
                raise ValueError("reviewed trajectory requires its generation calibration")
            if self.unavailable_reason is not None:
                raise ValueError("reviewed trajectory cannot carry an unavailable reason")
        elif (
            self.source is not FinalFieldSource.UNKNOWN
            or self.confidence != 0.0
            or self.calibration is not None
        ):
            raise ValueError("missing reviewed trajectory cannot carry provenance")
        if any(
            type(point.x_mm) is not int or type(point.y_mm) is not int or not point.is_on_table()
            for point in self.points
        ):
            raise ValueError("reviewed trajectory points must stay on the table")
        return self


class ShotReviewEvidence(ReviewStrictFrozenModel):
    shot_id: UUID4
    session_id: UUID4
    sequence: StrictInt = Field(ge=1)
    analysis_status: ShotAnalysisStatus
    review_status: ReviewStatus
    shot_created_at: datetime
    ai_values: ReviewShotValues
    actual_path: ReviewActualPath = ReviewActualPath(
        points=(),
        source=FinalFieldSource.UNKNOWN,
        confidence=0.0,
    )

    _shot_created_at_utc = field_validator("shot_created_at")(_require_utc)


class ReviewEvidence(ReviewStrictFrozenModel):
    session_id: UUID4
    shots: tuple[ShotReviewEvidence, ...]

    @model_validator(mode="after")
    def shot_set_is_consistent(self) -> ReviewEvidence:
        if not self.shots:
            raise ValueError("review evidence requires at least one shot")
        if any(shot.session_id != self.session_id for shot in self.shots):
            raise ValueError("review evidence shots must belong to its session")
        if len({shot.shot_id for shot in self.shots}) != len(self.shots):
            raise ValueError("review evidence shots must be unique")
        if {shot.sequence for shot in self.shots} != set(range(1, len(self.shots) + 1)):
            raise ValueError("review evidence requires a complete contiguous shot sequence")
        return self


class ReportRevisionReplacements(ReviewStrictFrozenModel):
    """Caller-verified report replacements; absent fields preserve the parent summary."""

    quality_summary: SessionQualitySummary | None = None
    session_statistics: CalibratedReportSessionStatistics | ReportSessionStatistics | None = None
    dimension_scores: SessionPerformanceScores | None = None
    typical_shots: tuple[TypicalShotSummary, ...] | None = None
    coaching_recommendations: tuple[CoachingRecommendation, ...] | None = None
    headline: ReportHeadline | None = None
    trust_scope: ReportTrustScope | None = None
    objective_statistics: ObjectiveStatisticsSection | None = None
    typical_shot_analysis: TypicalShotAnalysisSection | None = None
    training_recommendation: TrainingRecommendationSection | None = None
    quality_notice: ReportQualityNotice | None = None


class ReportNotDeliverableReason(StrEnum):
    REPORT_NOT_FOUND = "report_not_found"
    REVIEW_TASK_NOT_FOUND = "review_task_not_found"
    SESSION_MISMATCH = "session_mismatch"
    REVISION_MISMATCH = "revision_mismatch"
    REVIEW_PENDING = "review_pending"
    REVIEW_IN_PROGRESS = "review_in_progress"
    REVIEW_REJECTED = "review_rejected"
    UNKNOWN_REVIEW_STATUS = "unknown_review_status"
    REPORT_NOT_APPROVED = "report_not_approved"
    REPORT_STATUS_MISMATCH = "report_status_mismatch"
    SHOT_SET_INCOMPLETE = "shot_set_incomplete"
    UNPROCESSED_CRITICAL_SHOT = "unprocessed_critical_shot"
    UNKNOWN_SHOT_REVIEW_STATUS = "unknown_shot_review_status"


class ReviewTask(ReviewStrictFrozenModel):
    """Session authority bound to one immutable report revision."""

    id: UUID4
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)
    status: ReviewTaskStatus


class SessionRejection(ReviewStrictFrozenModel):
    """Immutable session-level rejection audit without raw or media fields."""

    rejection_id: UUID4
    review_task_id: UUID4
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)
    reason_code: SessionRejectionReason
    detail: SessionRejectionDetail
    reviewer: ReviewerIdentity
    rejected_at: datetime

    _rejected_at_utc = field_validator("rejected_at")(_require_utc)

    @model_validator(mode="after")
    def audit_is_authorized_and_controlled(self) -> SessionRejection:
        reviewer = ReviewerIdentity.model_validate(_stored_model_values(self.reviewer))
        if AccessActorRole.REVIEWER not in reviewer.roles:
            raise ValueError("session rejection requires the REVIEWER role")
        if self.detail is not self.reason_code.detail:
            raise ValueError("rejection detail must match its reason code")
        return self


class UserNotificationEvent(ReviewStrictFrozenModel):
    """A pending notification fact for a future delivery layer; no I/O occurs here."""

    event_id: UUID4
    session_id: UUID4
    reason_code: SessionRejectionReason
    message: UserNotificationMessage
    created_at: datetime
    status: UserNotificationStatus

    _created_at_utc = field_validator("created_at")(_require_utc)

    @model_validator(mode="after")
    def message_and_status_match_reason(self) -> UserNotificationEvent:
        if self.message is not self.reason_code.notification_message:
            raise ValueError("notification message must match its reason code")
        if self.status is not UserNotificationStatus.PENDING:
            raise ValueError("new rejection notifications must remain pending")
        return self


class SessionRejectionResult(ReviewStrictFrozenModel):
    """Atomic typed facts produced by the session rejection transition."""

    review_task: ReviewTask
    rejection: SessionRejection
    notification: UserNotificationEvent

    @model_validator(mode="after")
    def terminal_facts_are_consistent(self) -> SessionRejectionResult:
        task = ReviewTask.model_validate(_stored_model_values(self.review_task))
        rejection = SessionRejection.model_validate(_stored_model_values(self.rejection))
        notification = UserNotificationEvent.model_validate(_stored_model_values(self.notification))
        if task.status is not ReviewTaskStatus.REJECTED:
            raise ValueError("session rejection result requires a rejected review task")
        if (
            rejection.review_task_id != task.id
            or rejection.session_id != task.session_id
            or rejection.report_id != task.report_id
            or rejection.report_revision != task.report_revision
        ):
            raise ValueError("rejection audit must bind the exact rejected review task")
        if (
            notification.session_id != rejection.session_id
            or notification.reason_code is not rejection.reason_code
            or notification.created_at != rejection.rejected_at
        ):
            raise ValueError("notification must match the rejection session, reason, and time")
        return self


class RejectedSessionOutcome(ReviewStrictFrozenModel):
    """Safe query result for consumers that encounter a rejected session."""

    session_id: UUID4
    status: ReviewTaskStatus
    reason_code: SessionRejectionReason
    notification_event_id: UUID4
    notification_status: UserNotificationStatus

    @model_validator(mode="after")
    def statuses_are_terminal_and_pending_delivery(self) -> RejectedSessionOutcome:
        if self.status is not ReviewTaskStatus.REJECTED:
            raise ValueError("rejected session outcomes require rejected status")
        if self.notification_status is not UserNotificationStatus.PENDING:
            raise ValueError("rejected session notification must remain pending")
        return self


class ReportDeliveryDecision(ReviewStrictFrozenModel):
    deliverable: StrictBool
    reason: ReportNotDeliverableReason | None = None

    @model_validator(mode="after")
    def reason_matches_decision(self) -> ReportDeliveryDecision:
        if self.deliverable == (self.reason is not None):
            raise ValueError(
                "deliverable decisions must have no reason; blocked decisions need a reason"
            )
        return self


class ApprovedShotData(ReviewStrictFrozenModel):
    """Downstream projection; review/debug-only fields are deliberately absent."""

    shot_id: UUID4
    session_id: UUID4
    sequence: StrictInt = Field(ge=1)
    struck_ball: StrictInt | None
    potted_balls: tuple[StrictInt, ...]
    cushion_count: StrictInt | None
    travel_distance_mm: StrictInt | None
    is_foul: StrictBool | None
    deviation_mm: StrictInt | None
    target_zone_source: TargetZoneSource = TargetZoneSource.NONE
    confidence: Confidence
    analysis_status: ShotAnalysisStatus
    review_status: ReviewStatus


class ApprovedSessionData(ReviewStrictFrozenModel):
    """Only data that passed the unified report delivery gate."""

    session_id: UUID4
    report: ReportDocument
    shots: tuple[ApprovedShotData, ...]


class VisualizationReviewStrictFrozenModel(ReviewStrictFrozenModel):
    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        return type(self).model_validate(_stored_model_values(copied))


class ApprovedActualPathStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NOT_REVIEWED = "NOT_REVIEWED"


class ApprovedActualPathReason(StrEnum):
    NOT_OBSERVED = "NOT_OBSERVED"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NOT_REVIEWED = "NOT_REVIEWED"


class ApprovedActualPath(VisualizationReviewStrictFrozenModel):
    status: ApprovedActualPathStatus
    reason: ApprovedActualPathReason | None = None
    points: tuple[Point, ...] = ()
    source: FinalFieldSource | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def availability_and_geometry_are_consistent(self) -> ApprovedActualPath:
        if self.status is ApprovedActualPathStatus.AVAILABLE:
            if (
                self.reason is not None
                or len(set(self.points)) < 2
                or self.source in {None, FinalFieldSource.UNKNOWN}
                or self.confidence is None
                or self.confidence < CONFIDENCE_ABSTAIN_BELOW
            ):
                raise ValueError(
                    "available actual path requires reviewed geometry and threshold confidence"
                )
            if any(
                type(point.x_mm) is not int
                or type(point.y_mm) is not int
                or not point.is_on_table()
                for point in self.points
            ):
                raise ValueError("actual path must stay on the table")
        elif (
            self.reason is None
            or self.points
            or self.source is not None
            or self.confidence is not None
        ):
            raise ValueError("unavailable actual path cannot expose geometry or provenance")
        return self


class ApprovedVisualizationProjection(VisualizationReviewStrictFrozenModel):
    schema_version: str = "1.0"
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)
    shot_id: UUID4
    shot_sequence: StrictInt = Field(ge=1)
    struck_ball: StrictInt | None = Field(default=None, ge=1, le=15)
    ball_positions: tuple[BallPosition, ...]
    actual_path: ApprovedActualPath
    calibration_id: UUID4
    calibration_revision_id: UUID4
    calibration_revision_number: StrictInt = Field(ge=1)
    calibration_status: CalibrationStatus
    table_profile_id: ReviewNote
    table_length_mm: StrictInt
    table_width_mm: StrictInt
    table_configuration_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def source_is_versioned_canonical_and_supported(self) -> ApprovedVisualizationProjection:
        if self.schema_version != "1.0":
            raise ValueError("unsupported approved visualization projection schema")
        if (self.table_length_mm, self.table_width_mm) != (2540, 1270):
            raise ValueError("visualization projection requires the supported 2:1 table")
        ball_ids = tuple(item.ball_id for item in self.ball_positions)
        if ball_ids != tuple(sorted(set(ball_ids))):
            raise ValueError("visualization ball positions must be sorted and unique")
        if 0 not in ball_ids:
            raise ValueError("visualization projection requires the cue ball")
        if any(
            type(item.position.x_mm) is not int
            or type(item.position.y_mm) is not int
            or not item.position.is_on_table()
            for item in self.ball_positions
        ):
            raise ValueError("visualization ball positions must be integer table millimetres")
        if self.struck_ball is not None and self.struck_ball not in ball_ids:
            raise ValueError("struck ball must be present in the starting layout")
        cue_start = next(item.position for item in self.ball_positions if item.ball_id == 0)
        if (
            self.actual_path.status is ApprovedActualPathStatus.AVAILABLE
            and self.actual_path.points[0] != cue_start
        ):
            raise ValueError("actual path must start at the projected cue-ball position")
        return self


class VisualizationReviewAuthorization(VisualizationReviewStrictFrozenModel):
    review_task_id: UUID4
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)
    shot_id: UUID4
    shot_sequence: StrictInt = Field(ge=1)
    shot_created_at: datetime
    actual_path: ReviewActualPath
    decision: ShotReviewDecision

    _shot_created_at_utc = field_validator("shot_created_at")(_require_utc)

    @model_validator(mode="after")
    def decision_explicitly_authorizes_the_path(self) -> VisualizationReviewAuthorization:
        decision = ShotReviewDecision.model_validate(_stored_model_values(self.decision))
        if decision.outcome not in {
            ShotReviewOutcome.APPROVED,
            ShotReviewOutcome.CORRECTED,
        }:
            raise ValueError("visualization authorization requires an approved shot")
        if decision.shot_id != self.shot_id:
            raise ValueError("visualization authorization must bind the reviewed shot")
        if bool(self.actual_path.points) is not decision.trajectory_reviewed:
            raise ValueError("trajectory review flag must match path availability")
        return self
