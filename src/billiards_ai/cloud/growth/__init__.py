"""Approved-only growth evidence ingestion boundary."""

from billiards_ai.cloud.growth.errors import EvidenceIngestError, EvidenceIngestErrorCode
from billiards_ai.cloud.growth.metric_projector import (
    project_ability_metrics,
    project_ability_metrics_with_trace,
)
from billiards_ai.cloud.growth.outcome_repository import (
    PLAN_OUTCOME_EVENT_CLOCK_POLICY_VERSION,
    PLAN_OUTCOME_EVENT_CLOCK_SKEW_TOLERANCE,
    InMemoryPlanOutcomeRepository,
    OutcomeInboxReceipt,
    OutcomeIngestError,
    OutcomeIngestErrorCode,
    OutcomeIngestStatus,
    PlanOutcomeHistoryRecord,
    PlanOutcomeIngestResult,
    PlanOutcomeRepository,
)
from billiards_ai.cloud.growth.outcome_service import (
    AsyncPlanOutcomeIngestService,
    AsyncPlanOutcomeRepository,
    PlanOutcomeIngestService,
)
from billiards_ai.cloud.growth.projector import project_approved_shots
from billiards_ai.cloud.growth.read_service import (
    CurrentGrowthResult,
    CurrentGrowthStatus,
    UserGrowthReadService,
)
from billiards_ai.cloud.growth.repository import (
    ApprovedEvidenceLineage,
    ApprovedSourceEnvelopeLineage,
    EvidenceInboxReceipt,
    EvidenceIngestResult,
    EvidenceIngestStatus,
    EvidenceSourceSummary,
    GrowthEvidenceRepository,
    InMemoryGrowthEvidenceRepository,
)
from billiards_ai.cloud.growth.service import (
    ApprovedEvidenceLineageAuthority,
    ApprovedMetricEvidenceAuthority,
    AsyncEvidenceIngestService,
    AsyncGrowthEvidenceRepository,
    EvidenceIngestService,
)
from billiards_ai.cloud.growth.snapshot_consumer import (
    SnapshotConsumerCompatibilityError,
    growth_trend_lineage_payloads,
    prepare_growth_snapshot_for_consumer,
    prepare_growth_snapshot_v3_for_consumer,
)
from billiards_ai.cloud.growth.snapshot_projector import (
    project_growth_snapshot,
    project_growth_snapshot_v3,
)
from billiards_ai.cloud.growth.snapshot_repository import (
    GrowthSnapshotRepository,
    InMemoryGrowthSnapshotRepository,
    SnapshotIdempotencyConflict,
    SnapshotIntegrityError,
    SnapshotNotFound,
    SnapshotVersionConflict,
)
from billiards_ai.cloud.growth.snapshot_service import (
    AcceptedMetricEvidenceSource,
    AsyncAcceptedMetricEvidenceSource,
    AsyncGrowthSnapshotRepository,
    AsyncGrowthSnapshotService,
    AsyncGrowthSnapshotV3Service,
    GrowthSnapshotService,
    GrowthSnapshotV3Service,
    SnapshotReadFenceMismatch,
    SnapshotSourceError,
)

__all__ = [
    "ApprovedEvidenceLineage",
    "ApprovedEvidenceLineageAuthority",
    "ApprovedMetricEvidenceAuthority",
    "ApprovedSourceEnvelopeLineage",
    "AsyncAcceptedMetricEvidenceSource",
    "AsyncEvidenceIngestService",
    "AsyncGrowthEvidenceRepository",
    "AsyncGrowthSnapshotRepository",
    "AsyncGrowthSnapshotService",
    "AsyncGrowthSnapshotV3Service",
    "AsyncPlanOutcomeIngestService",
    "AsyncPlanOutcomeRepository",
    "EvidenceInboxReceipt",
    "EvidenceIngestError",
    "EvidenceIngestErrorCode",
    "EvidenceIngestResult",
    "EvidenceIngestService",
    "EvidenceIngestStatus",
    "EvidenceSourceSummary",
    "GrowthEvidenceRepository",
    "CurrentGrowthResult",
    "CurrentGrowthStatus",
    "GrowthSnapshotRepository",
    "GrowthSnapshotService",
    "GrowthSnapshotV3Service",
    "InMemoryGrowthEvidenceRepository",
    "InMemoryGrowthSnapshotRepository",
    "InMemoryPlanOutcomeRepository",
    "OutcomeInboxReceipt",
    "OutcomeIngestError",
    "OutcomeIngestErrorCode",
    "OutcomeIngestStatus",
    "PLAN_OUTCOME_EVENT_CLOCK_POLICY_VERSION",
    "PLAN_OUTCOME_EVENT_CLOCK_SKEW_TOLERANCE",
    "PlanOutcomeHistoryRecord",
    "PlanOutcomeIngestResult",
    "PlanOutcomeIngestService",
    "PlanOutcomeRepository",
    "AcceptedMetricEvidenceSource",
    "SnapshotIdempotencyConflict",
    "SnapshotConsumerCompatibilityError",
    "SnapshotIntegrityError",
    "SnapshotNotFound",
    "SnapshotReadFenceMismatch",
    "SnapshotSourceError",
    "SnapshotVersionConflict",
    "UserGrowthReadService",
    "project_approved_shots",
    "project_ability_metrics",
    "project_ability_metrics_with_trace",
    "project_growth_snapshot",
    "project_growth_snapshot_v3",
    "growth_trend_lineage_payloads",
    "prepare_growth_snapshot_for_consumer",
    "prepare_growth_snapshot_v3_for_consumer",
]
