"""Versioned drill catalog domain contracts and pure services."""

from billiards_ai.cloud.training.catalog import (
    BallPlacement,
    DrillCatalog,
    DrillDefinition,
    DrillEligibilityProfile,
    DrillReferenceValidation,
    DrillReviewStatus,
    DrillViolationCode,
    ProgressionRule,
    RegressionRule,
    ReviewLineage,
    SuccessMetricDefinition,
    TablePointMm,
    TableSetup,
    TableTargetZoneMm,
    list_eligible_drills,
    review_drill,
    validate_drill_reference,
)
from billiards_ai.cloud.training.planning_policy import TrainingPlanningPolicyV1

__all__ = [
    "BallPlacement",
    "DrillCatalog",
    "DrillDefinition",
    "DrillEligibilityProfile",
    "DrillReferenceValidation",
    "DrillReviewStatus",
    "DrillViolationCode",
    "ProgressionRule",
    "RegressionRule",
    "ReviewLineage",
    "SuccessMetricDefinition",
    "TablePointMm",
    "TableSetup",
    "TableTargetZoneMm",
    "list_eligible_drills",
    "review_drill",
    "validate_drill_reference",
    "TrainingPlanningPolicyV1",
]
