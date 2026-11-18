"""Drill policy services and compatible exports of shared catalog contracts."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from billiards_ai.shared.models.drill_catalog import (
    BallPlacement as BallPlacement,
)
from billiards_ai.shared.models.drill_catalog import (
    DrillCatalog as DrillCatalog,
)
from billiards_ai.shared.models.drill_catalog import (
    DrillContractModel as DrillContractModel,
)
from billiards_ai.shared.models.drill_catalog import (
    DrillDefinition as DrillDefinition,
)
from billiards_ai.shared.models.drill_catalog import (
    DrillEligibilityProfile as DrillEligibilityProfile,
)
from billiards_ai.shared.models.drill_catalog import (
    DrillReferenceValidation as DrillReferenceValidation,
)
from billiards_ai.shared.models.drill_catalog import (
    DrillReviewStatus as DrillReviewStatus,
)
from billiards_ai.shared.models.drill_catalog import (
    DrillViolationCode as DrillViolationCode,
)
from billiards_ai.shared.models.drill_catalog import (
    ProgressionRule as ProgressionRule,
)
from billiards_ai.shared.models.drill_catalog import (
    RegressionRule as RegressionRule,
)
from billiards_ai.shared.models.drill_catalog import (
    ReviewLineage as ReviewLineage,
)
from billiards_ai.shared.models.drill_catalog import (
    SuccessMetricDefinition as SuccessMetricDefinition,
)
from billiards_ai.shared.models.drill_catalog import (
    TablePointMm as TablePointMm,
)
from billiards_ai.shared.models.drill_catalog import (
    TableSetup as TableSetup,
)
from billiards_ai.shared.models.drill_catalog import (
    TableTargetZoneMm as TableTargetZoneMm,
)
from billiards_ai.shared.models.drill_catalog import (
    drill_model_values as _stored_model_values,
)
from billiards_ai.shared.models.training_plan import DrillReference


def review_drill(
    *,
    current: DrillDefinition,
    new_status: DrillReviewStatus,
    reviewer_id: UUID,
    reviewed_at: datetime,
    reason_code: str,
) -> DrillDefinition:
    """Create a review revision; only RETIRED is terminal in the frozen v1 policy."""

    current = DrillDefinition.model_validate(_stored_model_values(current))
    if current.review_status is DrillReviewStatus.RETIRED:
        raise ValueError("RETIRED is a terminal review status")
    lineage = ReviewLineage(
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        reason_code=reason_code,
    )
    if lineage.reviewed_at <= current.review_lineage.reviewed_at:
        raise ValueError("reviewed_at must advance the review lineage")
    values = current.model_dump(mode="python", exclude={"content_sha256"})
    values.update(
        revision=current.revision + 1,
        supersedes_revision=current.revision,
        supersedes_content_sha256=current.content_sha256,
        review_status=new_status,
        review_lineage=lineage,
    )
    return DrillDefinition.build(**values)


def validate_drill_reference(
    *, catalog: DrillCatalog, reference: DrillReference
) -> DrillReferenceValidation:
    """Resolve a C0 reference against one exact approved catalog head."""

    catalog = DrillCatalog.model_validate(_stored_model_values(catalog))
    reference = DrillReference.model_validate(_stored_model_values(reference))
    if reference.catalog_version != catalog.catalog_version:
        return DrillReferenceValidation(
            is_valid=False,
            violation_codes=(DrillViolationCode.CATALOG_VERSION_MISMATCH,),
        )
    drill = next((item for item in catalog.drills if item.drill_id == reference.drill_id), None)
    if drill is None:
        return DrillReferenceValidation(
            is_valid=False,
            violation_codes=(DrillViolationCode.DRILL_NOT_FOUND,),
        )
    if reference.drill_revision != drill.revision:
        return DrillReferenceValidation(
            is_valid=False,
            violation_codes=(DrillViolationCode.DRILL_REVISION_STALE,),
        )
    if reference.required_capabilities != drill.required_capabilities:
        return DrillReferenceValidation(
            is_valid=False,
            violation_codes=(DrillViolationCode.REQUIRED_CAPABILITIES_MISMATCH,),
        )
    if drill.review_status is not DrillReviewStatus.APPROVED:
        return DrillReferenceValidation(
            is_valid=False,
            violation_codes=(DrillViolationCode.DRILL_NOT_APPROVED,),
        )
    return DrillReferenceValidation(is_valid=True, drill=drill)


def list_eligible_drills(
    *, catalog: DrillCatalog, profile: DrillEligibilityProfile
) -> tuple[DrillDefinition, ...]:
    """Return approved, eligible current heads in catalog-stable order."""

    catalog = DrillCatalog.model_validate(_stored_model_values(catalog))
    profile = DrillEligibilityProfile.model_validate(_stored_model_values(profile))
    return tuple(drill for drill in catalog.drills if not drill.eligibility_violations(profile))
