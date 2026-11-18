"""Combined gate, asset, and purpose authorization policy."""

from uuid import UUID

from billiards_ai.shared.models import (
    AssetContext,
    BystanderDisposition,
    MaterialUseDecision,
    MaterialUsePurpose,
    RealVideoGateStatus,
    UseAuthorization,
)

from .errors import GateRepositoryError
from .repository import JsonGateRepository


def _classification_decision(
    disposition: BystanderDisposition, purpose: MaterialUsePurpose
) -> MaterialUseDecision:
    if disposition in {
        BystanderDisposition.NOT_IDENTIFIABLE,
        BystanderDisposition.ANONYMIZED,
    }:
        return MaterialUseDecision.ALLOW
    if disposition is BystanderDisposition.CONSENTED_FOR_SERVICE:
        if purpose in {
            MaterialUsePurpose.SERVICE_ANALYSIS,
            MaterialUsePurpose.NECESSARY_REVIEW,
        }:
            return MaterialUseDecision.ALLOW
        return MaterialUseDecision.REQUIRE_ANONYMIZATION_OR_CONSENT
    if disposition is BystanderDisposition.CONSENTED_FOR_TRAINING:
        if purpose in {MaterialUsePurpose.ANNOTATION, MaterialUsePurpose.TRAINING}:
            return MaterialUseDecision.ALLOW
        return MaterialUseDecision.REQUIRE_ANONYMIZATION_OR_CONSENT
    return MaterialUseDecision.REQUIRE_ANONYMIZATION_OR_CONSENT


def authorize_real_video_use(
    repository: JsonGateRepository,
    gate_revision_id: UUID,
    asset_context: AssetContext,
    requested_purpose: MaterialUsePurpose,
) -> UseAuthorization:
    """The sole public decision combining gate, asset and purpose evidence."""

    try:
        requested, current = repository.resolve_revision(gate_revision_id)
    except GateRepositoryError:
        return UseAuthorization(
            decision=MaterialUseDecision.DENY,
            reason_code="GATE_RECORD_UNAVAILABLE",
            requested_purpose=requested_purpose,
        )
    if requested.revision_id != current.revision_id:
        return UseAuthorization(
            decision=MaterialUseDecision.DENY,
            reason_code="STALE_GATE_REVISION",
            gate_revision_id=requested.revision_id,
            requested_purpose=requested_purpose,
        )
    if current.status is not RealVideoGateStatus.APPROVED:
        return UseAuthorization(
            decision=MaterialUseDecision.DENY,
            reason_code="GATE_NOT_APPROVED",
            gate_revision_id=current.revision_id,
            requested_purpose=requested_purpose,
        )
    if requested_purpose not in current.scope.approved_purposes:
        return UseAuthorization(
            decision=MaterialUseDecision.DENY,
            reason_code="PURPOSE_NOT_APPROVED_BY_GATE",
            gate_revision_id=current.revision_id,
            requested_purpose=requested_purpose,
        )
    if asset_context.captured_under_revision_id != current.revision_id:
        return UseAuthorization(
            decision=MaterialUseDecision.DENY,
            reason_code="ASSET_GATE_REVISION_MISMATCH",
            gate_revision_id=current.revision_id,
            requested_purpose=requested_purpose,
        )
    if asset_context.captured_at < current.approval.approved_at:
        return UseAuthorization(
            decision=MaterialUseDecision.DENY,
            reason_code="ASSET_CAPTURED_BEFORE_APPROVAL",
            gate_revision_id=current.revision_id,
            requested_purpose=requested_purpose,
        )
    if asset_context.gate_scope_fingerprint != current.scope.fingerprint:
        return UseAuthorization(
            decision=MaterialUseDecision.DENY,
            reason_code="ASSET_GATE_SCOPE_MISMATCH",
            gate_revision_id=current.revision_id,
            requested_purpose=requested_purpose,
        )
    source = asset_context.source_asset_provenance
    if source is not None:
        if source.gate_revision_id != current.revision_id:
            return UseAuthorization(
                decision=MaterialUseDecision.DENY,
                reason_code="ANONYMIZATION_SOURCE_GATE_REVISION_MISMATCH",
                gate_revision_id=current.revision_id,
                requested_purpose=requested_purpose,
            )
        if source.gate_scope_fingerprint != current.scope.fingerprint:
            return UseAuthorization(
                decision=MaterialUseDecision.DENY,
                reason_code="ANONYMIZATION_SOURCE_GATE_SCOPE_MISMATCH",
                gate_revision_id=current.revision_id,
                requested_purpose=requested_purpose,
            )
        if source.captured_at < current.approval.approved_at:
            return UseAuthorization(
                decision=MaterialUseDecision.DENY,
                reason_code="ANONYMIZATION_SOURCE_CAPTURED_BEFORE_APPROVAL",
                gate_revision_id=current.revision_id,
                requested_purpose=requested_purpose,
            )
    if not asset_context.bystanders_present:
        return UseAuthorization(
            decision=MaterialUseDecision.ALLOW,
            reason_code="NO_BYSTANDERS_PRESENT",
            gate_revision_id=current.revision_id,
            requested_purpose=requested_purpose,
        )

    decisions: list[MaterialUseDecision] = []
    for item in asset_context.classifications:
        if (
            item.disposition is BystanderDisposition.CONSENTED_FOR_SERVICE
            and item.consent_version != current.scope.service_consent_version
        ) or (
            item.disposition is BystanderDisposition.CONSENTED_FOR_TRAINING
            and item.consent_version != current.scope.training_consent_version
        ):
            decisions.append(MaterialUseDecision.DENY)
            continue
        decisions.append(_classification_decision(item.disposition, requested_purpose))
    if MaterialUseDecision.DENY in decisions:
        decision = MaterialUseDecision.DENY
        reason = "SUBJECT_PURPOSE_NOT_CONSENTED"
    elif MaterialUseDecision.REQUIRE_ANONYMIZATION_OR_CONSENT in decisions:
        decision = MaterialUseDecision.REQUIRE_ANONYMIZATION_OR_CONSENT
        reason = "IDENTIFIABLE_SUBJECT_REQUIRES_REMEDIATION"
    else:
        decision = MaterialUseDecision.ALLOW
        reason = "ALL_SUBJECTS_AUTHORIZED"
    return UseAuthorization(
        decision=decision,
        reason_code=reason,
        gate_revision_id=current.revision_id,
        requested_purpose=requested_purpose,
    )
