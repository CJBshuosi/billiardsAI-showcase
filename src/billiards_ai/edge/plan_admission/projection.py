"""Build the only executable edge projection from an exact verified v3 payload."""

from __future__ import annotations

from billiards_ai.edge.coach_runtime.execution_policy import (
    DrillExecutionProjection,
    ExecutionBlockProjection,
    MetricRuleProjection,
    SuccessPolicyProjection,
    VerifiedExecutionProjection,
    _issue_verified_execution_projection,
    evaluator_registry_compatible,
)
from billiards_ai.shared.plan_publication.contracts import DecodedPublishedPlanBundleV3


class UnsupportedExecutionPolicyError(ValueError):
    pass


def build_verified_execution_projection(
    decoded: DecodedPublishedPlanBundleV3,
    *,
    admission_id: str,
) -> VerifiedExecutionProjection:
    if type(decoded) is not DecodedPublishedPlanBundleV3:
        raise TypeError("projection source must be an exact DecodedPublishedPlanBundleV3")
    payload = decoded.payload
    policies = {
        (item.drill_id, item.drill_revision): item
        for item in payload.execution_policy_catalog.definitions
    }
    definitions = {(item.drill_id, item.revision): item for item in payload.drill_definitions}

    def project(key: tuple[str, int]) -> DrillExecutionProjection:
        policy = policies[key]
        definition = definitions[key]
        zone = definition.setup.target_zone
        return DrillExecutionProjection(
            drill_id=policy.drill_id,
            drill_revision=policy.drill_revision,
            drill_content_sha256=policy.drill_content_sha256,
            required_capabilities=policy.required_capabilities,
            initial_difficulty=policy.initial_difficulty,
            minimum_runtime_difficulty=policy.minimum_runtime_difficulty,
            maximum_runtime_difficulty=policy.maximum_runtime_difficulty,
            progression_count=policy.progression_rule.consecutive_count,
            progression_delta=policy.progression_rule.difficulty_delta,
            regression_count=policy.regression_rule.consecutive_count,
            regression_delta=policy.regression_rule.difficulty_delta,
            abstention_count=policy.abstention_rule.consecutive_count,
            abstention_action=policy.abstention_rule.action,
            success_policy=SuccessPolicyProjection(
                evaluator_policy_version=policy.success_policy.evaluator_policy_version,
                combination=policy.success_policy.combination,
                metric_rules=tuple(
                    MetricRuleProjection(
                        metric_code=item.metric_code,
                        target_attempts=item.target_attempts,
                        minimum_successes=item.minimum_successes,
                        evaluator_code=item.evaluator_code,
                        evaluator_version=item.evaluator_version,
                        required_capabilities=item.required_capabilities,
                    )
                    for item in policy.success_policy.metric_rules
                ),
            ),
            replan_reason_codes=policy.replan_rule.reason_codes,
            replan_continue_action=policy.replan_rule.continue_action,
            fallback_key=(
                None
                if policy.fallback_drill is None
                else (policy.fallback_drill.drill_id, policy.fallback_drill.drill_revision)
            ),
            target_zone=(
                None
                if zone is None
                else (
                    zone.minimum_x_mm,
                    zone.maximum_x_mm,
                    zone.minimum_y_mm,
                    zone.maximum_y_mm,
                )
            ),
        )

    blocks = tuple(
        ExecutionBlockProjection(
            block_id=block.block_id,
            order=block.order,
            target_repetitions=block.target_repetitions,
            drill=project((block.drill.drill_id, block.drill.drill_revision)),
        )
        for block in payload.plan.blocks
    )
    fallback_key = (
        payload.plan.fallback_drill.drill_id,
        payload.plan.fallback_drill.drill_revision,
    )
    manifest = payload.capability_manifest
    key = payload.publication_policy.signing_key
    projection = _issue_verified_execution_projection(
        admission_id=admission_id,
        bundle_id=payload.bundle_id,
        bundle_content_sha256=decoded.bundle_content_sha256,
        user_id=payload.plan.user_id,
        session_id=payload.session_id,
        plan_id=payload.plan.plan_id,
        plan_version=payload.plan.version,
        target_device_id=payload.target_device_id,
        publication_sequence=payload.publication_sequence,
        expires_at=payload.publication_policy.expires_at,
        plan_valid_until=payload.plan.valid_until,
        authorization_id=payload.authorization.authorization_id,
        authorization_version=payload.authorization.authorization_version,
        authorization_expires_at=payload.authorization.expires_at,
        signing_key_id=key.key_id,
        signing_key_version=key.key_version,
        signing_key_expires_at=key.expires_at,
        execution_policy_catalog_version=payload.execution_policy_catalog.catalog_version,
        execution_policy_catalog_sha256=payload.execution_policy_catalog.content_sha256,
        capability_manifest_id=manifest.manifest_id,
        capability_manifest_version=manifest.manifest_version,
        capability_manifest_content_sha256=manifest.content_sha256,
        capabilities=manifest.capabilities,
        equipment_codes=manifest.equipment_codes,
        blocks=blocks,
        terminal_fallback=project(fallback_key),
    )
    if not evaluator_registry_compatible(projection):
        raise UnsupportedExecutionPolicyError("unsupported execution evaluator policy")
    return projection


__all__ = ["UnsupportedExecutionPolicyError", "build_verified_execution_projection"]
