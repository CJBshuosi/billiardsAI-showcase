"""Consumer-neutral decoded view matching Story 10.1 verification material."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from billiards_ai.shared.plan_publication.codec import canonical_json_bytes
from billiards_ai.shared.plan_publication.contracts import DecodedPublishedPlanBundle


@dataclass(frozen=True, slots=True)
class DrillDefinitionClaim:
    drill_key: str
    content_sha256: str
    definition_bytes: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class CapabilityManifestClaim:
    manifest_id: str
    content_sha256: str
    manifest_bytes: bytes = field(repr=False)
    required_capabilities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AdmissionBundleView:
    schema_version: str
    bundle_id: UUID
    plan_id: UUID
    plan_version: int
    plan_user_id: UUID
    session_id: UUID
    target_device_id: str
    publication_sequence: int
    issued_at: datetime
    not_before: datetime
    expires_at: datetime
    plan_valid_until: datetime
    bundle_content_sha256: str
    signed_payload: bytes = field(repr=False)
    signature_algorithm: str
    signature_key_id: str
    signature: bytes = field(repr=False)
    planned_drill_keys: tuple[str, ...]
    drill_definitions: tuple[DrillDefinitionClaim, ...]
    capability_manifest: CapabilityManifestClaim


def admission_view(decoded: DecodedPublishedPlanBundle) -> AdmissionBundleView:
    payload = decoded.payload
    definitions = []
    for definition in payload.drill_definitions:
        definition_bytes = canonical_json_bytes(definition)
        definitions.append(
            DrillDefinitionClaim(
                drill_key=f"{definition.drill_id}@{definition.revision}",
                content_sha256=hashlib.sha256(definition_bytes).hexdigest(),
                definition_bytes=definition_bytes,
            )
        )
    manifest_bytes = canonical_json_bytes(payload.capability_manifest)
    return AdmissionBundleView(
        schema_version=payload.schema_version,
        bundle_id=payload.bundle_id,
        plan_id=payload.plan.plan_id,
        plan_version=payload.plan.version,
        plan_user_id=payload.plan.user_id,
        session_id=payload.session_id,
        target_device_id=payload.target_device_id,
        publication_sequence=payload.publication_sequence,
        issued_at=payload.publication_policy.issued_at,
        not_before=payload.publication_policy.not_before,
        expires_at=payload.publication_policy.expires_at,
        plan_valid_until=payload.plan.valid_until,
        bundle_content_sha256=decoded.bundle_content_sha256,
        signed_payload=decoded.signed_payload,
        signature_algorithm=payload.publication_policy.signing_key.algorithm,
        signature_key_id=payload.publication_policy.signing_key.key_id,
        signature=decoded.signature_bytes,
        planned_drill_keys=decoded.planned_drill_keys,
        drill_definitions=tuple(definitions),
        capability_manifest=CapabilityManifestClaim(
            manifest_id=payload.capability_manifest.manifest_id,
            content_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            manifest_bytes=manifest_bytes,
            required_capabilities=payload.capability_manifest.capabilities,
        ),
    )
