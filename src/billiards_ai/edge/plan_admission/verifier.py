"""Fail-closed orchestration for final-v2 published-plan admission."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

from pydantic import ValidationError

from billiards_ai.edge.coach_runtime.state_machine import VerifiedPlanAdmission
from billiards_ai.edge.plan_admission.codec import CanonicalPublishedPlanBundleCodec
from billiards_ai.edge.plan_admission.contracts import (
    AntiRollbackClaim,
    AntiRollbackDecision,
    AntiRollbackPort,
    AuthorizationClaim,
    CapabilityAuthorityPort,
    ClockPort,
    PlanAdmissionFailure,
    PlanAdmissionFailureCode,
    PlanAdmissionResult,
    PlanAuthorizationPort,
    PublishedPlanBundleCodecPort,
    SignatureVerifierPort,
    TrustedKeyringPort,
    UntrustedPlanBundle,
)
from billiards_ai.edge.plan_admission.projection import (
    UnsupportedExecutionPolicyError,
    build_verified_execution_projection,
)
from billiards_ai.shared.plan_publication.contracts import (
    DecodedPublishedPlanBundle,
    DecodedPublishedPlanBundleV3,
    PublicVerificationKey,
)
from billiards_ai.shared.plan_publication.crypto import (
    CryptographyEd25519Verifier,
    SignatureVerificationError,
)

_SUPPORTED_SCHEMA_VERSIONS = frozenset({"2.0", "3.0"})
_SUPPORTED_ALGORITHM = "ED25519"


def _failure(code: PlanAdmissionFailureCode, stage: str) -> PlanAdmissionResult:
    return PlanAdmissionResult(failure=PlanAdmissionFailure(code=code, stage=stage))


def _strict_utc(value: object) -> bool:
    return type(value) is datetime and value.tzinfo is UTC


def _revalidate_decoded(
    value: object,
) -> DecodedPublishedPlanBundleV3 | DecodedPublishedPlanBundle | None:
    try:
        if type(value) is DecodedPublishedPlanBundleV3:
            return DecodedPublishedPlanBundleV3.model_validate(value.model_dump(mode="python"))
        if type(value) is DecodedPublishedPlanBundle:
            return DecodedPublishedPlanBundle.model_validate(value.model_dump(mode="python"))
        return None
    except (AttributeError, TypeError, ValueError, ValidationError):
        return None


def _revalidate_key(value: object) -> PublicVerificationKey | None:
    if type(value) is not PublicVerificationKey:
        return None
    try:
        return PublicVerificationKey.model_validate(value.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError, ValidationError):
        return None


class PlanBundleVerifier:
    """Admit only bundles that pass every signed and live-authority gate."""

    def __init__(
        self,
        *,
        clock: ClockPort,
        keyring: TrustedKeyringPort,
        authorization: PlanAuthorizationPort,
        capabilities: CapabilityAuthorityPort,
        anti_rollback: AntiRollbackPort,
        codec: PublishedPlanBundleCodecPort | None = None,
        signature_verifier: SignatureVerifierPort | None = None,
    ) -> None:
        self._codec = CanonicalPublishedPlanBundleCodec() if codec is None else codec
        self._clock = clock
        self._keyring = keyring
        self._signature_verifier = (
            CryptographyEd25519Verifier() if signature_verifier is None else signature_verifier
        )
        self._authorization = authorization
        self._capabilities = capabilities
        self._anti_rollback = anti_rollback

    def verify(self, candidate: object) -> PlanAdmissionResult:
        if (
            type(candidate) is not UntrustedPlanBundle
            or type(candidate.envelope_bytes) is not bytes
            or not candidate.envelope_bytes
        ):
            return _failure(PlanAdmissionFailureCode.INVALID_INPUT, "input")

        try:
            decoded = self._codec.decode(candidate.envelope_bytes)
        except Exception:
            return _failure(PlanAdmissionFailureCode.MALFORMED_BUNDLE, "codec")
        decoded = _revalidate_decoded(decoded)
        if decoded is None:
            return _failure(PlanAdmissionFailureCode.MALFORMED_BUNDLE, "codec")

        payload = decoded.payload
        policy = payload.publication_policy
        embedded_key = policy.signing_key
        if payload.schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            return _failure(PlanAdmissionFailureCode.UNSUPPORTED_SCHEMA, "schema")
        if embedded_key.algorithm != _SUPPORTED_ALGORITHM:
            return _failure(
                PlanAdmissionFailureCode.UNSUPPORTED_SIGNATURE_ALGORITHM,
                "algorithm",
            )
        calculated_hash = hashlib.sha256(decoded.signed_payload).hexdigest()
        if not secrets.compare_digest(calculated_hash, decoded.bundle_content_sha256):
            return _failure(PlanAdmissionFailureCode.CONTENT_HASH_MISMATCH, "content")

        try:
            now = self._clock.now()
        except Exception:
            return _failure(PlanAdmissionFailureCode.AUTHORITY_UNAVAILABLE, "clock")
        if not _strict_utc(now):
            return _failure(PlanAdmissionFailureCode.AUTHORITY_UNAVAILABLE, "clock")

        if not all(
            _strict_utc(value)
            for value in (
                policy.issued_at,
                policy.not_before,
                policy.expires_at,
                payload.plan.valid_until,
            )
        ):
            return _failure(PlanAdmissionFailureCode.INVALID_VALIDITY_WINDOW, "validity")
        if not (
            payload.plan.created_at
            <= policy.issued_at
            <= policy.not_before
            < policy.expires_at
            <= payload.plan.valid_until
        ):
            return _failure(PlanAdmissionFailureCode.INVALID_VALIDITY_WINDOW, "validity")
        if now < policy.issued_at or now < policy.not_before:
            return _failure(PlanAdmissionFailureCode.NOT_YET_VALID, "validity")
        if now >= policy.expires_at:
            return _failure(PlanAdmissionFailureCode.EXPIRED, "validity")

        trusted_key = self._resolve_key(embedded_key, now)
        if type(trusted_key) is PlanAdmissionResult:
            return trusted_key

        try:
            self._signature_verifier.verify(
                trusted_key,
                decoded.signed_payload,
                decoded.signature_bytes,
                at=now,
            )
        except SignatureVerificationError as error:
            if str(error) == "SIGNATURE_INVALID":
                return _failure(PlanAdmissionFailureCode.SIGNATURE_INVALID, "authenticity")
            return _failure(PlanAdmissionFailureCode.KEY_NOT_TRUSTED, "authenticity")
        except Exception:
            return _failure(PlanAdmissionFailureCode.AUTHORITY_UNAVAILABLE, "authenticity")

        authorization_claim = AuthorizationClaim(
            authorization_id=payload.authorization.authorization_id,
            authorization_version=payload.authorization.authorization_version,
            user_id=payload.plan.user_id,
            session_id=payload.session_id,
            target_device_id=payload.target_device_id,
            plan_id=payload.plan.plan_id,
            bundle_id=payload.bundle_id,
        )
        try:
            authorized = self._authorization.authorize(authorization_claim)
        except Exception:
            return _failure(PlanAdmissionFailureCode.AUTHORITY_UNAVAILABLE, "authorization")
        if authorized is not True:
            return _failure(PlanAdmissionFailureCode.UNAUTHORIZED, "authorization")

        try:
            compatible = self._capabilities.compatible(
                target_device_id=payload.target_device_id,
                manifest=payload.capability_manifest,
            )
        except Exception:
            return _failure(PlanAdmissionFailureCode.AUTHORITY_UNAVAILABLE, "capability")
        if compatible is not True:
            return _failure(PlanAdmissionFailureCode.CAPABILITY_MISMATCH, "capability")

        rollback_claim = AntiRollbackClaim(
            bundle_id=payload.bundle_id,
            plan_id=payload.plan.plan_id,
            plan_version=payload.plan.version,
            user_id=payload.plan.user_id,
            session_id=payload.session_id,
            target_device_id=payload.target_device_id,
            authorization_id=payload.authorization.authorization_id,
            authorization_version=payload.authorization.authorization_version,
            publication_sequence=payload.publication_sequence,
            bundle_content_sha256=decoded.bundle_content_sha256,
        )
        try:
            rollback = self._anti_rollback.accept(rollback_claim)
        except Exception:
            return _failure(PlanAdmissionFailureCode.AUTHORITY_UNAVAILABLE, "rollback")
        if type(rollback) is not AntiRollbackDecision or rollback not in (
            AntiRollbackDecision.ACCEPTED,
            AntiRollbackDecision.REPLAYED,
        ):
            return _failure(PlanAdmissionFailureCode.ROLLBACK_REJECTED, "rollback")

        admission_id = f"bundle:{payload.bundle_id}:{decoded.bundle_content_sha256}"
        try:
            projection = (
                build_verified_execution_projection(decoded, admission_id=admission_id)
                if type(decoded) is DecodedPublishedPlanBundleV3
                else None
            )
        except UnsupportedExecutionPolicyError:
            return _failure(PlanAdmissionFailureCode.CAPABILITY_MISMATCH, "evaluator")
        return PlanAdmissionResult(
            admission=VerifiedPlanAdmission(
                admission_id=admission_id,
                session_id=payload.session_id,
                plan_id=payload.plan.plan_id,
                plan_version=payload.plan.version,
                execution_projection=projection,
            )
        )

    def _resolve_key(
        self, embedded_key: PublicVerificationKey, now: datetime
    ) -> PublicVerificationKey | PlanAdmissionResult:
        try:
            authority_key = self._keyring.resolve(
                algorithm=embedded_key.algorithm,
                key_id=embedded_key.key_id,
                key_version=embedded_key.key_version,
                purpose=embedded_key.purpose,
                at=now,
            )
        except Exception:
            return _failure(PlanAdmissionFailureCode.AUTHORITY_UNAVAILABLE, "keyring")
        authority_key = _revalidate_key(authority_key)
        if authority_key is None or authority_key != embedded_key:
            return _failure(PlanAdmissionFailureCode.KEY_NOT_TRUSTED, "keyring")
        return authority_key


__all__ = ["PlanBundleVerifier"]
