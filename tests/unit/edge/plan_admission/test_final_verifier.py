from __future__ import annotations

import base64
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from billiards_ai.edge import coach_runtime as runtime
from billiards_ai.edge.plan_admission import (
    AntiRollbackClaim,
    AntiRollbackDecision,
    PlanAdmissionFailureCode,
    PlanBundleVerifier,
    UntrustedPlanBundle,
)
from billiards_ai.shared.models.training_plan import PublishedPlanBundle
from billiards_ai.shared.plan_publication.codec import (
    canonical_json_bytes,
    decode_bundle,
    decode_bundle_v3,
    decode_canonical_json,
    encode_bundle,
)
from billiards_ai.shared.plan_publication.contracts import (
    DecodedPublishedPlanBundle,
    PublicVerificationKey,
    PublishedPlanPayload,
)
from billiards_ai.shared.plan_publication.crypto import (
    CryptographyEd25519Signer,
    CryptographyEd25519Verifier,
    OpaqueEd25519PrivateKey,
    public_verification_key,
)

ROOT = Path(__file__).parents[3] / "fixtures" / "plan_publication" / "v2"
C0_ROOT = Path(__file__).parents[3] / "fixtures" / "core_intelligence" / "v1"
GOLDEN_NOW = datetime(2026, 8, 5, 9, 0, tzinfo=UTC)
_DEFAULT_KEY = object()


def _wire(filename: str = "positive.json") -> bytes:
    vector = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    return base64.b64decode(vector["envelope_base64"], validate=True)


def _v3_wire() -> bytes:
    """Reuse the synthetic publisher builder without introducing a stored signed fixture."""
    path = Path(__file__).parents[2] / "cloud" / "plan_publication" / "test_publisher.py"
    spec = importlib.util.spec_from_file_location("_story_10_3_v3_publisher_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    candidate = module._candidate_v3()
    result = module._publisher(candidate).publish(candidate, now=module.NOW)
    assert result.envelope_bytes is not None
    return result.envelope_bytes


class Clock:
    def __init__(self, now: object, log: list[str]) -> None:
        self._now = now
        self._log = log

    def now(self) -> object:
        self._log.append("clock.now")
        return self._now


class Keyring:
    def __init__(
        self,
        key: object,
        log: list[str],
        *,
        error: Exception | None = None,
    ) -> None:
        self.key = key
        self.log = log
        self.error = error
        self.identity: dict[str, object] | None = None

    def resolve(self, **identity: object) -> object:
        self.log.append("keyring.resolve")
        self.identity = identity
        if self.error:
            raise self.error
        return self.key


class Authorization:
    def __init__(
        self,
        log: list[str],
        *,
        allowed: object = True,
        error: Exception | None = None,
    ) -> None:
        self.log = log
        self.allowed = allowed
        self.error = error
        self.claim: object | None = None

    def authorize(self, claim: object) -> object:
        self.log.append("authorization.authorize")
        self.claim = claim
        if self.error:
            raise self.error
        return self.allowed


class Capabilities:
    def __init__(
        self,
        log: list[str],
        *,
        compatible: object = True,
        error: Exception | None = None,
    ) -> None:
        self.log = log
        self.value = compatible
        self.error = error
        self.target_device_id: str | None = None
        self.manifest: object | None = None

    def compatible(self, *, target_device_id: str, manifest: object) -> object:
        self.log.append("capability.compatible")
        self.target_device_id = target_device_id
        self.manifest = manifest
        if self.error:
            raise self.error
        return self.value


class AntiRollback:
    def __init__(
        self,
        log: list[str],
        *,
        decision: object = AntiRollbackDecision.ACCEPTED,
        error: Exception | None = None,
    ) -> None:
        self.log = log
        self.decision = decision
        self.error = error
        self.claim: AntiRollbackClaim | None = None

    def accept(self, claim: AntiRollbackClaim) -> object:
        self.log.append("anti_rollback.accept")
        self.claim = claim
        if self.error:
            raise self.error
        return self.decision


class Codec:
    def __init__(
        self,
        decoded: object,
        log: list[str],
        *,
        error: Exception | None = None,
    ) -> None:
        self.decoded = decoded
        self.log = log
        self.error = error

    def decode(self, envelope_bytes: bytes) -> object:
        self.log.append("codec.decode")
        if self.error:
            raise self.error
        return self.decoded


class SignatureVerifier:
    def __init__(
        self,
        log: list[str],
        *,
        error: Exception | None = None,
    ) -> None:
        self.log = log
        self.error = error
        self.official = CryptographyEd25519Verifier()

    def verify(self, key, message, signature, *, at) -> None:
        self.log.append("signature.verify")
        if self.error:
            raise self.error
        self.official.verify(key, message, signature, at=at)


@dataclass
class Harness:
    log: list[str]
    clock: Clock
    keyring: Keyring
    authorization: Authorization
    capabilities: Capabilities
    anti_rollback: AntiRollback
    codec: object | None = None
    signature_verifier: object | None = None

    def verifier(self) -> PlanBundleVerifier:
        values: dict[str, object] = {
            "clock": self.clock,
            "keyring": self.keyring,
            "authorization": self.authorization,
            "capabilities": self.capabilities,
            "anti_rollback": self.anti_rollback,
        }
        if self.codec is not None:
            values["codec"] = self.codec
        if self.signature_verifier is not None:
            values["signature_verifier"] = self.signature_verifier
        return PlanBundleVerifier(**values)


def _harness(
    *,
    now: object = GOLDEN_NOW,
    key: object = _DEFAULT_KEY,
    authorization: object = True,
    capabilities: object = True,
    rollback: object = AntiRollbackDecision.ACCEPTED,
    codec: object | None = None,
    signature_verifier: object | None = None,
) -> Harness:
    log: list[str] = []
    decoded = decode_bundle(_wire())
    return Harness(
        log=log,
        clock=Clock(now, log),
        keyring=Keyring(
            decoded.payload.publication_policy.signing_key if key is _DEFAULT_KEY else key,
            log,
        ),
        authorization=Authorization(log, allowed=authorization),
        capabilities=Capabilities(log, compatible=capabilities),
        anti_rollback=AntiRollback(log, decision=rollback),
        codec=codec,
        signature_verifier=signature_verifier,
    )


def _resigned_wire(
    *,
    policy_issued_at: datetime = GOLDEN_NOW,
    policy_not_before: datetime = GOLDEN_NOW,
    policy_expires_at: datetime = datetime(2026, 8, 6, 9, tzinfo=UTC),
    key_expires_at: datetime = datetime(2026, 9, 4, 9, tzinfo=UTC),
    key_revoked_at: datetime | None = None,
    key_version: int = 1,
    predecessor: str | None = None,
) -> tuple[bytes, PublicVerificationKey]:
    base = decode_bundle(_wire()).payload.model_dump(mode="python")
    private = Ed25519PrivateKey.generate()
    key = public_verification_key(
        private.public_key(),
        key_id=f"consumer-test-key-{key_version}",
        key_version=key_version,
        issued_at=datetime(2026, 8, 4, 9, tzinfo=UTC),
        not_before=datetime(2026, 8, 4, 9, tzinfo=UTC),
        expires_at=key_expires_at,
        revoked_at=key_revoked_at,
        rotation_predecessor_key_id=predecessor,
    )
    base["publication_policy"] = {
        **base["publication_policy"],
        "issued_at": policy_issued_at,
        "not_before": policy_not_before,
        "expires_at": policy_expires_at,
        "signing_key": key,
    }
    payload = PublishedPlanPayload.model_validate(base)
    signed = canonical_json_bytes(payload)
    signature = CryptographyEd25519Signer().sign(OpaqueEd25519PrivateKey(private), signed)
    return encode_bundle(payload, signature), key


def test_formal_v2_codec_is_auditable_but_cannot_enter_10_2_ready() -> None:
    harness = _harness()

    result = harness.verifier().verify(UntrustedPlanBundle(_wire()))

    assert result.accepted is True
    decoded = decode_bundle(_wire())
    assert result.admission.session_id == decoded.payload.session_id
    assert result.admission.plan_id == decoded.payload.plan.plan_id
    loading = runtime.transition(
        runtime.RuntimeSnapshot.initial(), runtime.BeginPlanLoad()
    ).snapshot
    admitted = runtime.transition(loading, runtime.AdmitVerifiedPlan(result.admission))
    assert result.admission.execution_projection is None
    assert admitted.error.code is runtime.RuntimeErrorCode.INVALID_PLAN_ADMISSION
    assert admitted.snapshot.state is runtime.CoachRuntimeState.LOADING_PLAN


def test_exact_v3_bundle_builds_projection_then_enters_10_2_ready() -> None:
    wire = _v3_wire()
    decoded = decode_bundle_v3(wire)
    log: list[str] = []
    harness = Harness(
        log=log,
        clock=Clock(decoded.payload.publication_policy.not_before, log),
        keyring=Keyring(decoded.payload.publication_policy.signing_key, log),
        authorization=Authorization(log),
        capabilities=Capabilities(log),
        anti_rollback=AntiRollback(log),
    )

    result = harness.verifier().verify(UntrustedPlanBundle(wire))

    assert result.accepted and result.admission is not None
    projection = result.admission.execution_projection
    assert projection is not None
    assert projection.user_id == decoded.payload.plan.user_id
    assert projection.bundle_content_sha256 == decoded.bundle_content_sha256
    assert (
        projection.blocks[0].drill.drill_revision
        == decoded.payload.plan.blocks[0].drill.drill_revision
    )
    assert projection.terminal_fallback.key == (
        decoded.payload.plan.fallback_drill.drill_id,
        decoded.payload.plan.fallback_drill.drill_revision,
    )
    assert "execution_policy_catalog" not in repr(projection)
    loading = runtime.transition(
        runtime.RuntimeSnapshot.initial(), runtime.BeginPlanLoad()
    ).snapshot
    admitted = runtime.transition(loading, runtime.AdmitVerifiedPlan(result.admission))
    assert admitted.error is None
    assert admitted.snapshot.state is runtime.CoachRuntimeState.READY
    assert log == [
        "clock.now",
        "keyring.resolve",
        "authorization.authorize",
        "capability.compatible",
        "anti_rollback.accept",
    ]


def test_success_passes_exact_signed_authority_identities_in_frozen_order() -> None:
    log: list[str] = []
    decoded = decode_bundle(_wire())
    harness = _harness(
        codec=Codec(decoded, log),
        signature_verifier=SignatureVerifier(log),
    )
    harness.log = log
    harness.clock._log = log
    harness.keyring.log = log
    harness.authorization.log = log
    harness.capabilities.log = log
    harness.anti_rollback.log = log

    result = harness.verifier().verify(UntrustedPlanBundle(_wire()))

    assert result.accepted
    assert log == [
        "codec.decode",
        "clock.now",
        "keyring.resolve",
        "signature.verify",
        "authorization.authorize",
        "capability.compatible",
        "anti_rollback.accept",
    ]
    policy = decoded.payload.publication_policy
    assert harness.keyring.identity == {
        "algorithm": "ED25519",
        "key_id": policy.signing_key.key_id,
        "key_version": policy.signing_key.key_version,
        "purpose": "PLAN_BUNDLE_SIGNING",
        "at": GOLDEN_NOW,
    }
    assert (
        harness.authorization.claim.authorization_id
        == decoded.payload.authorization.authorization_id
    )
    assert harness.authorization.claim.authorization_version == 1
    assert harness.capabilities.target_device_id == decoded.payload.target_device_id
    assert harness.capabilities.manifest == decoded.payload.capability_manifest
    assert harness.anti_rollback.claim == AntiRollbackClaim(
        bundle_id=decoded.payload.bundle_id,
        plan_id=decoded.payload.plan.plan_id,
        plan_version=decoded.payload.plan.version,
        user_id=decoded.payload.plan.user_id,
        session_id=decoded.payload.session_id,
        target_device_id=decoded.payload.target_device_id,
        authorization_id=decoded.payload.authorization.authorization_id,
        authorization_version=decoded.payload.authorization.authorization_version,
        publication_sequence=decoded.payload.publication_sequence,
        bundle_content_sha256=decoded.bundle_content_sha256,
    )


@pytest.mark.parametrize(
    "candidate",
    (object(), {}, b"wire", decode_bundle(_wire())),
)
def test_only_nominal_untrusted_bytes_cross_the_public_input_boundary(candidate: object) -> None:
    harness = _harness()

    result = harness.verifier().verify(candidate)

    assert result.failure.code is PlanAdmissionFailureCode.INVALID_INPUT
    assert harness.log == []


def test_legacy_c0_bundle_cannot_cross_final_v2_consumer() -> None:
    bundle = PublishedPlanBundle.model_validate_json(
        (C0_ROOT / "published-plan-bundle.json").read_text(encoding="utf-8")
    )

    result = _harness().verifier().verify(bundle)

    assert result.failure.code is PlanAdmissionFailureCode.INVALID_INPUT


@pytest.mark.parametrize("filename", ("duplicate-key.json", "unknown-field.json", "non-nfc.json"))
def test_authoritative_negative_wire_vectors_fail_before_authority_use(filename: str) -> None:
    harness = _harness()

    result = harness.verifier().verify(UntrustedPlanBundle(_wire(filename)))

    assert result.failure.code is PlanAdmissionFailureCode.MALFORMED_BUNDLE
    assert harness.log == []


def test_noncanonical_base64_and_signature_malleability_fail_closed() -> None:
    raw = decode_canonical_json(_wire())
    raw["signature"] = raw["signature"].rstrip("=")
    malformed = canonical_json_bytes(raw)

    result = _harness().verifier().verify(UntrustedPlanBundle(malformed))

    assert result.failure.code is PlanAdmissionFailureCode.MALFORMED_BUNDLE


def test_canonical_signature_tampering_reaches_official_verify_and_is_rejected() -> None:
    raw = decode_canonical_json(_wire())
    raw["signature"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    tampered = canonical_json_bytes(raw)

    result = _harness().verifier().verify(UntrustedPlanBundle(tampered))

    assert result.failure.code is PlanAdmissionFailureCode.SIGNATURE_INVALID


def test_decoded_internal_inconsistency_from_codec_is_revalidated_and_rejected() -> None:
    valid = decode_bundle(_wire())
    forged = DecodedPublishedPlanBundle.model_construct(
        envelope=valid.envelope,
        signed_payload=b"different",
        signature_bytes=valid.signature_bytes,
        bundle_content_sha256=valid.bundle_content_sha256,
    )
    log: list[str] = []
    harness = _harness(codec=Codec(forged, log))
    harness.log = log
    harness.clock._log = log

    result = harness.verifier().verify(UntrustedPlanBundle(_wire()))

    assert result.failure.code is PlanAdmissionFailureCode.MALFORMED_BUNDLE
    assert log == ["codec.decode"]


@pytest.mark.parametrize(
    ("now", "expected"),
    (
        (datetime(2026, 8, 5, 8, 59, 59, tzinfo=UTC), PlanAdmissionFailureCode.NOT_YET_VALID),
        (datetime(2026, 8, 6, 9, tzinfo=UTC), PlanAdmissionFailureCode.EXPIRED),
        (datetime(2026, 8, 5, 9), PlanAdmissionFailureCode.AUTHORITY_UNAVAILABLE),
        (
            datetime(2026, 8, 5, 10, tzinfo=timezone(timedelta(hours=1))),
            PlanAdmissionFailureCode.AUTHORITY_UNAVAILABLE,
        ),
    ),
)
def test_consumer_uses_injected_strict_utc_clock_for_validity(now: datetime, expected) -> None:
    result = _harness(now=now).verifier().verify(UntrustedPlanBundle(_wire()))

    assert result.failure.code is expected


def test_not_before_is_independent_from_issued_at_and_blocks_early_admission() -> None:
    wire, key = _resigned_wire(
        policy_issued_at=GOLDEN_NOW,
        policy_not_before=GOLDEN_NOW + timedelta(hours=1),
    )
    harness = _harness(now=GOLDEN_NOW + timedelta(minutes=30), key=key)

    result = harness.verifier().verify(UntrustedPlanBundle(wire))

    assert result.failure.code is PlanAdmissionFailureCode.NOT_YET_VALID
    assert "keyring.resolve" not in harness.log


@pytest.mark.parametrize(
    "key_failure",
    ("unknown", "revoked-authority", "wrong-bytes", "wrong-version"),
)
def test_keyring_exact_identity_rotation_and_revocation_fail_closed(key_failure: str) -> None:
    decoded = decode_bundle(_wire())
    trusted = decoded.payload.publication_policy.signing_key
    key: object = None
    if key_failure == "wrong-bytes":
        key = trusted.model_copy(
            update={"public_key_base64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="}
        )
    elif key_failure == "wrong-version":
        key = trusted.model_copy(update={"key_version": 2, "rotation_predecessor_key_id": "old"})

    result = _harness(key=key).verifier().verify(UntrustedPlanBundle(_wire()))

    assert result.failure.code is PlanAdmissionFailureCode.KEY_NOT_TRUSTED


def test_key_revoked_after_publication_is_rejected_at_current_verification_time() -> None:
    revoked_at = GOLDEN_NOW + timedelta(minutes=30)
    wire, key = _resigned_wire(key_revoked_at=revoked_at)

    result = _harness(now=revoked_at, key=key).verifier().verify(UntrustedPlanBundle(wire))

    assert result.failure.code is PlanAdmissionFailureCode.KEY_NOT_TRUSTED


def test_rotated_key_version_and_predecessor_are_authoritatively_resolved() -> None:
    wire, key = _resigned_wire(key_version=2, predecessor="consumer-test-key-1")
    harness = _harness(key=key)

    result = harness.verifier().verify(UntrustedPlanBundle(wire))

    assert result.accepted
    assert harness.keyring.identity["key_version"] == 2


@pytest.mark.parametrize(
    ("gate", "expected"),
    (
        ("authorization", PlanAdmissionFailureCode.UNAUTHORIZED),
        ("capability", PlanAdmissionFailureCode.CAPABILITY_MISMATCH),
        ("rollback", PlanAdmissionFailureCode.ROLLBACK_REJECTED),
        ("rollback-string", PlanAdmissionFailureCode.ROLLBACK_REJECTED),
    ),
)
def test_authorization_capability_and_rollback_gates_fail_closed(gate: str, expected) -> None:
    harness = _harness(
        authorization=gate != "authorization",
        capabilities=gate != "capability",
        rollback=(
            "ACCEPTED"
            if gate == "rollback-string"
            else AntiRollbackDecision.REJECTED
            if gate == "rollback"
            else AntiRollbackDecision.ACCEPTED
        ),
    )

    result = harness.verifier().verify(UntrustedPlanBundle(_wire()))

    assert result.failure.code is expected


def test_exact_anti_rollback_replay_is_idempotent() -> None:
    accepted = _harness().verifier().verify(UntrustedPlanBundle(_wire()))
    replayed = (
        _harness(rollback=AntiRollbackDecision.REPLAYED)
        .verifier()
        .verify(UntrustedPlanBundle(_wire()))
    )

    assert accepted.admission == replayed.admission


@pytest.mark.parametrize(
    "stage",
    ("clock", "keyring", "signature", "authorization", "capability", "rollback"),
)
def test_port_exceptions_are_redacted_and_never_produce_admission(stage: str) -> None:
    secret = RuntimeError("DO-NOT-LEAK-KEY-SIGNATURE-OR-CLAIM")
    harness = _harness()
    if stage == "clock":
        harness.clock.now = lambda: (_ for _ in ()).throw(secret)
    elif stage == "keyring":
        harness.keyring.error = secret
    elif stage == "signature":
        harness.signature_verifier = SignatureVerifier(harness.log, error=secret)
    elif stage == "authorization":
        harness.authorization.error = secret
    elif stage == "capability":
        harness.capabilities.error = secret
    else:
        harness.anti_rollback.error = secret

    result = harness.verifier().verify(UntrustedPlanBundle(_wire()))

    assert result.failure.code is PlanAdmissionFailureCode.AUTHORITY_UNAVAILABLE
    assert "DO-NOT-LEAK" not in repr(result)
