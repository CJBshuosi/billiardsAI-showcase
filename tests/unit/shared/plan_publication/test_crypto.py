from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from billiards_ai.shared.plan_publication.contracts import PublicVerificationKey
from billiards_ai.shared.plan_publication.crypto import (
    CryptographyEd25519Signer,
    CryptographyEd25519Verifier,
    OpaqueEd25519PrivateKey,
    SignatureVerificationError,
    public_verification_key,
)


def _key(private: Ed25519PrivateKey, *, key_id: str = "key-1") -> PublicVerificationKey:
    now = datetime(2026, 8, 5, tzinfo=UTC)
    return public_verification_key(
        private.public_key(),
        key_id=key_id,
        key_version=1,
        issued_at=now,
        not_before=now,
        expires_at=now + timedelta(days=30),
    )


def test_official_ed25519_sign_and_verify_exact_message() -> None:
    private = Ed25519PrivateKey.generate()
    signer = CryptographyEd25519Signer()
    verifier = CryptographyEd25519Verifier()
    message = b'canonical:{"plan":1}'

    signature = signer.sign(OpaqueEd25519PrivateKey(private), message)

    assert len(signature) == 64
    verifier.verify(_key(private), message, signature, at=datetime(2026, 8, 6, tzinfo=UTC))


def test_wrong_key_signature_and_message_fail_closed() -> None:
    private = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    signature = CryptographyEd25519Signer().sign(OpaqueEd25519PrivateKey(private), b"message")
    verifier = CryptographyEd25519Verifier()
    now = datetime(2026, 8, 6, tzinfo=UTC)

    for key, message, candidate in (
        (_key(other), b"message", signature),
        (_key(private), b"changed", signature),
        (_key(private), b"message", b"\x00" * 64),
    ):
        with pytest.raises(SignatureVerificationError, match="SIGNATURE_INVALID"):
            verifier.verify(key, message, candidate, at=now)


def test_private_handle_and_errors_never_repr_key_material() -> None:
    private = Ed25519PrivateKey.generate()
    handle = OpaqueEd25519PrivateKey(private)

    assert repr(handle) == "OpaqueEd25519PrivateKey(<redacted>)"
    assert "private" not in str(SignatureVerificationError("SIGNATURE_INVALID")).lower()


def test_key_lifecycle_revocation_and_rotation_policy_fail_closed() -> None:
    private = Ed25519PrivateKey.generate()
    now = datetime(2026, 8, 5, tzinfo=UTC)
    key = public_verification_key(
        private.public_key(),
        key_id="key-2",
        key_version=2,
        issued_at=now,
        not_before=now + timedelta(hours=1),
        expires_at=now + timedelta(days=2),
        rotation_predecessor_key_id="key-1",
    )
    signature = CryptographyEd25519Signer().sign(OpaqueEd25519PrivateKey(private), b"message")

    with pytest.raises(SignatureVerificationError, match="KEY_NOT_YET_VALID"):
        CryptographyEd25519Verifier().verify(key, b"message", signature, at=now)

    revoked = public_verification_key(
        private.public_key(),
        key_id="key-2",
        key_version=2,
        issued_at=now,
        not_before=now,
        expires_at=now + timedelta(days=2),
        revoked_at=now + timedelta(hours=1),
        rotation_predecessor_key_id="key-1",
    )
    with pytest.raises(SignatureVerificationError, match="KEY_REVOKED"):
        CryptographyEd25519Verifier().verify(
            revoked, b"message", signature, at=now + timedelta(hours=1)
        )


def test_key_rotation_lineage_is_mandatory_after_version_one() -> None:
    private = Ed25519PrivateKey.generate()
    now = datetime(2026, 8, 5, tzinfo=UTC)

    with pytest.raises(ValueError, match="rotation lineage"):
        public_verification_key(
            private.public_key(),
            key_id="key-2",
            key_version=2,
            issued_at=now,
            not_before=now,
            expires_at=now + timedelta(days=2),
        )


@pytest.mark.parametrize(
    "invalid_at",
    [
        None,
        "2026-08-05T00:00:00Z",
        datetime(2026, 8, 5),
        datetime(2026, 8, 5, tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_verifier_rejects_non_exact_utc_time_with_stable_code(invalid_at) -> None:
    private = Ed25519PrivateKey.generate()
    message = b"message"
    signature = CryptographyEd25519Signer().sign(OpaqueEd25519PrivateKey(private), message)

    with pytest.raises(SignatureVerificationError, match="KEY_TIME_INVALID"):
        CryptographyEd25519Verifier().verify(_key(private), message, signature, at=invalid_at)
