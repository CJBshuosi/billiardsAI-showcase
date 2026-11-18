"""Official cryptography Ed25519 adapters with redacted key boundaries."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from billiards_ai.shared.plan_publication.contracts import PublicVerificationKey


class SignatureVerificationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)


@dataclass(frozen=True, slots=True, repr=False)
class OpaqueEd25519PrivateKey:
    _key: Ed25519PrivateKey = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._key, Ed25519PrivateKey):
            raise TypeError("opaque handle requires an Ed25519 private key object")

    def __repr__(self) -> str:
        return "OpaqueEd25519PrivateKey(<redacted>)"


class CryptographyEd25519Signer:
    def sign(self, handle: object, message: bytes) -> bytes:
        if type(handle) is not OpaqueEd25519PrivateKey or type(message) is not bytes:
            raise TypeError("SIGNING_INPUT_INVALID")
        return handle._key.sign(message)


class CryptographyEd25519Verifier:
    def verify(
        self,
        key: PublicVerificationKey,
        message: bytes,
        signature: bytes,
        *,
        at: datetime,
    ) -> None:
        if type(at) is not datetime or at.tzinfo is not UTC:
            raise SignatureVerificationError("KEY_TIME_INVALID")
        try:
            key = PublicVerificationKey.model_validate(key.model_dump(mode="python"))
            if key.purpose != "PLAN_BUNDLE_SIGNING":
                raise SignatureVerificationError("KEY_PURPOSE_INVALID")
            if at < key.not_before:
                raise SignatureVerificationError("KEY_NOT_YET_VALID")
            if at >= key.expires_at:
                raise SignatureVerificationError("KEY_EXPIRED")
            if key.revoked_at is not None and at >= key.revoked_at:
                raise SignatureVerificationError("KEY_REVOKED")
            public = Ed25519PublicKey.from_public_bytes(
                base64.b64decode(key.public_key_base64, validate=True)
            )
            public.verify(signature, message)
        except SignatureVerificationError:
            raise
        except (AttributeError, InvalidSignature, ValueError, TypeError):
            raise SignatureVerificationError("SIGNATURE_INVALID") from None


def public_verification_key(
    public_key: Ed25519PublicKey,
    *,
    key_id: str,
    key_version: int,
    issued_at: datetime,
    not_before: datetime,
    expires_at: datetime,
    revoked_at: datetime | None = None,
    rotation_predecessor_key_id: str | None = None,
) -> PublicVerificationKey:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return PublicVerificationKey(
        key_id=key_id,
        key_version=key_version,
        public_key_base64=base64.b64encode(raw).decode("ascii"),
        issued_at=issued_at,
        not_before=not_before,
        expires_at=expires_at,
        revoked_at=revoked_at,
        rotation_predecessor_key_id=rotation_predecessor_key_id,
    )
