"""Constrained canonical JSON codec for signed plan bundles."""

from __future__ import annotations

import base64
import hashlib
import json
import unicodedata
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError

from billiards_ai.shared.plan_publication.contracts import (
    DecodedPublishedPlanBundle,
    DecodedPublishedPlanBundleV3,
    PublishedPlanEnvelope,
    PublishedPlanEnvelopeV3,
    PublishedPlanPayload,
    PublishedPlanPayloadV3,
)

_MAX_SAFE_INTEGER = 2**53 - 1


class CanonicalCodecError(ValueError):
    def __init__(self, code: str = "MALFORMED_CANONICAL_BUNDLE") -> None:
        super().__init__(code)


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalCodecError()
        return value.isoformat().replace("+00:00", "Z")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if not -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
            raise CanonicalCodecError()
        return value
    if type(value) is float:
        raise CanonicalCodecError()
    if type(value) is str:
        if unicodedata.normalize("NFC", value) != value:
            raise CanonicalCodecError()
        return value
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise CanonicalCodecError()
        if any(unicodedata.normalize("NFC", key) != key for key in value):
            raise CanonicalCodecError()
        return {key: _json_value(value[key]) for key in sorted(value, key=lambda k: k.encode())}
    raise CanonicalCodecError()


def canonical_json_bytes(value: object) -> bytes:
    try:
        normalized = _json_value(value)
        return json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except CanonicalCodecError:
        raise
    except Exception:
        raise CanonicalCodecError() from None


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalCodecError()
        result[key] = value
    return result


def _reject_number(value: str) -> object:
    raise CanonicalCodecError()


def _parse_int(value: str) -> int:
    parsed = int(value)
    if not -_MAX_SAFE_INTEGER <= parsed <= _MAX_SAFE_INTEGER:
        raise CanonicalCodecError()
    return parsed


def decode_canonical_json(data: bytes) -> object:
    if type(data) is not bytes or not data:
        raise CanonicalCodecError()
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_float=_reject_number,
            parse_int=_parse_int,
            parse_constant=_reject_number,
        )
        if canonical_json_bytes(value) != data:
            raise CanonicalCodecError()
        return value
    except CanonicalCodecError:
        raise
    except Exception:
        raise CanonicalCodecError() from None


def encode_bundle(payload: PublishedPlanPayload, signature: bytes) -> bytes:
    envelope = {"payload": payload, "signature": base64.b64encode(signature).decode("ascii")}
    return canonical_json_bytes(envelope)


def decode_bundle(data: bytes) -> DecodedPublishedPlanBundle:
    decode_canonical_json(data)
    try:
        envelope = PublishedPlanEnvelope.model_validate_json(data)
        signed_payload = canonical_json_bytes(envelope.payload)
        signature = base64.b64decode(envelope.signature, validate=True)
        return DecodedPublishedPlanBundle(
            envelope=envelope,
            signed_payload=signed_payload,
            signature_bytes=signature,
            bundle_content_sha256=hashlib.sha256(signed_payload).hexdigest(),
        )
    except (ValidationError, ValueError, TypeError):
        raise CanonicalCodecError() from None


def encode_bundle_v3(payload: PublishedPlanPayloadV3, signature: bytes) -> bytes:
    envelope = {"payload": payload, "signature": base64.b64encode(signature).decode("ascii")}
    return canonical_json_bytes(envelope)


def decode_bundle_v3(data: bytes) -> DecodedPublishedPlanBundleV3:
    """Decode only an explicit 3.0 envelope; v2 is never default-upgraded."""

    decode_canonical_json(data)
    try:
        envelope = PublishedPlanEnvelopeV3.model_validate_json(data)
        signed_payload = canonical_json_bytes(envelope.payload)
        signature = base64.b64decode(envelope.signature, validate=True)
        return DecodedPublishedPlanBundleV3(
            envelope=envelope,
            signed_payload=signed_payload,
            signature_bytes=signature,
            bundle_content_sha256=hashlib.sha256(signed_payload).hexdigest(),
        )
    except (ValidationError, ValueError, TypeError):
        raise CanonicalCodecError() from None
