from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.shared.plan_publication.codec import (
    CanonicalCodecError,
    canonical_json_bytes,
    decode_bundle,
    decode_canonical_json,
)
from billiards_ai.shared.plan_publication.contracts import (
    DecodedPublishedPlanBundle,
    FinalTrainingPlan,
    PublicVerificationKey,
)


def test_canonical_json_has_one_cross_language_byte_representation() -> None:
    value = {"é": "训练", "a": [1, True, None], "z": {"x": "NFC"}}

    encoded = canonical_json_bytes(value)

    assert encoded == '{"a":[1,true,null],"z":{"x":"NFC"},"é":"训练"}'.encode()
    assert decode_canonical_json(encoded) == value


@pytest.mark.parametrize(
    "wire",
    [
        b'{"a":1,"a":1}',
        b'{"a":1.0}',
        b'{"a":9007199254740992}',
        '{"é":"e\u0301"}'.encode(),
        b'{ "a":1}',
        b'{"b":2,"a":1}',
        b'{"a":"\\u0061"}',
        '{"e\u0301":1}'.encode(),
    ],
)
def test_decoder_rejects_duplicate_numeric_unicode_and_byte_malleability(wire: bytes) -> None:
    with pytest.raises(CanonicalCodecError):
        decode_canonical_json(wire)


def test_public_key_contract_rejects_unknown_fields_and_bad_policy() -> None:
    now = datetime(2026, 8, 5, tzinfo=UTC)
    valid = {
        "schema_version": "1.0",
        "algorithm": "ED25519",
        "key_id": "plan-signing-2026-01",
        "key_version": 1,
        "purpose": "PLAN_BUNDLE_SIGNING",
        "public_key_base64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        "issued_at": now,
        "not_before": now,
        "expires_at": now + timedelta(days=30),
        "revoked_at": None,
        "rotation_predecessor_key_id": None,
    }

    key = PublicVerificationKey.model_validate(valid)

    assert key.key_id == "plan-signing-2026-01"
    with pytest.raises(ValidationError):
        PublicVerificationKey.model_validate(valid | {"secret": "forbidden"})
    with pytest.raises(ValidationError):
        PublicVerificationKey.model_validate(valid | {"purpose": "OTHER"})
    with pytest.raises(ValidationError):
        PublicVerificationKey.model_validate(valid | {"revoked_at": now - timedelta(seconds=1)})


def test_legacy_v1_contract_remains_available() -> None:
    from billiards_ai.shared.models.training_plan import PublishedPlanBundle, TrainingPlan

    assert TrainingPlan.model_fields["schema_version"].default == "1.0"
    assert PublishedPlanBundle.model_fields["schema_version"].default == "1.0"
    assert UUID("00000000-0000-4000-8000-000000000001").version == 4


def test_public_key_contract_rejects_rotation_self_loop() -> None:
    now = datetime(2026, 8, 5, tzinfo=UTC)
    with pytest.raises(ValidationError, match="rotation"):
        PublicVerificationKey(
            key_id="same",
            key_version=2,
            public_key_base64="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
            issued_at=now,
            not_before=now,
            expires_at=now + timedelta(days=1),
            rotation_predecessor_key_id="same",
        )


def test_final_plan_rejects_duplicate_block_ids() -> None:
    root = Path(__file__).parents[1] / "fixtures" / "plan_publication" / "v2"
    vector = json.loads((root / "positive.json").read_text())
    decoded = decode_bundle(base64.b64decode(vector["envelope_base64"]))
    plan = decoded.payload.plan
    duplicate = plan.blocks[0].model_copy(update={"order": 2})
    payload = plan.model_dump(mode="python", exclude={"content_sha256"})
    payload["blocks"] = (plan.blocks[0], duplicate)
    content_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    with pytest.raises(ValidationError, match="block"):
        FinalTrainingPlan.model_validate(payload | {"content_sha256": content_hash})


def test_decoded_bundle_rejects_claims_inconsistent_with_envelope() -> None:
    root = Path(__file__).parents[1] / "fixtures" / "plan_publication" / "v2"
    vector = json.loads((root / "positive.json").read_text())
    decoded = decode_bundle(base64.b64decode(vector["envelope_base64"]))

    with pytest.raises(ValidationError):
        DecodedPublishedPlanBundle(
            envelope=decoded.envelope,
            signed_payload=b"different",
            signature_bytes=decoded.signature_bytes,
            bundle_content_sha256=decoded.bundle_content_sha256,
        )
