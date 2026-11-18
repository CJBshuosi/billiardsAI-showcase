from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from billiards_ai.shared.plan_publication.codec import (
    CanonicalCodecError,
    canonical_json_bytes,
    decode_bundle,
    decode_canonical_json,
)
from billiards_ai.shared.plan_publication.contracts import PublishedPlanEnvelope
from billiards_ai.shared.plan_publication.crypto import CryptographyEd25519Verifier

ROOT = Path(__file__).parents[1] / "fixtures" / "plan_publication" / "v2"


def test_positive_golden_vector_decodes_hashes_and_verifies() -> None:
    vector = json.loads((ROOT / "positive.json").read_text())
    envelope = base64.b64decode(vector["envelope_base64"], validate=True)
    decoded = decode_bundle(envelope)

    assert hashlib.sha256(decoded.signed_payload).hexdigest() == vector["bundle_content_sha256"]
    assert base64.b64encode(decoded.signature_bytes).decode() == vector["signature_base64"]
    CryptographyEd25519Verifier().verify(
        decoded.payload.publication_policy.signing_key,
        decoded.signed_payload,
        decoded.signature_bytes,
        at=decoded.payload.publication_policy.issued_at,
    )


def test_authoritative_json_schema_matches_final_envelope_contract() -> None:
    schema = json.loads((ROOT / "schema.json").read_text())

    assert schema == PublishedPlanEnvelope.model_json_schema()


@pytest.mark.parametrize("filename", ["duplicate-key.json", "unknown-field.json", "non-nfc.json"])
def test_negative_golden_wire_vectors_fail_strict_decode(filename: str) -> None:
    vector = json.loads((ROOT / filename).read_text())
    with pytest.raises(CanonicalCodecError):
        decode_bundle(base64.b64decode(vector["envelope_base64"], validate=True))


def test_golden_vectors_never_contain_private_or_secret_material() -> None:
    text = "\n".join(path.read_text().lower() for path in ROOT.glob("*.json"))
    assert "private_key" not in text
    assert "private-key" not in text
    assert "secret" not in text


def test_codec_rejects_incomplete_drill_definitions_even_with_well_formed_signature() -> None:
    vector = json.loads((ROOT / "positive.json").read_text())
    envelope = base64.b64decode(vector["envelope_base64"], validate=True)
    raw = decode_canonical_json(envelope)
    assert isinstance(raw, dict)
    payload = raw["payload"]
    assert isinstance(payload, dict)
    payload["drill_definitions"] = payload["drill_definitions"][:-1]
    raw["signature"] = base64.b64encode(b"\x00" * 64).decode()

    with pytest.raises(CanonicalCodecError):
        decode_bundle(canonical_json_bytes(raw))
