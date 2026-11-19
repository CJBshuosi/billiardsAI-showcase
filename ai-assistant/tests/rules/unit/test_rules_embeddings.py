import hashlib

import pytest

from cueorbit_replay.rules.embeddings import validate_vector, verify_artifacts


@pytest.mark.parametrize(
    "vector",
    [[1.0] * 383, [float("nan")] * 384, [0.0] * 384, [float("inf")] * 384, [1.0] * 384],
)
def test_invalid_dimension_finite_and_normalization(vector):
    with pytest.raises(ValueError):
        validate_vector(vector)


def test_valid_unit_vector_and_artifact_tamper(tmp_path):
    assert validate_vector([1.0] + [0.0] * 383) == [1.0] + [0.0] * 383
    (tmp_path / "tokenizer.json").write_bytes(b"test")
    manifest = {
        "files": [
            {
                "file": "tokenizer.json",
                "sha256": hashlib.sha256(b"test").hexdigest(),
                "bytes": 4,
            }
        ]
    }
    verify_artifacts(tmp_path, manifest)
    (tmp_path / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_artifacts(tmp_path, manifest)


def test_official_source_cannot_claim_unverified_model_revision():
    from cueorbit_replay.rules.contracts import PDF_SHA256, RuleSourceManifestV1

    with pytest.raises(ValueError):
        RuleSourceManifestV1(
            source_id="official",
            ruleset_id="wpa-heyball",
            effective_date="2025-08-16",
            retrieved_at="2026-11-17T16:00:00Z",
            pdf_sha256=PDF_SHA256,
            extraction_version="test",
            embedding_revision="not-the-verified-model",
        )
