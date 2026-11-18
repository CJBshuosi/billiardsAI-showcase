import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from billiards_ai.lab.dependency_audit.index import resolve_target
from billiards_ai.lab.dependency_audit.models import DependencyAuditFailure, TargetPlatform
from billiards_ai.lab.dependency_audit.packet import _failure_payload_json
from billiards_ai.lab.windows_cuda_evidence import (
    IndexAccess,
    WindowsCudaEvidence,
    canonical_evidence_json,
    parse_target_anchor,
)

ROOT = Path("docs/engineering/algorithm/windows-cuda-rfdetr-smoke")
FAILURE_PATH = ROOT / (
    "windows-cuda-rfdetr-smoke-r1/failures/"
    "d6bf9ba5625e2914137c3f52b0fbaf45754cf18d87aadbd736e2bd2ba1da7b4d/failure.json"
)
REQUEST_PATH = Path("configs/algorithm/dependencies/windows-cuda-rfdetr-smoke.yaml")
PROVENANCE_COMMIT = "360a75257d75869ff2a0c53bc7452cfe07475314"


def _evidence_document() -> dict[str, object]:
    return json.loads((ROOT / "pytorch-index-evidence.json").read_text(encoding="utf-8"))


def test_windows_evidence_binds_the_typed_auditor_failure_to_the_request() -> None:
    request_bytes = REQUEST_PATH.read_bytes()
    failure_bytes = FAILURE_PATH.read_bytes()
    evidence = _evidence_document()
    failure = DependencyAuditFailure.model_validate(json.loads(failure_bytes))

    assert evidence["schema_version"] == "1.0"
    assert evidence["closure_status"] == "unresolved"
    assert evidence["request_sha256"] == hashlib.sha256(request_bytes).hexdigest()
    assert evidence["audit_failure_document_sha256"] == hashlib.sha256(failure_bytes).hexdigest()
    assert evidence["audit_failure_sha256"] == failure.failure_sha256
    assert evidence["audit_failure"]["failure_sha256"] == failure.failure_sha256
    assert evidence["side_effects"] == {
        "archive_downloads": 0,
        "installs": 0,
        "package_imports": 0,
        "weights_loaded": 0,
    }
    assert canonical_evidence_json(evidence) == (ROOT / "pytorch-index-evidence.json").read_bytes()
    assert failure.config_sha256 == hashlib.sha256(request_bytes).hexdigest()
    assert hashlib.sha256(_failure_payload_json(failure)).hexdigest() == failure.failure_sha256
    assert failure.code_commit == PROVENANCE_COMMIT
    assert FAILURE_PATH.parent.name == failure.failure_sha256
    assert len({source.path for source in failure.code_sources}) == len(failure.code_sources)
    for source in failure.code_sources:
        # This is historical evidence for PROVENANCE_COMMIT. Current source files may
        # legitimately evolve; binding them here would silently redefine the old audit.
        assert not Path(source.path).is_absolute()
        assert ".." not in Path(source.path).parts
        assert len(source.sha256) == 64
        int(source.sha256, 16)


def test_windows_evidence_is_deterministic_and_does_not_promote_partial_totals() -> None:
    evidence = _evidence_document()

    assert (
        evidence["evidence_sha256"]
        == hashlib.sha256(
            canonical_evidence_json({**evidence, "evidence_sha256": "0" * 64})
        ).hexdigest()
    )
    assert [item["url"] for item in evidence["index_accesses"]] == sorted(
        item["url"] for item in evidence["index_accesses"]
    )
    assert evidence["known_partial_bytes"]["complete_closure_network_bytes"] is None
    assert evidence["known_partial_bytes"]["complete_closure_installed_bytes"] is None
    assert evidence["known_partial_bytes"]["complete_closure_disk_bytes"] is None
    assert any(item["sha256_fragment"] is None for item in evidence["wheel_observations"])


def test_reverse_order_equivalent_evidence_has_identical_canonical_bytes_and_hash() -> None:
    evidence_data = _evidence_document()
    reversed_data = deepcopy(evidence_data)
    for field in ("index_accesses", "wheel_observations", "license_commercial_states"):
        reversed_data[field].reverse()
    reversed_data["audit_failure"]["metadata_accesses"].reverse()

    canonical = canonical_evidence_json({**evidence_data, "evidence_sha256": "0" * 64})
    reversed_canonical = canonical_evidence_json({**reversed_data, "evidence_sha256": "0" * 64})

    assert reversed_canonical == canonical
    assert hashlib.sha256(reversed_canonical).hexdigest() == evidence_data["evidence_sha256"]


@pytest.mark.parametrize(
    ("field", "nested_field"),
    [
        ("index_accesses", None),
        ("wheel_observations", None),
        ("license_commercial_states", None),
        ("audit_failure", "metadata_accesses"),
    ],
)
def test_windows_evidence_rejects_duplicate_semantic_keys(
    field: str, nested_field: str | None
) -> None:
    evidence_data = _evidence_document()
    collection = (
        evidence_data[field] if nested_field is None else evidence_data[field][nested_field]
    )
    collection.append(deepcopy(collection[0]))

    with pytest.raises(ValidationError, match="duplicate"):
        WindowsCudaEvidence.model_validate(evidence_data)


@pytest.mark.parametrize(
    "updates",
    [
        {"closure_status": "complete"},
        {"acquisition_status": "approved"},
        {"closure_status": "complete", "acquisition_status": "approved"},
    ],
)
def test_windows_evidence_rejects_every_attempt_to_promote_status(
    updates: dict[str, str],
) -> None:
    evidence_data = {**_evidence_document(), **updates}

    with pytest.raises(ValidationError):
        WindowsCudaEvidence.model_validate(evidence_data)


def test_windows_evidence_rejects_provenance_and_unretained_head_tampering() -> None:
    forged_failure = _evidence_document()
    forged_failure["audit_failure"]["failure_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="failure hash"):
        WindowsCudaEvidence.model_validate(forged_failure)

    forged_head = _evidence_document()
    forged_head["wheel_observations"][0]["head_content_length"] = 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WindowsCudaEvidence.model_validate(forged_head)


def test_retained_index_bodies_reparse_target_anchors_as_metadata_only() -> None:
    evidence_data = _evidence_document()
    evidence = WindowsCudaEvidence.model_validate(evidence_data)
    for package in ("torch", "torchvision"):
        raw_access = next(
            item for item in evidence_data["index_accesses"] if item["url"].endswith(f"/{package}/")
        )
        access = IndexAccess.model_validate(raw_access)
        body = (ROOT / f"{package}-index.html").read_bytes()
        parsed = parse_target_anchor(body, access, package)
        recorded = next(x for x in evidence.wheel_observations if x.package == package)
        assert (parsed.filename, parsed.archive_url, parsed.sha256_fragment) == (
            recorded.filename,
            recorded.archive_url,
            recorded.sha256_fragment,
        )


def test_windows_request_uses_the_frozen_cp312_win_amd64_target_tag() -> None:
    request = yaml.safe_load(REQUEST_PATH.read_text(encoding="utf-8"))
    target = TargetPlatform.model_validate(request["target"])
    assert resolve_target(target).wheel_tags[0] == "cp312-cp312-win_amd64"
