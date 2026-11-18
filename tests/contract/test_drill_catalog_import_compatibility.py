"""Shared catalog consumers keep wire identity without importing cloud services."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHARED = "billiards_ai.shared.models.drill_catalog"
TYPES = (
    "DrillContractModel",
    "DrillReviewStatus",
    "DrillViolationCode",
    "TablePointMm",
    "BallPlacement",
    "TableTargetZoneMm",
    "TableSetup",
    "SuccessMetricDefinition",
    "ProgressionRule",
    "RegressionRule",
    "ReviewLineage",
    "DrillEligibilityProfile",
    "DrillDefinition",
    "DrillCatalog",
    "DrillReferenceValidation",
)


def shared_catalog():
    assert importlib.util.find_spec(SHARED) is not None, "shared catalog contracts are missing"
    return importlib.import_module(SHARED)


@pytest.mark.parametrize("name", TYPES)
def test_existing_catalog_imports_are_the_same_types_used_by_shared_consumers(name):
    shared = shared_catalog()
    old = importlib.import_module("billiards_ai.cloud.training.catalog")
    assert getattr(old, name) is getattr(shared, name)
    if name != "DrillContractModel":
        package = importlib.import_module("billiards_ai.cloud.training")
        assert getattr(package, name) is getattr(shared, name)


def test_legacy_json_and_schema_keep_frozen_contract_bytes():
    shared = shared_catalog()
    old = importlib.import_module("billiards_ai.cloud.training.catalog")
    original = (ROOT / "tests/fixtures/core_intelligence/v1/drill-catalog.json").read_bytes()
    for model in (old.DrillCatalog, shared.DrillCatalog):
        catalog = model.model_validate_json(original)
        assert isinstance(catalog.drills[0], old.DrillDefinition)
        assert hashlib.sha256(catalog.model_dump_json().encode()).hexdigest() == (
            "776c8f918308d05a87496c3a964eb15bc766e5df78ca276ef6763379874cc554"
        )
        schema = json.dumps(model.model_json_schema(), sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(schema.encode()).hexdigest() == (
            "ad90def3f96b4e591c32215efcdcd5c71e29cd5b3001b28427573efe446233e3"
        )
        assert catalog.content_sha256 == (
            "412aef2d42a58f1b1d3692453f5c90cc29411d9cc6ff03b113b51e0bc3747957"
        )
    assert old._stored_model_values is shared.drill_model_values


def test_shared_only_payload_decodes_and_verifies_the_existing_signed_bundle(tmp_path):
    # A real package directory, outside the checkout, with no cloud files.
    payload = tmp_path / "shipped"
    package = payload / "billiards_ai"
    package.mkdir(parents=True)
    shutil.copyfile(ROOT / "src/billiards_ai/__init__.py", package / "__init__.py")
    shutil.copytree(
        ROOT / "src/billiards_ai/shared",
        package / "shared",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    golden = tmp_path / "positive.json"
    shutil.copyfile(ROOT / "tests/fixtures/plan_publication/v2/positive.json", golden)
    code = r"""
import base64, hashlib, importlib.abc, json, sys
from pathlib import Path
root, golden = map(Path, sys.argv[1:])
sys.path.insert(0, str(root))
class NoCloud(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "billiards_ai.cloud" or fullname.startswith("billiards_ai.cloud."):
            raise ImportError("shared contracts must not import cloud: " + fullname)
sys.meta_path.insert(0, NoCloud())
from billiards_ai.shared.plan_publication.codec import decode_bundle
from billiards_ai.shared.plan_publication.crypto import CryptographyEd25519Verifier
vector = json.loads(golden.read_text())
decoded = decode_bundle(base64.b64decode(vector["envelope_base64"], validate=True))
assert hashlib.sha256(decoded.signed_payload).hexdigest() == vector["bundle_content_sha256"]
CryptographyEd25519Verifier().verify(
    decoded.payload.publication_policy.signing_key, decoded.signed_payload,
    decoded.signature_bytes, at=decoded.payload.publication_policy.issued_at)
assert not any(
    n == "billiards_ai.cloud" or n.startswith("billiards_ai.cloud.") for n in sys.modules)
for name, module in tuple(sys.modules.items()):
    if name.startswith("billiards_ai") and getattr(module, "__file__", None):
        assert Path(module.__file__).resolve().is_relative_to(root.resolve()), name
print("shared-only decode and real signature verified")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(payload), str(golden)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "shared-only decode and real signature verified"
