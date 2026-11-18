from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from billiards_ai.lab.dependency_audit.models import load_audit_request


def _write_request(tmp_path: Path, **updates: object) -> Path:
    payload = {
        "schema_version": "1.0",
        "request_id": "macos-arm64-smoke-r1",
        "purpose": "synthetic_smoke",
        "target": {
            "os": "macos",
            "os_version": "26.0",
            "architecture": "arm64",
            "python_implementation": "cp",
            "python_version": "3.12.13",
        },
        "roots": [
            "rfdetr[train,onnx]==1.9.0",
            "trackers==2.5.0.post0",
            "albumentations",
            "onnxruntime",
        ],
        "allowed_licenses": ["Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "MIT"],
        "single_response_bytes_max": 12582912,
        "total_metadata_bytes_max": 67108864,
        "existing_payload_bytes": 366947924,
        "installed_size_multipliers": {"pure_python": 4.0, "native": 2.5},
        "free_space_safety_ratio": 0.20,
    }
    payload.update(updates)
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_request_freezes_exact_macos_arm64_target(tmp_path: Path) -> None:
    request = load_audit_request(_write_request(tmp_path))
    assert request.target.model_dump() == {
        "os": "macos",
        "os_version": "26.0",
        "architecture": "arm64",
        "python_implementation": "cp",
        "python_version": "3.12.13",
    }
    assert request.roots[0].name == "rfdetr"
    assert request.roots[0].extras == ("onnx", "train")
    assert request.roots[0].specifier == "==1.9.0"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("single_response_bytes_max", 0),
        ("total_metadata_bytes_max", 0),
        ("existing_payload_bytes", -1),
        ("free_space_safety_ratio", -0.01),
        ("roots", []),
    ],
)
def test_request_rejects_unsafe_or_empty_limits(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        load_audit_request(_write_request(tmp_path, **{field: value}))


def test_request_allows_total_metadata_limit_above_64_mib(tmp_path: Path) -> None:
    value = 64 * 1024 * 1024 + 1

    request = load_audit_request(_write_request(tmp_path, total_metadata_bytes_max=value))

    assert request.total_metadata_bytes_max == value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("single_response_bytes_max", 12 * 1024 * 1024 + 1),
        ("total_metadata_bytes_max", 128 * 1024 * 1024 + 1),
    ],
)
def test_request_rejects_metadata_limits_above_the_mandated_hard_caps(
    tmp_path: Path, field: str, value: int
) -> None:
    with pytest.raises(ValidationError):
        load_audit_request(_write_request(tmp_path, **{field: value}))


def test_request_rejects_unpinned_primary_packages(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="rfdetr and trackers roots must be exactly pinned"):
        load_audit_request(_write_request(tmp_path, roots=["rfdetr", "trackers>=2"]))


@pytest.mark.parametrize(
    "roots",
    [
        ["rfdetr[train,onnx]==1.9.0", "albumentations", "onnxruntime"],
        ["trackers==2.5.0.post0"],
    ],
    ids=["detector-only", "tracker-only"],
)
def test_request_allows_exactly_pinned_isolated_primary_package(
    tmp_path: Path,
    roots: list[str],
) -> None:
    request = load_audit_request(_write_request(tmp_path, roots=roots))

    assert [root.raw for root in request.roots] == roots


@pytest.mark.parametrize("roots", [["rfdetr"], ["trackers>=2"]])
def test_request_rejects_unpinned_isolated_primary_package(
    tmp_path: Path,
    roots: list[str],
) -> None:
    with pytest.raises(ValidationError, match="rfdetr and trackers roots must be exactly pinned"):
        load_audit_request(_write_request(tmp_path, roots=roots))


def test_request_still_rejects_roots_without_a_governed_primary_package(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="rfdetr and trackers roots must be exactly pinned"):
        load_audit_request(_write_request(tmp_path, roots=["albumentations", "onnxruntime"]))


@pytest.mark.parametrize(
    "roots",
    [
        ["rfdetr==1.*", "trackers==2.5.0.post0"],
        ["rfdetr==1.9.0", "trackers==2.*"],
        ["rfdetr==1.9.0,==2.0", "trackers==2.5.0.post0"],
        ["rfdetr==1.9.0", "trackers===2.5.0.post0"],
    ],
)
def test_request_rejects_wildcard_primary_pins(tmp_path: Path, roots: list[str]) -> None:
    with pytest.raises(ValidationError, match="rfdetr and trackers roots must be exactly pinned"):
        load_audit_request(_write_request(tmp_path, roots=roots))


@pytest.mark.parametrize(
    "multipliers",
    [
        {"pure_python": 0.0, "native": 2.5},
        {"pure_python": 4.0, "native": -2.5},
        {"pure_python": 4.0},
    ],
)
def test_request_rejects_unsafe_installed_size_multipliers(
    tmp_path: Path, multipliers: dict[str, float]
) -> None:
    with pytest.raises(ValidationError, match="installed size multipliers must be positive"):
        load_audit_request(_write_request(tmp_path, installed_size_multipliers=multipliers))
