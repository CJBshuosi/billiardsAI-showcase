import pytest
from pydantic import ValidationError

from billiards_ai.lab.pooltool_audit.models import PooltoolAuditRequest


def _request(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "request_id": "windows-pooltool-smoke-r2",
        "purpose": "synthetic_smoke",
        "target": {
            "os": "windows",
            "os_version": "11.0",
            "architecture": "x86_64",
            "python_implementation": "cp",
            "python_version": "3.12.10",
        },
        "root": "pooltool-billiards==0.6.0",
        "pypi_index": "https://pypi.org/simple/",
        "panda3d_index": "https://archive.panda3d.org/simple/",
        "single_response_bytes_max": 12 * 1024 * 1024,
        "total_metadata_bytes_max": 128 * 1024 * 1024,
    }
    payload.update(updates)
    return payload


def test_request_freezes_exact_pooltool_and_official_indexes() -> None:
    request = PooltoolAuditRequest.model_validate(_request())
    assert request.root == "pooltool-billiards==0.6.0"
    assert request.panda3d_index == "https://archive.panda3d.org/simple/"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root", "pooltool-billiards>=0.6.0"),
        ("pypi_index", "https://example.com/simple/"),
        ("panda3d_index", "http://archive.panda3d.org/simple/"),
        ("purpose", "local_training"),
    ],
)
def test_request_rejects_scope_expansion(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        PooltoolAuditRequest.model_validate(_request(**{field: value}))
