import json

import pytest

from billiards_ai.lab.pooltool_audit.models import PooltoolAuditRequest
from billiards_ai.lab.pooltool_audit.resolver import (
    PANDA_REQUIREMENT,
    PooltoolMetadataError,
    resolve_pooltool_pypi_closure,
)


def _simple(name: str, version: str) -> bytes:
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    return json.dumps(
        {
            "meta": {"api-version": "1.0"},
            "name": name,
            "files": [
                {
                    "filename": filename,
                    "url": (f"https://files.pythonhosted.org/packages/aa/bb/{'c' * 60}/{filename}"),
                    "hashes": {"sha256": "1" * 64},
                    "requires-python": ">=3.10",
                    "size": 1234,
                    "upload-time": "2026-01-01T00:00:00Z",
                    "yanked": False,
                }
            ],
        }
    ).encode()


def _version(name: str, version: str, dependencies: list[str]) -> bytes:
    return json.dumps(
        {
            "info": {
                "name": name,
                "version": version,
                "summary": "fixture",
                "requires_python": ">=3.10",
                "requires_dist": dependencies,
                "license_expression": "Apache-2.0",
                "classifiers": [],
                "license": None,
            }
        }
    ).encode()


class _Client:
    def __init__(
        self,
        panda_requirement: str = 'panda3d==1.11.0.dev3702; sys_platform != "darwin"',
    ) -> None:
        self.panda_requirement = panda_requirement

    def simple(self, name: str) -> bytes:
        versions = {"pooltool-billiards": "0.6.0", "child": "1.0.0"}
        return _simple(name, versions[name])

    def version(self, name: str, version: str) -> bytes:
        dependencies = (
            [self.panda_requirement, "child==1.0.0"]
            if name == "pooltool-billiards"
            else ["panda3d>=1.10.13"]
        )
        return _version(name, version, dependencies)


def _request() -> PooltoolAuditRequest:
    return PooltoolAuditRequest.model_validate(
        {
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
    )


def test_resolves_pypi_closure_and_retains_external_panda_pin() -> None:
    result = resolve_pooltool_pypi_closure(_request(), _Client())
    assert [(item.name, item.version) for item in result.packages] == [
        ("child", "1.0.0"),
        ("pooltool-billiards", "0.6.0"),
    ]
    assert result.panda3d_requirement == PANDA_REQUIREMENT
    assert PANDA_REQUIREMENT in result.root_requires_dist
    assert result.external_panda_constraints == (
        "child==1.0.0: panda3d>=1.10.13",
        f"pooltool-billiards==0.6.0: {PANDA_REQUIREMENT}",
    )


def test_fails_closed_if_upstream_panda_pin_changes() -> None:
    with pytest.raises(PooltoolMetadataError, match="Panda3D pin"):
        resolve_pooltool_pypi_closure(
            _request(),
            _Client("panda3d==1.11.0.dev9999 ; sys_platform != 'darwin'"),
        )


def test_fails_closed_if_transitive_constraint_rejects_frozen_panda() -> None:
    client = _Client()
    original = client.version

    def incompatible(name: str, version: str) -> bytes:
        if name == "child":
            return _version(name, version, ["panda3d<1.10"])
        return original(name, version)

    client.version = incompatible  # type: ignore[method-assign]
    with pytest.raises(PooltoolMetadataError, match="does not satisfy"):
        resolve_pooltool_pypi_closure(_request(), client)


def test_accepts_pypi_null_requires_dist_as_no_dependencies() -> None:
    client = _Client()
    original = client.version

    def null_dependencies(name: str, version: str) -> bytes:
        payload = json.loads(original(name, version))
        if name == "child":
            payload["info"]["requires_dist"] = None
        return json.dumps(payload).encode()

    client.version = null_dependencies  # type: ignore[method-assign]
    result = resolve_pooltool_pypi_closure(_request(), client)
    assert {package.name for package in result.packages} == {
        "child",
        "pooltool-billiards",
    }
