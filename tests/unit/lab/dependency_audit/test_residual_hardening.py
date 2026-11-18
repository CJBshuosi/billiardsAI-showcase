from __future__ import annotations

import hashlib
import json
import stat
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from billiards_ai.lab.dependency_audit import models as models_module
from billiards_ai.lab.dependency_audit import packet as packet_module
from billiards_ai.lab.dependency_audit.index import (
    WheelIndexError,
    parse_simple_index,
    resolve_target,
)
from billiards_ai.lab.dependency_audit.models import (
    ROOT_PARENT_SENTINEL,
    AuditRequest,
    CandidateDiagnostic,
    CodeProvenance,
    DependencyAuditResult,
    DependencyEdge,
    LicenseDecision,
    MetadataAccess,
    ProjectMetadata,
    ResolvedPackage,
    RootRequirement,
    SourceHash,
    TargetPlatform,
    ToolRuntime,
    WheelArtifact,
)
from billiards_ai.lab.dependency_audit.packet import build_approval_packet
from billiards_ai.lab.dependency_audit.resolver import (
    DependencyResolutionError,
    resolve_dependency_audit,
)

_SCRIPT = Path(__file__).parents[4] / "scripts" / "algorithm" / "dependency_audit.py"
_SPEC = spec_from_file_location("residual_dependency_audit_script", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("dependency audit script cannot be loaded")
_CLI = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CLI)

_RUNTIME = ToolRuntime(
    python="3.12.13",
    packaging="25.0",
    pydantic="2.11.0",
    pyyaml="6.0.2",
    pip="25.1",
    pip_vendored_resolvelib="1.1.0",
)
_ARCHIVE_HASH = "c" * 64
_EXPECTED_ALL = [
    ("rfdetr-1.0.0-py3-none-any.whl", "missing_sha256"),
    ("rfdetr-2.0.0-py3-none-any.whl", "invalid_size"),
    ("rfdetr-bad-one.whl", "malformed_filename"),
]


def _archive_url(filename: str, digest: str = _ARCHIVE_HASH) -> str:
    return (
        "https://files.pythonhosted.org/packages/"
        f"{digest[:2]}/{digest[2:4]}/{digest[4:]}/{filename}"
    )


def _file_record(
    filename: str,
    *,
    hashes: dict[str, str] | None = None,
    size: int = 100,
    url: str | None = None,
) -> dict[str, object]:
    return {
        "core-metadata": False,
        "data-dist-info-metadata": False,
        "filename": filename,
        "hashes": {"sha256": "a" * 64} if hashes is None else hashes,
        "requires-python": ">=3.12",
        "size": size,
        "upload-time": "2026-08-01T09:00:00.000000Z",
        "url": url or _archive_url(filename),
        "yanked": False,
    }


def _simple_payload(name: str, records: list[object]) -> bytes:
    return json.dumps(
        {"meta": {"api-version": "1.0"}, "name": name, "files": records},
        sort_keys=True,
    ).encode()


def _version_payload(name: str, version: str, requires: tuple[str, ...]) -> bytes:
    return json.dumps(
        {
            "info": {
                "name": name,
                "version": version,
                "summary": f"{name} fixture",
                "requires_python": ">=3.12",
                "requires_dist": list(requires),
                "license_expression": "MIT",
                "license": None,
                "classifiers": ["License :: OSI Approved :: MIT License"],
            },
            "urls": [],
        },
        sort_keys=True,
    ).encode()


@pytest.fixture
def audit_request() -> AuditRequest:
    return AuditRequest.model_construct(
        schema_version="1.0",
        request_id="fixture-request",
        purpose="synthetic_smoke",
        target=TargetPlatform(
            os="macos",
            os_version="26.0",
            architecture="arm64",
            python_implementation="cp",
            python_version="3.12.13",
        ),
        roots=(
            RootRequirement.parse("rfdetr==1.9.0"),
            RootRequirement.parse("trackers==2.5.0.post0"),
        ),
        allowed_licenses=("Apache-2.0", "MIT"),
        single_response_bytes_max=12 * 1024 * 1024,
        total_metadata_bytes_max=64 * 1024 * 1024,
        existing_payload_bytes=0,
        installed_size_multipliers={"pure_python": 4.0, "native": 2.5},
        free_space_safety_ratio=0.2,
    )


@pytest.fixture
def snapshot(audit_request: AuditRequest):
    return resolve_target(audit_request.target)


def _package(
    name: str,
    version: str,
    *,
    extras: tuple[str, ...] = (),
    parents: tuple[str, ...] = (ROOT_PARENT_SENTINEL,),
    dependencies: tuple[DependencyEdge, ...] = (),
) -> ResolvedPackage:
    filename = f"{name}-{version}-py3-none-any.whl"
    return ResolvedPackage(
        name=name,
        version=version,
        extras=extras,
        parents=parents,
        wheel=WheelArtifact(
            name=name,
            version=version,
            filename=filename,
            url=_archive_url(filename),
            tags=("py3-none-any",),
            bytes=100,
            sha256="a" * 64,
            requires_python=">=3.12",
            upload_time="2026-08-01T09:00:00Z",
            is_pure_python=True,
        ),
        dependencies=dependencies,
        project_metadata=ProjectMetadata(
            name=name,
            version=version,
            summary=f"{name} fixture",
            requires_python=">=3.12",
            requires_dist=tuple(edge.child for edge in dependencies),
        ),
        license=LicenseDecision(
            expression="MIT",
            evidence=("license_expression: MIT",),
            status="allowed",
            reason="allowlisted",
        ),
    )


def _with_dependency_metadata(
    package: ResolvedPackage,
    *,
    requires_dist: tuple[str, ...],
    dependencies: tuple[DependencyEdge, ...],
    extras: tuple[str, ...] | None = None,
) -> ResolvedPackage:
    metadata = package.project_metadata.model_copy(update={"requires_dist": requires_dist})
    update: dict[str, object] = {
        "dependencies": dependencies,
        "project_metadata": metadata,
    }
    if extras is not None:
        update["extras"] = extras
    return package.model_copy(update=update)


@pytest.fixture
def packages() -> tuple[ResolvedPackage, ...]:
    return (_package("rfdetr", "1.9.0"), _package("trackers", "2.5.0.post0"))


@pytest.fixture
def complete_audit(snapshot, packages) -> DependencyAuditResult:
    return DependencyAuditResult(target=snapshot, packages=packages, diagnostics=())


@pytest.fixture
def accesses() -> tuple[MetadataAccess, ...]:
    return (
        MetadataAccess(
            url="https://pypi.org/simple/rfdetr/",
            content_type="application/vnd.pypi.simple.v1+json",
            bytes=123,
            sha256="d" * 64,
        ),
    )


@pytest.fixture
def provenance() -> CodeProvenance:
    return CodeProvenance(
        commit="a" * 40,
        sources=(SourceHash(path="auditor.py", sha256="e" * 64),),
    )


def build_packet_fixture(
    request: AuditRequest,
    audit: DependencyAuditResult,
    accesses: tuple[MetadataAccess, ...],
    provenance: CodeProvenance,
):
    return build_approval_packet(
        request,
        audit,
        accesses,
        "a" * 40,
        "b" * 64,
        code_sources=provenance.sources,
        tool_runtime=_RUNTIME,
    )


@pytest.fixture
def packet(audit_request, complete_audit, accesses, provenance):
    return build_packet_fixture(audit_request, complete_audit, accesses, provenance)


class MarkerFixtureClient:
    metadata_accesses = ()

    def __init__(self) -> None:
        self.root_requires_dist: tuple[str, ...] = ()

    def simple(self, name: str) -> bytes:
        versions = {"rfdetr": "1.9.0", "trackers": "2.5.0.post0", "child": "1.0.0"}
        filename = f"{name}-{versions[name]}-py3-none-any.whl"
        return _simple_payload(name, [_file_record(filename)])

    def version(self, name: str, version: str) -> bytes:
        requires = self.root_requires_dist if name == "rfdetr" else ()
        return _version_payload(name, version, requires)


@pytest.fixture
def fixture_client() -> MarkerFixtureClient:
    return MarkerFixtureClient()


def test_index_collects_all_fatal_records_before_raising(snapshot) -> None:
    payload = _simple_payload(
        "demo",
        [
            _file_record("demo-bad-one.whl"),
            _file_record("demo-1.0.0-py3-none-any.whl", hashes={}),
            _file_record("demo-2.0.0-py3-none-any.whl", size=0),
        ],
    )

    with pytest.raises(WheelIndexError) as captured:
        parse_simple_index(payload, snapshot, expected_package="demo")

    assert [(item.filename, item.reason) for item in captured.value.diagnostics] == [
        ("demo-1.0.0-py3-none-any.whl", "missing_sha256"),
        ("demo-2.0.0-py3-none-any.whl", "invalid_size"),
        ("demo-bad-one.whl", "malformed_filename"),
    ]


@pytest.mark.parametrize("marker_name", ["platform_release", "platform_version"])
def test_unavailable_kernel_marker_fails_closed(
    marker_name: str,
    audit_request: AuditRequest,
    fixture_client: MarkerFixtureClient,
) -> None:
    fixture_client.root_requires_dist = (f'child; {marker_name} >= "1"',)

    with pytest.raises(
        DependencyResolutionError,
        match=f"unavailable target marker: {marker_name}",
    ):
        resolve_dependency_audit(audit_request, fixture_client)


@pytest.mark.parametrize(
    ("marker_name", "marker"),
    [
        (
            "platform_release",
            'python_version >= "3.12" or (os_name == "posix" and "1" <= platform_release)',
        ),
        (
            "platform_version",
            '(platform_version >= "1" and os_name == "posix") or python_version < "0"',
        ),
    ],
)
def test_unavailable_marker_is_found_recursively_on_either_comparison_side(
    marker_name: str,
    marker: str,
    audit_request: AuditRequest,
    fixture_client: MarkerFixtureClient,
) -> None:
    fixture_client.root_requires_dist = (f"child; {marker}",)

    with pytest.raises(
        DependencyResolutionError,
        match=f"unavailable target marker: {marker_name}",
    ):
        resolve_dependency_audit(audit_request, fixture_client)


def test_target_snapshot_records_kernel_markers_as_unavailable(snapshot) -> None:
    assert snapshot.marker_environment.platform_release is None
    assert snapshot.marker_environment.platform_version is None


def test_schema_two_packet_requires_complete_audit_result(
    audit_request: AuditRequest,
    packages: tuple[ResolvedPackage, ...],
    accesses: tuple[MetadataAccess, ...],
    provenance: CodeProvenance,
) -> None:
    with pytest.raises(TypeError):
        build_approval_packet(
            audit_request,
            packages,
            accesses,
            "a" * 40,
            "b" * 64,
            code_sources=provenance.sources,
            tool_runtime=_RUNTIME,
        )


def test_schema_two_packet_rejects_empty_packages(
    audit_request, complete_audit, accesses, provenance
) -> None:
    empty = complete_audit.model_copy(update={"packages": ()})

    with pytest.raises(ValidationError, match="packages"):
        build_packet_fixture(audit_request, empty, accesses, provenance)


def test_schema_two_packet_rejects_empty_metadata_accesses(
    audit_request, complete_audit, provenance
) -> None:
    with pytest.raises(ValidationError, match="metadata_accesses"):
        build_packet_fixture(audit_request, complete_audit, (), provenance)


@pytest.mark.parametrize(
    "forgery",
    ["request", "audit", "access", "source", "runtime"],
)
def test_packet_builder_recursively_revalidates_model_copy_inputs(
    forgery, audit_request, complete_audit, accesses, provenance
) -> None:
    forged_request = audit_request
    forged_audit = complete_audit
    forged_accesses = accesses
    forged_sources = provenance.sources
    forged_runtime = _RUNTIME
    if forgery == "request":
        forged_request = audit_request.model_copy(update={"roots": ()})
    elif forgery == "audit":
        first = complete_audit.packages[0]
        forged_wheel = first.wheel.model_copy(update={"bytes": 0})
        forged_audit = complete_audit.model_copy(
            update={
                "packages": (
                    first.model_copy(update={"wheel": forged_wheel}),
                    complete_audit.packages[1],
                )
            }
        )
    elif forgery == "access":
        forged_accesses = (accesses[0].model_copy(update={"bytes": 0}),)
    elif forgery == "source":
        forged_sources = (provenance.sources[0].model_copy(update={"sha256": "invalid"}),)
    elif forgery == "runtime":
        forged_runtime = _RUNTIME.model_copy(update={"python": None})

    with pytest.raises(ValidationError):
        build_approval_packet(
            forged_request,
            forged_audit,
            forged_accesses,
            "a" * 40,
            "b" * 64,
            code_sources=forged_sources,
            tool_runtime=forged_runtime,
        )


def test_schema_two_packet_rejects_target_request_mismatch(
    audit_request, complete_audit, accesses, provenance
) -> None:
    linux_target = audit_request.target.model_copy(update={"os": "linux"})
    wrong = complete_audit.model_copy(update={"target": resolve_target(linux_target)})

    with pytest.raises(ValidationError, match="target"):
        build_packet_fixture(audit_request, wrong, accesses, provenance)


def test_schema_two_packet_rejects_missing_root_package(
    audit_request, complete_audit, accesses, provenance
) -> None:
    missing = complete_audit.model_copy(update={"packages": complete_audit.packages[:1]})

    with pytest.raises(ValidationError, match="root"):
        build_packet_fixture(audit_request, missing, accesses, provenance)


def test_schema_two_packet_rejects_duplicate_canonical_package_names(
    audit_request, complete_audit, accesses, provenance
) -> None:
    duplicate = complete_audit.packages[1].model_copy(update={"name": "RFDETR"})
    audit = complete_audit.model_copy(update={"packages": (complete_audit.packages[0], duplicate)})

    with pytest.raises(ValidationError, match="canonical package names"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_rejects_project_identity_mismatch(
    audit_request, complete_audit, accesses, provenance
) -> None:
    first = complete_audit.packages[0]
    metadata = first.project_metadata.model_copy(update={"name": "different-project"})
    audit = complete_audit.model_copy(
        update={
            "packages": (
                first.model_copy(update={"project_metadata": metadata}),
                complete_audit.packages[1],
            )
        }
    )

    with pytest.raises(ValidationError, match="project_metadata"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_rejects_missing_dependency_reference(
    audit_request, complete_audit, accesses, provenance
) -> None:
    first = complete_audit.packages[0]
    edge = DependencyEdge(parent="rfdetr==1.9.0", child="missing-child>=1", marker=None)
    audit = complete_audit.model_copy(
        update={
            "packages": (
                first.model_copy(update={"dependencies": (edge,)}),
                complete_audit.packages[1],
            )
        }
    )

    with pytest.raises(ValidationError, match="dependency"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_rejects_missing_active_requires_dist_edge_and_child(
    audit_request, complete_audit, accesses, provenance
) -> None:
    parent = _with_dependency_metadata(
        complete_audit.packages[0],
        requires_dist=("transitive>=1",),
        dependencies=(),
    )
    audit = complete_audit.model_copy(update={"packages": (parent, complete_audit.packages[1])})

    with pytest.raises(ValidationError, match="requires_dist"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_rejects_a_fabricated_edge_absent_from_requires_dist(
    audit_request, complete_audit, accesses, provenance
) -> None:
    edge = DependencyEdge(parent="rfdetr==1.9.0", child="transitive>=1", marker=None)
    parent = _with_dependency_metadata(
        complete_audit.packages[0],
        requires_dist=(),
        dependencies=(edge,),
    )
    child = _package("transitive", "1.0.0", parents=(edge.parent,))
    audit = complete_audit.model_copy(
        update={"packages": (parent, complete_audit.packages[1], child)}
    )

    with pytest.raises(ValidationError, match="requires_dist"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_omits_inactive_extra_and_target_marker_requirements(
    audit_request, complete_audit, accesses, provenance
) -> None:
    parent = _with_dependency_metadata(
        complete_audit.packages[0],
        requires_dist=(
            'gpu-child>=1; extra == "gpu"',
            'windows-child>=1; sys_platform == "win32"',
        ),
        dependencies=(),
        extras=("cpu",),
    )
    audit = complete_audit.model_copy(update={"packages": (parent, complete_audit.packages[1])})

    built = build_packet_fixture(audit_request, audit, accesses, provenance)

    assert built.packages[0].dependencies == ()


def test_schema_two_packet_reconciles_semantic_requirements_for_multiple_extras(
    audit_request, complete_audit, accesses, provenance
) -> None:
    parent_identity = "rfdetr==1.9.0"
    cpu_marker = "extra == 'cpu'"
    gpu_marker = 'extra == "gpu"'
    cpu_edge = DependencyEdge(
        parent=parent_identity,
        child="cpu-child[alpha,beta]>=1",
        marker=cpu_marker,
    )
    gpu_edge = DependencyEdge(
        parent=parent_identity,
        child="gpu-child>=1",
        marker=gpu_marker,
    )
    parent = _with_dependency_metadata(
        complete_audit.packages[0],
        requires_dist=(
            f"CPU_Child[beta,alpha] >= 1; {cpu_marker}",
            f"gpu_child >= 1; {gpu_marker}",
            'tpu-child>=1; extra == "tpu"',
        ),
        dependencies=(cpu_edge, gpu_edge),
        extras=("cpu", "gpu"),
    )
    cpu_child = _package(
        "cpu-child",
        "1.0.0",
        extras=("alpha", "beta"),
        parents=(parent_identity,),
    )
    gpu_child = _package("gpu-child", "1.0.0", parents=(parent_identity,))
    audit = complete_audit.model_copy(
        update={
            "packages": (
                parent,
                complete_audit.packages[1],
                cpu_child,
                gpu_child,
            )
        }
    )

    built = build_packet_fixture(audit_request, audit, accesses, provenance)

    selected = next(item for item in built.packages if item.name == "rfdetr")
    assert selected.dependencies == (cpu_edge, gpu_edge)


def test_schema_two_packet_fails_closed_on_unavailable_requires_dist_marker(
    audit_request, complete_audit, accesses, provenance
) -> None:
    parent = _with_dependency_metadata(
        complete_audit.packages[0],
        requires_dist=('transitive>=1; platform_release >= "1"',),
        dependencies=(),
    )
    audit = complete_audit.model_copy(update={"packages": (parent, complete_audit.packages[1])})

    with pytest.raises(ValidationError, match="unavailable target marker: platform_release"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_rejects_missing_root_parent_sentinel(
    audit_request, complete_audit, accesses, provenance
) -> None:
    first = complete_audit.packages[0]
    audit = complete_audit.model_copy(
        update={"packages": (first.model_copy(update={"parents": ()}), complete_audit.packages[1])}
    )

    with pytest.raises(ValidationError, match="root parent"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_rejects_an_orphan_package(
    audit_request, complete_audit, accesses, provenance
) -> None:
    orphan = _package("orphan", "1.0.0", parents=())
    audit = complete_audit.model_copy(update={"packages": (*complete_audit.packages, orphan)})

    with pytest.raises(ValidationError, match="reachable"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_rejects_an_edge_with_the_parent_reversed(
    audit_request, complete_audit, accesses, provenance
) -> None:
    first, second = complete_audit.packages
    edge = DependencyEdge(
        parent="trackers==2.5.0.post0",
        child="rfdetr>=1",
        marker=None,
    )
    forged_parent = _with_dependency_metadata(
        first,
        requires_dist=(edge.child,),
        dependencies=(edge,),
    ).model_copy(update={"parents": (ROOT_PARENT_SENTINEL, edge.parent)})
    audit = complete_audit.model_copy(
        update={
            "packages": (
                forged_parent,
                second,
            )
        }
    )

    with pytest.raises(ValidationError, match="edge parent"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_rejects_an_edge_without_a_reciprocal_child_parent(
    audit_request, complete_audit, accesses, provenance
) -> None:
    first, second = complete_audit.packages
    edge = DependencyEdge(parent="rfdetr==1.9.0", child="trackers>=2", marker=None)
    parent = _with_dependency_metadata(
        first,
        requires_dist=(edge.child,),
        dependencies=(edge,),
    )
    audit = complete_audit.model_copy(
        update={
            "packages": (
                parent,
                second,
            )
        }
    )

    with pytest.raises(ValidationError, match="child parents"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_rejects_a_dangling_extra_parent(
    audit_request, complete_audit, accesses, provenance
) -> None:
    first, second = complete_audit.packages
    audit = complete_audit.model_copy(
        update={
            "packages": (
                first,
                second.model_copy(update={"parents": (ROOT_PARENT_SENTINEL, "rfdetr==1.9.0")}),
            )
        }
    )

    with pytest.raises(ValidationError, match="reciprocal edge"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


@pytest.mark.parametrize(
    ("child", "match"),
    [
        ("trackers<2", "selected child version"),
        ("trackers[gpu]>=2", "selected child extras"),
    ],
)
def test_schema_two_packet_rejects_unsatisfied_dependency_edges(
    child, match, audit_request, complete_audit, accesses, provenance
) -> None:
    first, second = complete_audit.packages
    edge = DependencyEdge(parent="rfdetr==1.9.0", child=child, marker=None)
    parent = _with_dependency_metadata(
        first,
        requires_dist=(edge.child,),
        dependencies=(edge,),
    )
    audit = complete_audit.model_copy(
        update={
            "packages": (
                parent,
                second.model_copy(update={"parents": (ROOT_PARENT_SENTINEL, edge.parent)}),
            )
        }
    )

    with pytest.raises(ValidationError, match=match):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_rejects_a_root_version_outside_its_specifier(
    audit_request, complete_audit, accesses, provenance
) -> None:
    wrong_root = _package("rfdetr", "9.0.0")
    audit = complete_audit.model_copy(update={"packages": (wrong_root, complete_audit.packages[1])})

    with pytest.raises(ValidationError, match="root requirement version"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def test_schema_two_packet_rejects_missing_root_extras(
    audit_request, complete_audit, accesses, provenance
) -> None:
    forged_request = audit_request.model_copy(
        update={
            "roots": (
                RootRequirement.parse("rfdetr[gpu]==1.9.0"),
                audit_request.roots[1],
            )
        }
    )

    with pytest.raises(ValidationError, match="root requirement extras"):
        build_packet_fixture(forged_request, complete_audit, accesses, provenance)


@pytest.mark.parametrize(
    "document",
    [
        {
            "expression": None,
            "evidence": (),
            "status": "allowed",
            "reason": "allowlisted",
        },
        {
            "expression": "MIT",
            "evidence": ("license_expression: MIT",),
            "status": "allowed",
            "reason": "not_allowlisted",
        },
        {
            "expression": "MIT",
            "evidence": (),
            "status": "allowed",
            "reason": "allowlisted",
        },
        {
            "expression": "MIT",
            "evidence": (
                "license_expression: MIT",
                "classifier: License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            ),
            "status": "allowed",
            "reason": "allowlisted",
        },
        {
            "expression": None,
            "evidence": (),
            "status": "legal_review_required",
            "reason": "not_allowlisted",
        },
        {
            "expression": "MIT",
            "evidence": ("license_expression: MIT",),
            "status": "unknown",
            "reason": "missing_or_unrecognized",
        },
        {
            "expression": None,
            "evidence": (),
            "status": "unknown",
            "reason": "allowlisted",
        },
        {
            "expression": None,
            "evidence": ("classifier: License :: OSI Approved :: MIT License",),
            "status": "unknown",
            "reason": "missing_or_unrecognized",
        },
    ],
)
def test_license_decision_rejects_impossible_relations(document) -> None:
    with pytest.raises(ValidationError, match="license decision"):
        LicenseDecision.model_validate(document)


def test_packet_builder_rejects_forged_unknown_licence_changed_to_allowed(
    audit_request, complete_audit, accesses, provenance
) -> None:
    first = complete_audit.packages[0]
    unknown = LicenseDecision(
        expression=None,
        evidence=(),
        status="unknown",
        reason="missing_or_unrecognized",
    )
    forged = unknown.model_copy(update={"status": "allowed"})
    audit = complete_audit.model_copy(
        update={
            "packages": (
                first.model_copy(update={"license": forged}),
                complete_audit.packages[1],
            )
        }
    )

    with pytest.raises(ValidationError, match="license decision"):
        build_packet_fixture(audit_request, audit, accesses, provenance)


def _validate_packet_update(packet, **update):
    document = packet.model_dump(mode="json")
    document.update(update)
    return models_module.DependencyApprovalPacket.model_validate(document)


def _rehash_model_copy(model, hash_field: str):
    payload = (
        json.dumps(
            model.model_dump(mode="json", exclude={hash_field}),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    return model.model_copy(update={hash_field: hashlib.sha256(payload).hexdigest()})


@pytest.mark.parametrize("forgery", ["empty", "relational"])
def test_packet_writer_revalidates_forged_model_copies(forgery, tmp_path: Path, packet) -> None:
    if forgery == "empty":
        forged = packet.model_copy(update={"packages": ()})
    else:
        first = packet.packages[0]
        forged = packet.model_copy(
            update={
                "packages": (
                    first.model_copy(update={"parents": ()}),
                    *packet.packages[1:],
                )
            }
        )
    forged = _rehash_model_copy(forged, "packet_sha256")

    with pytest.raises(ValidationError):
        packet_module.write_approval_packet(forged, tmp_path)
    assert not (tmp_path / packet.request.request_id).exists()


def test_packet_writer_rejects_rehashed_forged_allowed_licence(
    tmp_path: Path, audit_request, complete_audit, accesses, provenance
) -> None:
    first = complete_audit.packages[0]
    unknown = LicenseDecision(
        expression=None,
        evidence=(),
        status="unknown",
        reason="missing_or_unrecognized",
    )
    blocked_audit = complete_audit.model_copy(
        update={
            "packages": (
                first.model_copy(update={"license": unknown}),
                complete_audit.packages[1],
            )
        }
    )
    blocked = build_packet_fixture(audit_request, blocked_audit, accesses, provenance)
    forged_license = unknown.model_copy(update={"status": "allowed"})
    forged_package = blocked.packages[0].model_copy(update={"license": forged_license})
    forged = blocked.model_copy(
        update={
            "packages": (forged_package, *blocked.packages[1:]),
            "blockers": (),
            "license_preflight": "PASS",
        }
    )
    forged = _rehash_model_copy(forged, "packet_sha256")

    with pytest.raises(ValidationError, match="license decision"):
        packet_module.write_approval_packet(forged, tmp_path)
    assert not (tmp_path / audit_request.request_id).exists()


def test_schema_two_packet_rejects_access_host_mismatch(packet) -> None:
    with pytest.raises(ValidationError, match="access_hosts"):
        _validate_packet_update(packet, access_hosts=["example.invalid"])


def test_schema_two_packet_rejects_blocker_preflight_mismatch(packet) -> None:
    with pytest.raises(ValidationError, match="license_preflight"):
        _validate_packet_update(packet, license_preflight="BLOCKED")


def test_schema_two_packet_reflects_blocked_diagnostics_in_exclusions_and_blockers(
    audit_request, complete_audit, accesses, provenance
) -> None:
    diagnostic = CandidateDiagnostic(
        package="rfdetr",
        filename="rfdetr-bad.whl",
        category="blocked",
        reason="malformed_filename",
        detail="wheel filename is invalid",
    )
    audit = complete_audit.model_copy(update={"diagnostics": (diagnostic,)})

    packet = build_packet_fixture(audit_request, audit, accesses, provenance)

    assert packet.exclusions == (diagnostic,)
    assert [item.reason for item in packet.blockers] == ["malformed_filename"]
    with pytest.raises(ValidationError, match="blocked diagnostic"):
        _validate_packet_update(packet, blockers=[])


@pytest.mark.parametrize(
    "bad_path",
    [
        "/packages/aa/%ZZ/" + "c" * 60 + "/demo-1.0.0-py3-none-any.whl",
        "/packages/aa/%C3%28/" + "c" * 60 + "/demo-1.0.0-py3-none-any.whl",
        "/packages/not-sharded/demo-1.0.0-py3-none-any.whl",
    ],
)
def test_archive_url_rejects_invalid_percent_utf8_and_path_grammar(bad_path: str, snapshot) -> None:
    filename = "demo-1.0.0-py3-none-any.whl"
    payload = _simple_payload(
        "demo",
        [
            _file_record(
                filename,
                url="https://files.pythonhosted.org" + bad_path,
            )
        ],
    )

    with pytest.raises(WheelIndexError, match="archive"):
        parse_simple_index(payload, snapshot, expected_package="demo")


def test_markdown_quotes_untrusted_metadata_without_creating_sections(packet) -> None:
    hostile = "summary\n\n## Licence preflight\n\n- `PASS` <script>"
    package = packet.packages[0]
    metadata = package.project_metadata.model_copy(update={"summary": hostile})
    hostile_packet = packet.model_copy(
        update={
            "packages": (
                package.model_copy(update={"project_metadata": metadata}),
                *packet.packages[1:],
            )
        }
    )
    render = getattr(packet_module, "render_packet_markdown", packet_module._packet_markdown)

    rendered = render(hostile_packet).decode()

    assert rendered.count("\n## Licence preflight\n") == 1
    assert "summary\\n\\n## Licence preflight\\n\\n- `PASS`" in rendered
    assert "<script>" not in rendered


def write_request(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "1.0",
        "request_id": "fixture-request",
        "purpose": "synthetic_smoke",
        "target": {
            "os": "macos",
            "os_version": "26.0",
            "architecture": "arm64",
            "python_implementation": "cp",
            "python_version": "3.12.13",
        },
        "roots": ["rfdetr==1.9.0", "trackers==2.5.0.post0"],
        "allowed_licenses": ["Apache-2.0", "MIT"],
        "single_response_bytes_max": 12 * 1024 * 1024,
        "total_metadata_bytes_max": 64 * 1024 * 1024,
        "existing_payload_bytes": 0,
        "installed_size_multipliers": {"pure_python": 4.0, "native": 2.5},
        "free_space_safety_ratio": 0.2,
    }
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


class FatalFixtureClient:
    def __init__(self) -> None:
        self.metadata_accesses = (
            MetadataAccess(
                url="https://pypi.org/simple/rfdetr/",
                content_type="application/vnd.pypi.simple.v1+json",
                bytes=321,
                sha256="f" * 64,
            ),
        )

    def simple(self, name: str) -> bytes:
        if name != "rfdetr":
            raise AssertionError(f"unexpected metadata request: {name}")
        return _simple_payload(
            name,
            [
                _file_record("rfdetr-bad-one.whl"),
                _file_record("rfdetr-1.0.0-py3-none-any.whl", hashes={}),
                _file_record("rfdetr-2.0.0-py3-none-any.whl", size=0),
            ],
        )

    def version(self, name: str, version: str) -> bytes:
        raise AssertionError(f"unexpected version request: {name}=={version}")


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("injected missing file"), KeyError("injected missing key")],
    ids=["file-not-found", "key-error"],
)
def test_resolver_propagates_unexpected_client_lookup_defects(
    error: Exception,
    audit_request: AuditRequest,
) -> None:
    class BrokenClient:
        metadata_accesses = ()

        def simple(self, name: str) -> bytes:
            raise error

    with pytest.raises(type(error), match="injected missing"):
        resolve_dependency_audit(audit_request, BrokenClient())


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("injected missing file"), KeyError("injected missing key")],
    ids=["file-not-found", "key-error"],
)
def test_cli_propagates_unexpected_lookup_defects_without_failure_artifacts(
    error: Exception,
    tmp_path: Path,
) -> None:
    class BrokenClient:
        metadata_accesses = ()

        def simple(self, name: str) -> bytes:
            raise error

    output = tmp_path / "out"
    with pytest.raises(type(error), match="injected missing"):
        _CLI.main(
            ["--request", str(write_request(tmp_path)), "--output", str(output)],
            client_factory=lambda _request: BrokenClient(),
        )
    assert not output.exists()


@pytest.fixture
def fatal_fixture_factory():
    return lambda _request: FatalFixtureClient()


@pytest.fixture(autouse=True)
def _inject_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    provenance = CodeProvenance(
        commit="a" * 40,
        sources=(SourceHash(path="auditor.py", sha256="b" * 64),),
    )
    monkeypatch.setattr(_CLI, "live_code_provenance", lambda: provenance)


def test_cli_writes_nonapprovable_failure_artifact_with_all_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fatal_fixture_factory,
) -> None:
    exit_code = _CLI.main(
        ["--request", str(write_request(tmp_path)), "--output", str(tmp_path / "out")],
        client_factory=fatal_fixture_factory,
    )

    summary = json.loads(capsys.readouterr().out)
    failure_dir = Path(summary["failure_dir"])
    failure = json.loads((failure_dir / "failure.json").read_text())

    assert exit_code == 2
    assert summary["passed"] is False
    assert summary["blockers"] == [
        "candidate:rfdetr:missing_sha256",
        "candidate:rfdetr:invalid_size",
        "candidate:rfdetr:malformed_filename",
    ]
    assert failure["schema_version"] == "1.0"
    assert failure["approvable"] is False
    assert failure["error_category"] == "dependency_resolution"
    assert failure["side_effects"] == {
        "archive_downloads": 0,
        "installs": 0,
        "package_imports": 0,
        "weights_loaded": 0,
    }
    assert "packages" not in failure
    assert "license_preflight" not in failure
    assert failure_dir.parent.name == "failures"
    assert not (failure_dir / "packet.json").exists()
    assert [(item["filename"], item["reason"]) for item in failure["diagnostics"]] == (
        _EXPECTED_ALL
    )


def test_failure_artifact_is_content_addressed_readonly_and_idempotent(
    tmp_path: Path,
    audit_request: AuditRequest,
    snapshot,
    accesses: tuple[MetadataAccess, ...],
    provenance: CodeProvenance,
) -> None:
    error = DependencyResolutionError(
        "hostile\n\n## Approval\n\n<script>",
        (
            CandidateDiagnostic(
                package="rfdetr",
                filename="bad.whl",
                category="blocked",
                reason="missing_sha256",
                detail="detail\n\n## Injected\n\n`PASS`",
            ),
        ),
    )
    failure = packet_module.build_dependency_audit_failure(
        audit_request,
        snapshot,
        accesses,
        error,
        provenance.commit,
        "b" * 64,
        code_sources=provenance.sources,
        tool_runtime=_RUNTIME,
    )

    first = packet_module.write_dependency_audit_failure(failure, tmp_path)
    second = packet_module.write_dependency_audit_failure(failure, tmp_path)
    document = json.loads((first / "failure.json").read_text())
    unhashed = dict(document)
    unhashed.pop("failure_sha256")
    expected_payload = (
        json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()
    markdown = (first / "failure.md").read_text()

    assert second == first
    assert first == (
        tmp_path
        / audit_request.request_id
        / "failures"
        / hashlib.sha256(expected_payload).hexdigest()
    )
    assert document["failure_sha256"] == hashlib.sha256(expected_payload).hexdigest()
    assert stat.S_IMODE(first.stat().st_mode) == 0o555
    assert stat.S_IMODE((first / "failure.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((first / "failure.md").stat().st_mode) == 0o444
    assert markdown.count("\n## Failure\n") == 1
    assert "\n## Injected\n" not in markdown
    assert "<script>" not in markdown
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        failure.approvable = True

    first.chmod(0o755)
    failure_json = first / "failure.json"
    failure_json.chmod(0o644)
    failure_json.write_text("corrupt", encoding="utf-8")
    failure_json.chmod(0o444)
    first.chmod(0o555)
    with pytest.raises(packet_module.PacketIntegrityError, match="existing failure is corrupt"):
        packet_module.write_dependency_audit_failure(failure, tmp_path)
    assert failure_json.read_text(encoding="utf-8") == "corrupt"


def test_failure_writer_revalidates_forged_approvable_model_copy(
    tmp_path: Path,
    audit_request: AuditRequest,
    snapshot,
    accesses: tuple[MetadataAccess, ...],
    provenance: CodeProvenance,
) -> None:
    failure = packet_module.build_dependency_audit_failure(
        audit_request,
        snapshot,
        accesses,
        DependencyResolutionError("resolution failed"),
        provenance.commit,
        "b" * 64,
        code_sources=provenance.sources,
        tool_runtime=_RUNTIME,
    )
    forged = failure.model_copy(update={"approvable": True})
    forged = _rehash_model_copy(forged, "failure_sha256")

    with pytest.raises(ValidationError, match="approvable"):
        packet_module.write_dependency_audit_failure(forged, tmp_path)
    assert not (tmp_path / audit_request.request_id).exists()
