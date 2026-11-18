import json
from pathlib import Path

import pip
import pytest
from packaging.utils import canonicalize_name
from pip._vendor import resolvelib

from billiards_ai.lab.dependency_audit.index import resolve_target
from billiards_ai.lab.dependency_audit.models import (
    AuditRequest,
    DependencyEdge,
    RootRequirement,
    TargetPlatform,
)
from billiards_ai.lab.dependency_audit.resolver import (
    DependencyResolutionError,
    classify_license,
    resolve_dependencies,
    resolver_runtime_versions,
)
from billiards_ai.lab.dependency_audit.transport import MetadataSecurityError

FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "pypi"


class FixtureMetadataClient:
    def simple(self, name: str) -> bytes:
        return (FIXTURE_ROOT / f"{canonicalize_name(name)}-simple.json").read_bytes()

    def version(self, name: str, version: str) -> bytes:
        return (FIXTURE_ROOT / f"{canonicalize_name(name)}-{version}.json").read_bytes()


@pytest.fixture
def fixture_client() -> FixtureMetadataClient:
    return FixtureMetadataClient()


@pytest.fixture
def audit_request() -> AuditRequest:
    return AuditRequest.model_construct(
        schema_version="1.0",
        request_id="resolver-fixture",
        purpose="synthetic_smoke",
        target=TargetPlatform(
            os="macos",
            os_version="26.0",
            architecture="arm64",
            python_implementation="cp",
            python_version="3.12.13",
        ),
        roots=(RootRequirement.parse("Root[train]==1.0.0"),),
        allowed_licenses=("Apache-2.0", "MIT"),
        single_response_bytes_max=12 * 1024 * 1024,
        total_metadata_bytes_max=64 * 1024 * 1024,
        existing_payload_bytes=0,
        installed_size_multipliers={"pure_python": 4.0, "native": 2.5},
        free_space_safety_ratio=0.2,
    )


def test_resolver_applies_extras_and_target_markers(
    audit_request: AuditRequest, fixture_client: FixtureMetadataClient
) -> None:
    packages = resolve_dependencies(audit_request, fixture_client)

    assert [(item.name, item.version) for item in packages] == [
        ("child", "2.0.0"),
        ("root", "1.0.0"),
    ]
    root = next(item for item in packages if item.name == "root")
    assert root.parents == ("<root>",)
    assert root.extras == ("train",)
    assert root.dependencies == (
        DependencyEdge(
            parent="root==1.0.0",
            child="child>=2",
            marker='python_version >= "3.12" and extra == "train"',
        ),
    )


def test_resolver_fails_closed_on_incompatible_root_constraints(
    audit_request: AuditRequest, fixture_client: FixtureMetadataClient
) -> None:
    conflicting = audit_request.model_copy(
        update={
            "roots": (
                RootRequirement.parse("root[train]==1.0.0"),
                RootRequirement.parse("child<2"),
            )
        }
    )

    with pytest.raises(DependencyResolutionError, match="child"):
        resolve_dependencies(conflicting, fixture_client)


def test_resolver_combines_all_active_specifiers_and_requested_extras(
    audit_request: AuditRequest, fixture_client: FixtureMetadataClient
) -> None:
    combined = audit_request.model_copy(
        update={
            "roots": (
                RootRequirement.parse("child[foo]>=2"),
                RootRequirement.parse("Child[bar]<3"),
            )
        }
    )

    packages = resolve_dependencies(combined, fixture_client)

    assert [(item.name, item.version, item.extras) for item in packages] == [
        ("child", "2.0.0", ("bar", "foo")),
    ]


def test_resolver_fails_closed_when_no_compatible_candidate_exists(
    audit_request: AuditRequest, fixture_client: FixtureMetadataClient
) -> None:
    missing = audit_request.model_copy(update={"roots": (RootRequirement.parse("child>=3"),)})

    with pytest.raises(DependencyResolutionError, match=r"child.*>=3"):
        resolve_dependencies(missing, fixture_client)


def test_resolver_maps_malformed_requires_dist_to_contextual_error(
    audit_request: AuditRequest, fixture_client: FixtureMetadataClient
) -> None:
    class MalformedClient(FixtureMetadataClient):
        def version(self, name: str, version: str) -> bytes:
            payload = json.loads(super().version(name, version))
            if canonicalize_name(name) == "root":
                payload["info"]["requires_dist"] = ["not a valid requirement ==="]
            return json.dumps(payload).encode()

    with pytest.raises(DependencyResolutionError, match=r"root==1\.0\.0.*requirement"):
        resolve_dependencies(audit_request, MalformedClient())


def test_resolver_preserves_original_marker_text_on_the_dependency_edge(
    audit_request: AuditRequest,
) -> None:
    original_marker = "python_version >= '3.12' and extra == 'train'"

    class SingleQuotedMarkerClient(FixtureMetadataClient):
        def version(self, name: str, version: str) -> bytes:
            payload = json.loads(super().version(name, version))
            if canonicalize_name(name) == "root":
                payload["info"]["requires_dist"] = [f"child>=2; {original_marker}"]
            return json.dumps(payload).encode()

    packages = resolve_dependencies(audit_request, SingleQuotedMarkerClient())
    root = next(item for item in packages if item.name == "root")

    assert root.dependencies[0].marker == original_marker


def test_resolver_marker_environment_does_not_leak_the_host_python_version(
    audit_request: AuditRequest,
) -> None:
    class ImplementationMarkerClient(FixtureMetadataClient):
        def version(self, name: str, version: str) -> bytes:
            payload = json.loads(super().version(name, version))
            if canonicalize_name(name) == "root":
                payload["info"]["requires_dist"] = ['child>=2; implementation_version < "3.13"']
            return json.dumps(payload).encode()

    packages = resolve_dependencies(audit_request, ImplementationMarkerClient())

    assert [(item.name, item.version) for item in packages] == [
        ("child", "2.0.0"),
        ("root", "1.0.0"),
    ]
    marker_environment = resolve_target(audit_request.target).marker_environment
    assert marker_environment.platform_release is None
    assert marker_environment.platform_version is None


def test_resolver_rejects_a_simple_document_for_a_different_package(
    audit_request: AuditRequest,
) -> None:
    class MismatchedSimpleClient(FixtureMetadataClient):
        def simple(self, name: str) -> bytes:
            payload = json.loads(super().simple(name))
            if canonicalize_name(name) == "root":
                payload["name"] = "impostor"
                payload["files"][0]["filename"] = "impostor-1.0.0-py3-none-any.whl"
            return json.dumps(payload).encode()

    with pytest.raises(DependencyResolutionError, match=r"root.*package name"):
        resolve_dependencies(audit_request, MismatchedSimpleClient())


def test_resolver_maps_metadata_security_errors_to_contextual_error(
    audit_request: AuditRequest,
) -> None:
    class RejectingClient(FixtureMetadataClient):
        def simple(self, name: str) -> bytes:
            raise MetadataSecurityError("metadata response limit exceeded")

    with pytest.raises(DependencyResolutionError, match=r"root.*metadata response limit"):
        resolve_dependencies(audit_request, RejectingClient())


def test_resolver_uses_the_current_package_context_for_cached_metadata_errors(
    audit_request: AuditRequest,
) -> None:
    class MalformedRootLicenceClient(FixtureMetadataClient):
        def version(self, name: str, version: str) -> bytes:
            payload = json.loads(super().version(name, version))
            if canonicalize_name(name) == "root":
                payload["info"]["license_expression"] = []
            return json.dumps(payload).encode()

    with pytest.raises(
        DependencyResolutionError,
        match=r"root==1\.0\.0 version metadata.*license_expression",
    ):
        resolve_dependencies(audit_request, MalformedRootLicenceClient())


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("Apache-2.0", "allowed"),
        ("MIT OR MPL-2.0", "legal_review_required"),
        (None, "unknown"),
    ],
)
def test_license_classification_is_conservative(expression: str | None, expected: str) -> None:
    decision = classify_license(
        {"license_expression": expression, "classifiers": []},
        ("Apache-2.0", "MIT"),
    )

    assert decision.status == expected


def test_license_classification_allows_a_complete_allowlisted_compound_expression() -> None:
    decision = classify_license(
        {"license_expression": "MIT OR MPL-2.0", "classifiers": []},
        ("MIT OR MPL-2.0",),
    )

    assert decision.status == "allowed"


def test_license_classification_uses_a_standard_classifier_but_not_free_text() -> None:
    classifier = classify_license(
        {
            "license_expression": None,
            "classifiers": ["License :: OSI Approved :: MIT License"],
            "license": "untrusted free text",
        },
        ("MIT",),
    )
    free_text = classify_license(
        {
            "license_expression": None,
            "classifiers": [],
            "license": "MIT",
        },
        ("MIT",),
    )

    assert classifier.status == "allowed"
    assert classifier.expression == "MIT"
    assert free_text.status == "unknown"
    assert free_text.evidence == ("license: MIT",)


def test_license_classification_does_not_allow_conflicting_classifier_evidence() -> None:
    decision = classify_license(
        {
            "license_expression": None,
            "classifiers": [
                "License :: OSI Approved :: MIT License",
                "License :: Other/Proprietary License",
            ],
        },
        ("MIT",),
    )

    assert decision.status == "unknown"
    assert decision.expression is None


def test_license_classification_rejects_a_malformed_classifier_with_a_known_suffix() -> None:
    decision = classify_license(
        {
            "license_expression": None,
            "classifiers": ["License :: Other/Proprietary License :: MIT License"],
        },
        ("MIT",),
    )

    assert decision.status == "unknown"
    assert decision.expression is None


@pytest.mark.parametrize(
    "classifier",
    [
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "License :: Custom Approval :: Example Licence",
    ],
    ids=["gpl", "unrecognized"],
)
def test_license_expression_conflicts_with_any_nonmatching_standard_classifier(
    classifier: str,
) -> None:
    decision = classify_license(
        {
            "license_expression": "MIT",
            "classifiers": [classifier],
        },
        ("MIT",),
    )

    assert decision.status == "unknown"
    assert decision.expression == "MIT"
    assert decision.reason == "conflicting_evidence"
    assert decision.evidence == (
        "license_expression: MIT",
        f"classifier: {classifier}",
    )


def test_unmapped_gpl_classifier_without_expression_is_never_allowed() -> None:
    classifier = "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"

    decision = classify_license(
        {"license_expression": None, "classifiers": [classifier]},
        ("MIT",),
    )

    assert decision.status == "unknown"
    assert decision.expression is None
    assert decision.reason == "missing_or_unrecognized"
    assert decision.evidence == (f"classifier: {classifier}",)


def test_runtime_versions_report_the_installed_pip_and_its_vendored_resolvelib() -> None:
    versions = resolver_runtime_versions()

    assert versions == {
        "pip": pip.__version__,
        "pip_vendored_resolvelib": resolvelib.__version__,
    }
