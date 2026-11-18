import hashlib
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from packaging.utils import canonicalize_name
from pydantic import ValidationError

from billiards_ai.lab.dependency_audit import packet as packet_module
from billiards_ai.lab.dependency_audit.index import resolve_target
from billiards_ai.lab.dependency_audit.models import (
    AuditRequest,
    DependencyApprovalPacket,
    DependencyAuditResult,
    DependencyEdge,
    LicenseDecision,
    MetadataAccess,
    ProjectMetadata,
    ResolvedPackage,
    RootRequirement,
    SourceHash,
    TargetPlatform,
    WheelArtifact,
)
from billiards_ai.lab.dependency_audit.packet import (
    PacketIntegrityError,
    build_approval_packet,
    canonical_packet_json,
    write_approval_packet,
)
from billiards_ai.lab.dependency_audit.resolver import resolver_runtime_versions

_CODE_SOURCES = (SourceHash(path="auditor.py", sha256="f" * 64),)


@pytest.fixture
def audit_request() -> AuditRequest:
    return AuditRequest.model_construct(
        schema_version="1.0",
        request_id="macos-arm64-smoke-r1",
        purpose="synthetic_smoke",
        target=TargetPlatform(
            os="macos",
            os_version="26.0",
            architecture="arm64",
            python_implementation="cp",
            python_version="3.12.13",
        ),
        roots=(
            RootRequirement.parse("rfdetr==1.0.0"),
            RootRequirement.parse("trackers==1.0.0"),
            RootRequirement.parse("zulu-package==1.0.0"),
        ),
        allowed_licenses=("Apache-2.0", "MIT"),
        single_response_bytes_max=12 * 1024 * 1024,
        total_metadata_bytes_max=64 * 1024 * 1024,
        existing_payload_bytes=366_947_924,
        installed_size_multipliers={"pure_python": 4.0, "native": 2.5},
        free_space_safety_ratio=0.20,
    )


def _package(
    name: str,
    *,
    pure_python: bool,
    status: str = "allowed",
    parents: tuple[str, ...] = ("<root>",),
    dependencies: tuple[DependencyEdge, ...] = (),
) -> ResolvedPackage:
    canonical_name = canonicalize_name(name)
    filename = f"{canonical_name}-1.0.0-py3-none-any.whl"
    return ResolvedPackage(
        name=name,
        version="1.0.0",
        extras=(),
        parents=parents,
        wheel=WheelArtifact(
            name=canonical_name,
            version="1.0.0",
            filename=filename,
            url=(f"https://files.pythonhosted.org/packages/bb/bb/{'b' * 60}/{filename}"),
            tags=("py3-none-any",),
            bytes=100,
            sha256="b" * 64,
            requires_python=">=3.12",
            upload_time="2026-07-31T12:00:00Z",
            is_pure_python=pure_python,
        ),
        dependencies=dependencies,
        project_metadata=ProjectMetadata(
            name=canonical_name,
            version="1.0.0",
            summary=f"{canonical_name} fixture",
            requires_python=">=3.12",
            requires_dist=tuple(
                f"{edge.child}; {edge.marker}" if edge.marker is not None else edge.child
                for edge in dependencies
            ),
        ),
        license=LicenseDecision(
            expression="MIT" if status == "allowed" else None,
            evidence=(
                ("classifier: License :: OSI Approved :: MIT License",)
                if status == "allowed"
                else ()
            ),
            status=status,
            reason="allowlisted" if status == "allowed" else "missing_or_unrecognized",
        ),
    )


@pytest.fixture
def packages() -> tuple[ResolvedPackage, ...]:
    return (
        _package(
            "Zulu_Package",
            pure_python=True,
            dependencies=(
                DependencyEdge(
                    parent="zulu-package==1.0.0",
                    child="rfdetr>=1",
                    marker=None,
                ),
                DependencyEdge(
                    parent="zulu-package==1.0.0",
                    child="rfdetr>=1",
                    marker='python_version >= "3.12"',
                ),
            ),
        ),
        _package(
            "rfdetr",
            pure_python=True,
            parents=("<root>", "zulu-package==1.0.0"),
        ),
        _package("trackers", pure_python=False, status="unknown"),
    )


@pytest.fixture
def accesses() -> tuple[MetadataAccess, ...]:
    return (
        MetadataAccess(
            url="https://pypi.org/pypi/zulu-package/1.0.0/json",
            content_type="application/json",
            bytes=30,
            sha256="c" * 64,
        ),
        MetadataAccess(
            url="https://pypi.org/simple/rfdetr/",
            content_type="application/vnd.pypi.simple.v1+json",
            bytes=50,
            sha256="d" * 64,
        ),
    )


def _complete_audit(
    audit_request: AuditRequest,
    packages: tuple[ResolvedPackage, ...],
) -> DependencyAuditResult:
    return DependencyAuditResult(
        target=resolve_target(audit_request.target),
        packages=packages,
        diagnostics=(),
    )


@pytest.fixture
def packet(
    audit_request: AuditRequest,
    packages: tuple[ResolvedPackage, ...],
    accesses: tuple[MetadataAccess, ...],
) -> DependencyApprovalPacket:
    return build_approval_packet(
        audit_request,
        _complete_audit(audit_request, packages),
        accesses,
        code_commit="a" * 40,
        config_sha256="e" * 64,
        code_sources=_CODE_SOURCES,
    )


def test_packet_separates_exact_and_estimated_bytes(
    audit_request: AuditRequest,
    packages: tuple[ResolvedPackage, ...],
    accesses: tuple[MetadataAccess, ...],
) -> None:
    packet = build_approval_packet(
        audit_request,
        _complete_audit(audit_request, packages),
        accesses,
        code_commit="a" * 40,
        config_sha256="e" * 64,
        code_sources=_CODE_SOURCES,
    )

    assert packet.size.archive_bytes == 300
    assert packet.size.metadata_bytes == 80
    assert packet.size.existing_payload_bytes == 366_947_924
    assert packet.size.estimated_installed_bytes == 1_050
    assert packet.size.required_free_bytes == 440_339_225
    assert packet.side_effects.model_dump() == {
        "archive_downloads": 0,
        "installs": 0,
        "package_imports": 0,
        "weights_loaded": 0,
    }


def test_packet_uses_runtime_evidence_and_explicit_config_hash(
    packet: DependencyApprovalPacket,
) -> None:
    resolver_versions = resolver_runtime_versions()
    assert packet.tool_runtime.pip == resolver_versions["pip"]
    assert (
        packet.tool_runtime.pip_vendored_resolvelib == resolver_versions["pip_vendored_resolvelib"]
    )
    assert packet.tool_runtime.python
    assert packet.tool_runtime.packaging
    assert packet.tool_runtime.pydantic
    assert packet.tool_runtime.pyyaml
    assert packet.config_sha256 == "e" * 64
    assert [item.model_dump() for item in packet.blockers] == [
        {
            "category": "license",
            "package": "trackers==1.0.0",
            "filename": None,
            "reason": "unknown_license",
            "detail": "missing_or_unrecognized",
        }
    ]


def test_packet_requires_exact_source_config_hash(
    audit_request: AuditRequest,
    packages: tuple[ResolvedPackage, ...],
    accesses: tuple[MetadataAccess, ...],
) -> None:
    with pytest.raises(TypeError, match="config_sha256"):
        build_approval_packet(
            audit_request,
            _complete_audit(audit_request, packages),
            accesses,
            code_commit="a" * 40,
            code_sources=_CODE_SOURCES,
        )  # type: ignore[call-arg]


def test_packet_rejects_an_invalid_explicit_config_hash(
    audit_request: AuditRequest,
    packages: tuple[ResolvedPackage, ...],
    accesses: tuple[MetadataAccess, ...],
) -> None:
    with pytest.raises(ValidationError, match="config_sha256"):
        build_approval_packet(
            audit_request,
            _complete_audit(audit_request, packages),
            accesses,
            code_commit="a" * 40,
            config_sha256="",
            code_sources=_CODE_SOURCES,
        )


def test_packet_hash_excludes_only_its_own_field_and_json_is_canonical(
    packet: DependencyApprovalPacket,
) -> None:
    encoded = canonical_packet_json(packet)
    decoded = json.loads(encoded)
    unhashed = dict(decoded)
    unhashed.pop("packet_sha256")
    expected_payload = (
        json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    expected_document = (
        json.dumps(decoded, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")

    assert encoded.endswith(b"\n")
    assert encoded == expected_document
    assert packet.packet_sha256 == hashlib.sha256(expected_payload).hexdigest()


def test_packet_json_is_byte_identical_for_reordered_inputs(
    audit_request: AuditRequest,
    packages: tuple[ResolvedPackage, ...],
    accesses: tuple[MetadataAccess, ...],
) -> None:
    first = canonical_packet_json(
        build_approval_packet(
            audit_request,
            _complete_audit(audit_request, packages),
            accesses,
            "a" * 40,
            "e" * 64,
            code_sources=_CODE_SOURCES,
        )
    )
    second = canonical_packet_json(
        build_approval_packet(
            audit_request,
            _complete_audit(audit_request, tuple(reversed(packages))),
            tuple(reversed(accesses)),
            "a" * 40,
            "e" * 64,
            code_sources=_CODE_SOURCES,
        )
    )

    assert second == first
    decoded = json.loads(first)
    assert [item["name"] for item in decoded["packages"]] == [
        "rfdetr",
        "trackers",
        "Zulu_Package",
    ]
    assert [item["url"] for item in decoded["metadata_accesses"]] == sorted(
        item.url for item in accesses
    )
    assert [item["marker"] for item in decoded["packages"][2]["dependencies"]] == [
        None,
        'python_version >= "3.12"',
    ]


def test_packet_write_is_content_addressed_readonly_and_idempotent(
    tmp_path: Path, packet: DependencyApprovalPacket
) -> None:
    first = write_approval_packet(packet, tmp_path)

    assert first == tmp_path / packet.request.request_id / packet.packet_sha256
    assert first.name == packet.packet_sha256
    assert (first / "packet.json").read_bytes() == canonical_packet_json(packet)
    assert (
        (first / "packet.md")
        .read_text(encoding="utf-8")
        .startswith("# Dependency approval packet\n")
    )
    assert stat.S_IMODE(first.stat().st_mode) == 0o555
    assert stat.S_IMODE((first / "packet.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((first / "packet.md").stat().st_mode) == 0o444
    assert write_approval_packet(packet, tmp_path) == first


@pytest.mark.parametrize(
    ("relative_path", "wrong_mode"),
    [(".", 0o755), ("packet.json", 0o644), ("packet.md", 0o640)],
)
def test_packet_write_rejects_byte_identical_targets_with_wrong_modes_without_repair(
    tmp_path: Path,
    packet: DependencyApprovalPacket,
    relative_path: str,
    wrong_mode: int,
) -> None:
    target = write_approval_packet(packet, tmp_path)
    changed = target / relative_path
    changed.chmod(wrong_mode)

    with pytest.raises(PacketIntegrityError, match="existing packet is corrupt"):
        write_approval_packet(packet, tmp_path)

    assert stat.S_IMODE(changed.stat().st_mode) == wrong_mode
    assert (target / "packet.json").read_bytes() == canonical_packet_json(packet)


@pytest.mark.parametrize("filename", ["packet.json", "packet.md"])
def test_packet_write_refuses_corrupt_existing_files_without_repairing_them(
    tmp_path: Path, packet: DependencyApprovalPacket, filename: str
) -> None:
    target = write_approval_packet(packet, tmp_path)
    target.chmod(0o755)
    corrupt_file = target / filename
    corrupt_file.chmod(0o644)
    corrupt_file.write_text("corrupt", encoding="utf-8")
    corrupt_file.chmod(0o444)
    target.chmod(0o555)
    corrupt_bytes = corrupt_file.read_bytes()

    with pytest.raises(PacketIntegrityError, match="existing packet is corrupt"):
        write_approval_packet(packet, tmp_path)

    assert corrupt_file.read_bytes() == corrupt_bytes
    assert stat.S_IMODE(target.stat().st_mode) == 0o555
    assert stat.S_IMODE(corrupt_file.stat().st_mode) == 0o444
    assert not any(".tmp-" in path.name for path in target.parent.iterdir())


def test_temporary_cleanup_preserves_a_replacement_symlink_and_external_files(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "target-parent"
    parent.mkdir()
    temporary = parent / ".packet.tmp-owned"
    temporary.mkdir()
    (temporary / "packet.json").write_bytes(b"owned json")
    (temporary / "packet.md").write_bytes(b"owned markdown")
    owned = packet_module._owned_temporary(temporary)
    displaced = parent / ".displaced-owned-temp"
    temporary.rename(displaced)
    external = tmp_path / "external"
    external.mkdir()
    external_json = external / "packet.json"
    external_markdown = external / "packet.md"
    external_json.write_bytes(b"external json")
    external_markdown.write_bytes(b"external markdown")
    temporary.symlink_to(external, target_is_directory=True)

    packet_module._remove_owned_temporary(owned)

    assert temporary.is_symlink()
    assert external_json.read_bytes() == b"external json"
    assert external_markdown.read_bytes() == b"external markdown"
    assert (displaced / "packet.json").read_bytes() == b"owned json"
    assert (displaced / "packet.md").read_bytes() == b"owned markdown"


def test_temporary_cleanup_preserves_entry_replaced_after_validation(tmp_path: Path) -> None:
    parent = tmp_path / "target-parent"
    parent.mkdir()
    temporary = parent / ".packet.tmp-owned"
    temporary.mkdir()
    packet_json = temporary / "packet.json"
    packet_markdown = temporary / "packet.md"
    packet_json.write_bytes(b"owned json")
    packet_markdown.write_bytes(b"owned markdown")
    owned = packet_module._owned_temporary(temporary)
    displaced_json = temporary / "owned-packet.json"

    def replace_entry(event: str, entry: str | None) -> None:
        if event == "before_entry_unlink" and entry == "packet.json":
            packet_json.rename(displaced_json)
            packet_json.write_bytes(b"replacement json")

    packet_module._remove_owned_temporary(owned, hook=replace_entry)

    assert packet_json.read_bytes() == b"replacement json"
    assert displaced_json.read_bytes() == b"owned json"
    assert packet_markdown.read_bytes() == b"owned markdown"


def test_temporary_cleanup_preserves_directory_replaced_after_validation(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "target-parent"
    parent.mkdir()
    temporary = parent / ".packet.tmp-owned"
    temporary.mkdir()
    (temporary / "packet.json").write_bytes(b"owned json")
    (temporary / "packet.md").write_bytes(b"owned markdown")
    owned = packet_module._owned_temporary(temporary)
    displaced = parent / ".displaced-owned-temp"

    def replace_directory(event: str, entry: str | None) -> None:
        if event == "before_directory_rmdir":
            temporary.rename(displaced)
            temporary.mkdir()

    packet_module._remove_owned_temporary(owned, hook=replace_directory)

    assert temporary.is_dir()
    assert displaced.is_dir()


def test_windows_cleanup_capabilities_do_not_evaluate_posix_flags_or_break_publish(
    tmp_path: Path,
    packet: DependencyApprovalPacket,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows_capabilities = packet_module._CleanupCapabilities(
        descriptor_relative=False,
        directory_open_flags=None,
        file_nofollow_flag=None,
    )
    monkeypatch.setattr(packet_module, "_cleanup_capabilities", lambda: windows_capabilities)
    monkeypatch.delattr(os, "O_DIRECTORY")
    monkeypatch.delattr(os, "O_NOFOLLOW")

    target = write_approval_packet(packet, tmp_path)

    assert (target / "packet.json").read_bytes() == canonical_packet_json(packet)
    assert (target / "packet.md").is_file()
    assert not any(".tmp-" in path.name for path in target.parent.iterdir())


def test_concurrent_packet_writers_reuse_one_complete_target_and_preserve_siblings(
    tmp_path: Path, packet: DependencyApprovalPacket
) -> None:
    target_parent = tmp_path / packet.request.request_id
    target_parent.mkdir()
    unrelated = target_parent / f".{packet.packet_sha256}.tmp-do-not-touch"
    unrelated.mkdir()
    sentinel = unrelated / "sentinel"
    sentinel.write_text("owned elsewhere", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _: write_approval_packet(packet, tmp_path), range(24)))

    assert len(set(results)) == 1
    assert (results[0] / "packet.json").read_bytes() == canonical_packet_json(packet)
    assert (results[0] / "packet.md").is_file()
    assert sentinel.read_text(encoding="utf-8") == "owned elsewhere"
    assert [path for path in target_parent.iterdir() if ".tmp-" in path.name] == [unrelated]


def test_final_fix_11_partial_owned_temporary_acquisition_closes_every_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capabilities = packet_module._cleanup_capabilities()
    if not capabilities.descriptor_relative:
        pytest.skip("descriptor-relative ownership is unavailable on this platform")
    temporary = tmp_path / ".packet.tmp-owned"
    temporary.mkdir()
    (temporary / "packet.json").write_bytes(b"json")
    (temporary / "packet.md").write_bytes(b"markdown")
    original_owned_file = packet_module._owned_file
    acquired: list[int] = []

    def fail_second_acquisition(descriptor: int):
        acquired.append(descriptor)
        if len(acquired) == 2:
            raise PacketIntegrityError("deterministic acquisition failure")
        return original_owned_file(descriptor)

    monkeypatch.setattr(packet_module, "_owned_file", fail_second_acquisition)

    with pytest.raises(PacketIntegrityError, match="deterministic acquisition failure"):
        packet_module._owned_temporary(temporary)

    assert len(acquired) == 2
    for descriptor in acquired:
        with pytest.raises(OSError):
            os.fstat(descriptor)
