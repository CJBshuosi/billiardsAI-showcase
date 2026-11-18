from billiards_ai.lab.dependency_audit.models import (
    CandidateDiagnostic,
    LicenseDecision,
    ProjectMetadata,
    ResolvedPackage,
    WheelArtifact,
)
from billiards_ai.lab.pooltool_audit.models import (
    PandaArchiveArtifact,
    PooltoolAuditRequest,
    PooltoolMetadataDraft,
    PooltoolPyPIClosure,
    archive_access,
)
from billiards_ai.lab.pooltool_audit.resolver import PANDA_REQUIREMENT


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


def _package() -> ResolvedPackage:
    wheel = WheelArtifact(
        name="pooltool-billiards",
        version="0.6.0",
        filename="pooltool_billiards-0.6.0-py3-none-any.whl",
        url=(
            "https://files.pythonhosted.org/packages/aa/bb/"
            f"{'c' * 60}/pooltool_billiards-0.6.0-py3-none-any.whl"
        ),
        tags=("py3-none-any",),
        bytes=31_600_000,
        sha256="1" * 64,
        requires_python=">=3.10,<3.14",
        upload_time="2026-03-16T00:00:00Z",
        is_pure_python=True,
    )
    return ResolvedPackage(
        name="pooltool-billiards",
        version="0.6.0",
        extras=(),
        parents=("<root>",),
        wheel=wheel,
        dependencies=(),
        project_metadata=ProjectMetadata(
            name="pooltool-billiards",
            version="0.6.0",
            summary="fixture",
            requires_python=">=3.10,<3.14",
            requires_dist=(),
        ),
        license=LicenseDecision(
            expression="Apache-2.0",
            evidence=("license_expression: Apache-2.0",),
            status="allowed",
            reason="allowlisted",
        ),
    )


def test_draft_derives_exact_payload_totals_and_zero_side_effects() -> None:
    package = _package()
    closure = PooltoolPyPIClosure(
        root="pooltool-billiards==0.6.0",
        panda3d_requirement=PANDA_REQUIREMENT,
        root_requires_dist=(PANDA_REQUIREMENT,),
        external_panda_constraints=(f"pooltool-billiards==0.6.0: {PANDA_REQUIREMENT}",),
        packages=(package,),
        diagnostics=(
            CandidateDiagnostic(
                package="pooltool-billiards",
                filename="source.tar.gz",
                category="excluded",
                reason="source_distribution",
            ),
        ),
    )
    panda = PandaArchiveArtifact(
        filename="panda3d-1.11.0.dev3702-cp312-cp312-win_amd64.whl",
        url=(
            "https://buildbot.panda3d.org/downloads/"
            "8114971c56ce89781e58336be6d5b349aca07b5a/"
            "panda3d-1.11.0.dev3702-cp312-cp312-win_amd64.whl"
        ),
        tags=("cp312-cp312-win_amd64",),
        bytes=53_836_581,
        md5="d6763ec4bdae49bf34527a30d8a69424",
        sha256=None,
        last_modified="Wed, 29 Jan 2025 21:05:00 GMT",
        etag='"3357b25-62cdea999d9c4"',
        approval_status="sha256_required",
    )
    draft = PooltoolMetadataDraft(
        request=_request(),
        closure=closure,
        panda3d=panda,
        pypi_metadata_bytes=123,
        panda_index_access=archive_access(b"<html></html>", "text/html"),
        payload_count=2,
        payload_bytes=31_600_000 + 53_836_581,
        approval_status="panda3d_sha256_required",
        side_effects={
            "archive_downloads": 0,
            "installs": 0,
            "package_imports": 0,
            "weights_loaded": 0,
        },
    )
    assert draft.payload_bytes == 85_436_581
