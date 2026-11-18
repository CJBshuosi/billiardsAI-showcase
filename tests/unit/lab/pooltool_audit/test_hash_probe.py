import hashlib
import json
from pathlib import Path

import pytest

from billiards_ai.lab.dependency_audit.models import (
    LicenseDecision,
    ProjectMetadata,
    ResolvedPackage,
    WheelArtifact,
)
from billiards_ai.lab.pooltool_audit.hash_probe import (
    PandaHashProbeError,
    build_pooltool_packet,
    canonical_packet_bytes,
    download_panda_hash,
    probe_panda_hash,
    publish_pooltool_packet,
)
from billiards_ai.lab.pooltool_audit.models import (
    PandaArchiveArtifact,
    PooltoolAuditRequest,
    PooltoolMetadataDraft,
    PooltoolPyPIClosure,
    archive_access,
)
from billiards_ai.lab.pooltool_audit.resolver import PANDA_REQUIREMENT


def _draft(payload: bytes) -> PooltoolMetadataDraft:
    request = PooltoolAuditRequest.model_validate(
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
    package = ResolvedPackage(
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
    closure = PooltoolPyPIClosure(
        root="pooltool-billiards==0.6.0",
        panda3d_requirement=PANDA_REQUIREMENT,
        root_requires_dist=(PANDA_REQUIREMENT,),
        external_panda_constraints=(f"pooltool-billiards==0.6.0: {PANDA_REQUIREMENT}",),
        packages=(package,),
        diagnostics=(),
    )
    panda = PandaArchiveArtifact(
        filename="panda3d-1.11.0.dev3702-cp312-cp312-win_amd64.whl",
        url=(
            "https://buildbot.panda3d.org/downloads/"
            "8114971c56ce89781e58336be6d5b349aca07b5a/"
            "panda3d-1.11.0.dev3702-cp312-cp312-win_amd64.whl"
        ),
        tags=("cp312-cp312-win_amd64",),
        bytes=len(payload),
        md5=hashlib.md5(payload).hexdigest(),  # noqa: S324 - official archive contract
        sha256=None,
        last_modified="Wed, 29 Jan 2025 21:05:00 GMT",
        etag='"3357b25-62cdea999d9c4"',
        approval_status="sha256_required",
    )
    return PooltoolMetadataDraft(
        request=request,
        closure=closure,
        panda3d=panda,
        pypi_metadata_bytes=123,
        panda_index_access=archive_access(b"<html></html>", "text/html"),
        payload_count=2,
        payload_bytes=wheel.bytes + len(payload),
        approval_status="panda3d_sha256_required",
        side_effects={
            "archive_downloads": 0,
            "installs": 0,
            "package_imports": 0,
            "weights_loaded": 0,
        },
    )


def test_probe_hashes_exact_payload_and_finalizes_immutable_packet(tmp_path: Path) -> None:
    payload = b"approved-panda-wheel"
    draft = _draft(payload)

    evidence = probe_panda_hash(draft, (payload[:8], payload[8:]), tmp_path / "cache")
    packet = build_pooltool_packet(draft, evidence)
    destination = publish_pooltool_packet(packet, tmp_path / "packets")

    assert evidence.sha256 == hashlib.sha256(payload).hexdigest()
    assert evidence.wheel_path.read_bytes() == payload
    assert packet.panda3d.approval_status == "hash_complete"
    assert packet.payload_count == 2
    assert packet.side_effects.archive_downloads == 1
    assert packet.side_effects.installs == 0
    assert destination.name == packet.packet_sha256
    assert (destination / "packet.json").read_bytes() == canonical_packet_bytes(packet)
    document = json.loads((destination / "packet.json").read_bytes())
    unhashed = {key: value for key, value in document.items() if key != "packet_sha256"}
    expected = (
        json.dumps(unhashed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    assert packet.packet_sha256 == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize("chunks", [(b"short",), (b"approved-panda-wheel-extra",)])
def test_probe_rejects_wrong_size_and_removes_partial_file(
    tmp_path: Path, chunks: tuple[bytes, ...]
) -> None:
    draft = _draft(b"approved-panda-wheel")

    with pytest.raises(PandaHashProbeError, match="byte count"):
        probe_panda_hash(draft, chunks, tmp_path / "cache")

    assert list((tmp_path / "cache").glob("*")) == []


def test_probe_rejects_wrong_official_md5_and_removes_partial_file(tmp_path: Path) -> None:
    expected = b"approved-panda-wheel"
    draft = _draft(expected)
    replacement = b"tampered-panda-wheel"
    assert len(expected) == len(replacement)

    with pytest.raises(PandaHashProbeError, match="MD5"):
        probe_panda_hash(draft, (replacement,), tmp_path / "cache")

    assert list((tmp_path / "cache").glob("*")) == []


def test_packet_rejects_evidence_from_another_draft(tmp_path: Path) -> None:
    payload = b"approved-panda-wheel"
    first = _draft(payload)
    second = first.model_copy(update={"pypi_metadata_bytes": 124})
    evidence = probe_panda_hash(first, (payload,), tmp_path / "cache")

    with pytest.raises(PandaHashProbeError, match="draft identity"):
        build_pooltool_packet(second, evidence)


class _Response:
    def __init__(self, payload: bytes, *, url: str, length: int | None = None) -> None:
        self._payload = payload
        self._offset = 0
        self._url = url
        self.headers = {"Content-Length": str(length if length is not None else len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, size: int) -> bytes:
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class _Opener:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.requests = []

    def open(self, request, timeout: float):
        self.requests.append((request, timeout))
        return self.response


def test_download_uses_only_draft_url_and_exact_content_length(tmp_path: Path) -> None:
    payload = b"approved-panda-wheel"
    draft = _draft(payload)
    opener = _Opener(_Response(payload, url=draft.panda3d.url))

    evidence = download_panda_hash(draft, tmp_path / "cache", opener=opener)

    request, timeout = opener.requests[0]
    assert request.full_url == draft.panda3d.url
    assert request.method == "GET"
    assert timeout == 30
    assert evidence.sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("final_url,length", [("https://example.test/redirect", 20), (None, 21)])
def test_download_rejects_redirect_or_content_length_before_writing(
    tmp_path: Path, final_url: str | None, length: int
) -> None:
    payload = b"approved-panda-wheel"
    draft = _draft(payload)
    response = _Response(payload, url=final_url or draft.panda3d.url, length=length)

    with pytest.raises(PandaHashProbeError):
        download_panda_hash(draft, tmp_path / "cache", opener=_Opener(response))

    assert not (tmp_path / "cache").exists()
