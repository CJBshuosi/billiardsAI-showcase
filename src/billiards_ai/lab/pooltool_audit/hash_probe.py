"""Exact one-payload hash probe and immutable pooltool packet finalizer."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import ssl
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from billiards_ai.lab.pooltool_audit.models import (
    PandaHashEvidence,
    PooltoolApprovalPacket,
    PooltoolMetadataDraft,
)
from billiards_ai.lab.pooltool_audit.runner import canonical_draft_bytes


class PandaHashProbeError(RuntimeError):
    """Raised when the approved Panda3D payload or final packet fails closed."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise PandaHashProbeError(f"Panda3D payload redirect rejected: {code}")


def _draft_sha256(draft: PooltoolMetadataDraft) -> str:
    return hashlib.sha256(canonical_draft_bytes(draft)).hexdigest()


def _md5() -> Any:
    # MD5 is archive-supplied transport evidence only, never a security anchor.
    return hashlib.md5(usedforsecurity=False)


def _evidence_for_existing(
    draft: PooltoolMetadataDraft, destination: Path
) -> PandaHashEvidence | None:
    if not destination.exists():
        return None
    if not destination.is_file() or destination.stat().st_size != draft.panda3d.bytes:
        raise PandaHashProbeError("existing Panda3D cache target conflicts")
    md5_digest = _md5()
    sha256_digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5_digest.update(chunk)
            sha256_digest.update(chunk)
    if not hmac.compare_digest(md5_digest.hexdigest(), draft.panda3d.md5):
        raise PandaHashProbeError("existing Panda3D cache MD5 does not match")
    return PandaHashEvidence(
        draft_sha256=_draft_sha256(draft),
        url=draft.panda3d.url,
        filename=draft.panda3d.filename,
        bytes=draft.panda3d.bytes,
        md5=md5_digest.hexdigest(),
        sha256=sha256_digest.hexdigest(),
        wheel_path=destination.resolve(),
    )


def probe_panda_hash(
    draft: PooltoolMetadataDraft,
    chunks: Iterable[bytes],
    cache_root: Path,
) -> PandaHashEvidence:
    """Stream one approved wheel, enforce size/MD5, and calculate SHA-256."""

    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / draft.panda3d.filename
    existing = _evidence_for_existing(draft, destination)
    if existing is not None:
        return existing
    temporary = cache_root / f".{draft.panda3d.filename}.tmp-{uuid.uuid4()}"
    byte_count = 0
    md5_digest = _md5()
    sha256_digest = hashlib.sha256()
    try:
        with temporary.open("xb") as handle:
            for chunk in chunks:
                if not isinstance(chunk, bytes) or not chunk:
                    raise PandaHashProbeError("Panda3D payload yielded an invalid chunk")
                byte_count += len(chunk)
                if byte_count > draft.panda3d.bytes:
                    raise PandaHashProbeError("Panda3D payload byte count exceeds draft")
                md5_digest.update(chunk)
                sha256_digest.update(chunk)
                handle.write(chunk)
        if byte_count != draft.panda3d.bytes:
            raise PandaHashProbeError("Panda3D payload byte count does not match draft")
        if not hmac.compare_digest(md5_digest.hexdigest(), draft.panda3d.md5):
            raise PandaHashProbeError("Panda3D payload MD5 does not match official archive")
        temporary.rename(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return PandaHashEvidence(
        draft_sha256=_draft_sha256(draft),
        url=draft.panda3d.url,
        filename=draft.panda3d.filename,
        bytes=byte_count,
        md5=md5_digest.hexdigest(),
        sha256=sha256_digest.hexdigest(),
        wheel_path=destination,
    )


def _validate_download_url(draft: PooltoolMetadataDraft) -> None:
    parsed = urlsplit(draft.panda3d.url)
    expected_path = rf"/downloads/[0-9a-f]{{40}}/{re.escape(draft.panda3d.filename)}"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "buildbot.panda3d.org"
        or parsed.hostname != "buildbot.panda3d.org"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or re.fullmatch(expected_path, parsed.path) is None
    ):
        raise PandaHashProbeError("Panda3D payload URL is outside the approved archive boundary")


def download_panda_hash(
    draft: PooltoolMetadataDraft,
    cache_root: Path,
    *,
    opener: Any | None = None,
    timeout_seconds: float = 30,
) -> PandaHashEvidence:
    """GET exactly the draft wheel and pass its stream to the closed hash probe."""

    if not 0 < timeout_seconds <= 60:
        raise ValueError("Panda3D payload timeout is invalid")
    _validate_download_url(draft)
    client = opener or urllib.request.build_opener(_RejectRedirects())
    request = urllib.request.Request(
        draft.panda3d.url,
        headers={"Accept": "application/octet-stream"},
        method="GET",
    )
    try:
        with client.open(request, timeout=timeout_seconds) as response:
            if response.geturl() != draft.panda3d.url:
                raise PandaHashProbeError("Panda3D payload final URL mismatch")
            raw_length = response.headers.get("Content-Length")
            try:
                content_length = int(raw_length) if raw_length is not None else 0
            except ValueError as error:
                raise PandaHashProbeError("Panda3D payload Content-Length is invalid") from error
            if content_length != draft.panda3d.bytes:
                raise PandaHashProbeError("Panda3D payload Content-Length does not match draft")

            def chunks() -> Iterable[bytes]:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        return
                    yield chunk

            return probe_panda_hash(draft, chunks(), cache_root)
    except PandaHashProbeError:
        raise
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ssl.SSLError,
        ConnectionError,
        OSError,
    ) as error:
        raise PandaHashProbeError(
            f"Panda3D payload request failed: {type(error).__name__}"
        ) from error


def _canonical(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def build_pooltool_packet(
    draft: PooltoolMetadataDraft, evidence: PandaHashEvidence
) -> PooltoolApprovalPacket:
    draft_sha256 = _draft_sha256(draft)
    if evidence.draft_sha256 != draft_sha256:
        raise PandaHashProbeError("Panda3D evidence draft identity does not match")
    expected = draft.panda3d
    if (
        evidence.url != expected.url
        or evidence.filename != expected.filename
        or evidence.bytes != expected.bytes
        or not hmac.compare_digest(evidence.md5, expected.md5)
    ):
        raise PandaHashProbeError("Panda3D evidence does not match metadata draft")
    panda = expected.model_copy(
        update={"sha256": evidence.sha256, "approval_status": "hash_complete"}
    )
    document: dict[str, object] = {
        "schema_version": "1.0",
        "packet_sha256": "0" * 64,
        "draft_sha256": draft_sha256,
        "request": draft.request.model_dump(mode="json"),
        "closure": draft.closure.model_dump(mode="json"),
        "panda3d": panda.model_dump(mode="json"),
        "pypi_metadata_bytes": draft.pypi_metadata_bytes,
        "panda_index_access": draft.panda_index_access.model_dump(mode="json"),
        "payload_count": draft.payload_count,
        "payload_bytes": draft.payload_bytes,
        "approval_status": "hash_complete",
        "side_effects": {
            "archive_downloads": 1,
            "installs": 0,
            "package_imports": 0,
            "weights_loaded": 0,
        },
    }
    unhashed = {key: value for key, value in document.items() if key != "packet_sha256"}
    document["packet_sha256"] = hashlib.sha256(_canonical(unhashed)).hexdigest()
    return PooltoolApprovalPacket.model_validate(document)


def canonical_packet_bytes(packet: PooltoolApprovalPacket) -> bytes:
    return _canonical(packet.model_dump(mode="json"))


def publish_pooltool_packet(packet: PooltoolApprovalPacket, output_root: Path) -> Path:
    payload = canonical_packet_bytes(packet)
    unhashed = packet.model_dump(mode="json", exclude={"packet_sha256"})
    actual = hashlib.sha256(_canonical(unhashed)).hexdigest()
    if not hmac.compare_digest(actual, packet.packet_sha256):
        raise PandaHashProbeError("pooltool packet SHA-256 is invalid")
    parent = output_root.resolve() / packet.request.request_id
    destination = parent / packet.packet_sha256
    if destination.exists():
        packet_path = destination / "packet.json"
        if packet_path.is_file() and packet_path.read_bytes() == payload:
            return destination
        raise PandaHashProbeError("pooltool packet destination conflicts")
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{packet.packet_sha256}.tmp-{uuid.uuid4()}"
    temporary.mkdir()
    try:
        (temporary / "packet.json").write_bytes(payload)
        temporary.rename(destination)
    except OSError as error:
        raise PandaHashProbeError("pooltool packet publication failed") from error
    finally:
        if temporary.exists():
            temporary.rmdir()
    return destination
