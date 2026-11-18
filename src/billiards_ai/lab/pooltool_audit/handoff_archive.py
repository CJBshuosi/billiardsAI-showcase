"""Build a deterministic offline Raytheon handoff from verified pooltool wheels."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
import zipfile
from pathlib import Path

from billiards_ai.lab.pooltool_audit.handoff import (
    DownloadedWheel,
    PooltoolHandoffError,
    download_pooltool_wheels,
    load_pooltool_packet,
)

PREFIX = "billiardsAI-mainline1/"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    return info


def append_handoff_payloads(
    archive: Path,
    *,
    packet_path: Path,
    packet_sha256: str,
    source_commit: str,
    wheels: tuple[DownloadedWheel, ...],
) -> dict[str, object]:
    """Append verified immutable inputs to an existing source archive."""

    if re.fullmatch(r"[0-9a-f]{64}", packet_sha256) is None:
        raise PooltoolHandoffError("handoff packet identity is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise PooltoolHandoffError("handoff source commit is invalid")
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "source_commit": source_commit,
        "packet_sha256": packet_sha256,
        "payload_count": len(wheels),
        "payload_bytes": sum(wheel.bytes for wheel in wheels),
        "network_required_on_raytheon": False,
        "installs_performed": 0,
        "synthetic_smoke_performed": 0,
        "wheels": [
            {
                "filename": wheel.filename,
                "bytes": wheel.bytes,
                "sha256": wheel.sha256,
            }
            for wheel in sorted(wheels, key=lambda item: item.filename)
        ],
    }
    try:
        with zipfile.ZipFile(archive, "a", allowZip64=True) as output:
            names = set(output.namelist())
            additions = {
                f"{PREFIX}_handoff/packet.json",
                f"{PREFIX}_handoff/HANDOFF.json",
                *(f"{PREFIX}_handoff/wheelhouse/{wheel.filename}" for wheel in wheels),
            }
            if names & additions:
                raise PooltoolHandoffError("handoff archive contains duplicate destinations")
            output.writestr(_info(f"{PREFIX}_handoff/packet.json"), packet_path.read_bytes())
            output.writestr(_info(f"{PREFIX}_handoff/HANDOFF.json"), _canonical(manifest))
            for wheel in sorted(wheels, key=lambda item: item.filename):
                output.writestr(
                    _info(f"{PREFIX}_handoff/wheelhouse/{wheel.filename}"),
                    wheel.path.read_bytes(),
                )
    except (OSError, zipfile.BadZipFile) as error:
        raise PooltoolHandoffError("handoff archive could not be finalized") from error
    return manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_handoff_archive(
    *,
    project_root: Path,
    source_commit: str,
    packet_path: Path,
    packet_sha256: str,
    wheelhouse: Path,
    output_root: Path,
) -> Path:
    """Archive the exact Git commit and already-verified packet without network."""

    project_root = project_root.resolve()
    packet = load_pooltool_packet(packet_path, packet_sha256)

    def forbidden(_url: str) -> bytes:
        raise PooltoolHandoffError("network is forbidden while building handoff archive")

    wheels = download_pooltool_wheels(packet, wheelhouse, fetch=forbidden)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".pooltool-handoff.tmp-{uuid.uuid4()}.zip"
    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", source_commit],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual_commit != source_commit:
            raise PooltoolHandoffError("handoff source commit does not resolve exactly")
        subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "archive",
                "--format=zip",
                f"--prefix={PREFIX}",
                f"--output={temporary}",
                source_commit,
            ],
            check=True,
            capture_output=True,
        )
        append_handoff_payloads(
            temporary,
            packet_path=packet_path,
            packet_sha256=packet_sha256,
            source_commit=source_commit,
            wheels=wheels,
        )
        identity = _file_sha256(temporary)
        destination = output_root / f"billiardsAI-pooltool-handoff-{identity}.zip"
        if destination.exists():
            if _file_sha256(destination) != identity:
                raise PooltoolHandoffError("existing handoff archive conflicts")
            return destination
        os.replace(temporary, destination)
        return destination
    except subprocess.CalledProcessError as error:
        raise PooltoolHandoffError("Git source archive failed") from error
    finally:
        temporary.unlink(missing_ok=True)
