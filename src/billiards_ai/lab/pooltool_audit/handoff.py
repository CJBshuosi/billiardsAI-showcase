"""Standard-library-only handoff for an explicitly approved pooltool packet."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import ssl
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

Fetch = Callable[[str], bytes]


class PooltoolHandoffError(RuntimeError):
    """Raised when packet or wheel evidence fails closed."""


@dataclass(frozen=True, slots=True)
class DownloadedWheel:
    filename: str
    path: Path
    bytes: int
    sha256: str


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise PooltoolHandoffError(f"pooltool wheel redirect rejected: {code}")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _safe_filename(value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise PooltoolHandoffError("pooltool wheel filename is unsafe")
    return value


def _validate_url(url: object, filename: str, *, panda: bool) -> str:
    if not isinstance(url, str):
        raise PooltoolHandoffError("pooltool wheel URL is incomplete")
    parsed = urlsplit(url)
    expected_host = "buildbot.panda3d.org" if panda else "files.pythonhosted.org"
    if (
        parsed.scheme != "https"
        or parsed.netloc != expected_host
        or parsed.hostname != expected_host
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith(f"/{filename}")
    ):
        raise PooltoolHandoffError("pooltool wheel host or path is outside packet policy")
    panda_path = rf"/downloads/[0-9a-f]{{40}}/{re.escape(filename)}"
    if panda and re.fullmatch(panda_path, parsed.path) is None:
        raise PooltoolHandoffError("Panda3D wheel path is outside packet policy")
    if not panda and not parsed.path.startswith("/packages/"):
        raise PooltoolHandoffError("PyPI wheel path is outside packet policy")
    return url


def _specs(packet: dict[str, Any]) -> tuple[dict[str, object], ...]:
    closure = packet.get("closure")
    packages = closure.get("packages") if isinstance(closure, dict) else None
    if not isinstance(packages, list):
        raise PooltoolHandoffError("pooltool packet package closure is invalid")
    specs: list[dict[str, object]] = []
    for package in packages:
        wheel = package.get("wheel") if isinstance(package, dict) else None
        if not isinstance(wheel, dict):
            raise PooltoolHandoffError("pooltool packet wheel record is invalid")
        filename = _safe_filename(wheel.get("filename"))
        specs.append(
            {
                "filename": filename,
                "url": _validate_url(wheel.get("url"), filename, panda=False),
                "bytes": wheel.get("bytes"),
                "sha256": wheel.get("sha256"),
            }
        )
    panda = packet.get("panda3d")
    if not isinstance(panda, dict) or panda.get("approval_status") != "hash_complete":
        raise PooltoolHandoffError("pooltool packet Panda3D record is not hash-complete")
    filename = _safe_filename(panda.get("filename"))
    specs.append(
        {
            "filename": filename,
            "url": _validate_url(panda.get("url"), filename, panda=True),
            "bytes": panda.get("bytes"),
            "sha256": panda.get("sha256"),
        }
    )
    for spec in specs:
        if (
            not isinstance(spec["bytes"], int)
            or int(spec["bytes"]) <= 0
            or not isinstance(spec["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", str(spec["sha256"])) is None
        ):
            raise PooltoolHandoffError("pooltool packet payload evidence is incomplete")
    if len(specs) != packet.get("payload_count") or sum(
        int(spec["bytes"]) for spec in specs
    ) != packet.get("payload_bytes"):
        raise PooltoolHandoffError("pooltool packet payload totals do not match")
    return tuple(sorted(specs, key=lambda item: str(item["filename"])))


def load_pooltool_packet(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PooltoolHandoffError("pooltool packet cannot be read") from error
    if not isinstance(document, dict) or document.get("packet_sha256") != expected_sha256:
        raise PooltoolHandoffError("pooltool packet identity does not match approval")
    unhashed = {key: value for key, value in document.items() if key != "packet_sha256"}
    actual = hashlib.sha256(_canonical(unhashed)).hexdigest()
    if not hmac.compare_digest(actual, expected_sha256):
        raise PooltoolHandoffError("pooltool packet SHA-256 validation failed")
    if document.get("approval_status") != "hash_complete":
        raise PooltoolHandoffError("pooltool packet is not hash-complete")
    if document.get("side_effects") != {
        "archive_downloads": 1,
        "installs": 0,
        "package_imports": 0,
        "weights_loaded": 0,
    }:
        raise PooltoolHandoffError("pooltool packet side-effect evidence is invalid")
    _specs(document)
    return document


def _verify(path: Path, expected_bytes: int, expected_sha256: str) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_size != expected_bytes:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return hmac.compare_digest(digest.hexdigest(), expected_sha256)


def _stream(url: str, destination: Path, expected_bytes: int) -> None:
    opener = urllib.request.build_opener(_RejectRedirects())
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "billiardsAI-pooltool-handoff/1",
        },
        method="GET",
    )
    try:
        with opener.open(request, timeout=120) as response, destination.open("xb") as output:
            if response.geturl() != url:
                raise PooltoolHandoffError("pooltool wheel final URL mismatch")
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > expected_bytes:
                    raise PooltoolHandoffError("pooltool wheel exceeds packet byte count")
                output.write(chunk)
    except PooltoolHandoffError:
        raise
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ssl.SSLError,
        ConnectionError,
        OSError,
    ) as error:
        raise PooltoolHandoffError(
            f"pooltool wheel request failed: {type(error).__name__}"
        ) from error


def download_pooltool_wheels(
    packet: dict[str, Any],
    cache_root: Path,
    *,
    fetch: Fetch | None = None,
) -> tuple[DownloadedWheel, ...]:
    cache_root = cache_root.resolve()
    downloaded = []
    for spec in _specs(packet):
        filename = str(spec["filename"])
        expected_bytes = int(spec["bytes"])
        expected_sha256 = str(spec["sha256"])
        cache_root.mkdir(parents=True, exist_ok=True)
        destination = cache_root / filename
        if not _verify(destination, expected_bytes, expected_sha256):
            if destination.exists():
                raise PooltoolHandoffError(f"existing pooltool wheel conflicts: {filename}")
            temporary = cache_root / f".{filename}.part-{uuid.uuid4()}"
            try:
                if fetch is None:
                    _stream(str(spec["url"]), temporary, expected_bytes)
                else:
                    temporary.write_bytes(fetch(str(spec["url"])))
                if not _verify(temporary, expected_bytes, expected_sha256):
                    raise PooltoolHandoffError(f"pooltool wheel mismatch: {filename}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        downloaded.append(DownloadedWheel(filename, destination, expected_bytes, expected_sha256))
    return tuple(downloaded)
