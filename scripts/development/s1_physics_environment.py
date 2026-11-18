"""Download reviewed CPython 3.12 Windows physics wheels; never install/import them.

Installation into a new owned environment requires its own concrete execution
authorization. Existing model environments are not modified by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from billiards_ai.lab.pooltool_audit.handoff import (  # noqa: E402
    PooltoolHandoffError,
    _stream,
    _validate_url,
)
from scripts.development.blackbox_report import atomic_json  # noqa: E402
from scripts.development.blackbox_source import pin, plain  # noqa: E402

MANIFEST = Path(__file__).with_name("assets") / "s1-physics-windows.yaml"


def _manifest_bytes(path):
    path = plain(path)
    with path.open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("wheel manifest size exceeded")
    return raw


def _parse_manifest(raw):
    document = json.loads(raw)
    if (
        not isinstance(document, dict)
        or document.get("contract") != "s1-physics-windows-wheels/v1"
        or document.get("target") != "CPython312-win_amd64"
        or not re.fullmatch(r"[0-9a-f]{64}", str(document.get("source_uv_lock_sha256")))
    ):
        raise ValueError("wheel manifest identity invalid")
    wheels = document.get("wheels")
    if not isinstance(wheels, list) or not 0 < len(wheels) <= 32:
        raise ValueError("wheel inventory invalid")
    filenames, names, total = set(), set(), 0
    for wheel in wheels:
        if not isinstance(wheel, dict) or set(wheel) != {
            "name",
            "version",
            "filename",
            "url",
            "bytes",
            "sha256",
        }:
            raise ValueError("wheel record invalid")
        filename = wheel["filename"]
        if (
            not isinstance(filename, str)
            or not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename)
            or filename.casefold() in filenames
            or not isinstance(wheel["name"], str)
            or not re.fullmatch(r"[a-z0-9-]+", wheel["name"])
            or wheel["name"] in names
            or not isinstance(wheel["version"], str)
            or not re.fullmatch(r"[A-Za-z0-9_.+]+", wheel["version"])
            or type(wheel["bytes"]) is not int
            or not 0 < wheel["bytes"] <= 80 * 1024**2
            or not re.fullmatch(r"[0-9a-f]{64}", str(wheel["sha256"]))
        ):
            raise ValueError("wheel identity or budget invalid")
        try:
            _validate_url(wheel["url"], filename, panda=False)
        except PooltoolHandoffError as exc:
            raise ValueError("wheel URL outside official PyPI policy") from exc
        filenames.add(filename.casefold())
        names.add(wheel["name"])
        total += wheel["bytes"]
    if (
        type(document.get("payload_count")) is not int
        or document["payload_count"] != len(wheels)
        or type(document.get("payload_bytes")) is not int
        or document["payload_bytes"] != total
        or total > 220 * 1024**2
    ):
        raise ValueError("wheel inventory totals invalid")
    return document


def load_manifest(path):
    return _parse_manifest(_manifest_bytes(path))


def download(manifest, output, *, stream=None):
    raw = _manifest_bytes(manifest)
    document = _parse_manifest(raw)
    manifest_identity = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    output = plain(output)
    output.mkdir(parents=False)
    for wheel in document["wheels"]:
        temporary = output / (wheel["filename"] + ".part")
        try:
            (stream or _stream)(wheel["url"], temporary, wheel["bytes"])
            identity = {key: wheel[key] for key in ("bytes", "sha256")}
            if pin(temporary) != identity:
                raise ValueError("downloaded wheel identity mismatch")
            temporary.rename(output / wheel["filename"])
        finally:
            temporary.unlink(missing_ok=True)
    if pin(manifest) != manifest_identity:
        raise ValueError("wheel manifest changed during download")
    receipt = {
        "contract": "s1-physics-wheel-download/v1",
        "manifest": manifest_identity,
        "payload_count": document["payload_count"],
        "payload_bytes": document["payload_bytes"],
        "installs": 0,
        "imports": 0,
    }
    atomic_json(output / "wheels-verified.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(download(args.manifest, args.output)))


if __name__ == "__main__":
    main()
