"""Package the current source and immutable RER U3 calibration candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("output already exists")
    repository = Path(__file__).resolve().parents[2]
    packet = repository / "scripts" / "diagnostics" / "raytheon_mainline1_usb_gate"
    files: dict[str, bytes] = {}
    for path in (repository / "src" / "billiards_ai").rglob("*.py"):
        files[path.relative_to(repository).as_posix()] = path.read_bytes()
    for name in ("gate_check.py", "run.ps1", "README.txt"):
        files[name] = (packet / name).read_bytes()
    files["run.ps1"] = b"\xef\xbb\xbf" + files["run.ps1"].removeprefix(b"\xef\xbb\xbf")
    for path in arguments.candidate.rglob("*"):
        if path.is_file():
            files[(Path("candidate") / path.relative_to(arguments.candidate)).as_posix()] = (
                path.read_bytes()
            )
    manifest = {
        "schema_version": "1.0",
        "kind": "mainline1_usb_current_epoch_gate",
        "baseline_commit": "9a144955377b289ce47de71769d1305dd8585de4",
        "source_basis": "baseline_plus_current_usb_acceptance_patch",
        "release": False,
        "production_ready": False,
        "writes_installed_service_or_state_db": False,
        "files": {
            name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for name, value in sorted(files.items())
        },
    }
    files["packet-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arguments.output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(files.items()):
            archive.writestr(name, value)
    print(
        json.dumps(
            {
                "zip": str(arguments.output.resolve()),
                "bytes": arguments.output.stat().st_size,
                "sha256": _sha256(arguments.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
