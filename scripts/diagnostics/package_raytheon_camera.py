"""Build a small offline diagnostic ZIP; never package project data or credentials."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / "raytheon_camera"
    repository = source.parents[2]
    spec = importlib.util.spec_from_file_location("camera_check", source / "camera_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = module.BASELINE
    source_hashes = {}
    for name in module.SOURCE_FILES:
        result = subprocess.run(
            ["git", "-C", str(repository), "show", f"{baseline}:{name}"],
            capture_output=True,
            check=False,
        )
        source_hashes[name] = (
            hashlib.sha256(result.stdout).hexdigest() if not result.returncode else None
        )
    if not source_hashes["src/billiards_ai/lab/mainline1_frame_source.py"]:
        raise RuntimeError("Specified baseline source unavailable; cannot build a traceable packet")
    files = {
        name: (source / name).read_bytes() for name in ("run.ps1", "camera_check.py", "README.txt")
    }
    # Windows PowerShell 5.1 reads BOM-marked UTF-8 consistently.
    files["run.ps1"] = b"\xef\xbb\xbf" + files["run.ps1"].removeprefix(b"\xef\xbb\xbf")
    manifest = {
        "schema_version": "1.0",
        "packet_version": module.PACKET_VERSION,
        "kind": "raytheon_camera_diagnostic_only",
        "baseline_commit": baseline,
        "expected_camera": "RER-U3CAM4K60-CL100 / Sony IMX678",
        "acceptance": "NOT_EVALUATED",
        "windows_hardware_execution": "PENDING",
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in files.items()
        },
        "baseline_source_sha256": source_hashes,
    }
    files["packet-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    print(
        json.dumps(
            {
                "zip": str(args.output.resolve()),
                "sha256": module.sha256(args.output),
                "bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
