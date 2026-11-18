"""Build the offline, read-only Raytheon asset inventory packet."""

import argparse
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / "raytheon_inventory"
    camera_source = source.parent / "raytheon_camera" / "camera_check.py"
    spec = importlib.util.spec_from_file_location("inventory_check", source / "inventory_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {
        "run.ps1": (source / "run.ps1").read_bytes(),
        "inventory_check.py": (source / "inventory_check.py").read_bytes(),
        "camera_check.py": camera_source.read_bytes(),
        "README.txt": (source / "README.txt").read_bytes(),
    }
    files["run.ps1"] = b"\xef\xbb\xbf" + files["run.ps1"].removeprefix(b"\xef\xbb\xbf")
    manifest = {
        "schema_version": "1.0",
        "packet_version": module.PACKET_VERSION,
        "kind": "raytheon_asset_inventory_diagnostic_only",
        "baseline_commit": module.BASELINE,
        "read_only": True,
        "acceptance": "NOT_EVALUATED",
        "production_ready": False,
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in files.items()
        },
    }
    files["packet-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    result = {
        "zip": str(args.output.resolve()),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "bytes": args.output.stat().st_size,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
