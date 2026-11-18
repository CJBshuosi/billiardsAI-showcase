"""Build the offline Raytheon RF-DETR comparison packet."""

import argparse
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    source = Path(__file__).resolve().parent / "raytheon_rfdetr_compare"
    spec = importlib.util.spec_from_file_location("rfdetr_compare", source / "rfdetr_compare.py")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    files = {
        name: (source / name).read_bytes()
        for name in ("run.ps1", "rfdetr_compare.py", "README.txt")
    }
    files["run.ps1"] = b"\xef\xbb\xbf" + files["run.ps1"].removeprefix(b"\xef\xbb\xbf")
    manifest = {
        "schema_version": "1.0",
        "packet_version": module.PACKET_VERSION,
        "kind": "raytheon_rfdetr_comparison_diagnostic_only",
        "baseline_commit": module.BASELINE,
        "identity_evaluated": False,
        "official_results_written": False,
        "acceptance": "NOT_EVALUATED",
        "production_ready": False,
        "files": {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in files.items()
        },
    }
    files["packet-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arguments.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            archive.writestr(name, payload)
    result = {
        "zip": str(arguments.output.resolve()),
        "bytes": arguments.output.stat().st_size,
        "sha256": hashlib.sha256(arguments.output.read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
