"""Build the offline Raytheon RF-DETR FP32/FP16 diagnostic packet."""

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    source = Path(__file__).resolve().parent / "raytheon_rfdetr_fp16"
    module = _load(source / "rfdetr_fp16.py", "rfdetr_fp16")
    files = {
        name: (source / name).read_bytes() for name in ("run.ps1", "rfdetr_fp16.py", "README.txt")
    }
    files["rfdetr_compare.py"] = (
        Path(__file__).resolve().parent / "raytheon_rfdetr_compare" / "rfdetr_compare.py"
    ).read_bytes()
    files["run.ps1"] = b"\xef\xbb\xbf" + files["run.ps1"].removeprefix(b"\xef\xbb\xbf")
    manifest = {
        "schema_version": "1.0",
        "packet_version": module.PACKET_VERSION,
        "kind": "raytheon_rfdetr_fp32_fp16_diagnostic_only",
        "baseline_commit": module.BASELINE,
        "precision_modes": list(module.PRECISION_MODES),
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
    print(
        json.dumps(
            {
                "zip": str(arguments.output.resolve()),
                "bytes": arguments.output.stat().st_size,
                "sha256": hashlib.sha256(arguments.output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
