"""Package existing lens tools plus the opt-in RER capture compatibility profile."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    baseline = "9a144955377b289ce47de71769d1305dd8585de4"
    paths = [root / "src/billiards_ai/__init__.py"]
    paths += list((root / "src/billiards_ai/shared").rglob("*.py"))
    paths += [
        root / "src/billiards_ai/lab" / name
        for name in (
            "__init__.py",
            "contracts.py",
            "dataset.py",
            "mainline1_usb_capture.py",
            "mainline1_lens_sample_capture.py",
            "mainline1_lens_calibration.py",
        )
    ]
    paths += [
        root / "scripts/algorithm" / name
        for name in (
            "probe_mainline1_usb.py",
            "capture_mainline1_usb_lens_samples.py",
            "calibrate_mainline1_usb_lens.py",
            "windows/capture-mainline1-usb-lens-samples.ps1",
            "windows/calibrate-mainline1-usb-lens.ps1",
        )
    ]
    paths += [root / "docs/operations/rer-lens-reuse-20260905.md"]
    files, metadata = {}, {}
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Symlink not allowed: {path}")
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        if path.suffix == ".ps1":
            data = b"\xef\xbb\xbf" + data.removeprefix(b"\xef\xbb\xbf")
        files[name] = data
        original = subprocess.run(
            ["git", "-C", str(root), "show", f"{baseline}:{name}"], capture_output=True
        )
        metadata[name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "baseline_sha256": hashlib.sha256(original.stdout).hexdigest()
            if original.returncode == 0
            else None,
        }
    files["packet-manifest.json"] = json.dumps(
        {
            "baseline_commit": baseline,
            "packet_version": "rer-lens-reuse-20260905-v3",
            "kind": "existing_lens_tools_with_rer_capture_profile",
            "acceptance": "NOT_EVALUATED",
            "production_ready": False,
            "files": metadata,
            "limitations": [
                "operator-confirmed identity, not trusted handle binding",
                "no native camera-read cancellation guarantee",
                "readback checks are not a native controls lock",
                "sample capture is not calibration acceptance",
            ],
        },
        indent=2,
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    print(
        json.dumps(
            {
                "zip": str(args.output.resolve()),
                "bytes": args.output.stat().st_size,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
