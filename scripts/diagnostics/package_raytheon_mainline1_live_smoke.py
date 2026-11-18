"""Build the bounded, side-by-side Raytheon Mainline 1 live smoke packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

BASELINE = "9a144955377b289ce47de71769d1305dd8585de4"
DEVICE_LOCATOR = r"USB\VID_32E4&PID_6678&MI_00\6&B920475&0&0000"
DETECTOR_ROOT = (
    "bootstrap-training-runs/"
    "2759358fe07f948a3c056a18f2b592e0b0ebad76364159f0a774c4eca70ca186/"
    "rfdetr-seg-small-bootstrap-short-r1/"
    "967612dd02c72a6af9149b6790820c4e998068df4536d3b006b917c774ac4a51"
)
IDENTITY_ROOT = (
    "identity-group-training-runs/2098055fcba31010d88cda072e5ec33b46f125e91c8ea6512673cb2070349b97"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("output already exists")
    if not (arguments.candidate / "capture-setup.json").is_file():
        raise ValueError("candidate capture-setup.json is missing")
    if not (arguments.candidate / "calibrations" / "registry.json").is_file():
        raise ValueError("candidate calibration registry is missing")

    repository = Path(__file__).resolve().parents[2]
    packet = repository / "scripts" / "diagnostics" / "raytheon_mainline1_live_smoke"
    files: dict[str, bytes] = {}
    for path in (repository / "src" / "billiards_ai").rglob("*.py"):
        files[path.relative_to(repository).as_posix()] = path.read_bytes()
    for relative in (
        "scripts/algorithm/run_mainline1_usb_live.py",
        "configs/algorithm/shot-local.yaml",
    ):
        files[relative] = (repository / relative).read_bytes()
    for name in ("run.ps1", "README.txt"):
        files[name] = (packet / name).read_bytes()
    files["run.ps1"] = b"\xef\xbb\xbf" + files["run.ps1"].removeprefix(b"\xef\xbb\xbf")
    for path in arguments.candidate.rglob("*"):
        if path.is_file():
            relative = (Path("candidate") / path.relative_to(arguments.candidate)).as_posix()
            files[relative] = path.read_bytes()

    manifest = {
        "schema_version": "1.0",
        "packet_version": "raytheon-mainline1-live-smoke-20260906-v5",
        "kind": "mainline1_usb_live_single_shot_diagnostic",
        "baseline_commit": BASELINE,
        "source_basis": "baseline_plus_current_usb_acceptance_patch",
        "development_evidence_only": True,
        "production_ready": False,
        "writes_installed_service_or_state_db": False,
        "run": {
            "device_locator": DEVICE_LOCATOR,
            "runtime_index": 0,
            "max_frames": 600,
            "ready_after_inference_warmup": True,
            "capture_setup_id": "rer-u3-qiaoshi-overhead-20260905-v1",
            "calibration_artifact_id": (
                "ab5cf25b8d0b4fbed0e5ee8b8f7b1a050154b3b9e4ccc3395227862d5c255924"
            ),
        },
        "assets": {
            "detector": {
                "relative_root": DETECTOR_ROOT,
                "manifest_sha256": (
                    "355acf72c8e698e03a1f90265537a2e78e4cd883a1df93e8be8e40376e60934b"
                ),
                "checkpoint_sha256": (
                    "d9cd5efe6c6deb8b7762e08c0f720cb264a2f41bce9174ddc77154a56f434541"
                ),
            },
            "identity": {
                "relative_root": IDENTITY_ROOT,
                "manifest_sha256": (
                    "233a9e63829eceb674dbbe67741d4b4b9afffe84264831f6167e87934c685443"
                ),
                "head_sha256": ("2737c03cf7418da8fd308f44c58460261b4801791ce1d0808c0a589eadd73d03"),
                "metrics_sha256": (
                    "c9a86ad22a56a4351e9889bdb3375c2c3fa36b58c9ca632ffb184aa21fc182f6"
                ),
                "gate_status": "FAIL",
                "development_only": True,
            },
            "dinov2": {
                "relative_root": "dinov2-small-ed25f3a",
                "manifest_sha256": (
                    "875d184852d921f3a9cbee54f18771640a511bd71d6f47829dc5d6ecf30cf21c"
                ),
                "model_sha256": (
                    "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
                ),
                "config_sha256": (
                    "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
                ),
                "preprocessor_sha256": (
                    "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"
                ),
            },
        },
        "files": {
            name: {"bytes": len(data), "sha256": _sha256(data)}
            for name, data in sorted(files.items())
        },
    }
    files["packet-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arguments.output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
