"""Build the isolated table diagnostic; no production entry points are run."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lens-candidate", type=Path, required=True)
    parser.add_argument("--board-check", action="store_true")
    parser.add_argument("--frozen-candidate", type=Path)
    parser.add_argument("--motion-check", action="store_true")
    args = parser.parse_args()
    if args.motion_check and not args.frozen_candidate:
        parser.error("--motion-check requires --frozen-candidate")
    root = Path(__file__).resolve().parents[2]
    lens = json.loads(args.lens_candidate.read_text())
    if (
        lens.get("status") != "DIAGNOSTIC_CANDIDATE_NOT_REGISTERED"
        or lens.get("sample_count") != 36
        or lens.get("production_ready") is not False
        or lens.get("device_fingerprint")
        != "865a64805fad161c4f09aeb35214fa23b061be0d8a08c9e6fc99f39c86481a93"
    ):
        raise ValueError("This packet requires the reviewed 36-sample RER candidate")
    paths = [root / "src/billiards_ai/__init__.py"]
    paths += sorted((root / "src/billiards_ai/shared").rglob("*.py"))
    paths += [
        root / "src/billiards_ai/lab" / name
        for name in ("__init__.py", "contracts.py", "dataset.py", "mainline1_usb_capture.py")
    ]
    paths += [
        root / "scripts/diagnostics/raytheon_table" / name
        for name in ("table_check.py", "page.html")
    ]
    paths += [root / "scripts/diagnostics/raytheon_camera/camera_check.py"]
    if args.board_check or args.frozen_candidate:
        paths += [
            root / "src/billiards_ai/lab" / name
            for name in ("mainline1_lens_sample_capture.py", "mainline1_lens_calibration.py")
        ]
        paths += [
            root / "scripts/diagnostics/raytheon_table" / name
            for name in ("board_check.py", "rails-page.html")
        ]
    files = {p.relative_to(root).as_posix(): p.read_bytes() for p in paths if not p.is_symlink()}
    if len(files) != len(paths):
        raise ValueError("Symlinks are not allowed")
    files["run.ps1"] = b"\xef\xbb\xbf" + (
        root / "scripts/diagnostics/raytheon_table/run.ps1"
    ).read_bytes().removeprefix(b"\xef\xbb\xbf")
    doc = "rer-table-board-20260905.md" if args.board_check else "rer-table-check-20260905.md"
    if args.frozen_candidate:
        doc = "rer-plane-recheck-20260905.md"
    if args.motion_check:
        doc = "rer-motion-check-20260905.md"
    files["README.md"] = (root / "docs/operations" / doc).read_bytes()
    files["lens-candidate.json"] = args.lens_candidate.read_bytes()
    if args.frozen_candidate:
        candidate = json.loads(args.frozen_candidate.read_text())
        if (
            candidate.get("lens_file_sha256")
            != hashlib.sha256(files["lens-candidate.json"]).hexdigest()
        ):
            raise ValueError("Frozen plane must bind the exact lens file")
        files["frozen-plane-candidate.json"] = args.frozen_candidate.read_bytes()
        files["scripts/diagnostics/raytheon_table/frozen_check.py"] = (
            root / "scripts/diagnostics/raytheon_table/frozen_check.py"
        ).read_bytes()
    if args.motion_check:
        files["scripts/diagnostics/raytheon_table/motion_check.py"] = (
            root / "scripts/diagnostics/raytheon_table/motion_check.py"
        ).read_bytes()
    manifest = {
        "baseline_commit": "9a144955377b289ce47de71769d1305dd8585de4",
        "packet_version": "rer-table-board-20260905-v2"
        if args.board_check
        else "rer-table-check-20260905-v1",
        "kind": "manual_table_geometry_diagnostic_with_existing_capture_and_math",
        "acceptance": "NOT_EVALUATED",
        "production_ready": False,
        "active_calibration_written": False,
        "limitations": [
            "operator-confirmed identity, not trusted PnP capture-handle binding",
            "manual rail clicks, not a current-frame automatic corner detector",
            "control readbacks, not native control locking",
            "owned subprocess timeout, not native camera-read cancellation",
            "table nose-intersection origin is diagnostic only, "
            "not the production pocket-center contract",
        ],
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in files.items()
        },
    }
    if args.frozen_candidate:
        manifest["packet_version"] = "rer-plane-recheck-20260905-v1"
        manifest["kind"] = "fresh_frozen_plane_local_metric_recheck"
        manifest["refit_performed"] = False
        manifest["limitations"].append(
            "fresh frames evaluate frozen local distances only, "
            "not absolute position or ball-center plane"
        )
    if args.motion_check:
        manifest["packet_version"] = "rer-motion-check-20260905-v1"
        manifest["kind"] = "motion_media_for_offline_analysis"
        manifest["tracking_evaluated"] = False
    files["packet-manifest.json"] = json.dumps(manifest, indent=2).encode()
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
