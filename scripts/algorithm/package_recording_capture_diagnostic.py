"""Create a new diagnostic packet from pinned R4 data; never execute its old entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

from scripts.algorithm.package_recording_first_windows import freeze_stage
from scripts.algorithm.recording_packet import (
    CONTRACT,
    pin_file,
    plain,
    safe_member,
    verify_packet,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_ZIP_SHA = "f9e51b34c130dda7cee95dfa7c26aa6b942297c88cfbb9ddb1d15d60d6d4c0e3"
BASE_MANIFEST_SHA = "2f7fae42a75294886eea49bc3d6673835cd41740852066f08472c1a7fb1707b7"
PROBE = "scripts/algorithm/recording_capture_diagnostic.py"
PREPARE = "scripts/algorithm/windows/prepare-recording-profile.ps1"
BUILDER = "scripts/algorithm/package_recording_capture_diagnostic.py"


def _source(commit):
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("full committed diagnostic source required")
    files, blobs = {}, {}
    for name in (PROBE, PREPARE, BUILDER):
        raw = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        if plain(ROOT / name).read_bytes() != raw:
            raise ValueError("diagnostic source differs from committed blob")
        files[name] = raw
        blobs[name] = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    return files, blobs


def build_stage(stage_root, base_zip, commit):
    stage, base_zip = plain(stage_root), plain(base_zip)
    if stage.exists() or not stage.parent.is_dir():
        raise ValueError("fresh diagnostic stage required")
    if pin_file(base_zip)["sha256"] != BASE_ZIP_SHA:
        raise ValueError("fixed R4 data ZIP changed")
    source, blobs = _source(commit)
    with zipfile.ZipFile(base_zip) as archive:
        raw = archive.read("packet-manifest.json")
        if hashlib.sha256(raw).hexdigest() != BASE_MANIFEST_SHA:
            raise ValueError("fixed R4 manifest changed")
        base = json.loads(raw)
        expected = sorted([*base["members"], "packet-manifest.json"])
        if sorted(archive.namelist()) != expected:
            raise ValueError("base ZIP inventory mismatch")
        stage.mkdir()
        packet = stage / "packet"
        packet.mkdir()
        for name, pin in base["members"].items():
            safe_member(name)
            data = archive.read(name)
            if dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) != pin:
                raise ValueError("base member identity mismatch")
            if name in {"entry.py", "prepare.ps1"}:
                continue  # Replace executable roots; never invoke the old entry.
            destination = packet / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    for name, data in {
        "entry.py": source[PROBE],
        "prepare.ps1": source[PREPARE],
        "diagnostic-builder.py": source[BUILDER],
    }.items():
        (packet / name).write_bytes(data)
    write_json(
        packet / "diagnostic-source.json",
        dict(
            contract="recording-profile-source/v1",
            diagnostic_source_commit=commit,
            diagnostic_git_blobs=blobs,
            base_zip_sha256=BASE_ZIP_SHA,
            base_manifest_sha256=BASE_MANIFEST_SHA,
            base_source_commit=base["source_commit"],
        ),
    )
    members = {
        p.relative_to(packet).as_posix(): pin_file(p)
        for p in sorted(packet.rglob("*"))
        if p.is_file()
    }
    manifest = dict(
        contract=CONTRACT,
        purpose="capture-throughput-diagnostic/v1",
        source_commit=base["source_commit"],
        diagnostic_source_commit=commit,
        diagnostic_git_blobs=blobs,
        source_lock_sha256=base["source_lock_sha256"],
        input_sha256=base["input_sha256"],
        members=members,
        base_zip_sha256=BASE_ZIP_SHA,
        base_manifest_sha256=BASE_MANIFEST_SHA,
        dependency_site_packages=base["dependency_site_packages"],
        session_policy="recording-v1",
        capture_seconds=30,
        work_seconds=90,
        cleanup_seconds=10,
        browser_window_seconds=0,
        models_requested=False,
        cloud_requested=False,
        native_execution_authorized=False,
        windows_run=False,
        product_pass=False,
        model_pass=False,
        field_pass=False,
    )
    write_json(packet / "packet-manifest.json", manifest)
    digest = pin_file(packet / "packet-manifest.json")["sha256"]
    verify_packet(packet, digest)
    write_json(
        stage / "BUILD.json",
        dict(
            manifest_sha256=digest,
            members=len(members),
            native_execution_authorized=False,
            windows_run=False,
        ),
    )
    return digest


def freeze(stage_root, frozen_root, digest):
    archive = freeze_stage(stage_root, frozen_root, digest)
    target = archive.with_name("recording-capture-diagnostic.zip")
    archive.rename(target)
    manifest = verify_packet(plain(stage_root) / "packet", digest)
    receipt = Path(frozen_root) / "FROZEN.json"
    value = json.loads(receipt.read_bytes())
    value.update(
        purpose=manifest["purpose"],
        diagnostic_source_commit=manifest["diagnostic_source_commit"],
        capture_seconds=30,
        work_seconds=90,
        cleanup_seconds=10,
        browser_window_seconds=0,
    )
    write_json(receipt, value, replace=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--stage-root", type=Path, required=True)
    build.add_argument("--base-zip", type=Path, required=True)
    build.add_argument("--source-commit", required=True)
    seal = commands.add_parser("freeze")
    seal.add_argument("--stage-root", type=Path, required=True)
    seal.add_argument("--frozen-root", type=Path, required=True)
    seal.add_argument("--manifest-sha", required=True)
    args = parser.parse_args(argv)
    print(
        build_stage(args.stage_root, args.base_zip, args.source_commit)
        if args.command == "build"
        else freeze(args.stage_root, args.frozen_root, args.manifest_sha)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
