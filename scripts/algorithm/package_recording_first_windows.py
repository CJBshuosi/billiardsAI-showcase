"""Build/review/freeze an explicit R4 packet; never upload or execute Windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from scripts.algorithm.recording_packet import (
    CONTRACT,
    pin_file,
    plain,
    safe_member,
    verify_packet,
    write_json,
)
from scripts.algorithm.recording_windows_entry import BROWSER_WINDOW_SECONDS

ROOT = Path(__file__).resolve().parents[2]
PASSED_COMMIT = "d9d2c235773e983603efc0aea2f8dcd0d779261d"
PASSED_REPORT_SHA = "be8359836560c96ff52877720f1bf5d991bf8c2bc308bf4449ece2a503ad9bd5"
LOCAL_EVIDENCE = "docs/operations/durable-recording-d4-flush-local-evidence-2026-11-10.json"
OLD_ZIP_SHA = "f1eab79a3c7a2d73cc39a2d0e7589129800de16a18034871d4a34b48ff4cb2ee"
OLD_MANIFEST_SHA = "3933380a91462f4c5cb8d05642a193dc2274040caca99e2fdcf1dcc4b75f4c19"
DEPENDENCIES = "C:/Users/developer/Documents/billiardsAI-raytheon3/.venv-rfdetr/Lib/site-packages"
SOURCE_ROOTS = ("src", "scripts/algorithm", "configs/algorithm/shot-local.yaml")
HELPERS = (
    "scripts/algorithm/recording_packet.py",
    "scripts/algorithm/recording_windows_entry.py",
    "scripts/algorithm/package_recording_first_windows.py",
    "scripts/algorithm/run_recording_first_fullgame.py",
)
PREPARE = "scripts/algorithm/windows/prepare-recording-r4.ps1"


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _source(commit):
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("full frozen source commit required")
    if _git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip() != commit:
        raise ValueError("full frozen source commit required")
    changed = set(
        _git("diff", "--name-only", PASSED_COMMIT, commit, "--", *SOURCE_ROOTS)
        .decode()
        .splitlines()
    )
    if changed - {*HELPERS, PREPARE}:
        raise ValueError("product source changed after passing whole-game evidence")
    if subprocess.run(["git", "diff", "--quiet", commit, "--", *SOURCE_ROOTS], cwd=ROOT).returncode:
        raise ValueError("source differs from frozen commit")
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *SOURCE_ROOTS)
    if untracked.strip():
        raise ValueError("uncommitted source files are not admitted")
    values = {}
    blobs = {}
    for row in _git("ls-tree", "-rz", commit, "--", *SOURCE_ROOTS).split(b"\0"):
        if not row:
            continue
        metadata, raw_name = row.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        name = safe_member(raw_name.decode())
        if Path(name).suffix not in {".py", ".yaml", ".mako"}:
            continue
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError("source must be regular Git blobs")
        data = _git("show", f"{commit}:{name}")
        if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != oid:
            raise ValueError("source Git object mismatch")
        if plain(ROOT / name).read_bytes() != data:
            raise ValueError("working source differs from Git object")
        values[name], blobs[name] = data, oid
    if not set(HELPERS) <= values.keys():
        raise ValueError("current packet helpers must be committed")
    proof = json.loads(_git("show", f"{commit}:{LOCAL_EVIDENCE}"))
    if (
        proof.get("contract") != "durable-recording-d4-local-evidence/v1"
        or proof.get("source_commit") != PASSED_COMMIT
        or proof.get("browser", {}).get("source_commit") != PASSED_COMMIT
        or proof.get("source_report_sha256") != PASSED_REPORT_SHA
        or proof.get("result", {}).get("product_pass") is not True
        or proof.get("browser", {}).get("local_browser_pass") is not True
    ):
        raise ValueError("passing local evidence identity changed")
    for name, digest in proof["tested_source_files"].items():
        if name.startswith("src/") and hashlib.sha256(values[name]).hexdigest() != digest:
            raise ValueError("product source diverges from passing whole-game evidence")
    return values, blobs, proof


def build_stage(stage_root, input_root, runtime_zip, commit):
    from scripts.algorithm.run_recording_first_fullgame import SOURCE_SHA256, _validate_inputs

    stage = plain(stage_root)
    if stage.exists() or not stage.parent.is_dir():
        raise ValueError("fresh stage root required")
    inputs, preflight = _validate_inputs(input_root)
    source, blobs, proof = _source(commit)
    runtime_zip = plain(runtime_zip)
    if pin_file(runtime_zip)["sha256"] != OLD_ZIP_SHA:
        raise ValueError("reviewed predecessor ZIP changed")
    stage.mkdir()
    packet = stage / "packet"
    packet.mkdir()
    folded = set()

    def target(name):
        safe_member(name)
        if name.casefold() in folded:
            raise ValueError("packet path alias")
        folded.add(name.casefold())
        path = packet / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put(name, data):
        with target(name).open("xb") as stream:
            stream.write(data)

    for name, data in source.items():
        put("source/" + name, data)
    source_pins = {
        name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        for name, data in source.items()
    }
    put(
        "source/SOURCE-LOCK.json",
        (
            json.dumps(
                dict(
                    contract="recording-first-source-lock/v1",
                    commit=commit,
                    files=source_pins,
                ),
                indent=2,
            )
            + "\n"
        ).encode(),
    )
    for name in preflight["selected_files"]:
        destination = target("input/" + name)
        with (inputs / name).open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1024**2)
    put(
        "input/input-preflight.json",
        (json.dumps(dict(preflight, root="packet/input"), indent=2) + "\n").encode(),
    )
    put("local-software-evidence.json", (json.dumps(proof, indent=2) + "\n").encode())
    prepare = _git("show", f"{commit}:{PREPARE}")
    put("prepare.ps1", prepare)
    put("entry.py", source["scripts/algorithm/recording_windows_entry.py"])
    reused = {}
    with zipfile.ZipFile(runtime_zip) as archive:
        raw_manifest = archive.read("packet-manifest.json")
        if hashlib.sha256(raw_manifest).hexdigest() != OLD_MANIFEST_SHA:
            raise ValueError("predecessor manifest changed")
        old = json.loads(raw_manifest)
        names = [
            n
            for n in archive.namelist()
            if n.startswith("runtime/") and n != "runtime/python312._pth"
        ]
        names += [
            "ownership.py",
            "runtime-provenance.json",
            "runtime-origin/release.html",
            "runtime-origin/runtime.sigstore",
        ]
        for name in names:
            data = archive.read(name)
            if (
                dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                != old["members"][name]
            ):
                raise ValueError("predecessor runtime/helper content changed")
            put(name, data)
            reused[name] = old["members"][name]
    put(
        "runtime/python312._pth",
        ("python312.zip\n.\n../source/src\n../source\n..\n" + DEPENDENCIES + "\n").encode("ascii"),
    )  # No import-site, user-site or automatic .pth processing.
    inventory = {
        p.relative_to(packet).as_posix(): pin_file(p)
        for p in sorted(packet.rglob("*"))
        if p.is_file()
    }
    manifest = dict(
        contract=CONTRACT,
        source_commit=commit,
        source_git_blobs=blobs,
        input_sha256=SOURCE_SHA256,
        source_lock_sha256=inventory["source/SOURCE-LOCK.json"]["sha256"],
        members=inventory,
        predecessor_zip_sha256=OLD_ZIP_SHA,
        reused_members=reused,
        dependency_site_packages=DEPENDENCIES,
        session_policy="recording-v1",
        models_requested=False,
        cloud_requested=False,
        work_seconds=900,
        cleanup_seconds=10,
        browser_window_seconds=BROWSER_WINDOW_SECONDS,
        local_software_report_sha256=proof["source_report_sha256"],
        local_software_episode_pass=proof["episode_pass"] is True,
        local_software_cleanup_confirmed=proof["result"]["cleanup_confirmed"] is True,
        local_product_modules_unchanged=True,
        runtime_qualified=False,
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
            stage=str(stage),
            manifest_sha256=digest,
            members=len(inventory),
            bytes=sum(p["bytes"] for p in inventory.values()),
            native_execution_authorized=False,
            windows_run=False,
        ),
    )
    return digest


def freeze_stage(stage_root, frozen_root, manifest_sha):
    stage, frozen = plain(stage_root), plain(frozen_root)
    if frozen.exists() or not frozen.parent.is_dir():
        raise ValueError("fresh frozen directory required")
    manifest = verify_packet(stage / "packet", manifest_sha)
    frozen.mkdir()
    archive_path = frozen / "recording-first-r4-windows.zip"
    names = sorted([*manifest["members"], "packet-manifest.json"])
    with zipfile.ZipFile(
        archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in names:
            archive.write(stage / "packet" / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != names:
            raise ValueError("frozen ZIP inventory mismatch")
        prepare_data = None
        for name in names:
            digest, size = hashlib.sha256(), 0
            prepare_parts = [] if name == "prepare.ps1" else None
            with archive.open(name) as stream:  # Includes ZIP CRC validation.
                while block := stream.read(1024**2):
                    digest.update(block)
                    size += len(block)
                    if prepare_parts is not None:
                        if size > 1024**2:
                            raise ValueError("frozen preparation helper too large")
                        prepare_parts.append(block)
            expected = (
                dict(bytes=size, sha256=manifest_sha)
                if name == "packet-manifest.json"
                else manifest["members"][name]
            )
            if dict(bytes=size, sha256=digest.hexdigest()) != expected:
                raise ValueError("frozen ZIP content mismatch")
            if name == "prepare.ps1":
                prepare_data = b"".join(prepare_parts)
    if prepare_data is None:
        raise ValueError("frozen preparation helper missing")
    with (frozen / "prepare.ps1").open("xb") as stream:
        stream.write(prepare_data)
    write_json(
        frozen / "FROZEN.json",
        dict(
            contract="recording-first-frozen-packet/v1",
            source_commit=manifest["source_commit"],
            archive=pin_file(archive_path),
            manifest_sha256=manifest_sha,
            prepare=pin_file(frozen / "prepare.ps1"),
            native_execution_authorized=False,
            windows_run=False,
            product_pass=False,
            model_pass=False,
            field_pass=False,
        ),
    )
    return archive_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--stage-root", type=Path, required=True)
    build.add_argument("--input-root", type=Path, required=True)
    build.add_argument("--runtime-zip", type=Path, required=True)
    build.add_argument("--source-commit", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--stage-root", type=Path, required=True)
    freeze.add_argument("--frozen-root", type=Path, required=True)
    freeze.add_argument("--manifest-sha", required=True)
    args = parser.parse_args(argv)
    result = (
        build_stage(args.stage_root, args.input_root, args.runtime_zip, args.source_commit)
        if args.command == "build"
        else freeze_stage(args.stage_root, args.frozen_root, args.manifest_sha)
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
