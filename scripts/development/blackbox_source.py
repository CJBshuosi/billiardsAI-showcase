"""Export only committed executable sources; never traverse private working files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOTS = ("src", "scripts/algorithm", "scripts/development")
CONFIG = "configs/algorithm/shot-local.yaml"
SOURCE_ASSETS = {
    "scripts/development/assets/s1-felt.png": {
        "bytes": 2_775_905,
        "sha256": "5b87d7f16d8a83c0f5bdc9c670577af02adadea47d9c1313a981472ddefac0f3",
    }
}
CONTRACT = "recording-first-source-lock/v1"
ATTRIBUTES = b"* -text\n"


def git(repo, *args, executable="git"):
    return subprocess.check_output(
        [str(executable), "-C", str(repo), *args], stderr=subprocess.PIPE, timeout=20
    )


def safe_name(name):
    parts = name.split("/")
    if (
        not name
        or PurePosixPath(name).as_posix() != name
        or any(ord(c) < 32 or ord(c) > 126 or c in '\\:<>"|?*' for c in name)
        or any(
            p in {"", ".", "..", ".git", ".project-archive", ".superpowers"}
            or p[-1:] in {".", " "}
            or p.split(".")[0].upper()
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }
            for p in parts
        )
    ):
        raise ValueError("unsafe source name")
    return name


def allowed(name):
    return name in {CONFIG, ".gitattributes", *SOURCE_ASSETS} or (
        name.startswith(tuple(root + "/" for root in ROOTS))
        and Path(name).suffix in {".py", ".ps1", ".mako", ".yaml"}
    )


def plain(path):
    path = Path(path).absolute()
    if any(
        p.name.casefold() == ".project-archive"
        or p.name == ".."
        or p.is_symlink()
        or p.is_junction()
        for p in reversed((path, *path.parents))
    ):
        raise ValueError("unsafe source path")
    return path


def pin(path):
    path = plain(path)
    if not path.is_file():
        raise ValueError("regular file required")
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError("file changed while hashing")
    return {"bytes": after.st_size, "sha256": digest}


def export(repo, commit, destination):
    repo, destination = plain(repo), plain(destination)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full commit required")
    if git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip() != commit:
        raise ValueError("commit mismatch")
    if destination.exists():
        raise ValueError("fresh destination required")
    entries, folded, total = [], set(), len(ATTRIBUTES)
    for row in git(repo, "ls-tree", "-rz", commit, "--", *ROOTS, CONFIG).split(b"\0"):
        if not row:
            continue
        meta, raw = row.split(b"\t", 1)
        name = safe_name(raw.decode())
        if not allowed(name):
            continue
        mode, kind, oid = meta.decode().split()
        if mode not in {"100644", "100755"} or kind != "blob" or name.casefold() in folded:
            raise ValueError("source alias or nonregular entry")
        folded.add(name.casefold())
        size = int(git(repo, "cat-file", "-s", oid))
        total += size
        if size > 8 * 1024**2 or total > 32 * 1024**2:
            raise ValueError("source size budget exceeded")
        data = git(repo, "cat-file", "blob", oid)
        if len(data) != size:
            raise ValueError("source changed while exporting")
        if name in SOURCE_ASSETS and SOURCE_ASSETS[name] != {
            "bytes": size,
            "sha256": hashlib.sha256(data).hexdigest(),
        }:
            raise ValueError("approved source asset changed")
        entries.append((name, data))
    if not entries:
        raise ValueError("source inventory budget exceeded")
    destination.mkdir(parents=False)
    files = {}
    for name, data in entries:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    # Transport metadata is generated, not copied from private development history.
    (destination / ".gitattributes").write_bytes(ATTRIBUTES)
    files[".gitattributes"] = {
        "bytes": len(ATTRIBUTES),
        "sha256": hashlib.sha256(ATTRIBUTES).hexdigest(),
    }
    lock = {"contract": CONTRACT, "commit": commit, "files": files}
    (destination / "SOURCE-LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def verify(root, *, checkout=False):
    root = plain(root)
    path = plain(root / "SOURCE-LOCK.json")
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("source lock budget exceeded")
    lock = json.loads(path.read_bytes())
    if (
        lock.get("contract") != CONTRACT
        or not re.fullmatch(r"[0-9a-f]{40}", str(lock.get("commit")))
        or not isinstance(lock.get("files"), dict)
        or not lock["files"]
    ):
        raise ValueError("invalid source lock")
    expected = set(lock["files"])
    attributes = plain(root / ".gitattributes")
    if (
        ".gitattributes" not in expected
        or attributes.stat().st_size != len(ATTRIBUTES)
        or attributes.read_bytes() != ATTRIBUTES
    ):
        raise ValueError("source byte policy missing")
    folded, total = set(), 0
    for name, identity in lock["files"].items():
        safe_name(name)
        if (
            not isinstance(identity, dict)
            or set(identity) != {"bytes", "sha256"}
            or type(identity["bytes"]) is not int
            or not 0 <= identity["bytes"] <= 8 * 1024**2
            or not re.fullmatch(r"[0-9a-f]{64}", str(identity["sha256"]))
        ):
            raise ValueError("invalid source identity")
        total += identity["bytes"]
        if total > 32 * 1024**2:
            raise ValueError("source inventory budget exceeded")
        member = plain(root / name)
        if (
            not allowed(name)
            or (name in SOURCE_ASSETS and identity != SOURCE_ASSETS[name])
            or name.casefold() in folded
            or member.stat().st_size != identity["bytes"]
            or pin(member) != identity
        ):
            raise ValueError("source identity mismatch")
        folded.add(name.casefold())
    actual = set()

    # A sync clone may have Git metadata. Never recurse into it or follow a link.
    def walk(directory):
        for item in directory.iterdir():
            if checkout and item.parent == root and item.name == ".git":
                continue
            plain(item)
            if item.is_dir():
                walk(item)
            elif item.is_file():
                actual.add(item.relative_to(root).as_posix())
            else:
                raise ValueError("nonregular source member")

    walk(root)
    if actual != expected | {"SOURCE-LOCK.json"}:
        raise ValueError("source inventory mismatch")
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = export(args.repo, args.commit, args.output)
    verify(args.output)
    print(
        json.dumps(
            {
                "source_commit": lock["commit"],
                "files": len(lock["files"]),
                "lock": pin(args.output / "SOURCE-LOCK.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
