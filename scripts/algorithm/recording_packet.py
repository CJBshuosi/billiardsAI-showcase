"""Standard-library inventory checks shared by the R4 builder and owned launcher."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath

CONTRACT = "recording-first-windows-packet/v1"
MANIFEST_LIMIT = 2 * 1024**2
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_member(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or any(ord(c) < 32 or ord(c) > 126 for c in name)
        or any(c in name for c in '\\:\x00<>"|?*')
        or name.startswith("/")
        or name.endswith("/")
    ):
        raise ValueError("unsafe Windows packet member")
    parts = name.split("/")
    if (
        any(
            not part
            or part in {".", ".."}
            or part[-1] in ". "
            or part.casefold() in {".project-archive", ".git", ".env"}
            or part.split(".", 1)[0].upper() in _RESERVED
            for part in parts
        )
        or PurePosixPath(name).as_posix() != name
    ):
        raise ValueError("unsafe Windows packet member")
    return name


def plain(path: Path) -> Path:
    path = Path(path).absolute()
    if any(p.casefold() == ".project-archive" or p == ".." for p in path.parts) or any(
        p.is_symlink() or p.is_junction() for p in (path, *path.parents)
    ):
        raise ValueError("unsafe packet path")
    return path


def pin_file(path: Path) -> dict[str, object]:
    path = plain(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("unsafe packet file type")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("packet content changed while hashing")
    return dict(bytes=after.st_size, sha256=digest)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate packet JSON key")
        result[key] = value
    return result


def verify_packet(root: Path, manifest_sha256: str) -> dict:
    root = plain(root)
    if not re.fullmatch("[0-9a-f]{64}", manifest_sha256):
        raise ValueError("packet manifest digest invalid")
    manifest_path = plain(root / "packet-manifest.json")
    with manifest_path.open("rb") as stream:
        raw = stream.read(MANIFEST_LIMIT + 1)
    if len(raw) > MANIFEST_LIMIT or hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("packet manifest identity changed")
    manifest = json.loads(raw, object_pairs_hook=_unique_pairs)
    if manifest.get("contract") != CONTRACT or not isinstance(manifest.get("members"), dict):
        raise ValueError("packet manifest contract invalid")
    members = manifest["members"]
    folded = set()
    for name, pin in members.items():
        safe_member(name)
        if name == "packet-manifest.json" or name.casefold() in folded:
            raise ValueError("packet inventory alias")
        folded.add(name.casefold())
        if (
            not isinstance(pin, dict)
            or type(pin.get("bytes")) is not int
            or pin["bytes"] < 0
            or not isinstance(pin.get("sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", pin["sha256"])
        ):
            raise ValueError("packet content pin invalid")
    actual = set()
    directories = [root]
    while directories:
        for path in directories.pop().iterdir():
            plain(path)
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode):
                actual.add(path.relative_to(root).as_posix())
            elif stat.S_ISDIR(mode):
                directories.append(path)
            else:
                raise ValueError("unsafe packet entry type")
    if actual != set(members) | {"packet-manifest.json"}:
        raise ValueError("packet inventory changed")
    for name, pin in members.items():
        if pin_file(root / name) != pin:
            raise ValueError("packet content changed")
    return manifest


def write_json(path: Path, payload: dict, *, replace: bool = False) -> None:
    path = plain(path)
    target = path.with_name(path.name + ".pending") if replace else path
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)
