"""Grant one Windows identity read access to an immutable field artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, check=False, text=True)


def grant_artifact_read_execute(
    artifact: Path,
    *,
    identity: str,
    runner: CommandRunner | None = None,
) -> None:
    """Grant the scheduled-task identity recursive RX access with icacls."""

    if artifact.is_symlink() or not artifact.is_dir():
        raise ValueError("field artifact is missing or unsafe")
    if (
        not identity
        or identity != identity.strip()
        or any(character in identity for character in ("\0", "\r", "\n", ":"))
    ):
        raise ValueError("Windows identity is invalid")
    resolved = artifact.resolve(strict=True)
    command = [
        "icacls.exe",
        str(resolved),
        "/grant:r",
        f"{identity}:(OI)(CI)RX",
        "/T",
        "/Q",
    ]
    result = (runner or _default_runner)(command)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown icacls failure").strip()
        raise RuntimeError(f"icacls failed for {identity}: {detail}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--identity", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        grant_artifact_read_execute(
            arguments.artifact,
            identity=arguments.identity,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"field artifact ACL grant failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact": str(arguments.artifact.resolve()),
                "identity": arguments.identity,
                "status": "PASS",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
