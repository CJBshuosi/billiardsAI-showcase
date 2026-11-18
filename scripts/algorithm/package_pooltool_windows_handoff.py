"""Package current source and verified pooltool wheels for offline Raytheon use."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from billiards_ai.lab.pooltool_audit.handoff import PooltoolHandoffError
from billiards_ai.lab.pooltool_audit.handoff_archive import build_handoff_archive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        destination = build_handoff_archive(
            project_root=arguments.project_root,
            source_commit=arguments.source_commit,
            packet_path=arguments.packet,
            packet_sha256=arguments.packet_sha256,
            wheelhouse=arguments.wheelhouse,
            output_root=arguments.output_root,
        )
    except (OSError, ValueError, PooltoolHandoffError) as error:
        print(f"pooltool handoff packaging failed: {error}", file=sys.stderr)
        return 2
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
