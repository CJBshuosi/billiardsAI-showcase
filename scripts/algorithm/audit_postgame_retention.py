#!/usr/bin/env python3
"""Print a bounded, read-only postgame media policy simulation as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Direct execution works from any cwd without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from billiards_ai.edge.route_assistance.retention_inventory import (  # noqa: E402
    build_retention_inventory,
)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's normal errors can echo private paths or arbitrary argument text.
        raise ValueError("invalid_arguments")


def main(argv=None) -> int:
    parser = _Parser(description=__doc__, allow_abbrev=False)
    for name in ("state-db", "jobs-db", "artifact-root", "jobs-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    try:
        args = parser.parse_args(argv)
        report = build_retention_inventory(
            state_db=args.state_db,
            jobs_db=args.jobs_db,
            artifact_root=args.artifact_root,
            jobs_root=args.jobs_root,
            limit=args.limit,
        )
    except (OSError, ValueError, TypeError) as exc:
        allowed = {
            "invalid_arguments",
            "invalid_limit",
            "unsafe_roots",
            "database_unavailable",
            "unsupported_state_schema",
            "database_query_failed",
            "unsupported_platform",
        }
        code = str(exc) if str(exc) in allowed else "inventory_failed"
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "deletion_authorized": False,
                    "coverage_complete": False,
                    "error": code,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
