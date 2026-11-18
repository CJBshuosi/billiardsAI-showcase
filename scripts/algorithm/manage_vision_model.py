"""Inspect, activate, or roll back the verified local vision model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from billiards_ai.lab.vision.registry import (
    VisionRegistryError,
    activate_candidate,
    load_active_model,
    rollback_active,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--registry", type=Path, required=True)
    inspect.add_argument("--active", type=Path, required=True)
    for name in ("activate", "rollback"):
        command = subparsers.add_parser(name)
        command.add_argument("--registry", type=Path, required=True)
        command.add_argument("--active", type=Path, required=True)
        command.add_argument("--history-root", type=Path, required=True)
        command.add_argument("--activated-at", required=True)
        if name == "activate":
            command.add_argument("--verdict", type=Path, required=True)
        else:
            command.add_argument("--expected-activation-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "inspect":
            active = load_active_model(arguments.registry, arguments.active)
        elif arguments.command == "activate":
            active = activate_candidate(
                registry_path=arguments.registry,
                active_path=arguments.active,
                history_root=arguments.history_root,
                verdict_path=arguments.verdict,
                activated_at=arguments.activated_at,
            )
        else:
            active = rollback_active(
                registry_path=arguments.registry,
                active_path=arguments.active,
                history_root=arguments.history_root,
                expected_activation_id=arguments.expected_activation_id,
                activated_at=arguments.activated_at,
            )
    except VisionRegistryError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            active.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
