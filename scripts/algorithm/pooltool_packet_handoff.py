"""Download an explicitly approved pooltool wheel packet without installing it."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_handoff() -> ModuleType:
    source = Path(__file__).resolve().parents[2] / "src/billiards_ai/lab/pooltool_audit/handoff.py"
    spec = importlib.util.spec_from_file_location("_billiards_pooltool_handoff", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("pooltool handoff module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        handoff = _load_handoff()
        packet = handoff.load_pooltool_packet(arguments.packet, arguments.packet_sha256)
        wheels = handoff.download_pooltool_wheels(packet, arguments.wheelhouse)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"pooltool packet handoff failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "packet_sha256": arguments.packet_sha256,
                "wheelhouse": str(arguments.wheelhouse.resolve()),
                "payload_count": len(wheels),
                "payload_bytes": sum(item.bytes for item in wheels),
                "installs": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
