"""Hash the one approved Panda3D wheel and finalize the pooltool packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from billiards_ai.lab.pooltool_audit.hash_probe import (
    PandaHashProbeError,
    build_pooltool_packet,
    download_panda_hash,
    publish_pooltool_packet,
)
from billiards_ai.lab.pooltool_audit.models import PooltoolMetadataDraft
from billiards_ai.lab.pooltool_audit.runner import canonical_draft_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--draft-sha256", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        draft = PooltoolMetadataDraft.model_validate_json(arguments.draft.read_bytes())
        draft_sha256 = hashlib.sha256(canonical_draft_bytes(draft)).hexdigest()
        if draft_sha256 != arguments.draft_sha256:
            raise PandaHashProbeError("pooltool metadata draft SHA-256 does not match")
        evidence = download_panda_hash(draft, arguments.cache_root)
        packet = build_pooltool_packet(draft, evidence)
        destination = publish_pooltool_packet(packet, arguments.output_root)
    except (
        OSError,
        UnicodeError,
        ValidationError,
        ValueError,
        PandaHashProbeError,
    ) as error:
        print(f"pooltool hash probe failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "draft_sha256": draft_sha256,
                "panda3d_sha256": evidence.sha256,
                "packet_sha256": packet.packet_sha256,
                "output": str(destination),
                "wheel": str(evidence.wheel_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
