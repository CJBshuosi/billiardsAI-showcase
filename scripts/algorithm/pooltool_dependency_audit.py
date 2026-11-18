"""Generate a metadata-only pooltool dependency draft without payload downloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from billiards_ai.lab.dependency_audit.transport import (
    PyPIMetadataClient,
    UrllibMetadataTransport,
)
from billiards_ai.lab.pooltool_audit.runner import (
    PooltoolAuditError,
    build_metadata_draft,
    load_request,
    publish_draft,
)
from billiards_ai.lab.pooltool_audit.transport import PandaArchiveMetadataClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        request = load_request(args.request)
        pypi_client = PyPIMetadataClient(
            UrllibMetadataTransport(),
            request.single_response_bytes_max,
            request.total_metadata_bytes_max,
        )
        draft = build_metadata_draft(
            request,
            pypi_client=pypi_client,
            panda_client=PandaArchiveMetadataClient(),
        )
        destination = publish_draft(draft, args.output_root)
    except (PooltoolAuditError, ValueError, RuntimeError) as error:
        print(f"pooltool dependency audit failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "HASH_REQUIRED",
                "output": str(destination),
                "payload_count": draft.payload_count,
                "payload_bytes": draft.payload_bytes,
                "missing_hash": draft.panda3d.filename,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
