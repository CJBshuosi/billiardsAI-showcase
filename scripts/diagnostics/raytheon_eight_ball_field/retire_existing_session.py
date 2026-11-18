#!/usr/bin/env python3
"""Retire the one installed development field session before a USB sidecar run."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from billiards_ai.edge.route_assistance.sync import (
    UrllibSyncHttpTransport,
    WindowsCredentialManagerProvider,
)
from billiards_ai.lab.field_assistance.cloud_gateway import (
    ExistingDevelopmentControlGateway,
    SignedFieldCloudGateway,
)
from billiards_ai.lab.field_assistance.session_retirement import retire_existing_session


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", required=True, type=Path)
    parser.add_argument("--sync-base-url", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--credential-target", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        signed = SignedFieldCloudGateway(
            credentials=WindowsCredentialManagerProvider(
                device_id=arguments.device_id,
                target_name=arguments.credential_target,
            ),
            transport=UrllibSyncHttpTransport(
                arguments.sync_base_url,
                allow_redirects=False,
            ),
        )
        result = retire_existing_session(
            state_db=arguments.state_db,
            gateway=ExistingDevelopmentControlGateway(
                delegate=signed,
                expected_table_id=arguments.device_id,
            ),
            expected_table_id=arguments.device_id,
        )
        arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
        arguments.receipt.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except (OSError, sqlite3.DatabaseError, TypeError, ValueError) as exc:
        print(f"installed development session retirement failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
