#!/usr/bin/env python3
"""Run the offline cloud-training preflight; this script has no cloud client."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from billiards_ai.lab.cloud_preflight import create_job_evidence, run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args()

    result = run_preflight(args.spec)
    payload = asdict(result)
    if result.passed and args.evidence_root is not None:
        payload["evidence_dir"] = str(create_job_evidence(args.spec, args.evidence_root))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
