"""Operator CLI for the Phase 0 REAL_VIDEO_GATE."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from billiards_ai.cloud.privacy import (
    GateError,
    JsonGateRepository,
    RealVideoGateService,
    authorize_real_video_use,
)
from billiards_ai.shared.models import (
    AssetContext,
    EvidenceReference,
    GateScope,
    MaterialUseDecision,
    MaterialUsePurpose,
)

EXIT_OK = 0
EXIT_NOT_ALLOWED = 3
EXIT_REMEDIATION_REQUIRED = 4
EXIT_OPERATIONAL_ERROR = 5


def _datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an ISO 8601 datetime") from exc


def _uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a UUID") from exc


def _json_file(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="real-video-gate")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create a BLOCKED gate application")
    create.add_argument("--store", type=Path, required=True)
    create.add_argument("--application", type=Path, required=True)

    show = commands.add_parser("show", help="show the current public gate status")
    show.add_argument("--store", type=Path, required=True)
    show.add_argument("--gate-id", type=_uuid, required=True)

    approve = commands.add_parser("approve", help="append an APPROVED revision")
    approve.add_argument("--store", type=Path, required=True)
    approve.add_argument("--gate-id", type=_uuid, required=True)
    approve.add_argument("--approval-evidence-sha256", required=True)

    migrate = commands.add_parser("migrate", help="anchor a verified legacy gate store")
    migrate.add_argument("--store", type=Path, required=True)
    migrate.add_argument("--expected-store-sha256", required=True)
    migrate.add_argument("--dry-run", action="store_true")

    evaluate = commands.add_parser("evaluate", help="evaluate one asset and purpose")
    evaluate.add_argument("--store", type=Path, required=True)
    evaluate.add_argument("--gate-revision-id", type=_uuid, required=True)
    evaluate.add_argument("--asset-context", type=Path, required=True)
    evaluate.add_argument(
        "--purpose", type=MaterialUsePurpose, choices=list(MaterialUsePurpose), required=True
    )
    return parser


def _service(repository: JsonGateRepository) -> RealVideoGateService:
    return RealVideoGateService(repository)


def _create(args: argparse.Namespace, repository: JsonGateRepository) -> int:
    payload = _json_file(args.application)
    if not isinstance(payload, dict):
        raise ValueError("application must be a JSON object")
    scope = GateScope.model_validate(payload.get("scope"))
    evidence = tuple(EvidenceReference.model_validate(item) for item in payload.get("evidence", []))
    revision = _service(repository).create_gate(
        scope=scope,
        inspector_id=payload["inspector_id"],
        inspected_at=_datetime(payload["inspected_at"]),
        evidence=evidence,
    )
    print(revision.gate_id)
    return EXIT_OK


def _show(args: argparse.Namespace, repository: JsonGateRepository) -> int:
    revision = repository.get_current(args.gate_id)
    if revision.status.value == "APPROVED":
        print("门禁已批准")
    else:
        print("门禁未批准")
    return EXIT_OK


def _approve(args: argparse.Namespace, repository: JsonGateRepository) -> int:
    _service(repository).approve(
        gate_id=args.gate_id,
        approval_evidence_sha256=args.approval_evidence_sha256,
    )
    print("门禁已批准")
    return EXIT_OK


def _evaluate(args: argparse.Namespace, repository: JsonGateRepository) -> int:
    context = AssetContext.model_validate(_json_file(args.asset_context))
    authorization = authorize_real_video_use(
        repository,
        gate_revision_id=args.gate_revision_id,
        asset_context=context,
        requested_purpose=args.purpose,
    )
    if authorization.decision is MaterialUseDecision.ALLOW:
        print("可继续")
        return EXIT_OK
    if authorization.decision is MaterialUseDecision.REQUIRE_ANONYMIZATION_OR_CONSENT:
        print("需匿名化或补授权")
        return EXIT_REMEDIATION_REQUIRED
    print("不可处理")
    return EXIT_NOT_ALLOWED


def _migrate(args: argparse.Namespace, repository: JsonGateRepository) -> int:
    repository.migrate_legacy_store(
        expected_store_sha256=args.expected_store_sha256,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("迁移验证通过，未写入")
    else:
        print("迁移锚定完成")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = JsonGateRepository(args.store)
    try:
        if args.command == "create":
            return _create(args, repository)
        if args.command == "show":
            return _show(args, repository)
        if args.command == "approve":
            return _approve(args, repository)
        if args.command == "migrate":
            return _migrate(args, repository)
        return _evaluate(args, repository)
    except (
        GateError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        ValidationError,
    ):
        print("不可处理", file=sys.stderr)
        return EXIT_OPERATIONAL_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
