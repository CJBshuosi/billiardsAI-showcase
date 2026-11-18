"""Repeatable Phase 0 operator CLI for media deletion requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

from billiards_ai.shared.models import DeletionScope

from .media_store import ControlledFilesystemMediaStore
from .retention_repository import JsonRetentionRepository, RetentionError
from .retention_service import MediaRetentionService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="billiards-media-deletion")
    parser.add_argument("--store", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit")
    submit.add_argument("--user", required=True)
    submit.add_argument("--idempotency-key", required=True)
    scope = submit.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all-user-media", action="store_true")
    scope.add_argument("--session-id", action="append", type=UUID)
    scope.add_argument("--asset-id", action="append", type=UUID)

    status = sub.add_parser("status")
    status.add_argument("--user", required=True)
    status.add_argument("--request-id", required=True, type=UUID)

    due = sub.add_parser("list-due")
    due.add_argument("--owner", required=True, help="required operator-side owner filter")

    for command in ("execute", "retry"):
        action = sub.add_parser(command)
        action.add_argument("--request-id", required=True, type=UUID)
        action.add_argument("--executor", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--user", required=True)
    verify.add_argument("--request-id", required=True, type=UUID)
    reclaim = sub.add_parser("reclaim")
    reclaim.add_argument("--request-id", required=True, type=UUID)
    reclaim.add_argument("--executor", required=True)
    reclaim.add_argument("--reason", required=True)
    return parser


def _write(payload: dict | list) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = JsonRetentionRepository(args.store)
    try:
        if args.command in {"execute", "retry"}:
            service = MediaRetentionService(
                repository, media_store=ControlledFilesystemMediaStore.from_environment()
            )
        else:
            service = MediaRetentionService(repository)
        if args.command == "submit":
            scope = DeletionScope(
                all_user_media=args.all_user_media,
                session_ids=tuple(args.session_id or ()),
                asset_ids=tuple(args.asset_id or ()),
            )
            request = service.submit_deletion(
                current_user_id=args.user,
                scope=scope,
                idempotency_key=args.idempotency_key,
            )
            _write(
                {
                    "request_id": str(request.request_id),
                    "status": request.status.value,
                    "scope_summary": request.scope.summary,
                }
            )
        elif args.command == "status":
            public = service.get_public_status(
                current_user_id=args.user, request_id=args.request_id
            )
            _write(public.model_dump(mode="json"))
        elif args.command == "list-due":
            due = tuple(
                item for item in service.list_due_assets() if item.owner_user_id == args.owner
            )
            _write(
                [
                    {
                        "asset_id": str(item.asset_id),
                        "session_id": str(item.session_id),
                        "owner_user_id": item.owner_user_id,
                        "retention_expires_at": item.retention_expires_at.isoformat(),
                    }
                    for item in due
                ]
            )
        elif args.command in {"execute", "retry"}:
            request = service.execute_deletion(
                request_id=args.request_id, executor_id=args.executor
            )
            _write(
                {
                    "request_id": str(request.request_id),
                    "status": request.status.value,
                    "attempt_count": len(request.attempts),
                }
            )
            return 0 if request.status.value == "COMPLETED" else 3
        elif args.command == "verify":
            public = service.get_public_status(
                current_user_id=args.user, request_id=args.request_id
            )
            _write(public.model_dump(mode="json"))
        else:
            request = service.reclaim_in_progress(
                request_id=args.request_id,
                executor_id=args.executor,
                reason=args.reason,
            )
            _write(
                {
                    "request_id": str(request.request_id),
                    "status": request.status.value,
                    "attempt_count": len(request.attempts),
                }
            )
        return 0
    except (RetentionError, ValueError) as exc:
        print(
            json.dumps(
                {"error": type(exc).__name__, "message": "operation could not be completed"}
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
