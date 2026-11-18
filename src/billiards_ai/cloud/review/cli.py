"""Fail-closed local CLI adapter for controlled human review."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, TextIO
from urllib.parse import urlparse
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from billiards_ai.cloud.privacy.access_identity import FileReviewerIdentityProvider
from billiards_ai.cloud.table_binding.repository import JsonTableBindingRepository
from billiards_ai.shared.models.review import SessionRejectionReason

from .application_models import (
    ClaimReviewCommand,
    DecideReviewCommand,
    FinalizeReviewCommand,
    RejectReviewCommand,
    ShotDecisionIntent,
    TaskRef,
)
from .metric_authority import MetricShotReviewInput

_SAFE_MESSAGE = "operation could not be completed"
_MAX_INTENT_BYTES = 1_048_576


class ReviewApplicationServiceProtocol(Protocol):
    async def list_tasks(self, *, cursor: str | None = None, limit: int = 50) -> object: ...

    async def claim(self, command: ClaimReviewCommand) -> object: ...

    async def show(self, task: TaskRef) -> object: ...

    async def decide(self, command: DecideReviewCommand) -> object: ...

    async def finalize(self, command: FinalizeReviewCommand) -> object: ...

    async def reject(self, command: RejectReviewCommand) -> object: ...

    async def result(self, task: TaskRef) -> object: ...


class _CliContractError(ValueError):
    pass


class _DependencyUnavailable(RuntimeError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CliContractError("invalid command") from None


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="quality-gate-review", add_help=True)
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    list_command = commands.add_parser("list")
    list_command.add_argument("--cursor")
    list_command.add_argument("--limit", type=int, default=50)

    for name in ("claim", "show", "decide", "finalize", "reject", "result"):
        command = commands.add_parser(name)
        command.add_argument("--task-id", type=UUID, required=True)
        command.add_argument("--expected-lock-version", type=int, required=True)
        command.add_argument("--source-report-id", type=UUID, required=True)
        command.add_argument("--source-report-revision", type=int, required=True)
        if name in {"claim", "finalize", "reject"}:
            command.add_argument("--idempotency-key", required=True)
        if name in {"decide", "finalize"}:
            command.add_argument("--decisions", type=Path, required=True)
            command.add_argument("--metric-shots", type=Path, required=True)
        if name == "reject":
            command.add_argument("--reason", choices=tuple(SessionRejectionReason), required=True)
    return parser


def _task(args: argparse.Namespace) -> TaskRef:
    return TaskRef(
        task_id=args.task_id,
        expected_lock_version=args.expected_lock_version,
        source_report_id=args.source_report_id,
        source_report_revision=args.source_report_revision,
    )


def _validated_sequence(path: Path, adapter: TypeAdapter) -> tuple:
    raw = path.read_bytes()
    if len(raw) > _MAX_INTENT_BYTES:
        raise _CliContractError("intent input is too large")
    return adapter.validate_json(raw, strict=True)


_DECISIONS = TypeAdapter(tuple[ShotDecisionIntent, ...])
_METRICS = TypeAdapter(tuple[MetricShotReviewInput, ...])


def _configuration(environ: Mapping[str, str]) -> dict[str, object]:
    try:
        database_url = environ["QUALITY_GATE_DATABASE_URL"]
        table_binding_store = Path(environ["QUALITY_GATE_TABLE_BINDING_STORE"])
        calibration_store = Path(environ["QUALITY_GATE_CALIBRATION_STORE"])
        calibration_device_id = environ["QUALITY_GATE_CALIBRATION_DEVICE_ID"]
    except (KeyError, TypeError, ValueError) as exc:
        raise _DependencyUnavailable("quality-gate configuration is unavailable") from exc
    parsed = urlparse(database_url)
    if parsed.scheme != "postgresql+asyncpg" or not parsed.hostname or not calibration_device_id:
        raise _DependencyUnavailable("quality-gate configuration is invalid")
    return {
        "database_url": database_url,
        "identity_provider": FileReviewerIdentityProvider(),
        "table_binding_repository": JsonTableBindingRepository(table_binding_store),
        "calibration_store": calibration_store,
        "calibration_device_id": calibration_device_id,
    }


def _default_service_factory(**kwargs) -> ReviewApplicationServiceProtocol:
    from .application import build_local_review_application_service

    return build_local_review_application_service(**kwargs)


async def _dispatch(service: ReviewApplicationServiceProtocol, args: argparse.Namespace) -> object:
    if args.command == "list":
        return await service.list_tasks(cursor=args.cursor, limit=args.limit)
    task = _task(args)
    if args.command == "claim":
        return await service.claim(
            ClaimReviewCommand(task=task, idempotency_key=args.idempotency_key)
        )
    if args.command == "show":
        return await service.show(task)
    if args.command in {"decide", "finalize"}:
        decisions = _validated_sequence(args.decisions, _DECISIONS)
        metric_shots = _validated_sequence(args.metric_shots, _METRICS)
        if args.command == "decide":
            return await service.decide(
                DecideReviewCommand(task=task, decisions=decisions, metric_shots=metric_shots)
            )
        return await service.finalize(
            FinalizeReviewCommand(
                task=task,
                decisions=decisions,
                metric_shots=metric_shots,
                idempotency_key=args.idempotency_key,
            )
        )
    if args.command == "reject":
        return await service.reject(
            RejectReviewCommand(
                task=task,
                reason=SessionRejectionReason(args.reason),
                idempotency_key=args.idempotency_key,
            )
        )
    return await service.result(task)


async def _run(
    factory: Callable[..., ReviewApplicationServiceProtocol],
    composition: Mapping[str, object],
    args: argparse.Namespace,
) -> object:
    candidate = factory(**composition)
    service = await candidate if inspect.isawaitable(candidate) else candidate
    return await _dispatch(service, args)


def _payload(*, data: object | None = None, code: str | None = None) -> dict[str, object]:
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    return {
        "ok": code is None,
        "data": data if code is None else None,
        "error": None if code is None else {"code": code, "message": _SAFE_MESSAGE},
    }


def _write(stream: TextIO, payload: dict[str, object]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    stream.write("\n")


def _public_code(exc: BaseException) -> str | None:
    from .application import ReviewApplicationError, ReviewErrorCode

    if not isinstance(exc, ReviewApplicationError) or not isinstance(exc.code, ReviewErrorCode):
        return None
    return exc.code.value


def _exit_for(code: str) -> int:
    if code == "INVALID_CONTRACT" or code in {"INVALID_DECISION_SET", "INVALID_CORRECTION"}:
        return 2
    if code == "REVIEW_ACCESS_DENIED":
        return 3
    if code == "REVIEW_RESOURCE_UNAVAILABLE":
        return 4
    if code in {
        "REVISION_CONFLICT",
        "IDEMPOTENCY_CONFLICT",
        "NO_PROJECTABLE_APPROVED_EVIDENCE",
        "INVALID_AUTHORITY",
    }:
        return 5
    if code in {"DEPENDENCY_UNAVAILABLE", "COMMIT_STATE_UNKNOWN"}:
        return 6
    return 70


def main(
    argv: list[str] | None = None,
    *,
    service_factory: Callable[..., ReviewApplicationServiceProtocol] | None = None,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    try:
        args = _parser().parse_args(argv)
        composition = _configuration(os.environ if environ is None else environ)
        factory = service_factory or _default_service_factory
        result = asyncio.run(_run(factory, composition, args))
        _write(output, _payload(data=result))
        return 0
    except _DependencyUnavailable:
        code = "DEPENDENCY_UNAVAILABLE"
    except (
        _CliContractError,
        OSError,
        UnicodeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
    ):
        code = "INVALID_CONTRACT"
    except Exception as exc:  # noqa: BLE001 - transport must sanitize every failure
        code = _public_code(exc) or "INTERNAL_ERROR"
    _write(errors, _payload(code=code))
    return _exit_for(code)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
