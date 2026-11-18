"""Explicitly composed FastAPI adapter for the controlled review service."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Protocol
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    TypeAdapter,
    ValidationError,
    field_validator,
)

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


class ReviewApplicationServiceProtocol(Protocol):
    async def list_tasks(self, *, cursor: str | None = None, limit: int = 50) -> object: ...

    async def claim(self, command: ClaimReviewCommand) -> object: ...

    async def show(self, task: TaskRef) -> object: ...

    async def decide(self, command: DecideReviewCommand) -> object: ...

    async def finalize(self, command: FinalizeReviewCommand) -> object: ...

    async def reject(self, command: RejectReviewCommand) -> object: ...

    async def result(self, task: TaskRef) -> object: ...


class _TaskBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_lock_version: StrictInt = Field(ge=0)
    source_report_id: UUID4
    source_report_revision: StrictInt = Field(ge=1)

    def task_ref(self, task_id: UUID) -> TaskRef:
        return TaskRef(
            task_id=task_id,
            expected_lock_version=self.expected_lock_version,
            source_report_id=self.source_report_id,
            source_report_revision=self.source_report_revision,
        )


class _DecisionBody(_TaskBody):
    decisions: tuple[ShotDecisionIntent, ...] = Field(min_length=1)
    metric_shots: tuple[MetricShotReviewInput, ...] = Field(min_length=1)

    @field_validator("decisions", mode="before")
    @classmethod
    def strict_decisions(cls, value: object) -> tuple[ShotDecisionIntent, ...]:
        return _strict_json_tuple(value, _DECISION_ADAPTER)

    @field_validator("metric_shots", mode="before")
    @classmethod
    def strict_metrics(cls, value: object) -> tuple[MetricShotReviewInput, ...]:
        return _strict_json_tuple(value, _METRIC_ADAPTER)


class _RejectBody(_TaskBody):
    reason: SessionRejectionReason


_DECISION_ADAPTER = TypeAdapter(tuple[ShotDecisionIntent, ...])
_METRIC_ADAPTER = TypeAdapter(tuple[MetricShotReviewInput, ...])


def _strict_json_tuple(value: object, adapter: TypeAdapter) -> tuple:
    if isinstance(value, tuple) and all(isinstance(item, BaseModel) for item in value):
        return value
    return adapter.validate_json(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        strict=True,
    )


def _envelope(*, data: object | None = None, code: str | None = None) -> dict[str, object]:
    return {
        "ok": code is None,
        "data": jsonable_encoder(data) if code is None else None,
        "error": None if code is None else {"code": code, "message": _SAFE_MESSAGE},
    }


def _status_for(code: str) -> int:
    if code == "REVIEW_ACCESS_DENIED":
        return 403
    if code == "REVIEW_RESOURCE_UNAVAILABLE":
        return 404
    if code in {
        "REVISION_CONFLICT",
        "IDEMPOTENCY_CONFLICT",
        "NO_PROJECTABLE_APPROVED_EVIDENCE",
        "INVALID_AUTHORITY",
    }:
        return 409
    if code in {"INVALID_DECISION_SET", "INVALID_CORRECTION", "INVALID_CONTRACT"}:
        return 422
    if code in {"DEPENDENCY_UNAVAILABLE", "COMMIT_STATE_UNKNOWN"}:
        return 503
    return 500


def _task_from_query(
    task_id: UUID,
    expected_lock_version: int,
    source_report_id: UUID,
    source_report_revision: int,
) -> TaskRef:
    return TaskRef(
        task_id=task_id,
        expected_lock_version=expected_lock_version,
        source_report_id=source_report_id,
        source_report_revision=source_report_revision,
    )


def install_review_api(
    app: FastAPI,
    service_factory: Callable[[object], ReviewApplicationServiceProtocol],
    verified_identity_dependency: Callable[..., object],
) -> None:
    """Install review routes only when both trusted dependencies are explicit."""

    if (
        not isinstance(app, FastAPI)
        or not callable(service_factory)
        or not callable(verified_identity_dependency)
    ):
        raise TypeError("review API requires explicit trusted dependencies")

    from .application import ReviewApplicationError, ReviewErrorCode

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        if any(tuple(error.get("loc", ()))[:2] == ("path", "task_id") for error in errors):
            return JSONResponse(
                status_code=404,
                content=_envelope(code="REVIEW_RESOURCE_UNAVAILABLE"),
            )
        return JSONResponse(status_code=422, content=_envelope(code="INVALID_CONTRACT"))

    @app.exception_handler(ReviewApplicationError)
    async def domain_handler(_request: Request, exc: ReviewApplicationError) -> JSONResponse:
        code = exc.code.value if isinstance(exc.code, ReviewErrorCode) else "INTERNAL_ERROR"
        return JSONResponse(status_code=_status_for(code), content=_envelope(code=code))

    @app.exception_handler(Exception)
    async def unhandled_handler(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content=_envelope(code="INTERNAL_ERROR"))

    async def service(identity: object = Depends(verified_identity_dependency)):
        candidate = service_factory(identity)
        return await candidate if inspect.isawaitable(candidate) else candidate

    service_dependency = Depends(service)
    cursor_query = Query(default=None, min_length=1, max_length=512)
    limit_query = Query(default=50, ge=1, le=100)
    lock_version_query = Query(ge=0)
    source_report_query = Query()
    source_revision_query = Query(ge=1)
    idempotency_header = Header(
        alias="Idempotency-Key",
        min_length=32,
        max_length=128,
        pattern=r"^[A-Za-z0-9._~-]+$",
    )

    @app.get("/quality-gate/v1/review-tasks")
    async def list_tasks(
        cursor: str | None = cursor_query,
        limit: int = limit_query,
        review_service: ReviewApplicationServiceProtocol = service_dependency,
    ) -> dict[str, object]:
        return _envelope(data=await review_service.list_tasks(cursor=cursor, limit=limit))

    @app.post("/quality-gate/v1/review-tasks/{task_id}:claim")
    async def claim(
        task_id: UUID4,
        body: _TaskBody,
        idempotency_key: str = idempotency_header,
        review_service: ReviewApplicationServiceProtocol = service_dependency,
    ) -> dict[str, object]:
        try:
            command = ClaimReviewCommand(
                task=body.task_ref(task_id), idempotency_key=idempotency_key
            )
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from None
        return _envelope(data=await review_service.claim(command))

    @app.get("/quality-gate/v1/review-tasks/{task_id}")
    async def show(
        task_id: UUID4,
        expected_lock_version: int = lock_version_query,
        source_report_id: UUID4 = source_report_query,
        source_report_revision: int = source_revision_query,
        review_service: ReviewApplicationServiceProtocol = service_dependency,
    ) -> dict[str, object]:
        task = _task_from_query(
            task_id, expected_lock_version, source_report_id, source_report_revision
        )
        return _envelope(data=await review_service.show(task))

    @app.post("/quality-gate/v1/review-tasks/{task_id}:decide")
    async def decide(
        task_id: UUID4,
        body: _DecisionBody,
        review_service: ReviewApplicationServiceProtocol = service_dependency,
    ) -> dict[str, object]:
        command = DecideReviewCommand(
            task=body.task_ref(task_id),
            decisions=body.decisions,
            metric_shots=body.metric_shots,
        )
        return _envelope(data=await review_service.decide(command))

    @app.post("/quality-gate/v1/review-tasks/{task_id}:finalize")
    async def finalize(
        task_id: UUID4,
        body: _DecisionBody,
        idempotency_key: str = idempotency_header,
        review_service: ReviewApplicationServiceProtocol = service_dependency,
    ) -> dict[str, object]:
        try:
            command = FinalizeReviewCommand(
                task=body.task_ref(task_id),
                decisions=body.decisions,
                metric_shots=body.metric_shots,
                idempotency_key=idempotency_key,
            )
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from None
        return _envelope(data=await review_service.finalize(command))

    @app.post("/quality-gate/v1/review-tasks/{task_id}:reject")
    async def reject(
        task_id: UUID4,
        body: _RejectBody,
        idempotency_key: str = idempotency_header,
        review_service: ReviewApplicationServiceProtocol = service_dependency,
    ) -> dict[str, object]:
        try:
            command = RejectReviewCommand(
                task=body.task_ref(task_id),
                reason=body.reason,
                idempotency_key=idempotency_key,
            )
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from None
        return _envelope(data=await review_service.reject(command))

    @app.get("/quality-gate/v1/review-tasks/{task_id}/result")
    async def result(
        task_id: UUID4,
        expected_lock_version: int = lock_version_query,
        source_report_id: UUID4 = source_report_query,
        source_report_revision: int = source_revision_query,
        review_service: ReviewApplicationServiceProtocol = service_dependency,
    ) -> dict[str, object]:
        task = _task_from_query(
            task_id, expected_lock_version, source_report_id, source_report_revision
        )
        return _envelope(data=await review_service.result(task))
