from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import pytest
from fastapi import FastAPI

from billiards_ai.cloud.review.api import install_review_api

TASK_ID = "10000000-0000-4000-8000-000000000001"
REPORT_ID = "10000000-0000-4000-8000-000000000002"


async def _request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    query: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else b""
    raw_headers = [(b"content-type", b"application/json")]
    raw_headers.extend(
        (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
    )
    sent: list[dict[str, Any]] = []
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    try:
        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": path,
                "raw_path": path.encode(),
                "query_string": urlencode(query or {}).encode(),
                "root_path": "",
                "headers": raw_headers,
                "client": ("127.0.0.1", 12345),
                "server": ("test", 80),
            },
            receive,
            send,
        )
    except Exception:
        if not sent:
            raise
    start = next(message for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return start["status"], json.loads(response_body)


class _Service:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def list_tasks(self, *, cursor=None, limit=50):
        self.calls.append(("list", (cursor, limit)))
        return {"items": [], "next_cursor": None}

    async def claim(self, command):
        self.calls.append(("claim", command))
        return {"task_id": str(command.task.task_id), "status": "IN_REVIEW"}

    async def show(self, task):
        self.calls.append(("show", task))
        return {"task_id": str(task.task_id), "status": "IN_REVIEW"}

    async def decide(self, command):
        self.calls.append(("decide", command))
        return {"task_id": str(command.task.task_id), "status": "VALID"}

    async def finalize(self, command):
        self.calls.append(("finalize", command))
        return {"task_id": str(command.task.task_id), "status": "APPROVED"}

    async def reject(self, command):
        self.calls.append(("reject", command))
        return {"task_id": str(command.task.task_id), "status": "REJECTED"}

    async def result(self, task):
        self.calls.append(("result", task))
        return {"task_id": str(task.task_id), "status": "APPROVED"}


def _task_payload(**extra: Any) -> dict[str, Any]:
    return {
        "expected_lock_version": 3,
        "source_report_id": REPORT_ID,
        "source_report_revision": 1,
        **extra,
    }


def _decision_payload() -> dict[str, Any]:
    shot_id = "10000000-0000-4000-8000-000000000010"
    states = {
        name: "UNSUPPORTED"
        for name in (
            "geometry",
            "intent",
            "pot_attempt",
            "pot_result",
            "foul_result",
            "target_zone",
            "cue_ball_end",
        )
    }
    return {
        **_task_payload(),
        "decisions": [
            {
                "shot_id": shot_id,
                "outcome": "approved",
                "trajectory_reviewed": True,
                "corrections": [],
            }
        ],
        "metric_shots": [
            {
                "shot_id": shot_id,
                "cue_ball_start_mm": None,
                "object_ball_start_mm": None,
                "intended_object_ball": None,
                "intended_target_pocket": None,
                "pot_attempt_status": "UNSUPPORTED",
                "pot_result": None,
                "foul_result": "UNSUPPORTED",
                "target_zone_source": "none",
                "target_zone": None,
                "cue_ball_end_mm": None,
                "target_zone_evaluation_status": "UNSUPPORTED",
                "review_states": states,
            }
        ],
    }


def _app(service: _Service, *, identity: object | None = None) -> FastAPI:
    app = FastAPI()
    verified = identity or object()

    async def verified_identity_dependency() -> object:
        return verified

    install_review_api(
        app,
        lambda value: service if value is verified else None,
        verified_identity_dependency,
    )
    return app


@pytest.mark.asyncio
async def test_installer_injects_verified_identity_and_idempotency_header() -> None:
    service = _Service()
    status, body = await _request(
        _app(service),
        "POST",
        f"/quality-gate/v1/review-tasks/{TASK_ID}:claim",
        payload=_task_payload(),
        headers={"idempotency-key": "qg-review-claim-0123456789abcdef"},
    )

    assert status == 200
    assert body == {
        "ok": True,
        "data": {"task_id": TASK_ID, "status": "IN_REVIEW"},
        "error": None,
    }
    name, command = service.calls == [] and (None, None) or service.calls[-1]
    assert name == "claim"
    assert command.task.task_id == UUID(TASK_ID)
    assert command.idempotency_key == "qg-review-claim-0123456789abcdef"


@pytest.mark.asyncio
async def test_decide_has_no_idempotency_header_and_rejects_authority_injection() -> None:
    service = _Service()
    status, body = await _request(
        _app(service),
        "POST",
        f"/quality-gate/v1/review-tasks/{TASK_ID}:decide",
        payload={**_task_payload(), "decisions": [], "metric_shots": [], "reviewer": "forged"},
        headers={"idempotency-key": "must-not-be-consumed"},
    )

    assert status == 422
    assert body == {
        "ok": False,
        "data": None,
        "error": {"code": "INVALID_CONTRACT", "message": "operation could not be completed"},
    }
    assert service.calls == []
    serialized = json.dumps(body, sort_keys=True)
    assert all(token not in serialized for token in ("detail", "loc", "input", "forged"))


@pytest.mark.asyncio
async def test_unhandled_exception_is_sanitized() -> None:
    class FailingService(_Service):
        async def show(self, task):
            raise RuntimeError("sql review_tasks secret-principal")

    status, body = await _request(
        _app(FailingService()),
        "GET",
        f"/quality-gate/v1/review-tasks/{TASK_ID}",
        query=_task_payload(),
    )

    assert status == 500
    assert body["error"] == {
        "code": "INTERNAL_ERROR",
        "message": "operation could not be completed",
    }
    assert "review_tasks" not in json.dumps(body)
    assert "secret-principal" not in json.dumps(body)


@pytest.mark.asyncio
async def test_unhandled_exception_cannot_forge_a_public_error_code() -> None:
    class ForgedCodeError(RuntimeError):
        code = "REVIEW_ACCESS_DENIED"

    class FailingService(_Service):
        async def show(self, task):
            raise ForgedCodeError("not an application error")

    status, body = await _request(
        _app(FailingService()),
        "GET",
        f"/quality-gate/v1/review-tasks/{TASK_ID}",
        query=_task_payload(),
    )

    assert status == 500
    assert body["error"]["code"] == "INTERNAL_ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("a" * 32, "abcd" * 8, "0123456789abcdef0123456789abcde"))
async def test_api_rejects_low_entropy_shaped_idempotency_keys(key: str) -> None:
    service = _Service()
    status, body = await _request(
        _app(service),
        "POST",
        f"/quality-gate/v1/review-tasks/{TASK_ID}:claim",
        payload=_task_payload(),
        headers={"idempotency-key": key},
    )

    assert status == 422
    assert body["error"]["code"] == "INVALID_CONTRACT"
    assert service.calls == []


@pytest.mark.asyncio
async def test_all_routes_delegate_to_the_same_service_contract() -> None:
    service = _Service()
    app = _app(service)
    task_query = _task_payload()
    calls = (
        ("GET", "/quality-gate/v1/review-tasks", None, {}, {"cursor": "next", "limit": 7}),
        ("GET", f"/quality-gate/v1/review-tasks/{TASK_ID}", None, {}, task_query),
        (
            "POST",
            f"/quality-gate/v1/review-tasks/{TASK_ID}:decide",
            _decision_payload(),
            {},
            {},
        ),
        (
            "POST",
            f"/quality-gate/v1/review-tasks/{TASK_ID}:finalize",
            _decision_payload(),
            {"idempotency-key": "qg-api-finalize-0123456789abcdef"},
            {},
        ),
        (
            "POST",
            f"/quality-gate/v1/review-tasks/{TASK_ID}:reject",
            {**_task_payload(), "reason": "occluded"},
            {"idempotency-key": "qg-api-reject-00-0123456789abcdef"},
            {},
        ),
        (
            "GET",
            f"/quality-gate/v1/review-tasks/{TASK_ID}/result",
            None,
            {},
            task_query,
        ),
    )

    for method, path, payload, headers, query in calls:
        status, body = await _request(
            app, method, path, payload=payload, headers=headers, query=query
        )
        assert status == 200
        assert body["ok"] is True

    assert tuple(name for name, _value in service.calls) == (
        "list",
        "show",
        "decide",
        "finalize",
        "reject",
        "result",
    )


@pytest.mark.asyncio
async def test_nested_metric_authority_injection_is_rejected() -> None:
    service = _Service()
    payload = _decision_payload()
    payload["metric_shots"][0]["calibration_id"] = "10000000-0000-4000-8000-000000000099"

    status, body = await _request(
        _app(service),
        "POST",
        f"/quality-gate/v1/review-tasks/{TASK_ID}:decide",
        payload=payload,
    )

    assert status == 422
    assert body["error"]["code"] == "INVALID_CONTRACT"
    assert service.calls == []


@pytest.mark.asyncio
async def test_unknown_api_resource_uses_the_same_sanitized_envelope() -> None:
    status, body = await _request(
        _app(_Service()), "GET", "/quality-gate/v1/review-tasks/not-a-resource"
    )

    assert status == 404
    assert body == {
        "ok": False,
        "data": None,
        "error": {
            "code": "REVIEW_RESOURCE_UNAVAILABLE",
            "message": "operation could not be completed",
        },
    }
    assert "detail" not in body
