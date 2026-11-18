from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from billiards_ai.cloud.review.api import install_review_api


def _installed_app() -> FastAPI:
    app = FastAPI()

    async def verified_identity_dependency():
        return object()

    install_review_api(app, lambda _identity: object(), verified_identity_dependency)
    return app


def test_installer_registers_only_the_frozen_review_routes_and_handlers() -> None:
    app = _installed_app()
    routes = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if route.path.startswith("/quality-gate/")
    }

    assert routes == {
        ("GET", "/quality-gate/v1/review-tasks"),
        ("POST", "/quality-gate/v1/review-tasks/{task_id}:claim"),
        ("GET", "/quality-gate/v1/review-tasks/{task_id}"),
        ("POST", "/quality-gate/v1/review-tasks/{task_id}:decide"),
        ("POST", "/quality-gate/v1/review-tasks/{task_id}:finalize"),
        ("POST", "/quality-gate/v1/review-tasks/{task_id}:reject"),
        ("GET", "/quality-gate/v1/review-tasks/{task_id}/result"),
    }
    assert RequestValidationError in app.exception_handlers
    assert Exception in app.exception_handlers


def test_request_openapi_contains_business_intent_only() -> None:
    schema = _installed_app().openapi()
    forbidden = {
        "reviewer",
        "principal",
        "roles",
        "credential",
        "user_id",
        "ai_value",
        "approved",
        "media",
        "approval_kind",
        "approval_policy_version",
        "calibration_id",
        "batch_id",
        "event_id",
        "candidate_uuid",
    }
    request_schema_names = {
        reference["$ref"].rsplit("/", 1)[-1]
        for operation in schema["paths"].values()
        for endpoint in operation.values()
        if isinstance(endpoint, dict) and "requestBody" in endpoint
        for media in endpoint["requestBody"]["content"].values()
        for reference in (media["schema"],)
        if "$ref" in reference
    }
    request_properties = {
        property_name
        for name in request_schema_names
        for property_name in schema["components"]["schemas"][name].get("properties", {})
    }

    assert forbidden.isdisjoint(request_properties)


def test_only_mutating_terminal_routes_accept_idempotency_header() -> None:
    paths = _installed_app().openapi()["paths"]

    def headers(path: str, method: str) -> set[str]:
        return {
            parameter["name"]
            for parameter in paths[path][method]["parameters"]
            if parameter["in"] == "header"
        }

    assert headers("/quality-gate/v1/review-tasks/{task_id}:claim", "post") == {"Idempotency-Key"}
    assert headers("/quality-gate/v1/review-tasks/{task_id}:finalize", "post") == {
        "Idempotency-Key"
    }
    assert headers("/quality-gate/v1/review-tasks/{task_id}:reject", "post") == {"Idempotency-Key"}
    assert headers("/quality-gate/v1/review-tasks/{task_id}:decide", "post") == set()
