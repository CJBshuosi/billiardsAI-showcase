"""Explicit FastAPI adapter for the authorized committed growth read seam."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Protocol

from fastapi import Depends, FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from .persistence.errors import GrowthIdentityError, GrowthPersistenceIntegrityError


class GrowthReadServiceProtocol(Protocol):
    async def get_current(self) -> object: ...


def _envelope(*, data: object | None = None, code: str | None = None) -> dict[str, object]:
    return {
        "ok": code is None,
        "data": jsonable_encoder(data) if code is None else None,
        "error": None if code is None else {"code": code, "message": "request unavailable"},
    }


def install_growth_api(
    app: FastAPI,
    service_factory: Callable[[object], GrowthReadServiceProtocol],
    verified_user_dependency: Callable[..., object],
) -> None:
    if (
        not isinstance(app, FastAPI)
        or not callable(service_factory)
        or not callable(verified_user_dependency)
    ):
        raise TypeError("growth API requires explicit trusted dependencies")

    async def service(identity: object = Depends(verified_user_dependency)):
        candidate = service_factory(identity)
        return await candidate if inspect.isawaitable(candidate) else candidate

    service_dependency = Depends(service)

    @app.get("/v1/users/me/growth/current", response_model=None)
    async def current(
        growth_service: GrowthReadServiceProtocol = service_dependency,
    ) -> dict[str, object] | JSONResponse:
        try:
            return _envelope(data=await growth_service.get_current())
        except GrowthIdentityError:
            return JSONResponse(status_code=401, content=_envelope(code="AUTHENTICATION_REQUIRED"))
        except GrowthPersistenceIntegrityError:
            return JSONResponse(status_code=503, content=_envelope(code="GROWTH_READ_UNAVAILABLE"))
