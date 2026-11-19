import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .contracts import (
    AskRequest,
    AskResponse,
    ToolCall,
    ToolContext,
    ToolResult,
    UnifiedAskRequest,
)
from .fixtures import ROOT
from .media import MediaCatalog
from .providers.deepseek import DeepSeekModel
from .providers.offline import OfflineModel
from .repository import SessionRepository
from .settings import Settings
from .tools import execute_tool
from .workflow import ask


def create_app(
    settings: Settings,
    *,
    repository=None,
    model=None,
    rules_runtime=None,
    assistant_model=None,
) -> FastAPI:
    catalog = MediaCatalog(ROOT / "data/media")
    catalog.validate()
    owned = repository is None
    repo = (
        repository
        if repository is not None
        else SessionRepository(settings.database_url)
    )
    active_rules = None
    rules_reason = "DISABLED"
    if settings.rules_enabled:
        try:
            from .rules.runtime import RulesRuntime

            active_rules = (
                rules_runtime if rules_runtime is not None else RulesRuntime(settings)
            )
        except ImportError:
            rules_reason = "DEPENDENCY_MISSING"

    @asynccontextmanager
    async def lifespan(app):
        if active_rules is not None:
            await active_rules.start()
        try:
            yield
        finally:
            if active_rules is not None:
                await active_rules.aclose()
            if owned:
                await repo.close()

    app = FastAPI(title="CueOrbit Replay M1", lifespan=lifespan)
    app.state.repository = repo
    app.state.rules_runtime = active_rules

    @app.get("/api/capabilities")
    async def capabilities():
        return {
            "rules": active_rules.capability()
            if active_rules is not None
            else {
                "enabled": settings.rules_enabled,
                "ready": False,
                "reason": rules_reason,
                "rulesets": [],
            },
            "review": {
                "enabled": bool(
                    active_rules and getattr(active_rules, "review_enabled", False)
                ),
                "ready": bool(
                    active_rules
                    and getattr(active_rules, "review_enabled", False)
                    and active_rules.ready
                ),
            },
        }

    if active_rules is not None:
        from .rules.routes import create_rules_router

        app.include_router(create_rules_router(active_rules, settings, repo))
    elif settings.rules_enabled:

        @app.post("/api/rules/ask")
        async def unavailable_rules():
            raise HTTPException(503, "RULES_UNAVAILABLE")

        @app.post("/api/review/ask")
        async def unavailable_review():
            raise HTTPException(503, "RULES_UNAVAILABLE")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc):
        return JSONResponse(status_code=422, content={"error": "INVALID_REQUEST"})

    @app.exception_handler(Exception)
    async def unavailable(request: Request, exc):
        return JSONResponse(status_code=503, content={"error": "SERVICE_UNAVAILABLE"})

    @app.get("/health")
    async def health():
        try:
            await asyncio.wait_for(repo.list_sessions(), timeout=2)
            storage = "ready"
        except Exception:
            storage = "unavailable"
        return JSONResponse(
            status_code=200 if storage == "ready" else 503,
            content={
                "status": storage,
                "data_kind": "synthetic",
                "live_ready": settings.live_ready,
                "default_mode": "offline",
            },
        )

    @app.get("/api/sessions")
    async def sessions():
        return await asyncio.wait_for(repo.list_sessions(), timeout=2)

    @app.get("/api/sessions/{session_id}")
    async def summary(session_id: str):
        data = await asyncio.wait_for(repo.get_summary(session_id), timeout=2)
        if data is None:
            raise HTTPException(404, "RECORD_NOT_FOUND")
        return data

    @app.get("/api/sessions/{session_id}/shots", response_model=ToolResult)
    async def clips(
        session_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=20, ge=1, le=50),
    ):
        async def query():
            sessions = await repo.list_sessions()
            context = ToolContext(
                session_id=session_id,
                available_session_ids=tuple(s.session_id for s in sessions),
            )
            if session_id not in context.available_session_ids:
                raise HTTPException(404, "RECORD_NOT_FOUND")
            result = await execute_tool(
                ToolCall(
                    call_id="clip-browser",
                    name="list_shots",
                    arguments={"offset": offset, "limit": limit},
                ),
                context,
                repo,
            )
            if result.status == "not_found":
                raise HTTPException(404, "RECORD_NOT_FOUND")
            if result.status not in ("ok", "partial"):
                raise HTTPException(503, "SERVICE_UNAVAILABLE")
            return result

        return await asyncio.wait_for(query(), timeout=2)

    @app.get("/api/sessions/{session_id}/shots/{shot_id}")
    async def shot(session_id: str, shot_id: str):
        data = await asyncio.wait_for(
            repo.get_shot(session_id, shot_id=shot_id), timeout=2
        )
        if data is None:
            raise HTTPException(404, "RECORD_NOT_FOUND")
        return data

    @app.post("/api/assistant/ask")
    async def assistant_question(body: UnifiedAskRequest):
        from uuid import uuid4

        from .assistant import ask_assistant
        from .assistant_model import OfflineAssistantModel
        from .language import response_language
        from .rules.contracts import UnifiedAskResponse
        from .rules.model import RulesDeepSeekModel

        if body.mode == "live" and not settings.live_ready:
            status, text, _ = response_language(
                body.question, "error", "The live model is not configured."
            )
            return UnifiedAskResponse(
                request_id=str(uuid4()),
                mode=body.mode,
                status=status,
                answer=text,
                limitations=("LIVE_MODEL_NOT_CONFIGURED",),
            )
        selected = assistant_model
        if selected is None:
            selected = (
                OfflineAssistantModel()
                if body.mode == "offline"
                else active_rules.model_for("live")
                if active_rules is not None
                else RulesDeepSeekModel(settings.api_key, settings.model_name)
            )
        return await ask_assistant(
            body, repository=repo, model=selected, rules_runtime=active_rules
        )

    @app.post("/api/ask", response_model=AskResponse)
    async def question(body: AskRequest):
        if body.mode == "live" and not settings.live_ready:
            raise HTTPException(503, "LIVE_MODEL_NOT_CONFIGURED")
        selected = model
        if selected is None:
            selected = (
                OfflineModel()
                if body.mode == "offline"
                else DeepSeekModel(settings.api_key, settings.model_name)
            )
        return await ask(body, repository=repo, model=selected)

    @app.get("/api/media/{media_id}")
    async def media(media_id: str):
        try:
            return FileResponse(catalog.resolve(media_id), media_type="video/mp4")
        except ValueError:
            raise HTTPException(404, "MEDIA_UNAVAILABLE") from None

    static = ROOT / "src/cueorbit_replay/static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    async def index():
        return FileResponse(static / "index.html")

    return app


def default_app():
    return create_app(Settings.from_env())
