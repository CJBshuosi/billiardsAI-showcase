from fastapi import APIRouter, HTTPException

from .contracts import ReplayRulesAskRequest, RulesAskRequest, RulesAskResponse
from .review import ask_review
from .workflow import ask_rules


def create_rules_router(runtime, settings, replay_repository):
    router = APIRouter()

    @router.post("/api/rules/ask", response_model=RulesAskResponse)
    async def question(body: RulesAskRequest):
        if not runtime.ready:
            raise HTTPException(503, "RULES_UNAVAILABLE")
        if (body.ruleset_id, body.version) != (
            runtime.context.ruleset_id,
            runtime.context.version,
        ):
            return await ask_rules(
                body, context=runtime.context, repository=runtime.repository, model=None
            )
        if body.mode == "live" and not settings.live_ready:
            raise HTTPException(503, "LIVE_MODEL_NOT_CONFIGURED")
        return await ask_rules(
            body,
            context=runtime.context,
            repository=runtime.repository,
            model=runtime.model_for(body.mode),
            encoder=runtime.encoder,
            strategy=runtime.strategy,
        )

    @router.post("/api/review/ask", response_model=RulesAskResponse)
    async def review(body: ReplayRulesAskRequest):
        if not runtime.ready or not getattr(runtime, "review_enabled", False):
            raise HTTPException(503, "RULES_UNAVAILABLE")
        known = (body.ruleset_id, body.version) == (
            runtime.context.ruleset_id,
            runtime.context.version,
        )
        if known and body.mode == "live" and not settings.live_ready:
            raise HTTPException(503, "LIVE_MODEL_NOT_CONFIGURED")
        return await ask_review(
            body,
            context=runtime.context,
            replay_repository=replay_repository,
            rule_repository=runtime.repository,
            model=runtime.model_for(body.mode) if known else None,
            encoder=runtime.encoder,
            strategy=runtime.strategy,
        )

    return router
