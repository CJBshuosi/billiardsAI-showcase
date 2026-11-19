"""Scoped replay facts and independent rule evidence, without automatic adjudication."""

import asyncio
import json
import re
import time
from uuid import uuid4

from ..contracts import ToolContext, ToolResult
from ..execution import Budget
from ..language import response_language
from ..replay_context import select_clip_hint
from ..tools import execute_tool, tool_schemas
from .contracts import RulesAskResponse
from .prompts import RULES_PROMPT
from .workflow import TOOL_TIMEOUT, TOTAL_TIMEOUT, run_rule_dialog

REVIEW_PROMPT_VERSION = "m2-review-1"
REVIEW_PROMPT = (
    RULES_PROMPT.replace(
        "Rules-only mode has no replay observations. Do not invent shot evidence.",
        "Replay mode has synthetic observations only when a replay tool returns them. Do not invent shot evidence.",
    )
    + """
For replay questions first query the scoped replay record, then search relevant rules if needed. Explicit shot ID/number overrides selected clip; session-wide/list questions override selection. A bound get_shot schema must be obeyed. Rule-only questions may answer using rule tools alone.
A pocket_observed annotation means some pocket event was observed; it does not identify the ball, player, score, legal first contact, ball group or cue-ball scratch. Unknown is not no_pocket_observed. Do not assume missing conditions. When asked to judge a foul, legality, win/loss or cause from these records, explain only conditional rule requirements and list missing facts; use insufficient_evidence, never an unconditional verdict. No automatic adjudication is supported.
Cite replay evidence IDs separately in shot_evidence_refs and rule excerpts only in rule_citations. Never treat a source ID as a replay ID or vice versa. A replay factual assertion must cite successful replay tools from this request. Do not claim the model watched the video. The player is for human review; tools provide annotations only.
"""
)


def target_for(question):
    explicit_id = re.search(
        r"\bshot[_ ]id\s*[:=]?\s*([a-z0-9][a-z0-9_-]{0,79})\b|\bshot\s+([a-z0-9]+(?:[-_][a-z0-9]+)+)\b",
        question,
        re.I,
    )
    if explicit_id:
        return "shot_id", explicit_id.group(1) or explicit_id.group(2)
    sequence = re.search(
        r"(?:\bshot\s*(\d+)\b|第\s*(\d+)\s*(?:杆|次击球|次出杆|击球))", question, re.I
    )
    if sequence:
        return "sequence", int(sequence.group(1) or sequence.group(2))
    if re.search(
        r"\bsession\b|\brecorded shots\b|\b(?:list|show)\b[^?。？]{0,60}\bshots\b|整场|场次|所有单杆|这些击球|列出击球",
        question,
        re.I,
    ):
        return "session", None
    if re.search(
        r"\b(?:this|that|current|selected)\s+(?:replay\s+)?(?:shot|clip|one)\b|这一杆|这杆|那一杆|当前片段|这个片段",
        question,
        re.I,
    ):
        return "selected", None
    return None, None


async def ask_review(
    request,
    *,
    context,
    replay_repository,
    rule_repository,
    model,
    encoder=None,
    strategy="vector_glossary_v1",
    clock=time.monotonic,
):
    budget = Budget(TOTAL_TIMEOUT, clock)

    def early(status, text, limit=None):
        status, text, _ = response_language(request.question, status, text)
        return RulesAskResponse(
            request_id=str(uuid4()),
            mode=request.mode,
            status=status,
            answer=text,
            limitations=(limit,) if limit else (),
        )

    if (request.ruleset_id, request.version) != (context.ruleset_id, context.version):
        return early(
            "unsupported",
            "This ruleset or edition is not registered. Available: WPA Heyball 2025-08-16.",
            "RULESET_VERSION_NOT_REGISTERED",
        )
    try:
        sessions = await asyncio.wait_for(
            replay_repository.list_sessions(), budget.remaining(TOOL_TIMEOUT)
        )
        replay_context = ToolContext(
            session_id=request.session_id,
            available_session_ids=tuple(s.session_id for s in sessions),
        )
        if request.session_id not in replay_context.available_session_ids:
            return early(
                "insufficient_evidence",
                "The selected session is unavailable.",
                "RECORD_NOT_FOUND",
            )
        selected = None
        if request.selected_shot_id is not None:
            selected = await asyncio.wait_for(
                replay_repository.get_shot(
                    request.session_id, shot_id=request.selected_shot_id
                ),
                budget.remaining(TOOL_TIMEOUT),
            )
            if selected is None:
                return early(
                    "insufficient_evidence",
                    "The selected clip is unavailable in this session. Please select another clip.",
                    "SELECTED_SHOT_NOT_FOUND",
                )
        kind, value = target_for(request.question)
        if kind == "selected":
            if selected is None:
                return early("clarification", select_clip_hint(request.question))
            kind, value = "shot_id", selected.shot_id
        bound = {kind: value} if kind in ("shot_id", "sequence") else None
        schemas = tool_schemas()
        if bound:
            schemas = [s for s in schemas if s["function"]["name"] == "get_shot"]
            schemas[0]["function"]["parameters"] = {
                "type": "object",
                "properties": {
                    kind: {
                        "type": "string" if kind == "shot_id" else "integer",
                        "enum": [value],
                    }
                },
                "required": [kind],
                "additionalProperties": False,
            }

        async def execute(call):
            if bound and (call.name != "get_shot" or call.arguments != bound):
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    status="invalid_arguments",
                    limitations=("SELECTED_SHOT_SCOPE_MISMATCH",),
                )
            return await execute_tool(call, replay_context, replay_repository)

        extra = "\nReplay session: " + request.session_id
        if selected:
            extra += "\nSelected replay context: " + json.dumps(
                {"shot_id": selected.shot_id, "sequence": selected.sequence}
            )
        if bound:
            extra += "\nBound replay locator: " + json.dumps(bound)
        return await run_rule_dialog(
            request,
            context=context,
            repository=rule_repository,
            model=model,
            encoder=encoder,
            strategy=strategy,
            clock=clock,
            budget=budget,
            system_prompt=REVIEW_PROMPT,
            extra_schemas=schemas,
            extra_executor=execute,
            extra_system=extra,
            required_first_tool="get_shot" if bound else None,
            require_replay_evidence=bool(bound),
        )
    except TimeoutError:
        return early(
            "timeout", "The scoped record lookup timed out.", "REQUEST_OR_TOOL_TIMEOUT"
        )
    except Exception:
        return early(
            "error",
            "The replay data service is unavailable.",
            "REPLAY_STORAGE_UNAVAILABLE",
        )
