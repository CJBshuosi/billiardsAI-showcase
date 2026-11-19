import asyncio
import json
import time
from uuid import uuid4

from .contracts import (
    AskRequest,
    AskResponse,
    ToolContext,
    ToolResult,
    TraceEntry,
)
from .evidence import supported_refs
from .execution import Budget, UsageTotals
from .language import response_language
from .model import ModelError, ReplayModelPort
from .prompts import SYSTEM_PROMPT
from .replay_context import is_current_clip_question, select_clip_hint
from .repository import RepositoryPort
from .tools import ARGS, execute_tool, tool_schemas

TOTAL_TIMEOUT = 30
MODEL_TIMEOUT = 15
TOOL_TIMEOUT = 2


async def ask(
    request: AskRequest,
    *,
    repository: RepositoryPort,
    model: ReplayModelPort,
    clock=time.monotonic,
) -> AskResponse:
    request_id = str(uuid4())
    budget = Budget(TOTAL_TIMEOUT, clock)
    remaining = budget.remaining
    cards, trace, limitations = [], [], []
    used_ids = set()
    totals = UsageTotals()
    bound_shot = None

    def finish(status, answer, refs=()):
        status, answer, language_code = response_language(
            request.question, status, answer, enforce_answer=request.mode == "live"
        )
        if language_code:
            limitations.append(language_code)
        return AskResponse(
            request_id=request_id,
            mode=request.mode,
            status=status,
            answer=answer,
            fact_cards=tuple(cards),
            evidence_refs=tuple(refs),
            limitations=tuple(dict.fromkeys(limitations)),
            tool_trace=tuple(trace),
            usage=totals.snapshot(),
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.question},
    ]
    try:
        async with asyncio.timeout(TOTAL_TIMEOUT):
            sessions = await asyncio.wait_for(
                repository.list_sessions(), timeout=remaining(TOOL_TIMEOUT)
            )
            context = ToolContext(
                session_id=request.session_id,
                available_session_ids=tuple(s.session_id for s in sessions),
            )
            if request.session_id not in context.available_session_ids:
                limitations.append("RECORD_NOT_FOUND")
                return finish(
                    "insufficient_evidence", "The selected session is unavailable."
                )
            current_clip = is_current_clip_question(request.question)
            if current_clip and request.selected_shot_id is None:
                return finish("clarification", select_clip_hint(request.question))
            if request.selected_shot_id is not None:
                selected = await asyncio.wait_for(
                    repository.get_shot(
                        request.session_id, shot_id=request.selected_shot_id
                    ),
                    timeout=remaining(TOOL_TIMEOUT),
                )
                if selected is None:
                    limitations.append("SELECTED_SHOT_NOT_FOUND")
                    return finish(
                        "insufficient_evidence",
                        "The selected clip is unavailable in this session. Please select another clip.",
                    )
                messages[0]["content"] += "\nSelected replay context: " + json.dumps(
                    {"shot_id": selected.shot_id, "sequence": selected.sequence}
                )
                if current_clip:
                    bound_shot = selected.shot_id
            for round_index in range(3):
                timeout = remaining(MODEL_TIMEOUT)
                totals.begin()
                schemas = tool_schemas()
                options = {}
                if bound_shot is not None:
                    schemas = [
                        s for s in schemas if s["function"]["name"] == "get_shot"
                    ]
                    schemas[0]["function"]["parameters"] = {
                        "type": "object",
                        "properties": {
                            "shot_id": {"type": "string", "enum": [bound_shot]}
                        },
                        "required": ["shot_id"],
                        "additionalProperties": False,
                    }
                    if round_index == 0:
                        options["required_tool"] = "get_shot"
                turn = await asyncio.wait_for(
                    model.next_turn(messages, schemas, timeout_s=timeout, **options),
                    timeout=timeout,
                )
                remaining(TOTAL_TIMEOUT)
                totals.complete(turn.usage)
                if turn.tool_calls:
                    if turn.answer is not None:
                        limitations.append("INVALID_MODEL_TURN")
                        return finish("error", "The model returned an invalid turn.")
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": c.call_id,
                                    "type": "function",
                                    "function": {
                                        "name": c.name,
                                        "arguments": json.dumps(c.arguments),
                                    },
                                }
                                for c in turn.tool_calls
                            ],
                        }
                    )
                    for call in turn.tool_calls:
                        remaining(TOTAL_TIMEOUT)
                        if len(trace) >= 4:
                            limitations.append("TOOL_CALL_LIMIT")
                            return finish(
                                "limit_reached",
                                "The tool call limit was reached. Available facts are shown.",
                            )
                        if call.call_id in used_ids:
                            limitations.append("DUPLICATE_TOOL_CALL_ID")
                            return finish(
                                "error", "The model returned duplicate tool call IDs."
                            )
                        used_ids.add(call.call_id)
                        started = clock()
                        try:
                            if bound_shot is not None and (
                                call.name != "get_shot"
                                or call.arguments != {"shot_id": bound_shot}
                            ):
                                result = ToolResult(
                                    call_id=call.call_id,
                                    name=call.name,
                                    status="invalid_arguments",
                                    limitations=("SELECTED_SHOT_SCOPE_MISMATCH",),
                                )
                            else:
                                result = await asyncio.wait_for(
                                    execute_tool(call, context, repository),
                                    timeout=remaining(TOOL_TIMEOUT),
                                )
                        except TimeoutError:
                            result = ToolResult(
                                call_id=call.call_id,
                                name=call.name,
                                status="timeout",
                                limitations=("TOOL_TIMEOUT",),
                            )
                        except Exception:
                            result = ToolResult(
                                call_id=call.call_id,
                                name=call.name,
                                status="unavailable",
                                limitations=("TOOL_UNAVAILABLE",),
                            )
                        safe_args = (
                            dict(call.arguments)
                            if result.status not in ("invalid_arguments", "denied")
                            else {}
                        )
                        trace.append(
                            TraceEntry(
                                name=call.name if call.name in ARGS else "unknown_tool",
                                arguments=safe_args,
                                status=result.status,
                                elapsed_ms=max(0, (clock() - started) * 1000),
                            )
                        )
                        limitations.extend(result.limitations)
                        if result.status in ("ok", "partial"):
                            cards.append(result)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "content": result.model_dump_json(),
                            }
                        )
                    continue
                draft = turn.answer
                if draft is None:
                    limitations.append("EMPTY_MODEL_TURN")
                    return finish("error", "No answer was returned.")
                if not supported_refs(draft.evidence_refs, tuple(cards)):
                    limitations.append("UNSUPPORTED_REFERENCE")
                    return finish(
                        "insufficient_evidence",
                        "The answer cited evidence that was not returned by a successful tool.",
                    )
                if draft.status in ("answered", "partial") and (
                    not cards
                    or (any(c.evidence_refs for c in cards) and not draft.evidence_refs)
                ):
                    limitations.append("MISSING_EVIDENCE")
                    return finish(
                        "insufficient_evidence",
                        "No supported cited evidence was provided for the answer.",
                    )
                status = draft.status
                if status == "answered" and any(t.status != "ok" for t in trace):
                    status = "partial"
                return finish(status, draft.text, draft.evidence_refs)
            limitations.append("MODEL_ROUND_LIMIT")
            return finish(
                "limit_reached",
                "The model round limit was reached. Available facts are shown.",
            )
    except TimeoutError:
        limitations.append("REQUEST_OR_MODEL_TIMEOUT")
        return finish("timeout", "The request timed out. Available facts are shown.")
    except ModelError as error:
        limitations.append(error.code)
        return finish(
            "error", "The model response could not be completed. Please retry."
        )
    except Exception:
        limitations.append("MODEL_OR_STORAGE_UNAVAILABLE")
        return finish(
            "error", "The model or data service is unavailable. Please retry."
        )
