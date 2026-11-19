"""Rules dialog, shared by rules-only and scoped replay+rules modes."""

import asyncio
import json
import time
from uuid import uuid4

from ..contracts import ToolResult
from ..evidence import supported_refs
from ..execution import Budget, UsageTotals
from ..language import response_language
from ..model import ModelError
from .annotations import notices_for_citations
from .contracts import ExecutionEvent, RuleChunk, RulesAskResponse, RulesTraceEntry
from .evidence import project_rule_citations, validate_chunk_scope
from .prompts import RULES_PROMPT
from .tools import ARGS, execute_rules_tool, rules_tool_schemas

TOTAL_TIMEOUT = 30
MODEL_TIMEOUT = 15
TOOL_TIMEOUT = 2


async def ask_rules(
    request,
    *,
    context,
    repository,
    model,
    encoder=None,
    strategy="vector_glossary_v1",
    clock=time.monotonic,
):
    return await run_rule_dialog(
        request,
        context=context,
        repository=repository,
        model=model,
        encoder=encoder,
        strategy=strategy,
        clock=clock,
    )


async def run_rule_dialog(
    request,
    *,
    context,
    repository,
    model,
    encoder=None,
    strategy="vector_glossary_v1",
    clock=time.monotonic,
    budget=None,
    system_prompt=RULES_PROMPT,
    extra_schemas=(),
    extra_executor=None,
    extra_system="",
    required_first_tool=None,
    require_replay_evidence=False,
    response_factory=RulesAskResponse,
    include_rules=True,
    enforce_evidence_families=False,
    allow_empty_record=False,
):
    budget = budget or Budget(TOTAL_TIMEOUT, clock)
    totals = UsageTotals()
    request_id = str(uuid4())
    trace = []
    limits = []
    pool = {}
    cards = []
    used_ids = set()
    source_annotations = ()
    events = []
    model_rounds = 0

    def finish(status, text, citations=(), refs=()):
        status, text, language_code = response_language(
            request.question, status, text, enforce_answer=request.mode == "live"
        )
        if language_code:
            limits.append(language_code)
        return response_factory(
            request_id=request_id,
            mode=request.mode,
            status=status,
            answer=text,
            rule_citations=tuple(citations),
            shot_evidence_refs=tuple(refs),
            replay_fact_cards=tuple(cards),
            tool_trace=tuple(trace),
            usage=totals.snapshot(),
            source_annotations=source_annotations,
            limitations=tuple(dict.fromkeys(limits)),
            **(
                {"model_rounds": model_rounds, "execution_events": tuple(events)}
                if response_factory is not RulesAskResponse
                else {}
            ),
        )

    if (
        context is not None
        and hasattr(request, "ruleset_id")
        and (request.ruleset_id, request.version)
        != (context.ruleset_id, context.version)
    ):
        limits.append("RULESET_VERSION_NOT_REGISTERED")
        return finish(
            "unsupported",
            "This ruleset or edition is not registered. Available: WPA Heyball 2025-08-16, Chapter II and the provided Appendix A excerpts.",
        )
    schemas = (rules_tool_schemas() if include_rules else []) + list(extra_schemas)
    names = {s["function"]["name"] for s in schemas}
    messages = [
        {
            "role": "system",
            "content": system_prompt
            + "\nServer rule context: "
            + (
                context.model_dump_json()
                if context is not None
                else "Rules unavailable; do not answer rule facts from memory."
            )
            + extra_system,
        },
        {"role": "user", "content": request.question},
    ]
    try:
        async with asyncio.timeout(budget.remaining(TOTAL_TIMEOUT)):
            for round_index in range(3):
                timeout = budget.remaining(MODEL_TIMEOUT)
                totals.begin()
                options = (
                    {"required_tool": required_first_tool}
                    if required_first_tool and round_index == 0
                    else {}
                )
                model_rounds += 1
                model_started = clock()
                try:
                    turn = await asyncio.wait_for(
                        model.next_turn(
                            messages, schemas, timeout_s=timeout, **options
                        ),
                        timeout,
                    )
                except Exception as error:
                    events.append(
                        ExecutionEvent(
                            phase="model",
                            round=model_rounds,
                            status="timeout"
                            if isinstance(error, TimeoutError)
                            else "error",
                            elapsed_ms=max(0, (clock() - model_started) * 1000),
                            validation_errors=error.diagnostics
                            if isinstance(error, ModelError)
                            else (),
                            error_code="MODEL_TIMEOUT"
                            if isinstance(error, TimeoutError)
                            else error.code
                            if isinstance(error, ModelError)
                            else "MODEL_UNAVAILABLE",
                        )
                    )
                    raise
                events.append(
                    ExecutionEvent(
                        phase="model",
                        round=model_rounds,
                        status="completed",
                        elapsed_ms=max(0, (clock() - model_started) * 1000),
                    )
                )
                budget.remaining(TOTAL_TIMEOUT)
                totals.complete(turn.usage)
                if turn.tool_calls:
                    if turn.answer is not None:
                        limits.append("INVALID_MODEL_TURN")
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
                        budget.remaining(TOTAL_TIMEOUT)
                        if len(trace) >= 4:
                            limits.append("TOOL_CALL_LIMIT")
                            return finish(
                                "limit_reached",
                                "The tool limit was reached. Narrow the question and try again.",
                            )
                        if call.call_id in used_ids:
                            limits.append("DUPLICATE_TOOL_CALL_ID")
                            return finish(
                                "error", "The model returned duplicate tool IDs."
                            )
                        used_ids.add(call.call_id)
                        started = clock()
                        try:
                            operation = (
                                execute_rules_tool(
                                    call,
                                    context,
                                    repository,
                                    encoder=encoder,
                                    strategy=strategy,
                                )
                                if include_rules and call.name in ARGS
                                else extra_executor(call)
                                if extra_executor is not None and call.name in names
                                else None
                            )
                            result = (
                                await asyncio.wait_for(
                                    operation, budget.remaining(TOOL_TIMEOUT)
                                )
                                if operation is not None
                                else ToolResult(
                                    call_id=call.call_id,
                                    name=call.name,
                                    status="denied",
                                    limitations=("TOOL_DENIED",),
                                )
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
                        returned = []
                        if result.status in ("ok", "partial"):
                            if include_rules and call.name in ARGS:
                                if (
                                    result.evidence_refs
                                    or result.data.get("corpus_hash")
                                    != context.corpus_hash
                                ):
                                    limits.append("RULE_SOURCE_SCOPE_MISMATCH")
                                    return finish(
                                        "insufficient_evidence",
                                        "The returned source scope was invalid.",
                                    )
                                for raw in list(result.data.get("chunks", ())) + list(
                                    result.data.get("related_chunks", ())
                                ):
                                    chunk = validate_chunk_scope(
                                        RuleChunk.model_validate(raw), context
                                    )
                                    if (
                                        chunk.chunk_id in pool
                                        and pool[chunk.chunk_id] != chunk
                                    ):
                                        raise ValueError("CONFLICTING_RULE_SOURCE")
                                    pool[chunk.chunk_id] = chunk
                                    returned.append(chunk.chunk_id)
                            else:
                                cards.append(result)
                        trace.append(
                            RulesTraceEntry(
                                name=call.name
                                if call.name in names
                                else "unknown_tool",
                                arguments=dict(call.arguments)
                                if result.status not in ("denied", "invalid_arguments")
                                else {},
                                status=result.status,
                                elapsed_ms=max(0, (clock() - started) * 1000),
                                retrieval_strategy=result.data.get(
                                    "retrieval_strategy"
                                ),
                                executed_query=result.data.get("executed_query"),
                                returned_chunk_ids=tuple(dict.fromkeys(returned)),
                            )
                        )
                        limits.extend(result.limitations)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "content": result.model_dump_json(),
                            }
                        )
                        if result.status not in ("ok", "partial"):
                            return finish(
                                "timeout"
                                if result.status == "timeout"
                                else "error"
                                if result.status == "unavailable"
                                else "insufficient_evidence",
                                "The requested evidence is unavailable. No unsupported conclusion was generated.",
                            )
                    continue
                draft = turn.answer
                if draft is None:
                    limits.append("EMPTY_MODEL_TURN")
                    return finish("error", "No answer was returned.")
                try:
                    citations, full_quote_candidates, shortened = (
                        project_rule_citations(
                            draft.rule_citations, tuple(pool.values()), context
                        )
                    )
                except ValueError:
                    limits.append("UNSUPPORTED_RULE_CITATION")
                    return finish(
                        "insufficient_evidence",
                        "The answer did not provide valid verbatim citations from this request.",
                    )
                if shortened:
                    limits.append("QUOTE_EXCERPT_SHORTENED")
                if not supported_refs(draft.shot_evidence_refs, tuple(cards)):
                    limits.append("UNSUPPORTED_SHOT_REFERENCE")
                    return finish(
                        "insufficient_evidence",
                        "The answer cited replay evidence that was not queried.",
                    )
                if (
                    require_replay_evidence
                    and draft.status not in ("unsupported", "clarification")
                    and not (
                        cards
                        and (
                            draft.shot_evidence_refs
                            or (
                                allow_empty_record
                                and not any(c.evidence_refs for c in cards)
                            )
                        )
                    )
                ):
                    limits.append("MISSING_REPLAY_EVIDENCE")
                    return finish(
                        "insufficient_evidence",
                        "The scoped replay evidence was not queried and cited. No replay conclusion is supported.",
                    )
                if enforce_evidence_families and draft.status in (
                    "answered",
                    "partial",
                ):
                    if pool and not citations:
                        limits.append("MISSING_RULE_CITATION")
                        return finish(
                            "insufficient_evidence",
                            "No source-backed rule answer was provided.",
                        )
                    if (
                        cards
                        and any(c.evidence_refs for c in cards)
                        and not draft.shot_evidence_refs
                    ):
                        limits.append("MISSING_REPLAY_EVIDENCE")
                        return finish(
                            "insufficient_evidence",
                            "The answer omitted its queried replay evidence.",
                        )
                if (
                    draft.status in ("answered", "partial")
                    and not citations
                    and not (
                        cards
                        and (
                            draft.shot_evidence_refs
                            or (
                                allow_empty_record
                                and not any(c.evidence_refs for c in cards)
                            )
                        )
                    )
                ):
                    limits.append("MISSING_RULE_CITATION")
                    return finish(
                        "insufficient_evidence", "No source-backed answer was provided."
                    )
                source_annotations = notices_for_citations(
                    full_quote_candidates, tuple(pool.values())
                )
                if source_annotations:
                    limits.append("SOURCE_WORDING_AMBIGUITY")
                    text = (
                        "所引用的原文片段存在数量或程序含义的歧义。本次不采用模型的确定性扩述，请查看原文、PDF页码及图示；编辑核查注释不是官方规则。"
                        if any("\u4e00" <= c <= "\u9fff" for c in request.question)
                        else "The cited passage has ambiguous quantity or procedure wording. No definitive expansion is adopted. Inspect the original quote, PDF page and figure; editorial audit notes are not official rules."
                    )
                    return finish(
                        "clarification", text, citations, draft.shot_evidence_refs
                    )
                status = (
                    "partial"
                    if draft.status == "answered"
                    and any(t.status == "partial" for t in trace)
                    else draft.status
                )
                return finish(status, draft.text, citations, draft.shot_evidence_refs)
            limits.append("MODEL_ROUND_LIMIT")
            return finish(
                "limit_reached",
                "The model round limit was reached. Narrow the question and try again.",
            )
    except TimeoutError:
        limits.append("REQUEST_OR_MODEL_TIMEOUT")
        return finish(
            "timeout", "The request timed out. No unsupported conclusion was generated."
        )
    except ModelError as error:
        limits.append(error.code)
        return finish(
            "error", "The model response could not be completed. Please retry."
        )
    except Exception:
        limits.append("MODEL_OR_STORAGE_UNAVAILABLE")
        return finish(
            "error", "The model or data service is unavailable. Please retry."
        )
