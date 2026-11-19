from typing import Annotated

from pydantic import Field, ValidationError, model_validator

from .contracts import (
    ID,
    Contract,
    NonNegative,
    Outcome,
    Positive,
    ToolCall,
    ToolContext,
    ToolResult,
)
from .repository import RepositoryPort


class SummaryArgs(Contract):
    pass


class ListArgs(Contract):
    outcome: Outcome | None = None
    offset: NonNegative = 0
    limit: Annotated[Positive, Field(le=50)] = 20


class ShotArgs(Contract):
    shot_id: ID | None = None
    sequence: Positive | None = None

    @model_validator(mode="after")
    def exclusive(self):
        if (self.shot_id is None) == (self.sequence is None):
            raise ValueError("exactly one locator")
        return self


ARGS = {
    "get_session_summary": SummaryArgs,
    "list_shots": ListArgs,
    "get_shot": ShotArgs,
}


def tool_schemas() -> list[dict]:
    descriptions = {
        "get_session_summary": "Get recorded shot counts, observation counts, revision and limitations for the selected session.",
        "list_shots": "List recorded shots by observed outcome, with stable pagination. Unknown is distinct from observed absence.",
        "get_shot": "Get one recorded shot using exactly one of shot_id or sequence. Missing is not a missed shot.",
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": descriptions[name],
                "parameters": schema.model_json_schema(),
            },
        }
        for name, schema in ARGS.items()
    ]


async def execute_tool(
    call: ToolCall, context: ToolContext, repository: RepositoryPort
) -> ToolResult:
    def result(status, data=None, refs=(), limitations=()):
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            status=status,
            data=data or {},
            evidence_refs=tuple(refs),
            limitations=tuple(limitations),
        )

    if context.session_id not in context.available_session_ids or call.name not in ARGS:
        return result("denied", limitations=("TOOL_OR_SCOPE_DENIED",))
    try:
        args = ARGS[call.name].model_validate(call.arguments)
    except ValidationError:
        return result("invalid_arguments", limitations=("INVALID_ARGUMENTS",))
    try:
        if call.name == "get_session_summary":
            data = await repository.get_summary(context.session_id)
            if data is None:
                return result("not_found", limitations=("RECORD_NOT_FOUND",))
            partial = bool(
                data["unknown_count"]
                or "incomplete_observations" in data["limitations"]
            )
            return result(
                "partial" if partial else "ok",
                data,
                data["evidence_refs"],
                data["limitations"],
            )
        summary = await repository.get_summary(context.session_id)
        if summary is None:
            return result("not_found", limitations=("RECORD_NOT_FOUND",))
        scope_limits = tuple(summary["limitations"])
        scope_partial = bool(
            summary["unknown_count"] or "incomplete_observations" in scope_limits
        )
        if summary["unknown_count"]:
            scope_limits += ("session_contains_unknown_outcomes",)
        if call.name == "list_shots":
            rows = await repository.list_shots(
                context.session_id,
                outcome=args.outcome,
                offset=args.offset,
                limit=args.limit + 1,
            )
            more = len(rows) > args.limit
            rows = rows[: args.limit]
            data = {
                "shots": [s.model_dump(mode="json") for s in rows],
                "next_offset": args.offset + args.limit if more else None,
            }
            unknown = any(s.outcome == "unknown" for s in rows)
            partial = scope_partial or unknown
            row_limits = tuple(limit for shot in rows for limit in shot.limitations)
            return result(
                "partial" if partial else "ok",
                data,
                [s.evidence_id for s in rows],
                tuple(
                    dict.fromkeys(
                        scope_limits
                        + row_limits
                        + (("outcome_unknown",) if unknown else ())
                    )
                ),
            )
        shot = await repository.get_shot(
            context.session_id, shot_id=args.shot_id, sequence=args.sequence
        )
        if shot is None:
            return result("not_found", limitations=("RECORD_NOT_FOUND",))
        return result(
            "partial" if scope_partial or shot.outcome == "unknown" else "ok",
            shot.model_dump(mode="json"),
            (shot.evidence_id,),
            tuple(dict.fromkeys(scope_limits + shot.limitations)),
        )
    except Exception:
        return result("unavailable", limitations=("TOOL_UNAVAILABLE",))
