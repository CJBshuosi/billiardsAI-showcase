"""Strict native response decoding with content-free failure categories."""

import json

from pydantic import ValidationError

from ..contracts import ToolCall, Usage, ValidationDiagnostic
from ..model import ModelError


def decode_turn(data, draft_type, turn_type):
    try:
        choice = data["choices"][0]
        message = choice["message"]
        if not isinstance(message, dict):
            raise TypeError()
    except (KeyError, IndexError, TypeError):
        raise ModelError("MODEL_ENVELOPE_INVALID") from None
    if choice.get("finish_reason") == "length":
        raise ModelError("MODEL_OUTPUT_TRUNCATED")
    raw_usage = data.get("usage") or {}
    try:
        usage = Usage(
            input_tokens=raw_usage.get("prompt_tokens"),
            output_tokens=raw_usage.get("completion_tokens"),
        )
    except (AttributeError, ValidationError):
        raise ModelError("MODEL_USAGE_INVALID") from None
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise ModelError("MODEL_TOOL_CALLS_INVALID")
    calls = []
    for call in raw_calls:
        try:
            function = call["function"]
            raw_args = function["arguments"]
            if not isinstance(raw_args, str):
                raise ValueError()
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise ModelError("MODEL_TOOL_ARGUMENTS_INVALID") from None
        try:
            calls.append(
                ToolCall(call_id=call["id"], name=function["name"], arguments=args)
            )
        except (KeyError, TypeError, ValueError):
            raise ModelError("MODEL_TOOL_CALLS_INVALID") from None
    if calls:
        return turn_type(tool_calls=tuple(calls), usage=usage)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ModelError("MODEL_EMPTY_CONTENT")
    try:
        parsed = json.loads(content)
    except ValueError:
        raise ModelError("MODEL_FINAL_JSON_INVALID") from None
    try:
        answer = draft_type.model_validate(parsed)
    except ValidationError as error:
        raise ModelError(
            "MODEL_FINAL_CONTRACT_INVALID",
            diagnostics=safe_validation_diagnostics(error),
        ) from None
    return turn_type(answer=answer, usage=usage)


def safe_validation_diagnostics(error):
    fields = {
        "status",
        "text",
        "rule_citations",
        "shot_evidence_refs",
        "evidence_refs",
        "chunk_id",
        "quote",
    }
    types = set(ValidationDiagnostic.model_fields["type"].annotation.__args__)
    diagnostics = []
    for item in error.errors(
        include_input=False, include_context=False, include_url=False
    )[:8]:
        parts = []
        for value in item["loc"][:8]:
            parts.append(
                str(value)
                if type(value) is int and 0 <= value <= 999
                else value
                if type(value) is str and value in fields
                else "extra"
            )
        path = ".".join(parts) or "root"
        if len(path) > 160:
            path = "root"
        diagnostics.append(
            ValidationDiagnostic(
                path=path,
                type=item["type"] if item["type"] in types else "validation_error",
            )
        )
    return tuple(diagnostics)
