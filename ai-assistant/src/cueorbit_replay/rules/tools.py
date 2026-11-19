"""Read-only rule tools; models cannot choose corpus/version/source locations."""

import re
from typing import Annotated

from pydantic import Field, ValidationError, field_validator

from ..contracts import Contract, Positive, ToolCall, ToolResult
from .annotations import annotation_notices
from .contracts import RuleNumber, RulesContext
from .evidence import validate_chunk_scope
from .query import expand_query, glossary


class SearchArgs(Contract):
    query: Annotated[str, Field(min_length=1, max_length=1000)]
    top_k: Annotated[Positive, Field(le=5)] = 5

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("EMPTY_QUERY")
        return value


class RuleArgs(Contract):
    rule_number: RuleNumber


ARGS = {"search_rules": SearchArgs, "get_rule": RuleArgs}
PAIRS = {
    "A.1.4": "II.3",
    "A.1.6": "II.9",
    "A.1.7": "II.11",
    "A.1.8": "II.14",
    "A.1.10": "II.22",
    "A.1.12": "II.23",
    "A.1.13": "II.21",
    "A.3.9": "II.17",
    "A.3.10": "II.8",
    "A.3.11": "II.8",
    "A.3.16": "II.24",
}


def rules_tool_schemas() -> list[dict]:
    descriptions = {
        "search_rules": "Search the server-selected Heyball edition. top_k limits primary hits; linked specific/general exceptions may be returned separately. Similarity is not confidence. No other ruleset/version/URL is accepted.",
        "get_rule": "Get a chapter-qualified rule such as II.6 or A.1.6 and its child clauses. Known specific/general counterparts are separate evidence with explicit precedence.",
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": descriptions[name],
                "parameters": args.model_json_schema(),
            },
        }
        for name, args in ARGS.items()
    ]


async def execute_rules_tool(
    call: ToolCall,
    context: RulesContext,
    repository,
    *,
    encoder=None,
    strategy="vector_glossary_v1",
) -> ToolResult:
    def result(status, data=None, limits=()):
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            status=status,
            data=data or {},
            limitations=tuple(limits),
        )

    if call.name not in ARGS:
        return result("denied", limits=("RULE_TOOL_DENIED",))
    try:
        args = ARGS[call.name].model_validate(call.arguments)
    except ValidationError:
        return result("invalid_arguments", limits=("INVALID_RULE_ARGUMENTS",))
    try:
        executed_query = None
        if call.name == "get_rule":
            primary = list(await repository.get_rule(args.rule_number, context))
            used_strategy = "exact"
        else:
            query = args.query
            executed_query = (
                expand_query(query) if strategy == "vector_glossary_v1" else query
            )
            if strategy == "keyword" or (
                strategy == "vector_glossary_v1"
                and re.search(
                    r"\b(?:II|A)\.[1-9][0-9]*(?:\.[1-9][0-9]*)*\b", query, re.IGNORECASE
                )
            ):
                hits = await repository.search_keyword(query, context, args.top_k)
            elif strategy in ("vector", "vector_glossary_v1"):
                if encoder is None:
                    return result("unavailable", limits=("EMBEDDING_UNAVAILABLE",))
                vector = await encoder.embed_query(executed_query)
                hits = await repository.search_vector(vector, context, args.top_k)
            else:
                raise ValueError("UNKNOWN_RETRIEVAL_STRATEGY")
            primary = [h.chunk for h in hits]
            used_strategy = strategy
        primary = [validate_chunk_scope(c, context) for c in primary]
        if not primary:
            return result("not_found", limits=("NO_RULE_MATCH",))
        related = {}
        have = {c.chunk_id for c in primary}
        related_numbers = set()
        for chunk in primary:
            if chunk.rule_number in PAIRS:
                related_numbers.add(PAIRS[chunk.rule_number])
            related_numbers.update(
                a for a, b in PAIRS.items() if b == chunk.rule_number
            )
        for number in sorted(related_numbers):
            for value in await repository.get_rule(number, context):
                chunk = validate_chunk_scope(value, context)
                if chunk.chunk_id not in have:
                    related[chunk.chunk_id] = chunk
        return result(
            "ok",
            {
                "chunks": [c.model_dump(mode="json") for c in primary],
                "source_annotations": [
                    n.model_dump(mode="json")
                    for n in annotation_notices(
                        tuple(primary) + tuple(related.values())
                    )
                ],
                "related_chunks": [c.model_dump(mode="json") for c in related.values()],
                "ruleset_id": context.ruleset_id,
                "version": context.version,
                "corpus_hash": context.corpus_hash,
                "precedence": "heyball_specific_over_appendix_general",
                "retrieval_strategy": used_strategy,
                "executed_query": executed_query,
                "query_expansion_version": glossary()["version"]
                if strategy == "vector_glossary_v1"
                else None,
            },
        )
    except TimeoutError:
        raise
    except ValueError as error:
        if str(error) == "RULE_SOURCE_SCOPE_MISMATCH":
            return result("denied", limits=("RULE_SOURCE_SCOPE_MISMATCH",))
        return result("invalid_arguments", limits=("QUERY_OR_VECTOR_INVALID",))
    except Exception:
        return result("unavailable", limits=("RULE_STORAGE_UNAVAILABLE",))
