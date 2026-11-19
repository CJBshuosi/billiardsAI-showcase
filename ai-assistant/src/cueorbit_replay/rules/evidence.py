"""Only returned rule chunks and verbatim normalized substrings may be cited."""

from .contracts import RuleChunk, RuleCitation, RuleCitationDraft, RulesContext
from .corpus import normalize_text


def validate_chunk_scope(chunk, context):
    chunk = RuleChunk.model_validate_json(chunk.model_dump_json())
    if (chunk.ruleset_id, chunk.version, chunk.embedding_revision) != (
        context.ruleset_id,
        context.version,
        context.embedding_revision,
    ):
        raise ValueError("RULE_SOURCE_SCOPE_MISMATCH")
    return chunk


def hydrate_rule_citations(
    drafts, returned_chunks, context: RulesContext
) -> tuple[RuleCitation, ...]:
    pool = {}
    for value in returned_chunks:
        chunk = validate_chunk_scope(value, context)
        if chunk.chunk_id in pool and pool[chunk.chunk_id] != chunk:
            raise ValueError("CONFLICTING_RULE_SOURCE")
        pool[chunk.chunk_id] = chunk
    citations = []
    for value in drafts:
        draft = (
            RuleCitationDraft.model_validate_json(value.model_dump_json())
            if hasattr(value, "model_dump_json")
            else RuleCitationDraft.model_validate(value)
        )
        chunk = pool.get(draft.chunk_id)
        quote = normalize_text(draft.quote)
        if (
            chunk is None
            or not quote
            or len(quote) > 320
            or quote not in normalize_text(chunk.text)
        ):
            raise ValueError("UNSUPPORTED_RULE_CITATION")
        citations.append(
            RuleCitation(
                chunk_id=chunk.chunk_id,
                rule_number=chunk.rule_number,
                version=chunk.version,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                quote=quote,
                source_url=chunk.source_url,
            )
        )
    return tuple(citations)


def validate_rule_citations(citations, returned_chunks, context) -> bool:
    try:
        parsed = tuple(
            RuleCitation.model_validate_json(c.model_dump_json())
            if hasattr(c, "model_dump_json")
            else RuleCitation.model_validate(c)
            for c in citations
        )
        drafts = [{"chunk_id": c.chunk_id, "quote": c.quote} for c in parsed]
        return parsed == hydrate_rule_citations(drafts, returned_chunks, context)
    except (ValueError, TypeError, KeyError):
        return False


def project_rule_citations(candidates, returned_chunks, context):
    """Verify each entire bounded candidate before projecting a shorter exact quote."""
    from .contracts import RuleCitationCandidate

    pool = {}
    for value in returned_chunks:
        chunk = validate_chunk_scope(value, context)
        if chunk.chunk_id in pool and pool[chunk.chunk_id] != chunk:
            raise ValueError("CONFLICTING_RULE_SOURCE")
        pool[chunk.chunk_id] = chunk
    prepared = []
    full = []
    shortened = False
    for value in candidates:
        candidate = (
            RuleCitationCandidate.model_validate_json(value.model_dump_json())
            if hasattr(value, "model_dump_json")
            else RuleCitationCandidate.model_validate(value)
        )
        chunk = pool.get(candidate.chunk_id)
        quote = normalize_text(candidate.quote)
        if chunk is None or not quote or quote not in normalize_text(chunk.text):
            raise ValueError("UNSUPPORTED_RULE_CITATION")
        full.append(RuleCitationCandidate(chunk_id=candidate.chunk_id, quote=quote))
        excerpt = quote
        if len(excerpt) > 320:
            shortened = True
            excerpt = excerpt[:320]
            boundary = excerpt.rfind(" ")
            if quote[320] == " ":
                excerpt = excerpt.rstrip()
            elif boundary > 0:
                excerpt = excerpt[:boundary]
            else:
                raise ValueError("QUOTE_EXCERPT_NO_WORD_BOUNDARY")
        prepared.append(RuleCitationDraft(chunk_id=candidate.chunk_id, quote=excerpt))
    return (
        hydrate_rule_citations(prepared, returned_chunks, context),
        tuple(full),
        shortened,
    )
