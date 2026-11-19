import hashlib

import pytest
from pydantic import ValidationError

from cueorbit_replay.rules.contracts import (
    RuleAnswerCandidate,
    RuleCitationCandidate,
    RuleCitationDraft,
)
from cueorbit_replay.rules.evidence import (
    project_rule_citations,
    validate_rule_citations,
)
from tests.rules.unit.test_rules_tools import CONTEXT, FixtureRules


def source():
    text = "A supported original sentence with clear conditions. " * 12
    text = text.strip()
    return (
        FixtureRules()
        .chunks[0]
        .model_copy(
            update={
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    )


def test_full_candidate_verified_before_word_boundary_excerpt_and_final_contract():
    chunk = source()
    candidate = RuleCitationCandidate(chunk_id=chunk.chunk_id, quote=chunk.text)
    citations, full, shortened = project_rule_citations((candidate,), (chunk,), CONTEXT)
    assert shortened and len(citations) == 1 and len(citations[0].quote) <= 320
    assert chunk.text.startswith(citations[0].quote) and not citations[
        0
    ].quote.endswith("...")
    assert full[0].quote == chunk.text and validate_rule_citations(
        citations, (chunk,), CONTEXT
    )
    with pytest.raises(ValidationError):
        RuleCitationDraft(chunk_id=chunk.chunk_id, quote=chunk.text)


@pytest.mark.parametrize("bad", ["forged_tail", "joined_gap"])
def test_unsupported_tail_or_cross_span_join_cannot_be_laundered_by_truncation(bad):
    chunk = source()
    quote = (
        chunk.text + " FORGED TAIL"
        if bad == "forged_tail"
        else chunk.text[:340] + " gap invented " + chunk.text[390:]
    )
    with pytest.raises(ValueError, match="UNSUPPORTED_RULE_CITATION"):
        project_rule_citations(
            (RuleCitationCandidate(chunk_id=chunk.chunk_id, quote=quote),),
            (chunk,),
            CONTEXT,
        )


def test_bounded_candidates_unicode_normalization_and_foreign_scope():
    chunk = source()
    quote = chunk.text.replace(" ", "\u00a0")
    out, full, shortened = project_rule_citations(
        (RuleCitationCandidate(chunk_id=chunk.chunk_id, quote=quote),),
        (chunk,),
        CONTEXT,
    )
    assert (
        shortened
        and full[0].quote == chunk.text
        and validate_rule_citations(out, (chunk,), CONTEXT)
    )
    with pytest.raises(ValidationError):
        RuleCitationCandidate(chunk_id=chunk.chunk_id, quote="x" * 2001)
    with pytest.raises(ValueError):
        project_rule_citations(
            (RuleCitationCandidate(chunk_id=chunk.chunk_id, quote=chunk.text),),
            (chunk,),
            CONTEXT.model_copy(update={"version": "2024-01-01"}),
        )
    with pytest.raises(ValidationError):
        RuleAnswerCandidate.model_validate(
            {"status": "wrong", "text": "x", "extra": "forbidden"}
        )


def test_joining_different_returned_chunks_is_not_one_continuous_quote():
    chunk = source()
    other = FixtureRules().chunks[1]
    quote = chunk.text + " " + other.text
    with pytest.raises(ValueError, match="UNSUPPORTED_RULE_CITATION"):
        project_rule_citations(
            (RuleCitationCandidate(chunk_id=chunk.chunk_id, quote=quote),),
            (chunk, other),
            CONTEXT,
        )


def test_internal_candidate_instance_cannot_bypass_strict_draft_limits():
    from cueorbit_replay.rules.contracts import RuleAnswerDraft

    candidate = RuleCitationCandidate(chunk_id="safe-source", quote="x" * 321)
    with pytest.raises(ValidationError):
        RuleCitationDraft.model_validate(candidate)
    with pytest.raises(ValidationError):
        RuleAnswerDraft(status="answered", text="x", rule_citations=(candidate,))
    answer = RuleAnswerCandidate(
        status="answered", text="x", rule_citations=(candidate,)
    )
    with pytest.raises(ValidationError):
        RuleAnswerDraft.model_validate(answer)


def test_excerpt_uses_earlier_word_boundary_instead_of_cutting_a_long_token():
    text = "Short token " + ("x" * 400)
    chunk = source().model_copy(
        update={"text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
    )
    citations, _, shortened = project_rule_citations(
        (RuleCitationCandidate(chunk_id=chunk.chunk_id, quote=text),), (chunk,), CONTEXT
    )
    assert shortened and citations[0].quote == "Short token"


def test_single_overlong_token_has_no_valid_bounded_word_excerpt():
    text = "x" * 400
    chunk = source().model_copy(
        update={"text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
    )
    with pytest.raises(ValueError):
        project_rule_citations(
            (RuleCitationCandidate(chunk_id=chunk.chunk_id, quote=text),),
            (chunk,),
            CONTEXT,
        )
