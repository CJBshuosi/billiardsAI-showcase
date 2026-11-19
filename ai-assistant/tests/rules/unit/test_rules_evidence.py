import pytest

from cueorbit_replay.rules.evidence import (
    hydrate_rule_citations,
    validate_rule_citations,
)
from tests.rules.unit.test_rules_tools import CONTEXT, FixtureRules


def test_exact_quote_hydrates_canonical_metadata_and_not_global_db_membership():
    chunks = FixtureRules().chunks
    chunk = chunks[0]
    citations = hydrate_rule_citations(
        [{"chunk_id": chunk.chunk_id, "quote": "Synthetic specific exception."}],
        (chunk,),
        CONTEXT,
    )
    assert citations[0].version == "2025-08-16" and citations[0].rule_number == "II.11"
    assert citations[0].page_start == 1
    assert validate_rule_citations(citations, (chunk,), CONTEXT)
    assert not validate_rule_citations(citations, (chunks[1],), CONTEXT)


@pytest.mark.parametrize(
    "change",
    [
        {"version": "2024-01-01"},
        {"page_start": 9},
        {"page_end": 9},
        {"source_url": "https://forged.invalid"},
        {"quote": "Synthetic altered exception."},
        {"chunk_id": "other"},
        {"rule_number": "A.1.7"},
    ],
)
def test_forged_quote_or_source_metadata_fails(change):
    chunk = FixtureRules().chunks[0]
    citation = hydrate_rule_citations(
        [{"chunk_id": chunk.chunk_id, "quote": "Synthetic specific exception."}],
        (chunk,),
        CONTEXT,
    )[0]
    assert not validate_rule_citations(
        [citation.model_copy(update=change)], (chunk,), CONTEXT
    )


def test_ellipsized_or_oversized_quote_rejected():
    chunk = FixtureRules().chunks[0]
    for quote in ("Synthetic ... exception.", "x" * 321):
        with pytest.raises(ValueError):
            hydrate_rule_citations(
                [{"chunk_id": chunk.chunk_id, "quote": quote}], (chunk,), CONTEXT
            )
