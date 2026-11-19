import hashlib

import pytest

from cueorbit_replay.rules.contracts import RuleSourceManifestV1
from cueorbit_replay.rules.corpus import extract_chunks, normalize_text


def manifest():
    return RuleSourceManifestV1(
        source_id="synthetic-source",
        ruleset_id="synthetic-heyball",
        effective_date="2025-08-16",
        source_url="https://wpapool.com/wp-content/uploads/2025/08/250816-Rules-of-Heyball.pdf",
        retrieved_at="2026-11-17T15:00:00Z",
        pdf_sha256="a" * 64,
        scope_sections=("II", "A"),
        extraction_version="test-v1",
        data_kind="synthetic",
    )


class WordCounter:
    def count(self, text):
        return len(text.split())


def test_chapter_scope_cross_page_and_normalization():
    pages = [
        "CHAPTER I: General Principles\n1. Dress code\nNot in scope.\nCHAPTER II: RULES OF HEYBALL\n6. Break shot\nSynthetic first sentence.",
        "Synthetic continued text with a \ufb01rst example.\n7. Re-spotting\nSynthetic spotting.\nCHAPTER III: TIME-LIMITED\n1. Clock\nExcluded clock.",
        "Appendix A\nWPA General Rules\n1. General Rules\nGeneral synthetic text.\n1.6 Cue Ball in Hand\nSynthetic appendix condition.",
    ]
    chunks = extract_chunks(pages, manifest(), WordCounter())
    assert {c.rule_number for c in chunks} == {"II.6", "II.7", "A.1", "A.1.6"}
    first = next(c for c in chunks if c.rule_number == "II.6")
    assert first.page_start == 1 and first.page_end == 2
    assert "first example" in first.text
    assert all("Excluded" not in c.text and "Dress code" not in c.text for c in chunks)
    assert first.precedence == "heyball_specific"
    assert (
        next(c for c in chunks if c.rule_number == "A.1.6").precedence
        == "appendix_general"
    )
    assert first.text_sha256 == hashlib.sha256(first.text.encode()).hexdigest()
    assert extract_chunks(pages, manifest(), WordCounter()) == chunks


def test_long_clause_chunks_do_not_silently_truncate():
    text = " ".join("Synthetic sentence " + str(i) + "." for i in range(250))
    chunks = extract_chunks(
        ["CHAPTER II: RULES OF HEYBALL\n6. Break shot\n" + text],
        manifest(),
        WordCounter(),
    )
    assert len(chunks) > 1
    assert all(WordCounter().count(c.text) <= 400 for c in chunks)
    assert all(
        WordCounter().count("passage: " + c.heading + " " + c.text) <= 512
        for c in chunks
    )
    assert "249." in chunks[-1].text
    assert all(len(c.chunk_id) <= 80 for c in chunks)


def test_extraction_rejects_garbled_text_and_oversize_sentence():
    with pytest.raises(ValueError):
        normalize_text("bad\ufffdtext")
    with pytest.raises(ValueError):
        extract_chunks(
            ["CHAPTER II\n6. Break shot\n" + "word " * 600], manifest(), WordCounter()
        )


def test_cross_reference_inside_excluded_chapter_does_not_change_scope():
    pages = [
        "CHAPTER II\n6. Break shot\nSynthetic allowed rule.\nCHAPTER III\n1. Clock\nThis is the same as Rule 6 of Chapter II: Break Shot\n4. Timed match\nSynthetic excluded rule.\nAppendix A\n1.6 Cue Ball in Hand\nSynthetic allowed appendix."
    ]
    chunks = extract_chunks(pages, manifest(), WordCounter())
    assert {c.rule_number for c in chunks} == {"II.6", "A.1.6"}


def test_parent_heading_without_body_is_metadata_not_search_evidence():
    chunks = extract_chunks(
        [
            "CHAPTER II\n12. Playing Shots\n12.1 Synthetic first contact condition.\n15. Frozen Balls\n15.1 Synthetic frozen condition."
        ],
        manifest(),
        WordCounter(),
    )
    assert {c.rule_number for c in chunks} == {"II.12.1", "II.15.1"}
    assert (
        next(c for c in chunks if c.rule_number == "II.12.1").heading == "Playing Shots"
    )
