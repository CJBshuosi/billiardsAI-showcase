import hashlib

import pytest

from cueorbit_replay.rules.annotations import (
    AnnotationRegistry,
    SourceAnnotation,
    notices_for_citations,
)
from cueorbit_replay.rules.contracts import RuleAnswerDraft, RuleModelTurn
from tests.rules.unit.test_assistant import Runtime, call
from tests.rules.unit.test_rules_tools import FixtureRules


def fixture_source():
    text = "A clear position. Ambiguous collective wording. Players may request corrections."
    chunk = (
        FixtureRules()
        .chunks[0]
        .model_copy(
            update={
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    )
    ann = SourceAnnotation(
        annotation_id="fixture-ambiguity",
        chunk_id=chunk.chunk_id,
        text_sha256=chunk.text_sha256,
        source_id=chunk.source_id,
        fragment="Ambiguous collective wording.",
        kind="quantity_scope",
        note="Author audit: wording is ambiguous; inspect the original figure.",
    )
    return chunk, AnnotationRegistry(version="fixture-v1", annotations=(ann,))


def test_annotation_exact_span_and_hash_binding_leave_clear_quote_usable():
    chunk, registry = fixture_source()
    assert not notices_for_citations(
        [{"chunk_id": chunk.chunk_id, "quote": "A clear position."}],
        (chunk,),
        registry=registry,
    )
    notices = notices_for_citations(
        [{"chunk_id": chunk.chunk_id, "quote": "Ambiguous collective wording."}],
        (chunk,),
        registry=registry,
    )
    assert len(notices) == 1 and notices[0].author_kind == "ai_editorial_audit"
    assert (
        notices[0].source_url == chunk.source_url
        and notices[0].page_start == chunk.page_start
    )
    wrong = chunk.model_copy(update={"text_sha256": "f" * 64})
    with pytest.raises(ValueError, match="ANNOTATION_SOURCE_MISMATCH"):
        notices_for_citations(
            [{"chunk_id": chunk.chunk_id, "quote": "A clear position."}],
            (wrong,),
            registry=registry,
        )


@pytest.mark.parametrize("ambiguous", [True, False])
async def test_ambiguous_quote_removes_model_expansion_but_clear_source_stays_answered(
    monkeypatch, ambiguous
):
    chunk, registry = fixture_source()
    monkeypatch.setattr(
        "cueorbit_replay.rules.annotations.load_annotations", lambda: registry
    )
    runtime = Runtime()
    runtime.repository.chunks = (chunk,)
    quote = "Ambiguous collective wording." if ambiguous else "A clear position."
    draft = RuleModelTurn(
        answer=RuleAnswerDraft(
            status="answered",
            text="确定是六颗。" if ambiguous else "清晰的位置原文。",
            rule_citations=({"chunk_id": chunk.chunk_id, "quote": quote},),
        )
    )
    from cueorbit_replay.assistant import ask_assistant
    from cueorbit_replay.contracts import UnifiedAskRequest
    from tests.helpers import FixtureRepository
    from tests.rules.unit.test_rules_workflow import ScriptedRulesModel

    result = await ask_assistant(
        UnifiedAskRequest(question="请解释该条款"),
        repository=FixtureRepository(),
        rules_runtime=runtime,
        model=ScriptedRulesModel([call("get_rule", {"rule_number": "II.11"}), draft]),
    )
    if ambiguous:
        assert result.status == "clarification" and "六颗" not in result.answer
        assert (
            "歧义" in result.answer and result.source_annotations[0].fragment == quote
        )
        assert result.rule_citations[0].quote == quote
    else:
        assert (
            result.status == "answered"
            and result.answer == "清晰的位置原文。"
            and not result.source_annotations
        )


def test_orphaned_annotation_is_rejected_before_runtime_readiness():
    from cueorbit_replay.rules.annotations import validate_registry_metadata

    chunk, registry = fixture_source()
    registry = registry.model_copy(update={"corpus_hash": "a" * 64})
    metadata = {
        "active_context": {"corpus_hash": "a" * 64},
        "source": {
            "source_id": chunk.source_id,
            "pdf_sha256": registry.annotations[0].pdf_sha256,
        },
        "chunks": [{"chunk_id": chunk.chunk_id, "text_sha256": chunk.text_sha256}],
    }
    validate_registry_metadata(registry, metadata)
    orphan = registry.model_copy(
        update={
            "annotations": (
                registry.annotations[0].model_copy(update={"chunk_id": "orphaned-id"}),
            )
        }
    )
    with pytest.raises(ValueError, match="ANNOTATION_METADATA_MISMATCH"):
        validate_registry_metadata(orphan, metadata)


def test_duplicate_annotation_identity_is_rejected_instead_of_collapsing_notice():
    from pydantic import ValidationError

    _, registry = fixture_source()
    with pytest.raises(ValidationError):
        AnnotationRegistry.model_validate(
            {
                "version": "duplicate",
                "annotations": [
                    registry.annotations[0].model_dump(),
                    registry.annotations[0].model_dump(),
                ],
            }
        )


async def test_short_excerpt_cannot_hide_ambiguity_at_tail_of_full_candidate(
    monkeypatch,
):
    from cueorbit_replay.assistant import ask_assistant
    from cueorbit_replay.contracts import UnifiedAskRequest
    from cueorbit_replay.rules.contracts import RuleAnswerCandidate
    from tests.helpers import FixtureRepository
    from tests.rules.unit.test_rules_workflow import ScriptedRulesModel

    chunk, registry = fixture_source()
    text = "Clear source context sentence. " * 15 + "Ambiguous collective wording."
    chunk = chunk.model_copy(
        update={"text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
    )
    registry = registry.model_copy(
        update={
            "annotations": (
                registry.annotations[0].model_copy(
                    update={"text_sha256": chunk.text_sha256}
                ),
            )
        }
    )
    monkeypatch.setattr(
        "cueorbit_replay.rules.annotations.load_annotations", lambda: registry
    )
    runtime = Runtime()
    runtime.repository.chunks = (chunk,)
    draft = RuleModelTurn(
        answer=RuleAnswerCandidate(
            status="answered",
            text="确定六颗。",
            rule_citations=({"chunk_id": chunk.chunk_id, "quote": text},),
        )
    )
    result = await ask_assistant(
        UnifiedAskRequest(question="请解释"),
        repository=FixtureRepository(),
        rules_runtime=runtime,
        model=ScriptedRulesModel([call("get_rule", {"rule_number": "II.11"}), draft]),
    )
    assert result.status == "clarification" and "六颗" not in result.answer
    assert result.source_annotations and len(result.rule_citations[0].quote) <= 320
    assert (
        "QUOTE_EXCERPT_SHORTENED" in result.limitations
        and "SOURCE_WORDING_AMBIGUITY" in result.limitations
    )
