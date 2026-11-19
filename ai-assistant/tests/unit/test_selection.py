import json

import pytest

from cueorbit_replay.contracts import AnswerDraft, AskRequest, ModelTurn
from cueorbit_replay.providers.offline import OfflineModel
from cueorbit_replay.workflow import ask
from tests.helpers import FixtureRepository, ScriptedModel


@pytest.mark.parametrize(
    "question,tool,refs",
    [
        ("这一杆发生了什么？", "get_shot", ("ev-complete-7",)),
        ("What happened in this shot?", "get_shot", ("ev-complete-7",)),
        ("What happened on shot 2?", "get_shot", ("ev-complete-2",)),
        (
            "What happened in this session?",
            "get_session_summary",
            tuple(f"ev-complete-{i}" for i in range(1, 9)),
        ),
        (
            "What happened in this shots",
            "get_session_summary",
            tuple(f"ev-complete-{i}" for i in range(1, 9)),
        ),
        (
            "这些击球发生了什么？",
            "get_session_summary",
            tuple(f"ev-complete-{i}" for i in range(1, 9)),
        ),
    ],
)
async def test_selection_resolves_deictic_but_preserves_explicit_scope(
    question, tool, refs
):
    response = await ask(
        AskRequest(
            session_id="demo-complete", question=question, selected_shot_id="complete-7"
        ),
        repository=FixtureRepository(),
        model=OfflineModel(),
    )
    assert response.status == "answered"
    assert [t.name for t in response.tool_trace] == [tool]
    assert response.evidence_refs == refs


@pytest.mark.parametrize("selected", ["partial-1", "missing"])
async def test_foreign_or_missing_selection_stops_before_model(selected):
    model = ScriptedModel([])
    response = await ask(
        AskRequest(
            session_id="demo-complete", question="this shot", selected_shot_id=selected
        ),
        repository=FixtureRepository(),
        model=model,
    )
    assert response.status == "insufficient_evidence"
    assert "SELECTED_SHOT_NOT_FOUND" in response.limitations
    assert not model.messages and not response.fact_cards and not response.evidence_refs


async def test_selection_context_is_locator_only_and_not_evidence():
    model = ScriptedModel(
        [
            ModelTurn(
                answer=AnswerDraft(
                    status="answered",
                    text="Invented fact",
                    evidence_refs=("ev-complete-7",),
                )
            )
        ]
    )
    response = await ask(
        AskRequest(
            session_id="demo-complete",
            question="this shot",
            selected_shot_id="complete-7",
        ),
        repository=FixtureRepository(),
        model=model,
    )
    systems = [m["content"] for m in model.messages[0] if m["role"] == "system"]
    assert len(systems) == 1
    context = systems[0].split("\nSelected replay context: ", 1)[1]
    assert json.loads(context) == {
        "shot_id": "complete-7",
        "sequence": 7,
    }
    assert response.status == "insufficient_evidence"
    assert "UNSUPPORTED_REFERENCE" in response.limitations


async def test_unselected_chinese_question_invites_clicking_a_clip():
    response = await ask(
        AskRequest(session_id="demo-complete", question="这一杆发生了什么？"),
        repository=FixtureRepository(),
        model=OfflineModel(),
    )
    assert response.status == "clarification"
    assert "片段" in response.answer and "点击" in response.answer
    assert not response.tool_trace


async def test_unselected_current_clip_is_resolved_before_provider_access():
    model = ScriptedModel([])
    response = await ask(
        AskRequest(session_id="demo-complete", question="这一杆发生了什么？"),
        repository=FixtureRepository(),
        model=model,
    )
    assert response.status == "clarification"
    assert not model.messages


async def test_selected_current_clip_requires_native_lookup_of_bound_id():
    from cueorbit_replay.contracts import ToolCall

    class NativeSelectionModel:
        async def next_turn(self, messages, schemas, *, timeout_s, required_tool=None):
            if not any(m["role"] == "tool" for m in messages):
                assert required_tool == "get_shot"
                assert [s["function"]["name"] for s in schemas] == ["get_shot"]
                assert schemas[0]["function"]["parameters"]["properties"]["shot_id"][
                    "enum"
                ] == ["complete-7"]
                return ModelTurn(
                    tool_calls=(
                        ToolCall(
                            call_id="selected",
                            name="get_shot",
                            arguments={"shot_id": "complete-7"},
                        ),
                    )
                )
            assert required_tool is None
            return ModelTurn(
                answer=AnswerDraft(
                    status="answered",
                    text="Pocket observed.",
                    evidence_refs=("ev-complete-7",),
                )
            )

    response = await ask(
        AskRequest(
            session_id="demo-complete",
            question="这一杆发生了什么？",
            selected_shot_id="complete-7",
        ),
        repository=FixtureRepository(),
        model=NativeSelectionModel(),
    )
    assert response.status == "answered"
    assert response.fact_cards[0].data["shot_id"] == "complete-7"


async def test_selected_native_lookup_cannot_query_another_shot():
    from cueorbit_replay.contracts import ToolCall

    class WrongSelectionModel:
        def __init__(self):
            self.round = 0

        async def next_turn(self, messages, schemas, *, timeout_s, required_tool=None):
            self.round += 1
            if self.round == 1:
                return ModelTurn(
                    tool_calls=(
                        ToolCall(
                            call_id="wrong",
                            name="get_shot",
                            arguments={"shot_id": "complete-2"},
                        ),
                    )
                )
            return ModelTurn(
                answer=AnswerDraft(status="insufficient_evidence", text="No evidence.")
            )

    response = await ask(
        AskRequest(
            session_id="demo-complete",
            question="this shot",
            selected_shot_id="complete-7",
        ),
        repository=FixtureRepository(),
        model=WrongSelectionModel(),
    )
    assert not response.fact_cards
    assert response.tool_trace[0].status == "invalid_arguments"


@pytest.mark.parametrize(
    "question,status,tool,refs",
    [
        (
            "What happened in this shot (shot_id: complete-2)?",
            "answered",
            "get_shot",
            ("ev-complete-2",),
        ),
        (
            "What happened in this shot (shot_id: partial-1)?",
            "insufficient_evidence",
            "get_shot",
            (),
        ),
        (
            "Show recorded shots, including this shot.",
            "answered",
            "list_shots",
            tuple(f"ev-complete-{i}" for i in range(1, 9)),
        ),
        (
            "Show shots with no pocket observed, including this shot.",
            "answered",
            "list_shots",
            (
                "ev-complete-1",
                "ev-complete-3",
                "ev-complete-5",
                "ev-complete-6",
                "ev-complete-8",
            ),
        ),
        ("this clip", "answered", "get_shot", ("ev-complete-7",)),
        ("What happened in this clip?", "answered", "get_shot", ("ev-complete-7",)),
    ],
)
async def test_review_explicit_scope_and_clip_wording(question, status, tool, refs):
    response = await ask(
        AskRequest(
            session_id="demo-complete", question=question, selected_shot_id="complete-7"
        ),
        repository=FixtureRepository(),
        model=OfflineModel(),
    )
    assert response.status == status
    assert [entry.name for entry in response.tool_trace] == [tool]
    assert response.evidence_refs == refs
