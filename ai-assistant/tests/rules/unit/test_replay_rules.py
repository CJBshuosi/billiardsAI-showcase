import pytest
from pydantic import ValidationError

from cueorbit_replay.contracts import ToolCall
from cueorbit_replay.rules.contracts import (
    ReplayRulesAskRequest,
    RuleAnswerDraft,
    RuleModelTurn,
    RulesAskResponse,
)
from cueorbit_replay.rules.review import ask_review
from tests.helpers import FixtureRepository
from tests.rules.unit.test_rules_tools import CONTEXT, FixtureRules
from tests.rules.unit.test_rules_workflow import ScriptedRulesModel


def request(question="Can this clip establish a foul?", **changes):
    return ReplayRulesAskRequest(
        question=question,
        ruleset_id=CONTEXT.ruleset_id,
        version=CONTEXT.version,
        session_id="demo-complete",
        selected_shot_id="complete-7",
        **changes,
    )


async def invoke(req, model):
    return await ask_review(
        req,
        context=CONTEXT,
        replay_repository=FixtureRepository(),
        rule_repository=FixtureRules(),
        model=model,
        strategy="keyword",
    )


def turn(name, args, identity="replay-1"):
    return RuleModelTurn(
        tool_calls=(ToolCall(call_id=identity, name=name, arguments=args),)
    )


def answer(ref="ev-complete-7"):
    return RuleModelTurn(
        answer=RuleAnswerDraft(
            status="insufficient_evidence",
            text="Recorded pocket observation alone cannot establish first contact, ball group or cue-ball scratch. No verdict can be established.",
            shot_evidence_refs=(ref,),
        )
    )


class InspectModel(ScriptedRulesModel):
    def __init__(self, turns):
        super().__init__(turns)
        self.schemas = []
        self.required = []

    async def next_turn(self, messages, schemas, *, timeout_s, required_tool=None):
        self.schemas.append(schemas)
        self.required.append(required_tool)
        return await super().next_turn(
            messages, schemas, timeout_s=timeout_s, required_tool=required_tool
        )


async def test_selected_clip_binds_native_first_tool_and_keeps_unknown_conditions():
    model = InspectModel([turn("get_shot", {"shot_id": "complete-7"}), answer()])
    response = await invoke(request(), model)
    assert (
        response.status == "insufficient_evidence"
        and response.shot_evidence_refs == ("ev-complete-7",)
    )
    assert response.replay_fact_cards[0].data["sequence"] == 7
    assert not response.adjudication_supported and not response.rule_citations
    assert model.required[0] == "get_shot"
    schema = next(
        s["function"]["parameters"]
        for s in model.schemas[0]
        if s["function"]["name"] == "get_shot"
    )
    assert schema["properties"]["shot_id"]["enum"] == ["complete-7"]
    assert "first contact" in model.messages[0][0]["content"]


@pytest.mark.parametrize("selected", ["partial-2", "not-found"])
async def test_foreign_or_missing_selection_stops_before_provider(selected):
    model = InspectModel([])
    response = await invoke(
        request().model_copy(update={"selected_shot_id": selected}), model
    )
    assert response.status == "insufficient_evidence" and not model.messages


async def test_deictic_without_selection_is_actionable_clarification():
    model = InspectModel([])
    response = await invoke(
        request().model_copy(update={"selected_shot_id": None}), model
    )
    assert (
        response.status == "clarification"
        and "clip" in response.answer
        and not model.messages
    )


async def test_explicit_shot_and_session_summary_override_selection():
    model = InspectModel([turn("get_shot", {"sequence": 2}), answer("ev-complete-2")])
    response = await invoke(request("Could shot 2 be judged a foul?"), model)
    assert response.replay_fact_cards[0].data["sequence"] == 2
    schema = next(
        s["function"]["parameters"]
        for s in model.schemas[0]
        if s["function"]["name"] == "get_shot"
    )
    assert schema["properties"]["sequence"]["enum"] == [2]
    model = InspectModel([turn("get_session_summary", {}), answer()])
    response = await invoke(request("Summarize this session."), model)
    assert response.replay_fact_cards[0].data["recorded_shot_count"] == 8
    assert model.required[0] is None


async def test_bound_clip_cannot_be_replaced_by_provider_and_shot_refs_stay_separate():
    response = await invoke(
        request(), InspectModel([turn("get_shot", {"shot_id": "complete-2"})])
    )
    assert response.status == "insufficient_evidence" and not response.replay_fact_cards
    response = await invoke(
        request(),
        InspectModel(
            [turn("get_shot", {"shot_id": "complete-7"}), answer("rule-not-a-shot")]
        ),
    )
    assert (
        response.status == "insufficient_evidence"
        and "UNSUPPORTED_SHOT_REFERENCE" in response.limitations
    )


async def test_unknown_outcome_is_not_observed_absence():
    req = request().model_copy(
        update={"session_id": "demo-partial", "selected_shot_id": "partial-2"}
    )
    response = await invoke(
        req,
        InspectModel(
            [turn("get_shot", {"shot_id": "partial-2"}), answer("ev-partial-2")]
        ),
    )
    assert response.replay_fact_cards[0].data["outcome"] == "unknown"
    assert response.adjudication_supported is False
    with pytest.raises(ValidationError):
        RulesAskResponse.model_validate(
            {**response.model_dump(), "adjudication_supported": True}
        )
    with pytest.raises(ValidationError):
        RulesAskResponse.model_validate({**response.model_dump(), "foul_verdict": True})


async def test_bound_replay_question_cannot_answer_from_rules_alone():
    chunk = FixtureRules().chunks[0]
    model = InspectModel(
        [
            turn("get_rule", {"rule_number": "II.11"}),
            RuleModelTurn(
                answer=RuleAnswerDraft(
                    status="answered",
                    text="Unsupported unconditional verdict.",
                    rule_citations=({"chunk_id": chunk.chunk_id, "quote": chunk.text},),
                )
            ),
        ]
    )
    response = await invoke(request(), model)
    assert response.status == "insufficient_evidence"
    assert "MISSING_REPLAY_EVIDENCE" in response.limitations
    assert "unconditional verdict" not in response.answer


async def test_chinese_rule_number_does_not_force_replay_lookup():
    chunk = FixtureRules().chunks[0]
    model = InspectModel(
        [
            turn("get_rule", {"rule_number": "II.11"}),
            RuleModelTurn(
                answer=RuleAnswerDraft(
                    status="answered",
                    text="A rule explanation.",
                    rule_citations=({"chunk_id": chunk.chunk_id, "quote": chunk.text},),
                )
            ),
        ]
    )
    response = await invoke(request("请解释第11条规则（II.11）。"), model)
    assert response.status == "answered" and not response.replay_fact_cards
    assert model.required[0] is None


async def test_legacy_nonempty_replay_only_contract_remains_compatible():
    model = InspectModel(
        [
            turn("get_shot", {"shot_id": "complete-7"}),
            RuleModelTurn(
                answer=RuleAnswerDraft(
                    status="answered",
                    text="A claim about the applicable rule.",
                    shot_evidence_refs=("ev-complete-7",),
                )
            ),
        ]
    )
    response = await invoke(request("Explain which rule applies to this shot"), model)
    assert response.status == "answered" and response.shot_evidence_refs == (
        "ev-complete-7",
    )
