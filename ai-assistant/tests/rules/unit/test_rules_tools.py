import pytest

from cueorbit_replay.contracts import ToolCall
from cueorbit_replay.rules.contracts import RuleHit, RulesContext
from cueorbit_replay.rules.corpus import extract_chunks
from cueorbit_replay.rules.tools import execute_rules_tool
from tests.rules.unit.test_rules_corpus import WordCounter, manifest

CONTEXT = RulesContext(
    ruleset_id="synthetic-heyball", version="2025-08-16", corpus_hash="a" * 64
)


class FixtureRules:
    def __init__(self):
        self.calls = []
        self.chunks = extract_chunks(
            [
                "CHAPTER II\n11. No call\nSynthetic specific exception.\nAppendix A\n1.7 Standard call\nSynthetic general condition."
            ],
            manifest(),
            WordCounter(),
        )

    async def search_keyword(self, query, context, top_k):
        self.calls.append(("search", query, context))
        return tuple(
            RuleHit(chunk=c, score=1.0, method="keyword")
            for c in self.chunks
            if c.rule_number == "A.1.7"
        )

    async def get_rule(self, number, context):
        self.calls.append(("get", number, context))
        return tuple(c for c in self.chunks if c.rule_number == number)


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "x", "version": "2024"},
        {"query": "x", "ruleset_id": "other"},
        {"query": "x", "url": "https://example.invalid"},
        {"query": "x", "top_k": 6},
        {"query": "x", "top_k": True},
        {"query": " "},
        {"query": "x" * 1001},
    ],
)
async def test_model_cannot_choose_scope_or_invalid_search_args(arguments):
    repository = FixtureRules()
    result = await execute_rules_tool(
        ToolCall(call_id="a", name="search_rules", arguments=arguments),
        CONTEXT,
        repository,
        strategy="keyword",
    )
    assert result.status == "invalid_arguments" and not repository.calls


async def test_appendix_and_specific_exception_remain_distinct_evidence():
    repository = FixtureRules()
    result = await execute_rules_tool(
        ToolCall(
            call_id="a",
            name="search_rules",
            arguments={"query": "call shot", "top_k": 1},
        ),
        CONTEXT,
        repository,
        strategy="keyword",
    )
    assert result.status == "ok" and not result.evidence_refs
    assert result.data["chunks"][0]["rule_number"] == "A.1.7"
    assert result.data["related_chunks"][0]["rule_number"] == "II.11"
    assert result.data["related_chunks"][0]["precedence"] == "heyball_specific"
    assert result.data["precedence"] == "heyball_specific_over_appendix_general"
    assert result.data["corpus_hash"] == CONTEXT.corpus_hash


async def test_unknown_tool_and_unqualified_locator_are_rejected():
    repository = FixtureRules()
    for name, args, status in [
        ("execute_sql", {}, "denied"),
        ("get_rule", {"rule_number": "6"}, "invalid_arguments"),
        ("get_rule", {"rule_number": "II.99"}, "not_found"),
    ]:
        result = await execute_rules_tool(
            ToolCall(call_id="a", name=name, arguments=args),
            CONTEXT,
            repository,
            strategy="keyword",
        )
        assert result.status == status


async def test_repository_returning_another_version_is_denied():
    repository = FixtureRules()
    wrong = repository.chunks[0].model_copy(update={"version": "2024-01-01"})

    async def foreign(*args):
        return (wrong,)

    repository.get_rule = foreign
    result = await execute_rules_tool(
        ToolCall(call_id="a", name="get_rule", arguments={"rule_number": "II.11"}),
        CONTEXT,
        repository,
    )
    assert result.status == "denied" and not result.data
