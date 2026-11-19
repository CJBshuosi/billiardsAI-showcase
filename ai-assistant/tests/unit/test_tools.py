import pytest

from cueorbit_replay.contracts import ToolCall, ToolContext
from cueorbit_replay.tools import execute_tool, tool_schemas

CTX = ToolContext(session_id="demo-complete", available_session_ids=("demo-complete",))


class Tripwire:
    def __getattr__(self, name):
        raise AssertionError("invalid call reached repository")


@pytest.mark.parametrize(
    "name,args,status",
    [
        ("get_shot", {"session_id": "foreign", "sequence": 1}, "invalid_arguments"),
        ("get_shot", {"sequence": "1"}, "invalid_arguments"),
        ("get_shot", {"sequence": True}, "invalid_arguments"),
        ("get_shot", {"shot_id": "complete-1", "sequence": 1}, "invalid_arguments"),
        ("get_shot", {}, "invalid_arguments"),
        ("get_session_summary", {"sql": "select *"}, "invalid_arguments"),
        ("list_shots", {"url": "file:///secret"}, "invalid_arguments"),
        ("list_shots", {"limit": 51}, "invalid_arguments"),
        ("list_shots", {"offset": -1}, "invalid_arguments"),
        ("read_file", {"path": "/etc/passwd"}, "denied"),
    ],
)
async def test_reject_model_scope_and_bad_arguments(name, args, status):
    r = await execute_tool(
        ToolCall(call_id="a", name=name, arguments=args), CTX, Tripwire()
    )
    assert r.status == status
    assert not r.data and not r.evidence_refs


async def test_denied_context():
    context = ToolContext(
        session_id="foreign", available_session_ids=("demo-complete",)
    )
    r = await execute_tool(
        ToolCall(call_id="a", name="get_session_summary", arguments={}),
        context,
        Tripwire(),
    )
    assert r.status == "denied"


def test_only_three_safe_tool_schemas():
    schemas = tool_schemas()
    assert {s["function"]["name"] for s in schemas} == {
        "get_session_summary",
        "list_shots",
        "get_shot",
    }
    assert all(
        s["function"]["parameters"]["additionalProperties"] is False for s in schemas
    )


async def test_filtered_partial_session_preserves_session_and_shot_limits():
    from tests.helpers import FixtureRepository

    repo = FixtureRepository()
    b = repo.bundles["demo-partial"]
    repo.bundles["demo-partial"] = b.model_copy(
        update={
            "shots": (
                b.shots[0].model_copy(update={"limitations": ("occluded_boundary",)}),
                *b.shots[1:],
            )
        }
    )
    context = ToolContext(
        session_id="demo-partial", available_session_ids=("demo-partial",)
    )
    r = await execute_tool(
        ToolCall(
            call_id="a",
            name="list_shots",
            arguments={"outcome": "pocket_observed", "limit": 1},
        ),
        context,
        repo,
    )
    assert r.status == "partial"
    assert "incomplete_observations" in r.limitations
    assert "occluded_boundary" in r.limitations
    r = await execute_tool(
        ToolCall(call_id="b", name="get_shot", arguments={"sequence": 1}), context, repo
    )
    assert r.status == "partial"
    assert "incomplete_observations" in r.limitations
