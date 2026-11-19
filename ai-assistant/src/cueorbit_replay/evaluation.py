"""Structured regression scoring; this does not grade free-text semantic truth."""

from .contracts import AskResponse
from .evidence import supported_refs


def score_case(case: dict, response: AskResponse, *, latency_ms: float) -> dict:
    reasons = []
    if response.status != case["expected_status"]:
        reasons.append("STATUS_MISMATCH")
    tools = [t.name for t in response.tool_trace]
    if not set(tools) <= set(case["allowed_tools"]) or not set(
        case["required_tools"]
    ) <= set(tools):
        reasons.append("TOOL_MISMATCH")
    if not set(response.evidence_refs) <= set(case["allowed_evidence"]):
        reasons.append("EVIDENCE_SCOPE_MISMATCH")
    if not supported_refs(response.evidence_refs, response.fact_cards):
        reasons.append("UNSUPPORTED_REFERENCE")
    if set(response.evidence_refs) != set(case["expected_evidence"]):
        reasons.append("REFERENCE_MISMATCH")
    actual = {}
    for card in response.fact_cards:
        actual.update(card.data)
        if "shots" in card.data:
            actual["sequences"] = [s["sequence"] for s in card.data["shots"]]
    for key, expected in case["expected_facts"].items():
        if actual.get(key) != expected:
            reasons.append("FACT_MISMATCH")
    return dict(
        id=case["id"],
        language=case["language"],
        category=case["category"],
        result="failed" if reasons else "passed",
        reasons=list(dict.fromkeys(reasons)),
        tool_sequence=tools,
        latency_ms=latency_ms,
        usage=response.usage.model_dump(),
        cost=None,
        response=response.model_dump(mode="json"),
    )


def summarize(rows: list[dict]) -> dict:
    def counts(items):
        return dict(
            total=len(items),
            **{
                status: sum(r["result"] == status for r in items)
                for status in ("passed", "failed", "not_run")
            },
        )

    return {
        **counts(rows),
        "by_language": {
            key: counts([r for r in rows if r["language"] == key])
            for key in sorted({r["language"] for r in rows})
        },
        "by_category": {
            key: counts([r for r in rows if r["category"] == key])
            for key in sorted({r["category"] for r in rows})
        },
    }
