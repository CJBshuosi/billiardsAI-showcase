"""Scripted offline preview; live uses the same native rule/replay draft protocol."""

import json
import re

from .contracts import ToolCall
from .language import is_chinese
from .providers.offline import OfflineModel
from .rules.contracts import RuleAnswerDraft, RuleModelTurn
from .rules.model import OfflineRulesModel
from .rules.review import target_for


class OfflineAssistantModel:
    async def next_turn(self, messages, tool_schemas, *, timeout_s, required_tool=None):
        question = next(m["content"] for m in messages if m["role"] == "user")
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        names = {s["function"]["name"] for s in tool_schemas}
        replay = [
            r
            for r in results
            if r["name"] in ("get_shot", "list_shots", "get_session_summary")
        ]
        rule = [r for r in results if r["name"] in ("get_rule", "search_rules")]
        kind, value = target_for(question)
        mixed = (
            any(
                t in question.lower()
                for t in ("foul", "rule", "legal", "犯规", "规则", "合法")
            )
            and kind is not None
        )
        if not results and kind is not None:
            if not names.intersection(
                {"get_shot", "get_session_summary", "list_shots"}
            ):
                return RuleModelTurn(
                    answer=RuleAnswerDraft(
                        status="clarification",
                        text="Choose a session for recorded observations.",
                    )
                )
            if kind == "session":
                name = (
                    "list_shots"
                    if re.search(r"list|show.*shots|列出|所有", question, re.I)
                    else "get_session_summary"
                )
                args = {}
            else:
                name = "get_shot"
                schema = next(
                    s["function"]["parameters"]
                    for s in tool_schemas
                    if s["function"]["name"] == name
                )
                if any("enum" in v for v in schema["properties"].values()):
                    args = {
                        k: v["enum"][0]
                        for k, v in schema["properties"].items()
                        if "enum" in v
                    }
                else:
                    args = {kind: value}
            return RuleModelTurn(
                tool_calls=(
                    ToolCall(call_id="offline-replay-1", name=name, arguments=args),
                )
            )
        if replay and mixed:
            if not rule and "search_rules" in names:
                return RuleModelTurn(
                    tool_calls=(
                        ToolCall(
                            call_id="offline-rule-1",
                            name="search_rules",
                            arguments={
                                "query": "foul first contact cue ball scratch",
                                "top_k": 5,
                            },
                        ),
                    )
                )
            chunk = next((c for r in rule for c in r["data"].get("chunks", [])), None)
            return RuleModelTurn(
                answer=RuleAnswerDraft(
                    status="insufficient_evidence",
                    text="Scripted evidence preview: first contact, ball group and cue-ball status are missing. No automatic verdict is supported.",
                    shot_evidence_refs=tuple(
                        i for r in replay for i in r["evidence_refs"]
                    ),
                    rule_citations=(
                        {"chunk_id": chunk["chunk_id"], "quote": chunk["text"][:160]},
                    )
                    if chunk
                    else (),
                )
            )
        if replay:
            turn = await OfflineModel().next_turn(
                messages, tool_schemas, timeout_s=timeout_s, required_tool=required_tool
            )
            return RuleModelTurn(
                answer=RuleAnswerDraft(
                    status=turn.answer.status,
                    text=scripted_text(question, replay[-1])
                    if is_chinese(question)
                    else turn.answer.text,
                    shot_evidence_refs=turn.answer.evidence_refs,
                )
            )
        if "search_rules" in names:
            turn = await OfflineRulesModel().next_turn(
                messages, tool_schemas, timeout_s=timeout_s, required_tool=required_tool
            )
            if (
                turn.answer
                and turn.answer.status == "answered"
                and is_chinese(question)
            ):
                turn = turn.model_copy(
                    update={
                        "answer": turn.answer.model_copy(
                            update={
                                "text": "离线脚本仅展示已检索的规则原文，请对照引文与页码；这不是真实模型的规则解释。"
                            }
                        )
                    }
                )
            return turn
        return RuleModelTurn(
            answer=RuleAnswerDraft(
                status="unsupported",
                text="Rules are unavailable. Select a session to ask about its recorded observations.",
            )
        )


def scripted_text(question, result):
    data = result["data"]
    if "recorded_shot_count" in data:
        return f"离线脚本记录概览：共{data['recorded_shot_count']}杆，其中{data['pocket_observed_count']}杆观察到落袋、{data['no_pocket_observed_count']}杆未观察到落袋、{data['unknown_count']}杆未知。这些标注不能当作得分或球数。"
    if "shot_id" in data:
        outcome = {
            "pocket_observed": "观察到落袋",
            "no_pocket_observed": "未观察到落袋",
            "unknown": "观测未知",
        }[data["outcome"]]
        return f"离线脚本记录：第{data['sequence']}杆，{outcome}；证据范围{data['start_ms']}–{data['end_ms']}毫秒，不能据此判断技术原因。"
    return "离线脚本只展示本次查询到的击球记录，请查看事实卡；未知观测与没有落袋不同。"
