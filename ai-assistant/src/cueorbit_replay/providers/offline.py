"""Deterministic script for demos/regression. Not an LLM or quality benchmark."""

import json
import re

from ..contracts import AnswerDraft, ModelTurn, ToolCall


class OfflineModel:
    async def next_turn(self, messages, tool_schemas, *, timeout_s, required_tool=None):
        original_question = next(m["content"] for m in messages if m["role"] == "user")
        question = original_question.lower()
        selected = next(
            (
                json.loads(line.removeprefix("Selected replay context: "))
                for m in messages
                if m["role"] == "system"
                for line in m["content"].splitlines()
                if line.startswith("Selected replay context: ")
            ),
            None,
        )
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]

        def answer(status, text, refs=()):
            return ModelTurn(
                answer=AnswerDraft(status=status, text=text, evidence_refs=tuple(refs))
            )

        if any(
            t in question for t in ("rule", "foul", "规则", "犯规", "train", "训练")
        ):
            return answer(
                "unsupported",
                "M1 reviews recorded observations. Rules, foul judgments and training are outside this version.",
            )
        if not results:
            cause = any(t in question for t in ("why", "为什么", "为何"))
            locator = re.search(r"(?:shot\s*|第\s*)(\d+)", question)
            shot_id = re.search(
                r"\bshot[_ ]id\s*[:=]?\s*([a-z0-9][a-z0-9_-]{0,79})\b",
                original_question,
                re.IGNORECASE,
            )
            if shot_id is None:
                shot_id = re.search(
                    r"\bshot\s+([a-z0-9]+(?:[-_][a-z0-9]+)+)\b",
                    original_question,
                    re.IGNORECASE,
                )
            pockets = any(t in question for t in ("pocket", "进球", "落袋"))
            list_requested = bool(
                re.search(r"\b(?:list|show)\b.*\bshots\b", question)
            ) or any(t in question for t in ("recorded shots", "所有单杆", "列出击球"))
            deictic = bool(
                re.search(
                    r"\b(?:this|that)\s+(?:shot|clip|one)\b(?!s)|这一杆|这杆|那一杆|当前片段|这个片段",
                    question,
                )
            )
            summary = any(
                t in question
                for t in (
                    "session",
                    "summary",
                    "happened",
                    "场次",
                    "总结",
                    "概览",
                    "这些击球",
                )
            )
            filter_args = {}
            if any(t in question for t in ("unknown", "未知", "不确定")):
                filter_args = {"outcome": "unknown"}
            elif pockets:
                negative = any(
                    t in question
                    for t in (
                        "no pocket",
                        "not pocket",
                        "without",
                        "didn't",
                        "没有",
                        "未进",
                        "未落",
                        "无进",
                        "无落",
                        "no_pocket",
                    )
                )
                filter_args = {
                    "outcome": "no_pocket_observed" if negative else "pocket_observed"
                }
            if required_tool == "get_shot" and selected:
                name, args = "get_shot", {"shot_id": selected["shot_id"]}
            elif cause:
                name, args = "get_session_summary", {}
            elif list_requested:
                name, args = "list_shots", filter_args
            elif shot_id:
                name, args = "get_shot", {"shot_id": shot_id.group(1)}
            elif locator:
                name, args = "get_shot", {"sequence": int(locator.group(1))}
            elif any(
                t in question for t in ("session", "summary", "场次", "总结", "概览")
            ):
                name, args = "get_session_summary", {}
            elif filter_args:
                name, args = "list_shots", filter_args
            elif deictic:
                if selected:
                    name, args = "get_shot", {"shot_id": selected["shot_id"]}
                else:
                    return answer(
                        "clarification",
                        "请点击一个击球片段，再问这一杆发生了什么。"
                        if re.search(r"[\u4e00-\u9fff]", question)
                        else "Click a clip in the list, then ask about this shot.",
                    )
            elif summary:
                name, args = "get_session_summary", {}
            else:
                return answer(
                    "clarification",
                    "请点击一个击球片段，或询问整场概览。"
                    if re.search(r"[\u4e00-\u9fff]", question)
                    else "Click a clip in the list, or ask for a session summary.",
                )
            return ModelTurn(
                tool_calls=(ToolCall(call_id="offline-1", name=name, arguments=args),)
            )
        result = results[-1]
        refs = result["evidence_refs"]
        if result["status"] not in ("ok", "partial"):
            return answer(
                "insufficient_evidence",
                "The requested record is unavailable. This does not establish that a shot was missed.",
            )
        if any(t in question for t in ("why", "为什么", "为何")):
            return answer(
                "insufficient_evidence",
                "The records cannot establish why a shot missed. Technique, spin and player intent were not observed.",
                refs,
            )
        data = result["data"]
        if result["name"] == "get_session_summary":
            text = f"Recorded {data['recorded_shot_count']} shots: {data['pocket_observed_count']} with pocket observations, {data['no_pocket_observed_count']} with no pocket observed, and {data['unknown_count']} unknown. These are annotations, not player scores or ball totals."
        elif result["name"] == "list_shots":
            sequences = ", ".join(str(s["sequence"]) for s in data["shots"]) or "none"
            text = f"Shots with the selected observation: {sequences}. Only recorded observations are shown."
        else:
            text = f"Shot {data['sequence']}: {data['outcome'].replace('_', ' ')}. Evidence covers {data['start_ms']}–{data['end_ms']} ms; no technical cause is established."
        return answer(
            "partial" if result["status"] == "partial" else "answered", text, refs
        )
