"""Rules JSON contract over the same bounded native transport as replay."""

import json
import re

import httpx

from ..contracts import ToolCall
from ..model import ModelError
from ..providers.decoding import decode_turn
from ..providers.native import NativeChatTransport
from .contracts import RuleAnswerCandidate, RuleAnswerDraft, RuleModelTurn


class RulesDeepSeekModel:
    def __init__(self, api_key, model_name, *, client=None):
        self._transport = NativeChatTransport(api_key, model_name, client=client)

    async def next_turn(self, messages, tool_schemas, *, timeout_s, required_tool=None):
        try:
            data = await self._transport.complete(
                messages, tool_schemas, timeout_s=timeout_s, required_tool=required_tool
            )
            return decode_turn(data, RuleAnswerCandidate, RuleModelTurn)
        except (httpx.TimeoutException, TimeoutError):
            raise TimeoutError("MODEL_TIMEOUT") from None
        except ModelError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelError(
                "PROVIDER_NETWORK_ERROR"
                if isinstance(error, httpx.HTTPError)
                else "MODEL_ENVELOPE_INVALID"
            ) from None


class OfflineRulesModel:
    """Scripted source preview, never real generative answer performance."""

    async def next_turn(self, messages, tool_schemas, *, timeout_s, required_tool=None):
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        if not results:
            question = next(m["content"] for m in messages if m["role"] == "user")
            if len(question) > 1000:
                return RuleModelTurn(
                    answer=RuleAnswerDraft(
                        status="clarification",
                        text="Shorten the question to 1000 characters for this scripted source preview.",
                    )
                )
            if any(
                t in question.lower()
                for t in (
                    "dress",
                    "waistcoat",
                    "time-limited",
                    "drop table",
                    "american eight",
                    "\u670d\u88c5",
                    "\u9650\u65f6",
                    "\u7f8e\u5f0f",
                )
            ):
                return RuleModelTurn(
                    answer=RuleAnswerDraft(
                        status="unsupported",
                        text="This scripted mode previews the registered Heyball scope; this request is outside that scope.",
                    )
                )
            locator = re.search(
                r"\b(?:II|A)\.[1-9][0-9]*(?:\.[1-9][0-9]*)*\b", question, re.IGNORECASE
            )
            name, args = (
                ("get_rule", {"rule_number": locator.group().upper()})
                if locator
                else ("search_rules", {"query": question, "top_k": 5})
            )
            return RuleModelTurn(
                tool_calls=(
                    ToolCall(call_id="offline-rule-1", name=name, arguments=args),
                )
            )
        source = next(
            (chunk for r in results for chunk in r.get("data", {}).get("chunks", [])),
            None,
        )
        if source is None:
            return RuleModelTurn(
                answer=RuleAnswerDraft(
                    status="insufficient_evidence",
                    text="No registered source was returned.",
                )
            )
        return RuleModelTurn(
            answer=RuleAnswerDraft(
                status="answered",
                text="Scripted source preview for "
                + source["rule_number"]
                + ". Inspect the original excerpt; this is not a live model explanation.",
                rule_citations=(
                    {"chunk_id": source["chunk_id"], "quote": source["text"][:160]},
                ),
            )
        )


class OfflineReviewModel(OfflineRulesModel):
    """Scripted paired-evidence preview, never a model-quality result."""

    async def next_turn(self, messages, tool_schemas, *, timeout_s, required_tool=None):
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        combined = any(
            "Replay session: " in m.get("content", "")
            for m in messages
            if m["role"] == "system"
        )
        if not combined:
            return await super().next_turn(
                messages, tool_schemas, timeout_s=timeout_s, required_tool=required_tool
            )
        question = next(m["content"] for m in messages if m["role"] == "user")
        replay = [
            r
            for r in results
            if r["name"] in ("get_shot", "get_session_summary", "list_shots")
        ]
        if not results:
            if required_tool == "get_shot":
                schema = next(
                    s["function"]["parameters"]
                    for s in tool_schemas
                    if s["function"]["name"] == "get_shot"
                )
                args = {k: v["enum"][0] for k, v in schema["properties"].items()}
                return RuleModelTurn(
                    tool_calls=(
                        ToolCall(
                            call_id="offline-replay-1", name="get_shot", arguments=args
                        ),
                    )
                )
            from .review import target_for

            kind, _ = target_for(question)
            if kind == "session":
                return RuleModelTurn(
                    tool_calls=(
                        ToolCall(
                            call_id="offline-replay-1",
                            name="get_session_summary",
                            arguments={},
                        ),
                    )
                )
        if replay:
            rules = [r for r in results if r["name"] in ("get_rule", "search_rules")]
            if not rules:
                return RuleModelTurn(
                    tool_calls=(
                        ToolCall(
                            call_id="offline-review-rule-1",
                            name="search_rules",
                            arguments={
                                "query": "standard foul first contact cue ball in hand",
                                "top_k": 5,
                            },
                        ),
                    )
                )
            source = next(
                (c for r in rules for c in r.get("data", {}).get("chunks", [])), None
            )
            return RuleModelTurn(
                answer=RuleAnswerDraft(
                    status="insufficient_evidence",
                    text="Scripted paired-evidence preview: recorded pocket observations do not establish first contact, ball group, cue-ball scratch or a foul verdict. Inspect the record and rule excerpt separately; no automatic adjudication is supported.",
                    shot_evidence_refs=tuple(
                        dict.fromkeys(i for r in replay for i in r["evidence_refs"])
                    ),
                    rule_citations=(
                        {"chunk_id": source["chunk_id"], "quote": source["text"][:160]},
                    )
                    if source
                    else (),
                )
            )
        return await super().next_turn(
            messages, tool_schemas, timeout_s=timeout_s, required_tool=required_tool
        )
