"""Question-language public fallbacks, never a rule-answer translation dictionary."""

import re


def is_chinese(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def response_language(question, status, text, *, enforce_answer=False):
    if not is_chinese(question) or is_chinese(text):
        return status, text, None
    if status in ("answered", "partial"):
        if not enforce_answer:
            return status, text, None
        return (
            "insufficient_evidence",
            "模型未按要求返回中文回答。已保留查到的证据，本次不将未翻译内容当作有效中文回答。",
            "RESPONSE_LANGUAGE_MISMATCH",
        )
    messages = {
        "unsupported": "当前可用能力或收录范围不支持这个问题。规则仅覆盖已登记的WPA中式八球版本；请查看规则版本与可用证据。",
        "clarification": "请明确要查询的规则或记录；若问“这一杆”，请先选择一个场次和片段。",
        "error": "模型响应或数据查询未能完成；请查看执行记录中的安全错误分类，本次不生成无依据结论。",
        "timeout": "请求超时，未生成新的结论；已查询的证据仍可查看。",
        "limit_reached": "本次请求已达到模型轮次或工具次数上限，请缩小问题范围。",
        "insufficient_evidence": "现有证据不足以支持回答。记录可能缺少首碰、球组或母球状态，规则与片段证据请分开查看。",
    }
    return status, messages.get(status, "本次请求未能完成。"), None
