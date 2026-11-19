"""Resolve simple UI references; broader language remains with the model."""

import re


def is_current_clip_question(question: str) -> bool:
    text = re.sub(r"\s+", " ", question.strip().lower()).rstrip("?.!？。！ ")
    return text in {
        "this shot",
        "this clip",
        "that shot",
        "that one",
        "what happened in this shot",
        "what happened on this shot",
        "what happened to this shot",
        "what happened in this clip",
        "what happened in that shot",
        "tell me about this shot",
        "tell me about that one",
        "这一杆",
        "这杆",
        "那一杆",
        "这一杆发生了什么",
        "这杆发生了什么",
        "那一杆发生了什么",
        "这个片段发生了什么",
        "当前片段发生了什么",
    }


def select_clip_hint(question: str) -> str:
    return (
        "请点击一个击球片段，再问这一杆发生了什么。"
        if re.search(r"[\u4e00-\u9fff]", question)
        else "Click a clip in the list, then ask about this shot."
    )
