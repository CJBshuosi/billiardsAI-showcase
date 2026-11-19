"""Versioned synthetic-only data and immutable tool/model boundaries."""

import math
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    field_validator,
    model_validator,
)

ID = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")]
Positive = Annotated[StrictInt, Field(ge=1)]
NonNegative = Annotated[StrictInt, Field(ge=0)]
Outcome = Literal["pocket_observed", "no_pocket_observed", "unknown"]
AnswerStatus = Literal[
    "answered", "partial", "clarification", "unsupported", "insufficient_evidence"
]
MANIFEST = {
    "clip-complete": "complete.mp4",
    "clip-partial": "partial.mp4",
    "clip-empty": "empty.mp4",
}


MEDIA_DURATIONS = {"clip-complete": 16000, "clip-partial": 6000, "clip-empty": 2000}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def immutable(*args, **kwargs):
    raise TypeError("JSON boundary is immutable")


class FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = (
        __ior__
    ) = immutable


class FrozenList(list):
    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = (
        sort
    ) = reverse = __iadd__ = __imul__ = immutable


def freeze(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON")
    if isinstance(value, dict):
        return FrozenDict({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze(v) for v in value)
    return value


class ReplaySession(Contract):
    session_id: ID
    title: Annotated[str, Field(min_length=1, max_length=160)]
    limitations: tuple[str, ...] = ()


class ReplayShot(Contract):
    shot_id: ID
    session_id: ID
    sequence: Positive
    start_ms: NonNegative
    end_ms: Positive
    outcome: Outcome
    evidence_id: ID
    observation_source: Literal["synthetic_annotation"] = "synthetic_annotation"
    limitations: tuple[str, ...] = ()
    media_id: ID | None = None

    @model_validator(mode="after")
    def valid_times(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("end must follow start")
        return self


class ReplayMedia(Contract):
    media_id: ID
    filename: str
    duration_ms: Positive

    @model_validator(mode="after")
    def fixed_manifest(self):
        if (
            MANIFEST.get(self.media_id) != self.filename
            or MEDIA_DURATIONS.get(self.media_id) != self.duration_ms
        ):
            raise ValueError("media must match application manifest")
        return self


class ReplayBundleV1(Contract):
    schema_version: Literal["1.0"]
    bundle_id: ID
    revision: Positive
    data_kind: Literal["synthetic"]
    session: ReplaySession
    shots: tuple[ReplayShot, ...]
    media: tuple[ReplayMedia, ...]

    @model_validator(mode="after")
    def consistent(self):
        for values in (
            [s.shot_id for s in self.shots],
            [s.sequence for s in self.shots],
            [s.evidence_id for s in self.shots],
            [m.media_id for m in self.media],
        ):
            if len(values) != len(set(values)):
                raise ValueError("duplicate identity")
        media = {m.media_id: m for m in self.media}
        for shot in self.shots:
            if shot.session_id != self.session.session_id:
                raise ValueError("foreign session")
            if shot.media_id is not None:
                if (
                    shot.media_id not in media
                    or shot.end_ms > media[shot.media_id].duration_ms
                ):
                    raise ValueError("unresolved media or clip range")
        return self


class ImportResult(Contract):
    status: Literal["imported", "unchanged"]


class ToolContext(Contract):
    session_id: ID
    available_session_ids: tuple[ID, ...]


class ToolCall(Contract):
    call_id: Annotated[str, Field(min_length=1, max_length=200)]
    name: Annotated[str, Field(min_length=1, max_length=100)]
    arguments: dict[str, JsonValue]

    @field_validator("arguments")
    @classmethod
    def frozen_arguments(cls, v):
        return freeze(v)


class ToolResult(Contract):
    call_id: str
    name: str
    status: Literal[
        "ok",
        "partial",
        "not_found",
        "denied",
        "invalid_arguments",
        "unavailable",
        "timeout",
    ]
    data: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_refs: tuple[ID, ...] = ()
    limitations: tuple[str, ...] = ()

    @field_validator("data")
    @classmethod
    def frozen_data(cls, v):
        return freeze(v)


class AnswerDraft(Contract):
    status: AnswerStatus
    text: Annotated[str, Field(min_length=1, max_length=10000)]
    evidence_refs: tuple[ID, ...] = ()


class Usage(Contract):
    input_tokens: NonNegative | None = None
    output_tokens: NonNegative | None = None


class ModelTurn(Contract):
    tool_calls: tuple[ToolCall, ...] = ()
    answer: AnswerDraft | None = None
    usage: Usage = Field(default_factory=Usage)


class AskRequest(Contract):
    session_id: ID
    question: Annotated[str, Field(min_length=1, max_length=2000)]
    mode: Literal["offline", "live"] = "offline"
    selected_shot_id: ID | None = None

    @field_validator("question")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("question is blank")
        return v


class TraceEntry(Contract):
    name: str
    arguments: dict[str, JsonValue]
    status: str
    elapsed_ms: float


class AskResponse(Contract):
    request_id: str
    mode: Literal["offline", "live"]
    status: Literal[
        "answered",
        "partial",
        "clarification",
        "unsupported",
        "insufficient_evidence",
        "timeout",
        "error",
        "limit_reached",
    ]
    answer: str
    fact_cards: tuple[ToolResult, ...] = ()
    evidence_refs: tuple[ID, ...] = ()
    limitations: tuple[str, ...] = ()
    tool_trace: tuple[TraceEntry, ...] = ()
    usage: Usage = Field(default_factory=Usage)


class UnifiedAskRequest(Contract):
    question: Annotated[str, Field(min_length=1, max_length=2000)]
    mode: Literal["offline", "live"] = "offline"
    session_id: ID | None = None
    selected_shot_id: ID | None = None

    @model_validator(mode="after")
    def valid_scope(self):
        if not self.question.strip():
            raise ValueError("EMPTY_QUESTION")
        if self.selected_shot_id is not None and self.session_id is None:
            raise ValueError("SELECTION_REQUIRES_SESSION")
        return self


class ValidationDiagnostic(Contract):
    path: Annotated[
        str,
        Field(
            pattern=r"^(?:root|extra|status|text|rule_citations|shot_evidence_refs|evidence_refs|chunk_id|quote)(?:\.(?:[0-9]{1,3}|extra|status|text|rule_citations|shot_evidence_refs|evidence_refs|chunk_id|quote))*$",
            max_length=160,
        ),
    ]
    type: Literal[
        "string_too_long",
        "string_too_short",
        "extra_forbidden",
        "literal_error",
        "string_pattern_mismatch",
        "string_type",
        "tuple_type",
        "list_type",
        "model_type",
        "missing",
        "too_long",
        "too_short",
        "value_error",
        "validation_error",
    ]
