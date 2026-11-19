import hashlib
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from ..contracts import (
    ID,
    AnswerStatus,
    Contract,
    NonNegative,
    Positive,
    ToolCall,
    ToolResult,
    TraceEntry,
    Usage,
    ValidationDiagnostic,
)

SOURCE_URL = (
    "https://wpapool.com/wp-content/uploads/2025/08/250816-Rules-of-Heyball.pdf"
)
PDF_SHA256 = "2b92dd5088ea630337326e26b7a2a7a4499a9801f2c5da7e5cc40421d5424a24"
MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
RuleNumber = Annotated[str, Field(pattern=r"^(?:II|A)\.[1-9][0-9]*(?:\.[1-9][0-9]*)*$")]


class RuleSourceManifestV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    source_id: ID
    ruleset_id: ID
    effective_date: str
    source_url: Literal[SOURCE_URL] = SOURCE_URL
    retrieved_at: str
    pdf_sha256: SHA
    scope_sections: tuple[Literal["II", "A"], ...] = ("II", "A")
    extraction_version: str
    data_kind: Literal["official", "synthetic"] = "official"
    embedding_model: Literal[MODEL_NAME] = MODEL_NAME
    embedding_revision: str = MODEL_REVISION
    dimension: Literal[384] = 384

    @field_validator("effective_date")
    @classmethod
    def date_valid(cls, value):
        date.fromisoformat(value)
        return value

    @field_validator("retrieved_at")
    @classmethod
    def timestamp_valid(cls, value):
        if datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("TIMEZONE_REQUIRED")
        return value

    @model_validator(mode="after")
    def source_identity(self):
        if not self.scope_sections or len(set(self.scope_sections)) != len(
            self.scope_sections
        ):
            raise ValueError("INVALID_SCOPE")
        if self.data_kind == "official" and (
            self.ruleset_id != "wpa-heyball"
            or self.effective_date != "2025-08-16"
            or self.pdf_sha256 != PDF_SHA256
            or self.embedding_revision != MODEL_REVISION
        ):
            raise ValueError("UNREGISTERED_SOURCE")
        return self


class RuleChunk(Contract):
    chunk_id: ID
    source_id: ID
    ruleset_id: ID
    version: str
    rule_number: RuleNumber
    heading: Annotated[str, Field(min_length=1, max_length=500)]
    page_start: Positive
    page_end: Positive
    text: Annotated[str, Field(min_length=1, max_length=20000)]
    text_sha256: SHA
    source_url: Literal[SOURCE_URL] = SOURCE_URL
    embedding_model: Literal[MODEL_NAME] = MODEL_NAME
    embedding_revision: str = MODEL_REVISION
    dimension: Literal[384] = 384
    precedence: Literal["heyball_specific", "appendix_general"]
    data_kind: Literal["official", "synthetic"] = "official"

    @model_validator(mode="after")
    def integrity(self):
        if (
            self.page_end < self.page_start
            or hashlib.sha256(self.text.encode()).hexdigest() != self.text_sha256
        ):
            raise ValueError("CHUNK_INTEGRITY")
        if (self.rule_number.startswith("II.")) != (
            self.precedence == "heyball_specific"
        ):
            raise ValueError("RULE_PRECEDENCE")
        return self


class RulesContext(Contract):
    ruleset_id: ID
    version: str
    corpus_hash: SHA
    embedding_revision: str = MODEL_REVISION


class RuleHit(Contract):
    chunk: RuleChunk
    score: float
    method: Literal["keyword", "vector", "exact"]


class RuleCitationDraft(Contract):
    chunk_id: ID
    quote: Annotated[str, Field(min_length=1, max_length=320)]


class RuleCitation(Contract):
    chunk_id: ID
    rule_number: RuleNumber
    version: str
    page_start: Positive
    page_end: Positive
    quote: Annotated[str, Field(min_length=1, max_length=320)]
    source_url: Literal[SOURCE_URL] = SOURCE_URL


class RulesAskRequest(Contract):
    question: Annotated[str, Field(min_length=1, max_length=2000)]
    ruleset_id: ID
    version: Annotated[str, Field(min_length=1, max_length=30)]
    mode: Literal["offline", "live"] = "offline"

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("EMPTY_QUESTION")
        return value


class RuleAnswerDraft(Contract):
    status: AnswerStatus
    text: Annotated[str, Field(min_length=1, max_length=8000)]
    rule_citations: Annotated[tuple[RuleCitationDraft, ...], Field(max_length=5)] = ()
    shot_evidence_refs: tuple[ID, ...] = ()


class RuleCitationCandidate(Contract):
    """Internal only; no success until complete source membership is checked."""

    chunk_id: ID
    quote: Annotated[str, Field(min_length=1, max_length=2000)]


class RuleAnswerCandidate(Contract):
    status: AnswerStatus
    text: Annotated[str, Field(min_length=1, max_length=8000)]
    rule_citations: Annotated[
        tuple[RuleCitationCandidate, ...], Field(max_length=5)
    ] = ()
    shot_evidence_refs: tuple[ID, ...] = ()


class ReplayRulesAskRequest(RulesAskRequest):
    session_id: ID
    selected_shot_id: ID | None = None


class RuleModelTurn(Contract):
    tool_calls: tuple[ToolCall, ...] = ()
    answer: RuleAnswerDraft | RuleAnswerCandidate | None = None
    usage: Usage = Field(default_factory=Usage)


class RulesTraceEntry(TraceEntry):
    retrieval_strategy: str | None = None
    executed_query: str | None = None
    returned_chunk_ids: tuple[ID, ...] = ()


class SourceAnnotation(Contract):
    annotation_id: ID
    chunk_id: ID
    source_id: ID
    pdf_sha256: Literal[PDF_SHA256] = PDF_SHA256
    text_sha256: SHA
    fragment: Annotated[str, Field(min_length=1, max_length=300)]
    kind: Literal["quantity_scope", "procedure_scope"]
    note: Annotated[str, Field(min_length=1, max_length=800)]
    author_kind: Literal["ai_editorial_audit"] = "ai_editorial_audit"


class AnnotationRegistry(Contract):
    corpus_hash: SHA | None = None
    version: Annotated[str, Field(min_length=1, max_length=80)]
    annotations: tuple[SourceAnnotation, ...]

    @model_validator(mode="after")
    def unique_identity(self):
        if len({a.annotation_id for a in self.annotations}) != len(self.annotations):
            raise ValueError("DUPLICATE_ANNOTATION_ID")
        return self


class SourceAnnotationNotice(Contract):
    annotation_id: ID
    registry_version: str
    chunk_id: ID
    rule_number: RuleNumber
    version: str
    fragment: str
    kind: Literal["quantity_scope", "procedure_scope"]
    note: str
    author_kind: Literal["ai_editorial_audit"] = "ai_editorial_audit"
    page_start: Positive
    page_end: Positive
    source_url: Literal[SOURCE_URL] = SOURCE_URL


class RulesAskResponse(Contract):
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
    rule_citations: tuple[RuleCitation, ...] = ()
    shot_evidence_refs: tuple[ID, ...] = ()
    replay_fact_cards: tuple[ToolResult, ...] = ()
    tool_trace: tuple[RulesTraceEntry, ...] = ()
    usage: Usage = Field(default_factory=Usage)
    limitations: tuple[str, ...] = ()
    adjudication_supported: Literal[False] = False
    source_annotations: tuple[SourceAnnotationNotice, ...] = ()


class ExecutionEvent(Contract):
    phase: Literal["model"]
    round: Positive
    status: Literal["completed", "error", "timeout"]
    elapsed_ms: float
    error_code: str | None = None
    validation_errors: tuple[ValidationDiagnostic, ...] = ()


class UnifiedAskResponse(RulesAskResponse):
    model_rounds: NonNegative = 0
    execution_events: tuple[ExecutionEvent, ...] = ()
