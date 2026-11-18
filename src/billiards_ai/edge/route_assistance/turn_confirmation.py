"""Internal references to unresolved turns, never proof of consent or fact review."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TurnConfirmationRef:
    session_id: str
    shot_id: str
    evidence_digest: str
    match_revision: int

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.shot_id.strip():
            raise ValueError("confirmation identity is required")
        if len(self.evidence_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.evidence_digest
        ):
            raise ValueError("confirmation evidence must be SHA-256")
        if type(self.match_revision) is not int or self.match_revision < 0:
            raise ValueError("confirmation revision is invalid")


@dataclass(frozen=True, slots=True)
class TurnConfirmationReceipt:
    reference: TurnConfirmationRef
    source_mainline1_artifact_id: str | None
    status: Literal["pending", "stale", "resolved"]
    stale_reason: str | None
    resolved_command_id: str | None
    created_at_utc: str
    action: Literal["abstain"] = "abstain"
    reason: str = "AUTHORITATIVE_RULE_FACTS_INCOMPLETE"
