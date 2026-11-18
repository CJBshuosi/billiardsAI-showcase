"""Typed failures returned by the CoachRuntime reducer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RuntimeErrorCode(StrEnum):
    INVALID_TRANSITION = "INVALID_TRANSITION"
    INVALID_COMMAND = "INVALID_COMMAND"
    INVALID_SNAPSHOT = "INVALID_SNAPSHOT"
    INVALID_OBSERVATION = "INVALID_OBSERVATION"
    INVALID_PLAN_ADMISSION = "INVALID_PLAN_ADMISSION"
    SESSION_MISMATCH = "SESSION_MISMATCH"
    OBSERVATION_OUT_OF_STATE = "OBSERVATION_OUT_OF_STATE"
    # Defensive code for externally reconstructed snapshots whose receipts may be incomplete.
    STALE_OBSERVATION = "STALE_OBSERVATION"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    RECOVERY_PROOF_MISMATCH = "RECOVERY_PROOF_MISMATCH"
    INVALID_ADVANCE = "INVALID_ADVANCE"
    TERMINAL_STATE = "TERMINAL_STATE"


@dataclass(frozen=True, slots=True)
class RuntimeTransitionError:
    code: RuntimeErrorCode
    state: str
    command_type: str
    message: str
