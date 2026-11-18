"""Typed fail-closed errors for approved evidence ingestion."""

from __future__ import annotations

from enum import StrEnum


class EvidenceIngestErrorCode(StrEnum):
    INVALID_BATCH = "INVALID_BATCH"
    NOT_APPROVED = "NOT_APPROVED"
    REVISION_MISMATCH = "REVISION_MISMATCH"
    LINEAGE_MISMATCH = "LINEAGE_MISMATCH"
    DIGEST_MISMATCH = "DIGEST_MISMATCH"
    STATISTICS_MISMATCH = "STATISTICS_MISMATCH"
    SCORES_MISMATCH = "SCORES_MISMATCH"
    PROJECTION_UNAVAILABLE = "PROJECTION_UNAVAILABLE"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"


class EvidenceIngestError(ValueError):
    """Safe public failure without embedding rejected payloads."""

    def __init__(self, code: EvidenceIngestErrorCode):
        self.code = code
        super().__init__(code.value)
