"""Fail-closed personal-memory admission using Stage 2's real read-only ledger.

This is not an ownership state machine. No match/roster lookup, repair, ID
conversion, or write is permitted here. Existing replay facts/media stay intact.
"""

from __future__ import annotations

from dataclasses import replace

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import ShotOwnershipReceipt


class FieldMemoryAdmissionError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def require_personal_memory_admission(
    store: SQLiteFieldRecordStore,
    session_id: str,
) -> None:
    """Reject unsupported field sources before the growth persistence boundary.

    Current receipt v1 supports development_replay only. There is deliberately
    no positive production branch until real-subject/source and UUID mapping
    authorities are approved. A later schema addition does not implicitly opt in.
    """
    try:
        session = store.load_session(session_id)
        shots = store.list_shots(session_id)
        if session.session_id != session_id or not shots:
            raise FieldMemoryAdmissionError("OWNERSHIP_UNKNOWN")
        replay = False
        for shot in shots:
            receipt = store.get_shot_ownership(session_id, shot.shot_id)
            if receipt is None:
                raise FieldMemoryAdmissionError("OWNERSHIP_UNKNOWN")
            if not isinstance(receipt, ShotOwnershipReceipt):
                raise FieldMemoryAdmissionError("OWNERSHIP_LINEAGE_MISMATCH")
            receipt = replace(receipt)  # Revalidate even a mutated dataclass instance.
            if (
                receipt.session_id != session_id
                or receipt.shot_id != shot.shot_id
                or receipt.sequence != shot.sequence
                or receipt.source_artifact_id != session.artifact_id
                or receipt.source_artifact_id != shot.media.artifact_id
                or receipt.source_started_ms != shot.started_ms
            ):
                raise FieldMemoryAdmissionError("OWNERSHIP_LINEAGE_MISMATCH")
            if receipt.evidence_status != "bound_at_start":
                raise FieldMemoryAdmissionError("OWNERSHIP_UNKNOWN")
            if not receipt.source_mainline1_artifact_id or not receipt.source_start_event_id:
                raise FieldMemoryAdmissionError("OWNERSHIP_SOURCE_UNVERIFIED")
            if receipt.actor_role is not shot.actor_role:
                raise FieldMemoryAdmissionError("OWNERSHIP_LINEAGE_MISMATCH")
            replay |= receipt.source_kind == "development_replay"
        if replay:
            raise FieldMemoryAdmissionError("DEVELOPMENT_REPLAY_EXCLUDED")
    except FieldMemoryAdmissionError:
        raise
    except (ValueError, TypeError, AttributeError) as exc:
        raise FieldMemoryAdmissionError("OWNERSHIP_UNAVAILABLE") from exc
    raise FieldMemoryAdmissionError("IDENTITY_MAPPING_REQUIRED")
