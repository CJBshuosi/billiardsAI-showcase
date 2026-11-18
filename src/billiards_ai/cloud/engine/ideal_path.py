"""Fail-closed ideal-path orchestration over a replaceable physics adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from billiards_ai.shared.models.calibration import CalibrationStatus
from billiards_ai.shared.models.ideal_path import (
    IdealPathEvent,
    IdealPathEventKind,
    IdealPathIntent,
    IdealPathReason,
    IdealPathRequest,
    IdealPathResult,
    IdealPathStatus,
)


class EngineUnavailableError(RuntimeError):
    """The configured physics engine cannot run in this environment."""


class IdealPathAdapter(Protocol):
    adapter_version: str
    engine_name: str
    engine_version: str

    def calculate(self, request: IdealPathRequest) -> tuple[IdealPathEvent, ...]: ...


@dataclass(frozen=True, slots=True)
class _AdapterMetadata:
    adapter_version: str
    engine_name: str
    engine_version: str


_UNAVAILABLE_METADATA = _AdapterMetadata(
    adapter_version="unavailable",
    engine_name="unavailable",
    engine_version="unavailable",
)


def _snapshot_metadata(adapter: IdealPathAdapter) -> _AdapterMetadata:
    values = (
        adapter.adapter_version,
        adapter.engine_name,
        adapter.engine_version,
    )
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError("adapter metadata must contain non-empty strings")
    return _AdapterMetadata(*values)


def _result(
    request: IdealPathRequest,
    metadata: _AdapterMetadata,
    status: IdealPathStatus,
    reason: IdealPathReason | None,
    events: tuple[IdealPathEvent, ...] = (),
) -> IdealPathResult:
    return IdealPathResult(
        status=status,
        reason=reason,
        adapter_version=metadata.adapter_version,
        engine_name=metadata.engine_name,
        engine_version=metadata.engine_version,
        source_configuration_version=request.table.configuration_version,
        provenance=request.provenance if status is IdealPathStatus.SUCCEEDED else None,
        events=events,
    )


def _unsupported_reason(request: IdealPathRequest) -> IdealPathReason | None:
    if (
        request.target_ball_id is None
        or request.target_pocket is None
        or request.intent is not IdealPathIntent.DIRECT_POT
    ):
        return IdealPathReason.INTENT_NOT_EXPLICIT
    if not (
        request.target_ball_reviewed and request.target_pocket_reviewed and request.intent_reviewed
    ):
        return IdealPathReason.INTENT_NOT_REVIEWED
    if request.calibration_status is not CalibrationStatus.ACTIVE:
        return IdealPathReason.CALIBRATION_NOT_ACTIVE
    object_ids = {ball.ball_id for ball in request.object_balls}
    if request.target_ball_id == 0 or request.target_ball_id not in object_ids:
        return IdealPathReason.TARGET_BALL_MISSING
    return None


def _events_satisfy_explicit_intent(
    request: IdealPathRequest,
    events: tuple[IdealPathEvent, ...],
) -> bool:
    if not any(
        event.kind is IdealPathEventKind.CUE_STRIKE and event.ball_id == 0 for event in events
    ):
        return False

    cue_contacts = (
        event
        for event in events
        if event.kind is IdealPathEventKind.BALL_CONTACT
        and (event.ball_id == 0 or event.other_ball_id == 0)
    )
    first_cue_contact = next(cue_contacts, None)
    if first_cue_contact is None:
        return False
    contacted_ball = (
        first_cue_contact.other_ball_id
        if first_cue_contact.ball_id == 0
        else first_cue_contact.ball_id
    )
    if contacted_ball != request.target_ball_id:
        return False

    return any(
        event.kind is IdealPathEventKind.POCKETED
        and event.ball_id == request.target_ball_id
        and event.pocket is request.target_pocket
        for event in events
    )


def calculate_ideal_path(
    request: IdealPathRequest,
    adapter: IdealPathAdapter,
) -> IdealPathResult:
    """Calculate one explicit reviewed path without searching alternative intents."""

    try:
        metadata = _snapshot_metadata(adapter)
    except Exception:
        return _result(
            request,
            _UNAVAILABLE_METADATA,
            IdealPathStatus.SIMULATION_FAILED,
            IdealPathReason.ADAPTER_FAILURE,
        )

    unsupported = _unsupported_reason(request)
    if unsupported is not None:
        return _result(
            request,
            metadata,
            IdealPathStatus.UNSUPPORTED_INTENT,
            unsupported,
        )
    try:
        raw_events = adapter.calculate(request)
        events = tuple(IdealPathEvent.model_validate(event) for event in raw_events)
        if not _events_satisfy_explicit_intent(request, events):
            raise ValueError("adapter events do not satisfy the explicit intent")
        return _result(request, metadata, IdealPathStatus.SUCCEEDED, None, events)
    except EngineUnavailableError:
        return _result(
            request,
            metadata,
            IdealPathStatus.ENGINE_UNAVAILABLE,
            IdealPathReason.ENGINE_UNAVAILABLE,
        )
    except Exception:
        return _result(
            request,
            metadata,
            IdealPathStatus.SIMULATION_FAILED,
            IdealPathReason.ADAPTER_FAILURE,
        )
