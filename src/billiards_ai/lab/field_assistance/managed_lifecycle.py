"""Retained managed identity checks and capture-free command closure."""

from __future__ import annotations

from dataclasses import asdict

from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionCloseGateway,
    ManagedSessionCloseState,
)
from billiards_ai.edge.route_assistance.records import FieldSessionStatus
from billiards_ai.lab.field_assistance.live_source import confirmation_reference


def resolve_command_confirmation(envelope):
    """Parse only the durable envelope, preserving its augmented ledger digest."""
    if envelope.command_type not in {"continue_turn", "switch_turn"}:
        return None
    try:
        return confirmation_reference(envelope.payload.get("confirmation_ref"))
    except ValueError:
        return None


def poll_managed_close(store, request, gateway, match_lock) -> ManagedSessionCloseState:
    """Validate retained historical authority, then poll without a local lock."""
    with match_lock:
        persisted = store.get_managed_open_request(request.scope, request.open_request_id)
        if persisted != request:
            raise ValueError("managed close request changed; recovery required")
        session = store.validate_managed_local_session(request.scope, request.open_request_id)
        if session.status is not FieldSessionStatus.ENDED:
            raise ValueError("managed close requires retained ended local session")
    if not isinstance(gateway, ManagedSessionCloseGateway):
        raise OSError("cloud gateway has no managed session close capability")
    result = gateway.close_managed_session(binding=request.binding)
    if not isinstance(result, ManagedSessionCloseState):
        raise OSError("cloud gateway returned invalid managed session close")
    return ManagedSessionCloseState.from_payload(asdict(result), binding=request.binding)


def close_and_drain_once(store, request, gateway, match_lock) -> ManagedSessionCloseState:
    """Poll the real close fence, then apply at most one real ordered command.

    The ordinary background loop already claims after maintenance: it uses only
    poll_managed_close. Bootstrap owns this one-step worker, never starts it, and
    uses no projection or terminal port.
    """
    result = poll_managed_close(store, request, gateway, match_lock)
    if result.coverage == "sealed":
        return result

    def before_application(envelope):
        if envelope.session_id != request.binding.session_id:
            raise ValueError("managed close claimed foreign session; recovery required")
        session = store.validate_managed_local_session(request.scope, request.open_request_id)
        if session.status is not FieldSessionStatus.ENDED:
            raise ValueError("managed close command requires retained ended local session")

    # Local import avoids coupling server initialization to the bootstrap path.
    from billiards_ai.lab.field_assistance.server import _BackgroundFieldCommandWorker

    worker = _BackgroundFieldCommandWorker(
        gateway=gateway,
        store=store,
        match_lock=match_lock,
        before_application=before_application,
        confirmation_resolver=resolve_command_confirmation,
    )
    worker.claim_and_apply_once(raise_errors=True)
    return result
