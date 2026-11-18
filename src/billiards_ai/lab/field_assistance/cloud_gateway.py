"""Cloud gateway and local authority boundary for field-session joins."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandApplication,
    FieldCommandEnvelope,
)
from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionBinding,
    ManagedSessionCloseGateway,
    ManagedSessionCloseState,
    ManagedSessionGateway,
    validate_managed_scope_id,
    validate_managed_uuid,
)
from billiards_ai.edge.route_assistance.sync import (
    DeviceCredentialProvider,
    SyncHttpTransport,
    SyncRequestToSign,
    sign_sync_request,
)
from billiards_ai.shared.models import SessionAnalysisProjectionV1


@dataclass(frozen=True, slots=True)
class FieldSessionLease:
    session_id: str
    table_id: str
    status: str


@dataclass(frozen=True, slots=True)
class FieldMatchResultProposal:
    session_id: str
    status: str
    winner_role: WristbandRole
    red_score: int
    blue_score: int
    finish_reason: Literal["normal", "concession"]

    def __post_init__(self) -> None:
        if (
            not self.session_id
            or self.status
            not in {
                "awaiting_result_confirmations",
                "fact_package_ready",
                "disputed",
                "confirmation_expired",
            }
            or isinstance(self.red_score, bool)
            or not isinstance(self.red_score, int)
            or self.red_score < 0
            or isinstance(self.blue_score, bool)
            or not isinstance(self.blue_score, int)
            or self.blue_score < 0
        ):
            raise ValueError("field match result proposal is invalid")


@dataclass(frozen=True, slots=True)
class RuntimeAttestation:
    runtime_environment: str
    cloudbase_environment_id: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if (
            self.schema_version != "1.0"
            or not isinstance(self.runtime_environment, str)
            or not self.runtime_environment
            or not isinstance(self.cloudbase_environment_id, str)
            or not self.cloudbase_environment_id
        ):
            raise ValueError("runtime attestation is invalid")


@dataclass(frozen=True, slots=True)
class JoinRequest:
    request_id: str
    display_name: str
    status: str
    role: WristbandRole | None


@dataclass(frozen=True, slots=True)
class ConfirmedParticipant:
    request_id: str
    user_id: str
    role: WristbandRole


class FieldControlGateway(Protocol):
    """Capability boundary for the previously deployed join/control API only."""

    def open_field_session(self, session_id: str) -> FieldSessionLease: ...

    def close_field_session(self, session_id: str) -> FieldSessionLease: ...

    def propose_match_result(
        self,
        *,
        session_id: str,
        winner_role: WristbandRole,
        red_score: int,
        blue_score: int,
        finish_reason: Literal["normal", "concession"],
    ) -> FieldMatchResultProposal: ...

    def claim_field_command(self) -> FieldCommandEnvelope | None: ...

    def complete_field_command(
        self, application: FieldCommandApplication
    ) -> FieldCommandApplication: ...

    def refresh_join_requests(self, session_id: str) -> tuple[JoinRequest, ...]: ...

    def confirm_participant_role(
        self, *, session_id: str, request_id: str, role: WristbandRole
    ) -> ConfirmedParticipant: ...


class SessionAnalysisGateway(Protocol):
    def runtime_attestation(self) -> RuntimeAttestation: ...

    def publish_session_analysis(
        self, payload: Mapping[str, object]
    ) -> SessionAnalysisProjectionV1: ...


class FieldCloudGateway(FieldControlGateway, SessionAnalysisGateway, Protocol):
    """Full gateway used only when the new attested cloud runtime is deployed."""


@dataclass(slots=True)
class ExistingDevelopmentControlGateway:
    """Expose only legacy control calls and pin every returned session lease."""

    delegate: FieldControlGateway
    expected_table_id: str

    def __post_init__(self) -> None:
        if not self.expected_table_id:
            raise ValueError("expected development table ID is required")

    def _lease(
        self, lease: FieldSessionLease, *, session_id: str, expected_status: str
    ) -> FieldSessionLease:
        if (
            lease.session_id != session_id
            or lease.table_id != self.expected_table_id
            or lease.status != expected_status
        ):
            raise OSError("cloud returned an invalid pinned development table lease")
        return lease

    def open_field_session(self, session_id: str) -> FieldSessionLease:
        return self._lease(
            self.delegate.open_field_session(session_id),
            session_id=session_id,
            expected_status="open",
        )

    def close_field_session(self, session_id: str) -> FieldSessionLease:
        return self._lease(
            self.delegate.close_field_session(session_id),
            session_id=session_id,
            expected_status="ended",
        )

    def claim_field_command(self) -> FieldCommandEnvelope | None:
        return self.delegate.claim_field_command()

    def complete_field_command(
        self, application: FieldCommandApplication
    ) -> FieldCommandApplication:
        return self.delegate.complete_field_command(application)

    def refresh_join_requests(self, session_id: str) -> tuple[JoinRequest, ...]:
        return self.delegate.refresh_join_requests(session_id)

    def confirm_participant_role(
        self, *, session_id: str, request_id: str, role: WristbandRole
    ) -> ConfirmedParticipant:
        return self.delegate.confirm_participant_role(
            session_id=session_id, request_id=request_id, role=role
        )


@dataclass(slots=True)
class DevelopmentAttestedFieldCloudGateway:
    """Reject every field call unless the endpoint is the exact dev deployment."""

    delegate: FieldCloudGateway
    expected_cloudbase_environment_id: str

    def __post_init__(self) -> None:
        if not self.expected_cloudbase_environment_id:
            raise ValueError("expected development CloudBase environment ID is required")

    def runtime_attestation(self) -> RuntimeAttestation:
        observed = self.delegate.runtime_attestation()
        if (
            observed.runtime_environment != "development"
            or observed.cloudbase_environment_id != self.expected_cloudbase_environment_id
        ):
            raise OSError("development runtime attestation did not match the expected deployment")
        return observed

    def open_field_session(self, session_id: str) -> FieldSessionLease:
        self.runtime_attestation()
        return self.delegate.open_field_session(session_id)

    def open_managed_session(
        self, *, open_request_id: str, expected_table_id: str
    ) -> ManagedSessionBinding:
        validate_managed_uuid(open_request_id, name="open request ID")
        validate_managed_scope_id(expected_table_id, name="expected table ID")
        self.runtime_attestation()
        if not isinstance(self.delegate, ManagedSessionGateway):
            raise OSError("cloud gateway has no managed session capability")
        return self.delegate.open_managed_session(
            open_request_id=open_request_id, expected_table_id=expected_table_id
        )

    def close_managed_session(self, *, binding: ManagedSessionBinding) -> ManagedSessionCloseState:
        if not isinstance(binding, ManagedSessionBinding):
            raise ValueError("invalid managed session binding")
        binding = ManagedSessionBinding.from_payload(
            asdict(binding),
            expected_table_id=binding.table_id,
            expected_device_id=binding.device_id,
        )
        self.runtime_attestation()
        if not isinstance(self.delegate, ManagedSessionCloseGateway):
            raise OSError("cloud gateway has no managed session close capability")
        result = self.delegate.close_managed_session(binding=binding)
        if not isinstance(result, ManagedSessionCloseState):
            raise OSError("cloud gateway returned invalid managed session close")
        try:
            return ManagedSessionCloseState.from_payload(asdict(result), binding=binding)
        except ValueError:
            raise OSError("cloud gateway returned invalid managed session close") from None

    def close_field_session(self, session_id: str) -> FieldSessionLease:
        self.runtime_attestation()
        return self.delegate.close_field_session(session_id)

    def propose_match_result(
        self,
        *,
        session_id: str,
        winner_role: WristbandRole,
        red_score: int,
        blue_score: int,
        finish_reason: Literal["normal", "concession"],
    ) -> FieldMatchResultProposal:
        self.runtime_attestation()
        return self.delegate.propose_match_result(
            session_id=session_id,
            winner_role=winner_role,
            red_score=red_score,
            blue_score=blue_score,
            finish_reason=finish_reason,
        )

    def claim_field_command(self) -> FieldCommandEnvelope | None:
        self.runtime_attestation()
        return self.delegate.claim_field_command()

    def complete_field_command(
        self, application: FieldCommandApplication
    ) -> FieldCommandApplication:
        self.runtime_attestation()
        return self.delegate.complete_field_command(application)

    def refresh_join_requests(self, session_id: str) -> tuple[JoinRequest, ...]:
        self.runtime_attestation()
        return self.delegate.refresh_join_requests(session_id)

    def confirm_participant_role(
        self, *, session_id: str, request_id: str, role: WristbandRole
    ) -> ConfirmedParticipant:
        self.runtime_attestation()
        return self.delegate.confirm_participant_role(
            session_id=session_id,
            request_id=request_id,
            role=role,
        )

    def publish_session_analysis(
        self, payload: Mapping[str, object]
    ) -> SessionAnalysisProjectionV1:
        self.runtime_attestation()
        return self.delegate.publish_session_analysis(payload)


class FieldJoinCoordinator:
    """Keep scans pending until the tablet confirms one cloud binding."""

    def __init__(self, *, session_id: str, gateway: FieldControlGateway) -> None:
        self.session_id = session_id
        self.gateway = gateway
        self.pending: tuple[JoinRequest, ...] = ()
        self.confirmed_requests: tuple[JoinRequest, ...] = ()
        self.confirmed_participants: tuple[ConfirmedParticipant, ...] = ()

    def refresh(self) -> tuple[JoinRequest, ...]:
        requests = self.gateway.refresh_join_requests(self.session_id)
        self.pending = tuple(item for item in requests if item.status == "pending")
        self.confirmed_requests = tuple(
            item for item in requests if item.status == "confirmed" and item.role is not None
        )
        return self.pending

    def confirm(self, request_id: str, role: WristbandRole) -> ConfirmedParticipant:
        pending_request = next(
            (item for item in self.pending if item.request_id == request_id), None
        )
        if pending_request is None:
            raise ValueError("join request is not pending on this tablet")
        confirmed = self.gateway.confirm_participant_role(
            session_id=self.session_id,
            request_id=request_id,
            role=role,
        )
        if confirmed.role is not role:
            raise ValueError("cloud confirmation role does not match tablet request")
        by_role = {item.role: item for item in self.confirmed_participants if item.role is not role}
        by_role[role] = confirmed
        self.confirmed_participants = tuple(
            by_role[item] for item in (WristbandRole.RED, WristbandRole.BLUE) if item in by_role
        )
        self.pending = tuple(item for item in self.pending if item.request_id != request_id)
        confirmed_request = JoinRequest(
            request_id=request_id,
            display_name=pending_request.display_name,
            status="confirmed",
            role=role,
        )
        by_request_role = {
            item.role: item for item in self.confirmed_requests if item.role is not role
        }
        by_request_role[role] = confirmed_request
        self.confirmed_requests = tuple(
            by_request_role[item]
            for item in (WristbandRole.RED, WristbandRole.BLUE)
            if item in by_request_role
        )
        return confirmed


class SignedFieldCloudGateway:
    """Use the same device HMAC contract as the edge sync client."""

    def __init__(
        self,
        *,
        credentials: DeviceCredentialProvider,
        transport: SyncHttpTransport,
        now_seconds: Callable[[], int] = lambda: int(time.time()),
        nonce_factory: Callable[[], str] = lambda: f"nonce-{secrets.token_hex(16)}",
    ) -> None:
        self._credentials = credentials
        self._transport = transport
        self._now_seconds = now_seconds
        self._nonce_factory = nonce_factory

    def _request_data(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object],
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> object:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        timestamp = str(self._now_seconds())
        nonce = self._nonce_factory()
        body_hash = hashlib.sha256(body).hexdigest()
        signed = sign_sync_request(
            SyncRequestToSign(method, path, timestamp, nonce, body_hash),
            secret=self._credentials.secret(),
        )
        headers = {
            "content-type": "application/json",
            "x-device-id": self._credentials.device_id(),
            "x-timestamp": timestamp,
            "x-nonce": nonce,
            "x-body-sha256": body_hash,
            "x-signature": signed.signature,
            **dict(extra_headers or {}),
        }
        if method == "POST":
            result = self._transport.post_json(path, headers, body)
        elif method == "PUT":
            result = self._transport.put_json(path, headers, body)
        else:  # pragma: no cover - private callers are fixed above
            raise ValueError("unsupported cloud gateway method")
        try:
            response = json.loads(result.body)
        except json.JSONDecodeError as exc:
            raise OSError("cloud gateway returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise OSError("cloud gateway returned invalid response")
        if not 200 <= result.status_code < 300 or response.get("ok") is not True:
            error = response.get("error")
            code = error.get("code") if isinstance(error, dict) else None
            if (
                not isinstance(code, str)
                or not code
                or len(code) > 128
                or not code.isascii()
                or not code.replace("_", "").isalnum()
                or code.upper() != code
            ):
                code = f"HTTP_{result.status_code}"
            raise OSError(code)
        return response.get("data")

    def _post_data(self, path: str, payload: Mapping[str, object]) -> object:
        return self._request_data("POST", path, payload)

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        data = self._post_data(path, payload)
        if not isinstance(data, dict):
            raise OSError("cloud gateway returned invalid data")
        return data

    def _session_lease(self, session_id: str, operation: str) -> FieldSessionLease:
        data = self._post(f"/v1/field-sessions/{session_id}/{operation}", {})
        try:
            lease = FieldSessionLease(
                session_id=str(data["session_id"]),
                table_id=str(data["table_id"]),
                status=str(data["status"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OSError("cloud gateway returned invalid field session") from exc
        if lease.session_id != session_id:
            raise OSError("cloud gateway returned mismatched field session")
        return lease

    def runtime_attestation(self) -> RuntimeAttestation:
        data = self._post("/v1/runtime-attestation", {})
        try:
            return RuntimeAttestation(
                schema_version=data["schema_version"],
                runtime_environment=data["runtime_environment"],
                cloudbase_environment_id=data["cloudbase_environment_id"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OSError("cloud gateway returned invalid runtime attestation") from exc

    def open_field_session(self, session_id: str) -> FieldSessionLease:
        return self._session_lease(session_id, "open")

    def open_managed_session(
        self, *, open_request_id: str, expected_table_id: str
    ) -> ManagedSessionBinding:
        validate_managed_uuid(open_request_id, name="open request ID")
        validate_managed_scope_id(expected_table_id, name="expected table ID")
        expected_device_id = validate_managed_scope_id(
            self._credentials.device_id(), name="expected device ID"
        )
        data = self._post(
            "/v2/field-sessions/open",
            {"schema_version": "1.0", "open_request_id": open_request_id},
        )
        try:
            return ManagedSessionBinding.from_payload(
                data, expected_table_id=expected_table_id, expected_device_id=expected_device_id
            )
        except ValueError:
            raise OSError("cloud gateway returned invalid managed session") from None

    def close_managed_session(self, *, binding: ManagedSessionBinding) -> ManagedSessionCloseState:
        if not isinstance(binding, ManagedSessionBinding):
            raise ValueError("invalid managed session binding")
        binding = ManagedSessionBinding.from_payload(
            asdict(binding),
            expected_table_id=binding.table_id,
            expected_device_id=self._credentials.device_id(),
        )
        data = self._post(
            f"/v2/field-sessions/{binding.session_id}/close", {"schema_version": "1.0"}
        )
        try:
            return ManagedSessionCloseState.from_payload(data, binding=binding)
        except ValueError:
            raise OSError("cloud gateway returned invalid managed session close") from None

    def close_field_session(self, session_id: str) -> FieldSessionLease:
        return self._session_lease(session_id, "close")

    def propose_match_result(
        self,
        *,
        session_id: str,
        winner_role: WristbandRole,
        red_score: int,
        blue_score: int,
        finish_reason: Literal["normal", "concession"],
    ) -> FieldMatchResultProposal:
        if (
            not session_id
            or isinstance(red_score, bool)
            or not isinstance(red_score, int)
            or red_score < 0
            or isinstance(blue_score, bool)
            or not isinstance(blue_score, int)
            or blue_score < 0
            or finish_reason not in {"normal", "concession"}
        ):
            raise ValueError("field match result proposal is invalid")
        data = self._post(
            f"/v1/field-sessions/{session_id}/ranking-result",
            {
                "winner_role": winner_role.value,
                "red_score": red_score,
                "blue_score": blue_score,
                "finish_reason": finish_reason,
            },
        )
        result = data.get("result")
        if not isinstance(result, dict):
            raise OSError("cloud gateway returned invalid match result")
        try:
            proposal = FieldMatchResultProposal(
                session_id=str(data["session_id"]),
                status=str(data["status"]),
                winner_role=WristbandRole(result["winner_role"]),
                red_score=result["red_score"],
                blue_score=result["blue_score"],
                finish_reason=result["finish_reason"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OSError("cloud gateway returned invalid match result") from exc
        expected = FieldMatchResultProposal(
            session_id=session_id,
            status=proposal.status,
            winner_role=winner_role,
            red_score=red_score,
            blue_score=blue_score,
            finish_reason=finish_reason,
        )
        if proposal != expected:
            raise OSError("cloud gateway returned mismatched match result")
        return proposal

    def claim_field_command(self) -> FieldCommandEnvelope | None:
        data = self._post_data("/v1/field-commands/claim", {})
        if data is None:
            return None
        if not isinstance(data, dict):
            raise OSError("cloud gateway returned invalid field command")
        try:
            return FieldCommandEnvelope(
                command_id=str(data["command_id"]),
                session_id=str(data["session_id"]),
                actor_user_id=str(data["actor_user_id"]),
                command_type=str(data["command_type"]),
                payload=dict(data["validated_payload"]),
                expected_revision=int(data["expected_revision"]),
                server_sequence=int(data["server_sequence"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OSError("cloud gateway returned invalid field command") from exc

    def complete_field_command(
        self, application: FieldCommandApplication
    ) -> FieldCommandApplication:
        data = self._post(
            f"/v1/field-commands/{application.command_id}/complete",
            {
                "status": application.outcome,
                "public_result_code": application.result_code,
                "resulting_revision": application.resulting_revision,
            },
        )
        try:
            completed = FieldCommandApplication(
                command_id=str(data["command_id"]),
                outcome=str(data["status"]),
                result_code=str(data["public_result_code"]),
                resulting_revision=int(data["resulting_revision"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OSError("cloud gateway returned invalid command completion") from exc
        if completed != application:
            raise OSError("cloud gateway returned mismatched command completion")
        return completed

    def refresh_join_requests(self, session_id: str) -> tuple[JoinRequest, ...]:
        data = self._post(f"/v1/field-sessions/{session_id}/join-requests/query", {})
        requests = data.get("requests")
        if not isinstance(requests, list):
            raise OSError("cloud gateway returned invalid join requests")
        return tuple(
            JoinRequest(
                request_id=str(item["request_id"]),
                display_name=str(item["display_name"]),
                status=str(item["status"]),
                role=None if item.get("role") is None else WristbandRole(item["role"]),
            )
            for item in requests
        )

    def confirm_participant_role(
        self, *, session_id: str, request_id: str, role: WristbandRole
    ) -> ConfirmedParticipant:
        data = self._post(
            f"/v1/field-sessions/{session_id}/join-requests/{request_id}/confirm",
            {"role": role.value},
        )
        return ConfirmedParticipant(
            request_id=str(data["request_id"]),
            user_id=str(data["user_id"]),
            role=WristbandRole(data["role"]),
        )

    def publish_session_analysis(
        self, payload: Mapping[str, object]
    ) -> SessionAnalysisProjectionV1:
        session_id = payload.get("session_id")
        revision = payload.get("revision")
        if not isinstance(session_id, str) or not session_id or not isinstance(revision, int):
            raise ValueError("analysis projection requires session ID and revision")
        data = self._request_data(
            "PUT",
            f"/v1/field-sessions/{session_id}/analysis",
            payload,
            extra_headers={"x-idempotency-key": f"analysis-{session_id}-revision-{revision}"},
        )
        if not isinstance(data, dict):
            raise OSError("cloud gateway returned invalid analysis projection")
        try:
            projection = SessionAnalysisProjectionV1.model_validate(data)
            expected = SessionAnalysisProjectionV1.model_validate(
                {
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {
                        "frozen_shots_digest",
                        "frozen_shot_ids",
                        "published_shots",
                        "published_clips",
                    }
                }
            )
        except ValueError as exc:
            raise OSError("cloud gateway returned invalid analysis projection") from exc
        if projection != expected:
            raise OSError("cloud gateway returned mismatched analysis projection")
        return projection
