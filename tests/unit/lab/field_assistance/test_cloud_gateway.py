from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandApplication
from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.sync import HttpResult
from billiards_ai.lab.field_assistance.cloud_gateway import (
    ConfirmedParticipant,
    DevelopmentAttestedFieldCloudGateway,
    ExistingDevelopmentControlGateway,
    FieldJoinCoordinator,
    FieldMatchResultProposal,
    FieldSessionLease,
    JoinRequest,
    SignedFieldCloudGateway,
)


def test_existing_development_control_gateway_exposes_only_pinned_legacy_control() -> None:
    class Delegate:
        def open_field_session(self, session_id):
            return FieldSessionLease(session_id, "raytheon-01", "open")

        def close_field_session(self, session_id):
            return FieldSessionLease(session_id, "raytheon-01", "ended")

        def claim_field_command(self):
            return None

        def complete_field_command(self, application):
            return application

        def refresh_join_requests(self, session_id):
            return ()

        def confirm_participant_role(self, *, session_id, request_id, role):
            return ConfirmedParticipant(request_id, "user-a", role)

        def runtime_attestation(self):
            raise AssertionError("new attestation endpoint must not be called")

        def publish_session_analysis(self, payload):
            raise AssertionError("new analysis endpoint must not be called")

    gateway = ExistingDevelopmentControlGateway(
        delegate=Delegate(), expected_table_id="raytheon-01"
    )

    assert gateway.open_field_session("session-1").status == "open"
    assert gateway.close_field_session("session-1").status == "ended"
    assert gateway.claim_field_command() is None
    assert gateway.refresh_join_requests("session-1") == ()
    confirmed = gateway.confirm_participant_role(
        session_id="session-1", request_id="request-1", role=WristbandRole.RED
    )
    assert confirmed.user_id == "user-a"
    assert not hasattr(gateway, "runtime_attestation")
    assert not hasattr(gateway, "publish_session_analysis")
    assert not hasattr(gateway, "open_managed_session")


@pytest.mark.parametrize(
    ("table_id", "status"),
    [("another-table", "open"), ("raytheon-01", "ended")],
)
def test_existing_development_control_gateway_rejects_unpinned_open_lease(
    table_id: str, status: str
) -> None:
    class Delegate:
        def open_field_session(self, session_id):
            return FieldSessionLease(session_id, table_id, status)

    gateway = ExistingDevelopmentControlGateway(
        delegate=Delegate(), expected_table_id="raytheon-01"
    )

    with pytest.raises(OSError, match="pinned development table lease"):
        gateway.open_field_session("session-1")


def test_signed_gateway_puts_analysis_with_revision_idempotency() -> None:
    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def __init__(self) -> None:
            self.request = None

        def put_json(self, path, headers, body):
            self.request = (path, headers, body)
            return HttpResult(
                200,
                b'{"ok":true,"data":{"schema_version":"1.0",'
                b'"session_id":"session-1","revision":2,"status":"PARTIAL",'
                b'"total_shots":2,"available_shots":1,"retrying_shots":1,'
                b'"report_available":true,"deadline_at":"2026-09-06T12:02:00Z",'
                b'"updated_at":"2026-09-06T12:01:00Z"}}',
            )

        def post_json(self, path, headers, body):
            raise AssertionError((path, headers, body))

    payload = {
        "schema_version": "1.0",
        "session_id": "session-1",
        "revision": 2,
        "status": "PARTIAL",
        "total_shots": 2,
        "available_shots": 1,
        "retrying_shots": 1,
        "report_available": True,
        "deadline_at": "2026-09-06T12:02:00Z",
        "updated_at": "2026-09-06T12:01:00Z",
        "frozen_shots_digest": "a" * 64,
        "frozen_shot_ids": ["shot-1", "shot-2"],
        "published_shots": [{"session_id": "session-1", "shot_id": "shot-1"}],
        "published_clips": [{"session_id": "session-1", "shot_id": "shot-1", "clip_id": "clip-1"}],
    }
    transport = Transport()
    gateway = SignedFieldCloudGateway(
        credentials=Credentials(),
        transport=transport,
        now_seconds=lambda: 1_700_000_000,
        nonce_factory=lambda: "nonce-analysis-0001",
    )

    result = gateway.publish_session_analysis(payload)

    assert result.session_id == "session-1"
    assert result.status == "PARTIAL"
    path, headers, body = transport.request
    assert path == "/v1/field-sessions/session-1/analysis"
    assert headers["x-idempotency-key"] == "analysis-session-1-revision-2"
    assert headers["x-device-id"] == "device-1"
    assert b'"frozen_shots_digest"' in body
    assert "fixture-secret" not in repr(transport.request)


@pytest.mark.parametrize("runtime_environment", ["production", "unknown"])
def test_development_attestation_blocks_untrusted_transport_before_any_mutation(
    runtime_environment: str,
) -> None:
    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def post_json(self, path, _headers, _body):
            self.paths.append(path)
            assert path == "/v1/runtime-attestation"
            body = (
                '{"ok":true,"data":{"schema_version":"1.0",'
                f'"runtime_environment":"{runtime_environment}",'
                '"cloudbase_environment_id":"cloud-production"}}'
            ).encode()
            return HttpResult(200, body)

        def put_json(self, path, _headers, _body):
            self.paths.append(path)
            raise AssertionError("must not mutate an untrusted runtime")

    transport = Transport()
    delegate = SignedFieldCloudGateway(credentials=Credentials(), transport=transport)
    gateway = DevelopmentAttestedFieldCloudGateway(
        delegate=delegate,
        expected_cloudbase_environment_id="cloud-development",
    )

    with pytest.raises(OSError, match="development runtime attestation"):
        gateway.publish_session_analysis({"session_id": "session-1", "revision": 1})

    assert transport.paths == ["/v1/runtime-attestation"]


class _Gateway:
    def __init__(self) -> None:
        self.confirmed = False

    def refresh_join_requests(self, session_id: str) -> tuple[JoinRequest, ...]:
        assert session_id == "session-1"
        return (JoinRequest("request-1", "球友 FC9529", "pending", None),)

    def confirm_participant_role(
        self, *, session_id: str, request_id: str, role: WristbandRole
    ) -> ConfirmedParticipant:
        assert (session_id, request_id, role) == (
            "session-1",
            "request-1",
            WristbandRole.RED,
        )
        self.confirmed = True
        return ConfirmedParticipant("request-1", "user-a", WristbandRole.RED)


def test_scanning_does_not_change_local_match_until_tablet_confirms() -> None:
    gateway = _Gateway()
    coordinator = FieldJoinCoordinator(session_id="session-1", gateway=gateway)

    pending = coordinator.refresh()
    assert pending[0].display_name == "球友 FC9529"
    assert coordinator.confirmed_participants == ()

    confirmed = coordinator.confirm("request-1", WristbandRole.RED)
    assert gateway.confirmed is True
    assert confirmed.user_id == "user-a"
    assert coordinator.confirmed_participants == (confirmed,)


def test_failed_cloud_confirmation_never_creates_local_confirmation() -> None:
    class Offline(_Gateway):
        def confirm_participant_role(self, **_kwargs):
            raise OSError("offline")

    coordinator = FieldJoinCoordinator(session_id="session-1", gateway=Offline())
    coordinator.refresh()

    try:
        coordinator.confirm("request-1", WristbandRole.RED)
    except OSError:
        pass
    else:
        raise AssertionError("offline confirmation must fail")
    assert coordinator.confirmed_participants == ()


def test_refresh_separates_pending_scans_from_confirmed_player_names() -> None:
    class Gateway(_Gateway):
        def refresh_join_requests(self, session_id: str) -> tuple[JoinRequest, ...]:
            assert session_id == "session-1"
            return (
                JoinRequest("request-new", "球友 NEW", "pending", None),
                JoinRequest("request-red", "球友 87BC6C", "confirmed", WristbandRole.RED),
                JoinRequest("request-blue", "球友 42AF10", "confirmed", WristbandRole.BLUE),
            )

    coordinator = FieldJoinCoordinator(session_id="session-1", gateway=Gateway())

    assert coordinator.refresh() == (JoinRequest("request-new", "球友 NEW", "pending", None),)
    assert coordinator.confirmed_requests == (
        JoinRequest("request-red", "球友 87BC6C", "confirmed", WristbandRole.RED),
        JoinRequest("request-blue", "球友 42AF10", "confirmed", WristbandRole.BLUE),
    )


def test_signed_gateway_uses_device_scope_without_exposing_secret() -> None:
    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def __init__(self) -> None:
            self.requests = []

        def post_json(self, path, headers, body):
            self.requests.append((path, headers, body))
            if path.endswith("/open"):
                return HttpResult(
                    200,
                    b'{"ok":true,"data":{"schema_version":"1.0",'
                    b'"session_id":"session-1","table_id":"table-1",'
                    b'"status":"open"}}',
                )
            if path.endswith("/close"):
                return HttpResult(
                    200,
                    b'{"ok":true,"data":{"schema_version":"1.0",'
                    b'"session_id":"session-1","table_id":"table-1",'
                    b'"status":"ended"}}',
                )
            return HttpResult(200, b'{"ok":true,"data":{"requests":[]}}')

        def put_file(self, url, path, content_type):
            raise AssertionError((url, path, content_type))

    transport = Transport()
    gateway = SignedFieldCloudGateway(
        credentials=Credentials(),
        transport=transport,
        now_seconds=lambda: 1_700_000_000,
        nonce_factory=iter(
            ("nonce-open-000001", "nonce-refresh-001", "nonce-close-00001")
        ).__next__,
    )

    assert gateway.open_field_session("session-1") == FieldSessionLease(
        "session-1", "table-1", "open"
    )
    assert gateway.refresh_join_requests("session-1") == ()
    assert gateway.close_field_session("session-1") == FieldSessionLease(
        "session-1", "table-1", "ended"
    )
    assert [path for path, _, _ in transport.requests] == [
        "/v1/field-sessions/session-1/open",
        "/v1/field-sessions/session-1/join-requests/query",
        "/v1/field-sessions/session-1/close",
    ]
    assert all("x-signature" in headers for _, headers, _ in transport.requests)
    assert "fixture-secret" not in repr(transport.requests)


def test_signed_gateway_proposes_the_terminal_result_on_the_existing_sync_origin() -> None:
    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def __init__(self) -> None:
            self.request = None

        def post_json(self, path, headers, body):
            self.request = (path, headers, body)
            return HttpResult(
                200,
                b'{"ok":true,"data":{"session_id":"session-1",'
                b'"status":"awaiting_result_confirmations",'
                b'"result":{"result_revision":1,"winner_role":"red",'
                b'"red_score":1,"blue_score":0,"finish_reason":"normal"}}}',
            )

    transport = Transport()
    gateway = SignedFieldCloudGateway(
        credentials=Credentials(),
        transport=transport,
        now_seconds=lambda: 1_700_000_000,
        nonce_factory=lambda: "nonce-result-0001",
    )

    result = gateway.propose_match_result(
        session_id="session-1",
        winner_role=WristbandRole.RED,
        red_score=1,
        blue_score=0,
        finish_reason="normal",
    )

    assert result == FieldMatchResultProposal(
        session_id="session-1",
        status="awaiting_result_confirmations",
        winner_role=WristbandRole.RED,
        red_score=1,
        blue_score=0,
        finish_reason="normal",
    )
    path, headers, body = transport.request
    assert path == "/v1/field-sessions/session-1/ranking-result"
    assert headers["x-device-id"] == "device-1"
    assert b'"winner_role":"red"' in body


def test_signed_gateway_reads_public_pending_identity_without_internal_user_id() -> None:
    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def post_json(self, path, headers, body):
            return HttpResult(
                200,
                b'{"ok":true,"data":{"requests":[{"request_id":"request-1",'
                b'"display_name":"\xe7\x90\x83\xe5\x8f\x8b FC9529",'
                b'"status":"pending","role":null}]}}',
            )

        def put_file(self, url, path, content_type):
            raise AssertionError((url, path, content_type))

    gateway = SignedFieldCloudGateway(
        credentials=Credentials(),
        transport=Transport(),
        now_seconds=lambda: 1_700_000_000,
        nonce_factory=lambda: "nonce-refresh-001",
    )

    assert gateway.refresh_join_requests("session-1") == (
        JoinRequest("request-1", "球友 FC9529", "pending", None),
    )


def test_signed_gateway_claims_and_completes_device_command() -> None:
    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def __init__(self) -> None:
            self.requests = []

        def post_json(self, path, headers, body):
            self.requests.append((path, body))
            if path == "/v1/field-commands/claim":
                return HttpResult(
                    200,
                    b'{"ok":true,"data":{"command_id":"command-1",'
                    b'"session_id":"session-1","actor_user_id":"user-a",'
                    b'"command_type":"claim_role","validated_payload":{"role":"red"},'
                    b'"expected_revision":0,"server_sequence":1}}',
                )
            return HttpResult(
                200,
                b'{"ok":true,"data":{"command_id":"command-1",'
                b'"status":"applied","public_result_code":"APPLIED",'
                b'"resulting_revision":1}}',
            )

        def put_file(self, url, path, content_type):
            raise AssertionError((url, path, content_type))

    transport = Transport()
    gateway = SignedFieldCloudGateway(
        credentials=Credentials(),
        transport=transport,
        now_seconds=lambda: 1_700_000_000,
        nonce_factory=iter(("nonce-command-claim", "nonce-command-complete")).__next__,
    )

    command = gateway.claim_field_command()
    assert command is not None
    assert command.command_id == "command-1"
    assert command.payload == {"role": "red"}
    completed = gateway.complete_field_command(
        FieldCommandApplication("command-1", "applied", "APPLIED", 1)
    )

    assert completed.resulting_revision == 1
    assert [request[0] for request in transport.requests] == [
        "/v1/field-commands/claim",
        "/v1/field-commands/command-1/complete",
    ]


def test_signed_gateway_returns_none_when_no_command_is_pending() -> None:
    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def post_json(self, path, headers, body):
            return HttpResult(200, b'{"ok":true,"data":null}')

        def put_file(self, url, path, content_type):
            raise AssertionError((url, path, content_type))

    gateway = SignedFieldCloudGateway(
        credentials=Credentials(),
        transport=Transport(),
        nonce_factory=lambda: "nonce-command-empty",
    )

    assert gateway.claim_field_command() is None


_MANAGED_REQUEST_ID = "598f7a02-c120-43de-bb14-037defb5c362"
_MANAGED_PAYLOAD = {
    "schema_version": "1.0",
    "session_id": "managed-43a89b7e-40ac-49d0-a28b-2e5d0b0a8d72",
    "birth_id": "43a89b7e-40ac-49d0-a28b-2e5d0b0a8d72",
    "birth_contract": "managed_session_birth_v1",
    "table_id": "table-1",
    "device_id": "device-1",
    "status": "open",
}


class _ManagedCredentials:
    def __init__(self, device="device-1"):
        self.device = device
        self.secret_reads = 0

    def device_id(self):
        return self.device

    def secret(self):
        self.secret_reads += 1
        return b"fixture-secret"


class _ManagedTransport:
    def __init__(self, data=_MANAGED_PAYLOAD, *, environment="development", env_id="dev"):
        self.data = data
        self.environment = environment
        self.env_id = env_id
        self.requests = []

    def post_json(self, path, headers, body):
        self.requests.append((path, headers, body))
        if path == "/v1/runtime-attestation":
            data = dict(
                schema_version="1.0",
                runtime_environment=self.environment,
                cloudbase_environment_id=self.env_id,
            )
        else:
            assert path == "/v2/field-sessions/open", "must not fall back to legacy open"
            data = self.data
        return HttpResult(200, json.dumps(dict(ok=True, data=data)).encode())


def _managed_gateway(transport, credentials=None):
    return SignedFieldCloudGateway(
        credentials=credentials or _ManagedCredentials(),
        transport=transport,
        now_seconds=lambda: 1700000000,
        nonce_factory=lambda: "nonce-managed-0001",
    )


@pytest.mark.parametrize("status", ["open", "ended"])
def test_managed_open_signs_exact_route_body_and_device_pin(status):
    transport = _ManagedTransport({**_MANAGED_PAYLOAD, "status": status})
    gateway = _managed_gateway(transport)
    binding = gateway.open_managed_session(
        open_request_id=_MANAGED_REQUEST_ID, expected_table_id="table-1"
    )
    assert binding.session_id == "managed-43a89b7e-40ac-49d0-a28b-2e5d0b0a8d72"
    assert binding.device_id == "device-1"
    assert binding.status == status
    assert len(transport.requests) == 1
    path, headers, body = transport.requests[0]
    assert path == "/v2/field-sessions/open"
    assert (
        body == b'{"open_request_id":"598f7a02-c120-43de-bb14-037defb5c362","schema_version":"1.0"}'
    )
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = (
        f"POST\n/v2/field-sessions/open\n1700000000\nnonce-managed-0001\n{body_hash}".encode()
    )
    assert headers == {
        "content-type": "application/json",
        "x-device-id": "device-1",
        "x-timestamp": "1700000000",
        "x-nonce": "nonce-managed-0001",
        "x-body-sha256": body_hash,
        "x-signature": hmac.new(b"fixture-secret", canonical, hashlib.sha256).hexdigest(),
    }


@pytest.mark.parametrize(
    ("request_id", "table_id", "device_id"),
    [
        (None, "table-1", "device-1"),
        ([], "table-1", "device-1"),
        (_MANAGED_REQUEST_ID.upper(), "table-1", "device-1"),
        (_MANAGED_REQUEST_ID.replace("-43de-", "-13de-"), "table-1", "device-1"),
        (_MANAGED_REQUEST_ID, None, "device-1"),
        (_MANAGED_REQUEST_ID, "..", "device-1"),
        (_MANAGED_REQUEST_ID, "table/1", "device-1"),
        (_MANAGED_REQUEST_ID, "table-1", []),
        (_MANAGED_REQUEST_ID, "table-1", ".."),
    ],
)
def test_managed_open_rejects_local_input_without_secret_or_http(request_id, table_id, device_id):
    credentials = _ManagedCredentials(device_id)
    transport = _ManagedTransport()
    with pytest.raises(ValueError):
        _managed_gateway(transport, credentials).open_managed_session(
            open_request_id=request_id, expected_table_id=table_id
        )
    assert credentials.secret_reads == 0
    assert transport.requests == []


@pytest.mark.parametrize("key", list(_MANAGED_PAYLOAD))
@pytest.mark.parametrize("value", [None, [], 123, True, "wrong"])
def test_managed_open_rejects_malformed_or_mismatched_binding_without_fallback(key, value):
    transport = _ManagedTransport({**_MANAGED_PAYLOAD, key: value})
    with pytest.raises(OSError, match="managed session"):
        _managed_gateway(transport).open_managed_session(
            open_request_id=_MANAGED_REQUEST_ID, expected_table_id="table-1"
        )
    assert len(transport.requests) == 1


@pytest.mark.parametrize("data", [None, [], {**_MANAGED_PAYLOAD, "extra": "private-response"}])
def test_managed_open_rejects_wrong_shape_and_extras(data):
    transport = _ManagedTransport(data)
    with pytest.raises(OSError) as error:
        _managed_gateway(transport).open_managed_session(
            open_request_id=_MANAGED_REQUEST_ID, expected_table_id="table-1"
        )
    assert "private-response" not in str(error.value)
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    ("environment", "env_id"), [("production", "dev"), ("unknown", "dev"), ("development", "other")]
)
def test_managed_open_attestation_denies_before_mutation(environment, env_id):
    transport = _ManagedTransport(environment=environment, env_id=env_id)
    gateway = DevelopmentAttestedFieldCloudGateway(_managed_gateway(transport), "dev")
    with pytest.raises(OSError, match="development runtime attestation"):
        gateway.open_managed_session(
            open_request_id=_MANAGED_REQUEST_ID, expected_table_id="table-1"
        )
    assert [request[0] for request in transport.requests] == ["/v1/runtime-attestation"]


def test_managed_open_attests_before_each_mutation_and_preserves_request_id():
    transport = _ManagedTransport()
    gateway = DevelopmentAttestedFieldCloudGateway(_managed_gateway(transport), "dev")
    first = gateway.open_managed_session(
        open_request_id=_MANAGED_REQUEST_ID, expected_table_id="table-1"
    )
    second = gateway.open_managed_session(
        open_request_id=_MANAGED_REQUEST_ID, expected_table_id="table-1"
    )
    assert first == second
    assert [request[0] for request in transport.requests] == [
        "/v1/runtime-attestation",
        "/v2/field-sessions/open",
        "/v1/runtime-attestation",
        "/v2/field-sessions/open",
    ]
    assert transport.requests[1][2] == transport.requests[3][2]


def test_attested_gateway_without_managed_capability_fails_without_legacy_open():
    class Legacy:
        def runtime_attestation(self):
            from billiards_ai.lab.field_assistance.cloud_gateway import RuntimeAttestation

            return RuntimeAttestation("development", "dev")

        def open_field_session(self, session_id):
            raise AssertionError("legacy open is not a managed capability")

    gateway = DevelopmentAttestedFieldCloudGateway(Legacy(), "dev")
    with pytest.raises(OSError, match="managed session capability"):
        gateway.open_managed_session(
            open_request_id=_MANAGED_REQUEST_ID, expected_table_id="table-1"
        )


@pytest.mark.parametrize(
    ("request_id", "table_id"),
    [(None, "table-1"), (_MANAGED_REQUEST_ID.upper(), "table-1"), (_MANAGED_REQUEST_ID, "..")],
)
def test_attested_managed_open_rejects_invalid_input_before_attestation(request_id, table_id):
    credentials = _ManagedCredentials()
    transport = _ManagedTransport()
    gateway = DevelopmentAttestedFieldCloudGateway(_managed_gateway(transport, credentials), "dev")
    with pytest.raises(ValueError):
        gateway.open_managed_session(open_request_id=request_id, expected_table_id=table_id)
    assert credentials.secret_reads == 0
    assert transport.requests == []


@pytest.mark.parametrize("status", [404, 409, 503])
def test_managed_open_remote_error_never_tries_legacy_route(status):
    class Transport:
        def __init__(self):
            self.paths = []

        def post_json(self, path, headers, body):
            self.paths.append(path)
            return HttpResult(status, b'{"ok":false,"error":{"code":"SYNC_CONFLICT"}}')

    transport = Transport()
    with pytest.raises(OSError, match="SYNC_CONFLICT"):
        _managed_gateway(transport).open_managed_session(
            open_request_id=_MANAGED_REQUEST_ID, expected_table_id="table-1"
        )
    assert transport.paths == ["/v2/field-sessions/open"]


_CLOSE_PAYLOAD = {
    **{k: v for k, v in _MANAGED_PAYLOAD.items() if k != "status"},
    "coverage": "closing",
    "closed_target": 3,
    "contiguous_ack_sequence": 1,
}


class _CloseTransport(_ManagedTransport):
    def post_json(self, path, headers, body):
        if path == "/v1/runtime-attestation":
            return super().post_json(path, headers, body)
        self.requests.append((path, headers, body))
        assert path == f"/v2/field-sessions/{_MANAGED_PAYLOAD['session_id']}/close"
        return HttpResult(200, json.dumps(dict(ok=True, data=self.data)).encode())


def _close_binding():
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionBinding

    return ManagedSessionBinding(**_MANAGED_PAYLOAD)


def test_managed_close_signs_exact_route_body_and_scope():
    transport = _CloseTransport(_CLOSE_PAYLOAD)
    result = _managed_gateway(transport).close_managed_session(binding=_close_binding())
    assert result.coverage == "closing" and result.closed_target == 3
    path, headers, body = transport.requests[0]
    assert body == b'{"schema_version":"1.0"}'
    digest = hashlib.sha256(body).hexdigest()
    expected = hmac.new(
        b"fixture-secret",
        f"POST\n{path}\n1700000000\nnonce-managed-0001\n{digest}".encode(),
        hashlib.sha256,
    ).hexdigest()
    assert headers["x-signature"] == expected
    assert headers["x-device-id"] == "device-1"


@pytest.mark.parametrize(
    "changes",
    [
        {"device_id": "other"},
        {"table_id": "other"},
        {"closed_target": True},
        {"extra": 1},
        {"coverage": "sealed"},
    ],
)
def test_managed_close_refuses_invalid_response_without_fallback(changes):
    transport = _CloseTransport({**_CLOSE_PAYLOAD, **changes})
    with pytest.raises(OSError, match="managed session"):
        _managed_gateway(transport).close_managed_session(binding=_close_binding())
    assert len(transport.requests) == 1


def test_managed_close_rejects_credential_drift_before_signing():
    transport = _CloseTransport(_CLOSE_PAYLOAD)
    credentials = _ManagedCredentials("other")
    with pytest.raises(ValueError):
        _managed_gateway(transport, credentials).close_managed_session(binding=_close_binding())
    assert not transport.requests and credentials.secret_reads == 0


@pytest.mark.parametrize("environment,env_id", [("production", "dev"), ("development", "other")])
def test_managed_close_attests_before_mutation(environment, env_id):
    transport = _CloseTransport(_CLOSE_PAYLOAD, environment=environment, env_id=env_id)
    gateway = DevelopmentAttestedFieldCloudGateway(_managed_gateway(transport), "dev")
    with pytest.raises(OSError, match="development runtime attestation"):
        gateway.close_managed_session(binding=_close_binding())
    assert [r[0] for r in transport.requests] == ["/v1/runtime-attestation"]


def test_managed_close_attested_exact_retry():
    transport = _CloseTransport(
        {**_CLOSE_PAYLOAD, "coverage": "sealed", "contiguous_ack_sequence": 3}
    )
    gateway = DevelopmentAttestedFieldCloudGateway(_managed_gateway(transport), "dev")
    assert gateway.close_managed_session(binding=_close_binding()) == gateway.close_managed_session(
        binding=_close_binding()
    )
    assert [r[0] for r in transport.requests] == [
        "/v1/runtime-attestation",
        f"/v2/field-sessions/{_MANAGED_PAYLOAD['session_id']}/close",
    ] * 2


@pytest.mark.parametrize(
    "response",
    [
        [],
        None,
        {"ok": False, "error": None},
        {"ok": False, "error": {"code": {"private": "secret"}}},
    ],
)
def test_gateway_malformed_outer_envelope_is_controlled_oserror(response):
    class Transport:
        def post_json(self, path, headers, body):
            return HttpResult(400, json.dumps(response).encode())

    with pytest.raises(OSError) as error:
        _managed_gateway(Transport()).open_managed_session(
            open_request_id=_MANAGED_REQUEST_ID, expected_table_id="table-1"
        )
    assert "private" not in str(error.value) and "secret" not in str(error.value)


def test_attested_managed_close_requires_separate_capability_without_legacy_close():
    class Delegate:
        def runtime_attestation(self):
            from billiards_ai.lab.field_assistance.cloud_gateway import RuntimeAttestation

            return RuntimeAttestation("development", "dev")

        def close_field_session(self, session_id):
            pytest.fail("legacy close is not managed close capability")

    gateway = DevelopmentAttestedFieldCloudGateway(Delegate(), "dev")
    with pytest.raises(OSError, match="managed session close capability"):
        gateway.close_managed_session(binding=_close_binding())
    assert not hasattr(
        ExistingDevelopmentControlGateway(Delegate(), "table-1"), "close_managed_session"
    )


@pytest.mark.parametrize(
    "schema,environment,env_id", [(1.0, "development", "123"), ("1.0", "development", 123)]
)
def test_managed_close_rejects_coerced_attestation_before_mutation(schema, environment, env_id):
    class Transport(_CloseTransport):
        def post_json(self, path, headers, body):
            if path == "/v1/runtime-attestation":
                self.requests.append((path, headers, body))
                return HttpResult(
                    200,
                    json.dumps(
                        dict(
                            ok=True,
                            data=dict(
                                schema_version=schema,
                                runtime_environment=environment,
                                cloudbase_environment_id=env_id,
                            ),
                        )
                    ).encode(),
                )
            return super().post_json(path, headers, body)

    transport = Transport(_CLOSE_PAYLOAD)
    gateway = DevelopmentAttestedFieldCloudGateway(_managed_gateway(transport), "123")
    with pytest.raises(OSError, match="runtime attestation"):
        gateway.close_managed_session(binding=_close_binding())
    assert [r[0] for r in transport.requests] == ["/v1/runtime-attestation"]
