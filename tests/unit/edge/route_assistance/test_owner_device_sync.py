"""Exact owner deletion wire and HMAC using synthetic credentials/SQLite only."""

# ruff: noqa: F811
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.sync import FieldSyncClient, HttpResult
from tests.unit.edge.route_assistance.test_owner_device_delivery import ack, finish, queue
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    NOW,
    SCOPE,
    scene,  # noqa: F401
    task,
)


class Credentials:
    def device_id(self):
        return SCOPE.device_id

    def secret(self):
        return b"synthetic-long-key"


class Transport:
    def __init__(self, data=None, status=200):
        self.data, self.status, self.calls = data, status, []

    def post_json(self, path, headers, body):
        self.calls.append((path, headers, body))
        return HttpResult(self.status, json.dumps({"ok": True, "data": self.data}).encode())


def client(tx, before=lambda: None):
    return FieldSyncClient(
        credentials=Credentials(),
        transport=tx,
        owner_scope=SCOPE,
        before_mutation=before,
        now_seconds=lambda: 42,
        nonce_factory=lambda: "synthetic-nonce",
    )


def delivery(t, **changes):
    return (
        dict(
            contract="owner_delete_device_delivery_v1",
            task=json.loads(t.canonical_json()),
            task_sha256=t.task_sha256,
            delivery_token="11111111-1111-4111-8111-111111111111",
            lease_until_utc="2026-11-01T00:00:00.000Z",
        )
        | changes
    )


def test_signed_exact_raw_poll_and_scope_bound_task(scene):
    tx = Transport(delivery(task(scene)))
    attested = []
    result = client(tx, lambda: attested.append(True)).poll_owner_delete_once(SCOPE)
    assert result.outcome == "task" and result.task == task(scene)
    assert attested == [True]
    path, headers, body = tx.calls[0]
    assert path == "/v1/owner-media-deletions/claim"
    assert body == b'{"contract":"owner_delete_device_poll_v1"}'
    digest = hashlib.sha256(body).hexdigest()
    expected = hmac.new(
        Credentials().secret(),
        f"POST\n{path}\n42\nsynthetic-nonce\n{digest}".encode(),
        hashlib.sha256,
    ).hexdigest()
    assert headers["x-body-sha256"] == digest and headers["x-signature"] == expected


@pytest.mark.parametrize(
    "change",
    [
        {"task_sha256": "a" * 64},
        {"contract": "bad"},
        {"extra": True},
        {"delivery_token": "other"},
        {"lease_until_utc": "2026-02-30T00:00:00.000Z"},
    ],
)
def test_strict_delivery_failure_has_no_task(scene, change):
    result = client(Transport(delivery(task(scene), **change))).poll_owner_delete_once(SCOPE)
    assert result.outcome == "blocked" and result.task is None
    assert result.error_code == "invalid_response"


def test_wrong_original_scope_blocks(scene):
    result = client(Transport(delivery(task(scene)))).poll_owner_delete_once(
        replace(SCOPE, table_id="other")
    )
    assert result.outcome == "blocked" and result.error_code == "scope_conflict"


@pytest.mark.parametrize(
    "status, outcome, error",
    [
        (401, "retry", "unauthorized"),
        (503, "retry", "transport_unavailable"),
        (409, "blocked", "remote_conflict"),
        (302, "blocked", "invalid_response"),
    ],
)
def test_fixed_http_errors(status, outcome, error):
    result = client(Transport(status=status)).poll_owner_delete_once(SCOPE)
    assert (result.outcome, result.error_code) == (outcome, error)


def test_idle_and_attestation_failure_never_post():
    assert client(Transport()).poll_owner_delete_once(SCOPE).outcome == "idle"
    tx = Transport()

    def fail():
        raise OSError("sensitive raw endpoint")

    result = client(tx, fail).poll_owner_delete_once(SCOPE)
    assert result.error_code == "attestation_failed" and result.outcome == "retry"
    assert tx.calls == []


def test_exact_completion_and_strict_ack(scene):
    finish(scene)
    item = queue(scene).claim_due(NOW)
    tx = Transport(ack(item))
    result = client(tx).complete_owner_delete_once(item)
    assert result.outcome == "ack" and result.ack == ack(item)
    path, headers, body = tx.calls[0]
    assert path == f"/v1/owner-media-deletions/{item.deletion_job_id}/complete"
    assert body == item.completion_json.encode("ascii")
    assert headers["x-body-sha256"] == item.completion_sha256
    tx.data = ack(item, clip_id="other")
    result = client(tx).complete_owner_delete_once(item)
    assert result.outcome == "blocked" and result.error_code == "invalid_ack" and result.ack is None
    tx.calls.clear()
    result = client(tx).complete_owner_delete_once(replace(item, completion_json="{}"))
    assert result.outcome == "blocked" and result.error_code == "invalid_completion"
    assert tx.calls == []


def test_stop_after_attestation_suppresses_owner_http():
    stopped = []
    tx = Transport()
    c = FieldSyncClient(
        credentials=Credentials(),
        transport=tx,
        before_mutation=lambda: stopped.append(True),
        owner_cancelled=lambda: bool(stopped),
    )
    assert c.poll_owner_delete_once(SCOPE).outcome == "retry"
    assert tx.calls == []


@pytest.mark.parametrize(
    "mode",
    ["success", "slow_attestation", "oversize_attestation", "redirect_attestation", "slow_owner"],
)
def test_real_bounded_attestation_and_owner_post_share_transport(scene, mode):
    import time

    from billiards_ai.edge.route_assistance.owner_device_transport import (
        OwnerDeviceMetadataHttpTransport,
    )
    from billiards_ai.lab.field_assistance.cloud_gateway import (
        DevelopmentAttestedFieldCloudGateway,
        SignedFieldCloudGateway,
    )
    from tests.unit.edge.route_assistance.test_owner_device_transport import endpoint

    t = task(scene)

    def respond(handler, seen):
        attestation = handler.path.endswith("/runtime-attestation")
        if mode == "slow_attestation" and attestation:
            time.sleep(0.8)
        payload = (
            {
                "schema_version": "1.0",
                "runtime_environment": "development",
                "cloudbase_environment_id": SCOPE.environment_id,
            }
            if attestation
            else delivery(t)
        )
        raw = json.dumps({"ok": True, "data": payload}).encode()
        if mode == "oversize_attestation" and attestation:
            raw = b"x" * 16385
        handler.send_response(302 if mode == "redirect_attestation" and attestation else 200)
        if mode == "redirect_attestation" and attestation:
            handler.send_header("Location", "/followed")
        handler.end_headers()
        if mode == "slow_owner" and not attestation:
            for byte in raw:
                handler.wfile.write(bytes([byte]))
                handler.wfile.flush()
                time.sleep(0.04)
        else:
            handler.wfile.write(raw)
            handler.wfile.flush()

    with endpoint(respond) as (url, seen, _):
        tx = OwnerDeviceMetadataHttpTransport(url, timeout_seconds=0.45)
        gateway = DevelopmentAttestedFieldCloudGateway(
            SignedFieldCloudGateway(credentials=Credentials(), transport=tx), SCOPE.environment_id
        )
        c = client(tx, gateway.runtime_attestation)
        result = c.poll_owner_delete_once(SCOPE)
        assert result.outcome == ("task" if mode == "success" else "retry")
        assert len(seen) == (2 if mode in {"success", "slow_owner"} else 1)
        for path, headers, body in seen:
            headers = {key.lower(): value for key, value in headers.items()}
            assert headers["x-body-sha256"] == hashlib.sha256(body).hexdigest()
            wire = (
                f"POST\n{path.removeprefix('/sync')}\n{headers['x-timestamp']}\n"
                f"{headers['x-nonce']}\n{headers['x-body-sha256']}"
            ).encode()
            assert (
                headers["x-signature"]
                == hmac.new(Credentials().secret(), wire, hashlib.sha256).hexdigest()
            )
        assert tx._active is None
        tx.close()


@pytest.mark.parametrize(
    "change", [{"environment_id": "foreign"}, {"table_id": "foreign"}, {"device_id": "foreign"}]
)
def test_completion_requires_exact_original_configured_scope(scene, change):
    finish(scene)
    item = replace(queue(scene).claim_due(NOW), **change)
    tx = Transport()
    result = client(tx).complete_owner_delete_once(item)
    assert (result.outcome, result.error_code) == ("blocked", "scope_conflict")
    assert tx.calls == []


@pytest.mark.parametrize(
    "raw",
    [
        b'{"ok":true,"ok":true,"data":null}',
        b'{"ok":true,"data":null,"extra":0}',
        b"\xff",
        b"x" * 16385,
    ],
)
def test_malformed_duplicate_extra_and_oversized_wrapper_never_deliver(raw):
    tx = Transport()
    tx.post_json = lambda *a: HttpResult(200, raw)
    result = client(tx).poll_owner_delete_once(SCOPE)
    assert result.outcome == "blocked" and result.error_code == "invalid_response"
