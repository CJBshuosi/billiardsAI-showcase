from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedOpenRequest,
    ManagedSessionBinding,
    ManagedSessionScope,
)

BIRTH_ID = "43a89b7e-40ac-49d0-a28b-2e5d0b0a8d72"
REQUEST_ID = "598f7a02-c120-43de-bb14-037defb5c362"
PAYLOAD = {
    "schema_version": "1.0",
    "session_id": f"managed-{BIRTH_ID}",
    "birth_id": BIRTH_ID,
    "birth_contract": "managed_session_birth_v1",
    "table_id": "table-1",
    "device_id": "device-1",
    "status": "open",
}


def load(payload=PAYLOAD, **expected):
    return ManagedSessionBinding.from_payload(
        payload,
        expected_table_id=expected.get("table_id", "table-1"),
        expected_device_id=expected.get("device_id", "device-1"),
    )


@pytest.mark.parametrize("status", ["open", "ended"])
def test_binding_preserves_exact_server_identity_and_is_immutable(status):
    binding = load({**PAYLOAD, "status": status})
    assert asdict(binding) == {**PAYLOAD, "status": status}
    with pytest.raises(FrozenInstanceError):
        binding.status = "open"
    assert not hasattr(binding, "__dict__")


@pytest.mark.parametrize("key", list(PAYLOAD))
@pytest.mark.parametrize("value", [None, 1, True, [], {}])
def test_binding_rejects_nonstring_fields_without_coercion(key, value):
    with pytest.raises(ValueError):
        load({**PAYLOAD, key: value})
    with pytest.raises(ValueError):
        ManagedSessionBinding(**{**PAYLOAD, key: value})


@pytest.mark.parametrize("key", list(PAYLOAD))
def test_binding_rejects_every_missing_field(key):
    with pytest.raises(ValueError):
        load({name: value for name, value in PAYLOAD.items() if name != key})


@pytest.mark.parametrize("payload", [None, [], "binding", 1, {**PAYLOAD, "_id": "extra"}])
def test_binding_rejects_nonobjects_and_extra_fields(payload):
    with pytest.raises(ValueError):
        load(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", "2.0"),
        ("birth_contract", "legacy"),
        ("birth_id", BIRTH_ID.upper()),
        ("birth_id", BIRTH_ID.replace("-49d0-", "-19d0-")),
        ("birth_id", BIRTH_ID.replace("-a28b-", "-728b-")),
        ("birth_id", BIRTH_ID.replace("-", "")),
        ("birth_id", f" {BIRTH_ID}"),
        ("session_id", "managed-other"),
        ("session_id", f"MANAGED-{BIRTH_ID}"),
        ("status", "closing"),
        ("table_id", "other"),
        ("device_id", "other"),
    ],
)
def test_binding_rejects_wrong_contract_identity_status_or_scope(key, value):
    with pytest.raises(ValueError):
        load({**PAYLOAD, key: value})


@pytest.mark.parametrize("field", ["environment_id", "device_id", "table_id"])
@pytest.mark.parametrize("value", [None, 1, [], "", ".", "..", "a/b", "a b", "é", "a\n", "a" * 129])
def test_scope_rejects_unsafe_ids(field, value):
    with pytest.raises(ValueError):
        ManagedSessionScope(
            **{**dict(environment_id="dev", device_id="device-1", table_id="table-1"), field: value}
        )


def request(**changes):
    values = dict(
        scope=ManagedSessionScope("dev", "device-1", "table-1"),
        artifact_id="artifact-1",
        rule_profile_id=PROFILE_ID,
        open_request_id=REQUEST_ID,
        phase="pending",
        binding=None,
        created_at_utc="2026-10-30T10:00:00Z",
        updated_at_utc="2026-10-30T10:00:01Z",
    )
    return ManagedOpenRequest(**{**values, **changes})


@pytest.mark.parametrize("status", ["open", "ended"])
def test_request_accepts_pending_and_bound_views_without_creating_identity(status):
    pending = request(rule_profile_id=None)
    assert pending.binding is None
    assert pending.open_request_id == REQUEST_ID
    for phase in ("bound", "materialized"):
        bound = replace(pending, phase=phase, binding=load({**PAYLOAD, "status": status}))
        assert bound.binding.session_id == f"managed-{BIRTH_ID}"
        assert bound.binding.status == status
    with pytest.raises(FrozenInstanceError):
        pending.phase = "bound"
    assert not hasattr(pending, "__dict__")


@pytest.mark.parametrize(
    "changes",
    [
        {"scope": None},
        {"scope": {}},
        {"artifact_id": ""},
        {"artifact_id": []},
        {"rule_profile_id": "other"},
        {"rule_profile_id": []},
        {"open_request_id": REQUEST_ID.upper()},
        {"open_request_id": []},
        {"open_request_id": REQUEST_ID.replace("-43de-", "-13de-")},
        {"phase": "unknown"},
        {"phase": []},
        {"phase": "bound"},
        {"phase": "materialized"},
        {"binding": "bad"},
        {"created_at_utc": 123},
        {"created_at_utc": "2026-02-30T10:00:00Z"},
        {"created_at_utc": "yesterdayZ"},
        {"updated_at_utc": "2026-10-30"},
        {"updated_at_utc": "2026-10-30T10:00:00+01:00"},
    ],
)
def test_request_rejects_invalid_persisted_views(changes):
    with pytest.raises(ValueError):
        request(**changes)


def test_request_rejects_pending_binding_and_scope_drift():
    with pytest.raises(ValueError):
        request(binding=load())
    for binding in (replace(load(), table_id="other"), replace(load(), device_id="other")):
        with pytest.raises(ValueError):
            request(phase="bound", binding=binding)


def test_managed_contracts_import_without_loading_lab_or_persistence():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import billiards_ai.edge.route_assistance.managed_sessions; "
                "assert not any(name.startswith('billiards_ai.lab') or "
                "name == 'billiards_ai.edge.route_assistance.persistence' for name in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


CLOSE_PAYLOAD = {
    **{k: v for k, v in PAYLOAD.items() if k != "status"},
    "coverage": "closing",
    "closed_target": 3,
    "contiguous_ack_sequence": 1,
}


def close_load(payload=CLOSE_PAYLOAD):
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionCloseState

    return ManagedSessionCloseState.from_payload(payload, binding=load())


@pytest.mark.parametrize("coverage,ack", [("closing", 1), ("sealed", 3), ("sealed", 0)])
def test_close_state_preserves_exact_identity_and_safe_prefix(coverage, ack):
    payload = {**CLOSE_PAYLOAD, "coverage": coverage, "contiguous_ack_sequence": ack}
    if ack == 0:
        payload["closed_target"] = 0
    state = close_load(payload)
    assert asdict(state) == payload
    with pytest.raises(FrozenInstanceError):
        state.coverage = "sealed"


@pytest.mark.parametrize("key", list(CLOSE_PAYLOAD))
def test_close_state_rejects_missing_fields(key):
    with pytest.raises(ValueError):
        close_load({k: v for k, v in CLOSE_PAYLOAD.items() if k != key})


@pytest.mark.parametrize("key", ["closed_target", "contiguous_ack_sequence"])
@pytest.mark.parametrize("value", [True, 1.0, "1", None, -1, 9007199254740992])
def test_close_state_rejects_non_safe_integer_prefix(key, value):
    with pytest.raises(ValueError):
        close_load({**CLOSE_PAYLOAD, key: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 1.0},
        {"schema_version": "2.0"},
        {"birth_contract": "other"},
        {"session_id": "managed-other"},
        {"birth_id": BIRTH_ID.upper()},
        {"table_id": "other"},
        {"device_id": "other"},
        {"device_id": True},
        {"coverage": "open"},
        {"coverage": []},
        {"contiguous_ack_sequence": 4},
        {"coverage": "sealed"},
        {"extra": 1},
    ],
)
def test_close_state_rejects_wrong_scope_contract_or_coverage(changes):
    with pytest.raises(ValueError):
        close_load({**CLOSE_PAYLOAD, **changes})


@pytest.mark.parametrize("payload", [None, [], "close"])
def test_close_state_rejects_nonobjects(payload):
    with pytest.raises(ValueError):
        close_load(payload)
