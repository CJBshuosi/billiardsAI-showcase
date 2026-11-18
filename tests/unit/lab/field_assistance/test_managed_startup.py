"""Real SQLite/worker/signed gateway startup; only remote HTTP is controlled."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import importlib.util
import json
import sqlite3
import threading
from dataclasses import asdict, replace

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope, reduce_field_command
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionBinding,
    ManagedSessionScope,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import FieldSessionStatus
from billiards_ai.edge.route_assistance.sync import HttpResult
from billiards_ai.lab.field_assistance.cloud_gateway import (
    DevelopmentAttestedFieldCloudGateway,
    SignedFieldCloudGateway,
)

SCOPE = ManagedSessionScope("dev", "device-1", "table-1")
REQUEST = "11111111-1111-4111-8111-111111111111"
NEXT_REQUEST = "33333333-3333-4333-8333-333333333333"
BIRTH = "22222222-2222-4222-8222-222222222222"
NEXT_BIRTH = "44444444-4444-4444-8444-444444444444"
BINDING = ManagedSessionBinding(
    "1.0", f"managed-{BIRTH}", BIRTH, "managed_session_birth_v1", "table-1", "device-1", "open"
)


def startup_module():
    name = "billiards_ai.lab.field_assistance.managed_startup"
    assert importlib.util.find_spec(name) is not None, "real managed startup coordinator is missing"
    return importlib.import_module(name)


class Credentials:
    def device_id(self):
        return "device-1"

    def secret(self):
        return b"synthetic-device-secret"


class Remote:
    """A signed HTTP boundary, not a fake local lifecycle/store."""

    def __init__(self):
        self.bindings = {REQUEST: BINDING}
        self.commands = []
        self.acks = []
        self.calls = []
        self.target = 0
        self.ack = 0
        self.hook = lambda _path: None
        self.lost = None

    def post_json(self, path, headers, body):
        digest = hashlib.sha256(body).hexdigest()
        signed = "\n".join(["POST", path, headers["x-timestamp"], headers["x-nonce"], digest])
        assert hmac.compare_digest(
            headers["x-signature"],
            hmac.new(Credentials().secret(), signed.encode(), hashlib.sha256).hexdigest(),
        )
        self.calls.append((path, json.loads(body)))
        self.hook(path)
        if path == "/v1/runtime-attestation":
            data = dict(
                schema_version="1.0",
                runtime_environment="development",
                cloudbase_environment_id="dev",
            )
        elif path == "/v2/field-sessions/open":
            request_id = json.loads(body)["open_request_id"]
            if request_id not in self.bindings:
                self.bindings[request_id] = replace(
                    BINDING, session_id=f"managed-{NEXT_BIRTH}", birth_id=NEXT_BIRTH
                )
            data = asdict(self.bindings[request_id])
        elif path.endswith("/close") and path.startswith("/v2/"):
            assert path == f"/v2/field-sessions/{BINDING.session_id}/close"
            data = {k: v for k, v in asdict(BINDING).items() if k != "status"}
            data.update(
                coverage="sealed" if self.ack == self.target else "closing",
                closed_target=self.target,
                contiguous_ack_sequence=self.ack,
            )
        elif path == "/v1/field-commands/claim":
            data = (
                None
                if not self.commands
                else {
                    name: (
                        dict(self.commands[0].payload)
                        if name == "payload"
                        else getattr(self.commands[0], name)
                    )
                    for name in (
                        "command_id",
                        "session_id",
                        "actor_user_id",
                        "command_type",
                        "payload",
                        "expected_revision",
                        "server_sequence",
                    )
                }
            )
            if data:
                data["validated_payload"] = data.pop("payload")
        elif path.endswith("/complete"):
            result = json.loads(body)
            data = dict(command_id=path.split("/")[-2], **result)
            self.acks.append(data)
            if self.commands:
                self.ack = self.commands.pop(0).server_sequence
        else:
            pytest.fail(f"unexpected/legacy session RPC: {path}")
        if self.lost == path:
            self.lost = None
            raise OSError("response lost after server commit")
        return HttpResult(200, json.dumps(dict(ok=True, data=data)).encode())


def store_at(path, *, materialized=True):
    ids = iter([REQUEST, NEXT_REQUEST])
    store = SQLiteFieldRecordStore(
        path,
        managed_open_request_id_factory=lambda: next(ids),
        session_id_factory=lambda: pytest.fail("local session UUID"),
    )
    if materialized:
        store.reserve_managed_open(SCOPE, artifact_id="old")
        store.record_managed_binding(SCOPE, REQUEST, BINDING)
        store.materialize_managed_session(SCOPE, REQUEST)
    return store


def coordinator(remote, **kwargs):
    module = startup_module()
    gateway = DevelopmentAttestedFieldCloudGateway(
        SignedFieldCloudGateway(credentials=Credentials(), transport=remote), "dev"
    )
    return module.ManagedStartupCoordinator(
        scope=SCOPE, gateway_factory=lambda **_budget: gateway, **kwargs
    )


def prepare(c, store, *, policy="blackbox_fresh", artifact="new", **kwargs):
    return c.prepare(store, artifact_id=artifact, rule_profile_id=None, policy=policy, **kwargs)


def command(sequence=1, **kwargs):
    return FieldCommandEnvelope(
        **(
            dict(
                command_id=f"command-{sequence}",
                session_id=BINDING.session_id,
                actor_user_id="alice",
                command_type="claim_role",
                payload={"role": "red"},
                expected_revision=0,
                server_sequence=sequence,
            )
            | kwargs
        )
    )


def test_fresh_really_ends_drains_retires_and_keeps_business_intent(tmp_path):
    store = store_at(tmp_path / "field.db")
    applied = store.apply_field_command(command(), reduce_field_command)
    remote = Remote()
    remote.target = 2
    remote.commands = [command(), command(2, command_type="release_role")]
    result = prepare(coordinator(remote), store)
    assert result.session_id == f"managed-{NEXT_BIRTH}" and result.status is FieldSessionStatus.OPEN
    old = store.load_session(BINDING.session_id)
    assert old.status is FieldSessionStatus.ENDED and old.match.red_user_id == "alice"
    assert [(a["public_result_code"], a["resulting_revision"]) for a in remote.acks] == [
        (applied.result_code, 1),
        ("SESSION_ENDED", 1),
    ]
    retired = store.get_latest_retired_managed_open_request(SCOPE)
    assert retired.open_request_id == REQUEST and retired.sealed_close.closed_target == 2
    assert store.list_pending_cloud_close_session_ids() == (BINDING.session_id,)
    assert store.get_postgame_manifest(BINDING.session_id).session_id == BINDING.session_id


def test_usb_resumes_only_exact_valid_open_and_replays_same_request(tmp_path):
    store = store_at(tmp_path / "field.db")
    remote = Remote()
    result = prepare(coordinator(remote), store, policy="usb_resume", artifact="old")
    assert result.session_id == BINDING.session_id
    assert [p for p, _ in remote.calls if p != "/v1/runtime-attestation"] == [
        "/v2/field-sessions/open"
    ]
    assert store.get_active_managed_open_request(SCOPE).open_request_id == REQUEST


@pytest.mark.parametrize("phase", ["pending", "bound"])
def test_retry_original_unmaterialized_uuid_even_blackbox(tmp_path, phase):
    store = store_at(tmp_path / "field.db", materialized=False)
    store.reserve_managed_open(SCOPE, artifact_id="old")
    if phase == "bound":
        store.record_managed_binding(SCOPE, REQUEST, BINDING)
    remote = Remote()
    remote.lost = "/v2/field-sessions/open"
    c = coordinator(remote)
    with pytest.raises(OSError):
        prepare(c, store, artifact="old")
    result = prepare(c, SQLiteFieldRecordStore(store.path), artifact="old")
    assert result.session_id == BINDING.session_id and result.status is FieldSessionStatus.OPEN
    assert all(
        body["open_request_id"] == REQUEST for path, body in remote.calls if path.endswith("/open")
    )


@pytest.mark.parametrize("kind", ["foreign", "missing", "unexpected_open", "gap", "collision"])
def test_drain_failure_does_not_ack_or_allocate(tmp_path, kind):
    store = store_at(tmp_path / "field.db")
    remote = Remote()
    cmd = command()
    if kind in {"foreign", "missing", "unexpected_open"}:
        cmd = replace(cmd, session_id=f"managed-{NEXT_BIRTH}")
        if kind == "unexpected_open":
            # A local legacy-shaped same-name open row is still no managed authority.
            other = SQLiteFieldRecordStore(store.path, session_id_factory=lambda: cmd.session_id)
            other.open_or_resume_session(artifact_id="unrelated")
    if kind == "foreign":
        cmd = replace(cmd, session_id="legacy-foreign")
    if kind == "gap":
        cmd = command(2)
    if kind == "collision":
        store.apply_field_command(command(), reduce_field_command)
        cmd = command(payload={"role": "blue"})
    remote.target = 2 if kind == "gap" else 1
    remote.commands = [cmd]
    with pytest.raises((ValueError, OSError)):
        prepare(coordinator(remote), store)
    assert not remote.acks
    assert not any(path.endswith("/open") for path, _ in remote.calls)
    assert store.get_active_managed_open_request(SCOPE).open_request_id == REQUEST


def test_closing_without_claim_is_not_inferred_sealed(tmp_path):
    store = store_at(tmp_path / "field.db")
    remote = Remote()
    remote.target = 1
    tick = iter([0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2])
    c = coordinator(remote, monotonic=lambda: next(tick, 2))
    with pytest.raises(TimeoutError):
        prepare(c, store, deadline=1)
    assert store.get_active_managed_open_request(SCOPE).lifecycle == "active"
    assert not remote.acks


def test_retired_predecessor_is_positively_repolled_before_new_reservation(tmp_path):
    store = store_at(tmp_path / "field.db")
    remote = Remote()
    # Stop after retirement by failing the new reservation, then retry from durable history.
    c = coordinator(remote)
    store._managed_open_request_id_factory = lambda: (_ for _ in ()).throw(
        OSError("crash before reserve")
    )
    with pytest.raises(OSError, match="crash before reserve"):
        prepare(c, store)
    assert store.get_active_managed_open_request(SCOPE) is None
    remote.calls.clear()
    restarted = SQLiteFieldRecordStore(
        store.path, managed_open_request_id_factory=lambda: NEXT_REQUEST
    )
    result = prepare(c, restarted)
    operations = [p for p, _ in remote.calls if p != "/v1/runtime-attestation"]
    assert operations == [
        f"/v2/field-sessions/{BINDING.session_id}/close",
        "/v2/field-sessions/open",
    ]
    assert result.session_id == f"managed-{NEXT_BIRTH}"


def test_cancel_before_start_keeps_all_durable_state_and_makes_no_http(tmp_path):
    store = store_at(tmp_path / "field.db")
    remote = Remote()
    with pytest.raises(OSError, match="cancel"):
        prepare(coordinator(remote), store, cancelled=lambda: True)
    assert store.load_session(BINDING.session_id).status is FieldSessionStatus.OPEN
    assert remote.calls == []


def test_network_never_holds_match_lock_or_sqlite_transaction(tmp_path):
    store = store_at(tmp_path / "field.db")
    remote = Remote()
    remote.target = 1
    remote.commands = [command()]
    lock = threading.RLock()
    module = startup_module()
    request = store.get_active_managed_open_request(SCOPE)
    store.end_session(BINDING.session_id)

    def check(_path):
        acquired = []

        def another_thread():
            with lock:
                with sqlite3.connect(store.path, timeout=0.1) as db:
                    db.execute("BEGIN IMMEDIATE")
                    acquired.append(True)

        t = threading.Thread(target=another_thread)
        t.start()
        t.join(1)
        assert acquired == [True]

    remote.hook = check
    gateway = SignedFieldCloudGateway(credentials=Credentials(), transport=remote)
    result = module.close_and_drain_once(store, request, gateway, lock)
    assert result.coverage == "closing" and remote.acks[0]["public_result_code"] == "SESSION_ENDED"
    assert module.close_and_drain_once(store, request, gateway, lock).coverage == "sealed"


def test_confirmation_replay_keeps_augmented_digest_after_projection_is_gone(tmp_path):
    from tests.unit.edge.route_assistance.test_turn_ownership import match, shot

    store = store_at(tmp_path / "field.db")
    store.save_match(BINDING.session_id, match(), expected_revision=0)
    start = replace(
        shot(), session_id=BINDING.session_id, media=replace(shot().media, artifact_id="old")
    )
    store.record_shot_start(start, source_mainline1_artifact_id="a" * 64)
    store.record_shot_settle(replace(start, settled_ms=150))
    reference = store.get_turn_confirmation(BINDING.session_id, start.shot_id).reference
    cmd = command(
        command_type="switch_turn",
        actor_user_id="user-red",
        expected_revision=1,
        payload={"confirmation_ref": asdict(reference)},
    )

    # Live reduction strips the already-validated reference before the pure
    # business reducer; storage retains the original envelope and digest.
    def reducer(match, envelope, **kwargs):
        return reduce_field_command(match, replace(envelope, payload={}), **kwargs)

    applied = store.apply_field_command(cmd, reducer, confirmation_ref=reference)
    assert applied.result_code == "APPLIED" and applied.resulting_revision == 2
    remote = Remote()
    remote.target = 1
    remote.commands = [cmd]
    prepare(coordinator(remote), store)
    assert remote.acks[0]["public_result_code"] == "APPLIED"
    assert remote.acks[0]["resulting_revision"] == applied.resulting_revision


@pytest.mark.parametrize("stop_after", ["first_attest", "second_attest", "mutation"])
def test_budget_is_rechecked_for_every_http_and_attestation(tmp_path, stop_after):
    module = startup_module()
    clock = [0.0]
    timeouts = []
    remote = Remote()

    def factory(_url, *, timeout_seconds, allow_redirects):
        assert allow_redirects is False
        timeouts.append(timeout_seconds)

        class Transport:
            def post_json(self, path, headers, body):
                result = remote.post_json(path, headers, body)
                clock[0] += 0.4
                return result

        return Transport()

    transport = module.BoundedStartupTransport(
        "https://synthetic.invalid",
        transport_factory=factory,
        deadline={"first_attest": 0.3, "second_attest": 0.7, "mutation": 1.1}[stop_after],
        cancelled=lambda: False,
        monotonic=lambda: clock[0],
    )
    gateway = DevelopmentAttestedFieldCloudGateway(
        SignedFieldCloudGateway(credentials=Credentials(), transport=transport), "dev"
    )
    with pytest.raises(TimeoutError):
        gateway.runtime_attestation()
        gateway.open_managed_session(open_request_id=REQUEST, expected_table_id="table-1")
    expected_calls = {"first_attest": 1, "second_attest": 2, "mutation": 3}[stop_after]
    assert len(remote.calls) == expected_calls
    assert timeouts == pytest.approx(
        [
            {"first_attest": 0.3, "second_attest": 0.7, "mutation": 1.1}[stop_after] - i * 0.4
            for i in range(expected_calls)
        ]
    )


def test_transport_cancellation_between_attestation_and_mutation(tmp_path):
    module = startup_module()
    remote = Remote()
    cancelled = [False]
    remote.hook = lambda _path: cancelled.__setitem__(0, True)
    transport = module.BoundedStartupTransport(
        "https://synthetic.invalid",
        transport_factory=lambda *_args, **_kwargs: remote,
        deadline=60,
        cancelled=lambda: cancelled[0],
        monotonic=lambda: 0,
    )
    gateway = DevelopmentAttestedFieldCloudGateway(
        SignedFieldCloudGateway(credentials=Credentials(), transport=transport), "dev"
    )
    with pytest.raises(OSError, match="cancel"):
        gateway.open_managed_session(open_request_id=REQUEST, expected_table_id="table-1")
    assert [p for p, _ in remote.calls] == ["/v1/runtime-attestation"]


@pytest.mark.parametrize(
    "damage",
    [
        "DELETE FROM field_sessions",
        "DELETE FROM field_match_state",
        "UPDATE field_match_state SET revision=999",
        "UPDATE field_sessions SET artifact_id='wrong'",
        "UPDATE field_sessions SET ended_at_utc='bad'",
        "DELETE FROM field_cloud_close_intents",
        "UPDATE field_cloud_close_intents SET status='open'",
        "UPDATE field_cloud_close_intents SET updated_at_utc='bad'",
    ],
)
def test_invalid_retained_scene_or_marker_refuses_before_network(tmp_path, damage):
    store = store_at(tmp_path / "field.db")
    store.end_session(BINDING.session_id)
    with sqlite3.connect(store.path) as db:
        db.execute(damage)
    remote = Remote()
    with pytest.raises(ValueError):
        prepare(coordinator(remote), store)
    assert remote.calls == []


@pytest.mark.parametrize("phase", ["pending", "bound"])
def test_remote_ended_without_local_scene_keeps_original_recovery_request(tmp_path, phase):
    store = store_at(tmp_path / "field.db", materialized=False)
    store.reserve_managed_open(SCOPE, artifact_id="old")
    if phase == "bound":
        store.record_managed_binding(SCOPE, REQUEST, BINDING)
    remote = Remote()
    remote.bindings[REQUEST] = replace(BINDING, status="ended")
    with pytest.raises(ValueError):
        prepare(coordinator(remote), store, artifact="old")
    request = store.get_active_managed_open_request(SCOPE)
    assert request.open_request_id == REQUEST and request.binding.status == "ended"
    assert store.find_open_session("old") is None


@pytest.mark.parametrize("phase", ["pending", "bound"])
def test_pending_resource_drift_refuses_before_http(tmp_path, phase):
    store = store_at(tmp_path / "field.db", materialized=False)
    store.reserve_managed_open(SCOPE, artifact_id="old")
    if phase == "bound":
        store.record_managed_binding(SCOPE, REQUEST, BINDING)
    remote = Remote()
    with pytest.raises(ValueError):
        prepare(coordinator(remote), store, artifact="new")
    assert remote.calls == []


def test_default_deadline_is_sixty_seconds_and_factory_gets_cancellation(tmp_path):
    module = startup_module()
    store = store_at(tmp_path / "field.db", materialized=False)
    seen = []

    def cancel():
        return False

    remote = Remote()
    gateway = SignedFieldCloudGateway(credentials=Credentials(), transport=remote)
    c = module.ManagedStartupCoordinator(
        scope=SCOPE,
        monotonic=lambda: 100,
        gateway_factory=lambda **kwargs: seen.append(kwargs) or gateway,
    )
    prepare(c, store, artifact="old", cancelled=cancel)
    assert seen == [dict(deadline=160, cancelled=cancel)]


def test_each_transport_request_clamps_to_twenty_seconds():
    module = startup_module()
    remote = Remote()
    timeouts = []

    def factory(_url, **kwargs):
        timeouts.append(kwargs["timeout_seconds"])
        return remote

    transport = module.BoundedStartupTransport(
        "https://synthetic.invalid",
        transport_factory=factory,
        deadline=100,
        cancelled=lambda: False,
        monotonic=lambda: 0,
    )
    SignedFieldCloudGateway(credentials=Credentials(), transport=transport).runtime_attestation()
    assert timeouts == [20]


def test_invalid_retained_state_refuses_before_constructing_attested_gateway(tmp_path):
    store = store_at(tmp_path / "field.db")
    with sqlite3.connect(store.path) as db:
        db.execute("DELETE FROM field_cloud_close_intents")

    def factory(**_kwargs):
        pytest.fail("attested gateway constructed before retained state validation")

    c = startup_module().ManagedStartupCoordinator(scope=SCOPE, gateway_factory=factory)
    with pytest.raises(ValueError):
        prepare(c, store)


def test_claim_sdk_failure_keeps_actual_failure_and_never_allocates(tmp_path):
    store = store_at(tmp_path / "field.db")
    remote = Remote()
    remote.target = 1

    def hook(path):
        if path == "/v1/field-commands/claim":
            raise OSError("SDK_QUERY_FAILED")

    remote.hook = hook
    with pytest.raises(OSError, match="SDK_QUERY_FAILED"):
        prepare(coordinator(remote), store)
    assert remote.acks == []
    assert store.get_active_managed_open_request(SCOPE).open_request_id == REQUEST


@pytest.mark.parametrize(
    "operation",
    [
        "end_session",
        "retire_managed_open",
        "reserve_managed_open",
        "record_managed_binding",
        "materialize_managed_session",
    ],
)
def test_restart_after_each_local_commit_keeps_original_request_and_history(tmp_path, operation):
    store = store_at(tmp_path / "field.db")
    remote = Remote()
    original = getattr(store, operation)

    def lost_return(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("crash after durable commit")

    setattr(store, operation, lost_return)
    with pytest.raises(OSError, match="crash after durable commit"):
        prepare(coordinator(remote), store)
    restarted = SQLiteFieldRecordStore(
        store.path,
        managed_open_request_id_factory=lambda: NEXT_REQUEST,
        session_id_factory=lambda: pytest.fail("local successor identity"),
    )
    result = prepare(coordinator(remote), restarted, policy="usb_resume")
    assert result.session_id == f"managed-{NEXT_BIRTH}"
    assert restarted.get_latest_retired_managed_open_request(SCOPE).open_request_id == REQUEST
    assert restarted.get_active_managed_open_request(SCOPE).open_request_id == NEXT_REQUEST
    assert restarted.load_session(BINDING.session_id).status is FieldSessionStatus.ENDED
    assert BINDING.session_id in restarted.list_pending_cloud_close_session_ids()


@pytest.mark.parametrize("policy", ["usb_resume", "blackbox_fresh"])
def test_legal_trailing_interrupted_shot_is_retained_without_fake_settle(tmp_path, policy):
    from tests.unit.edge.route_assistance.test_turn_ownership import shot

    store = store_at(tmp_path / "field.db")
    start = replace(
        shot(), session_id=BINDING.session_id, media=replace(shot().media, artifact_id="old")
    )
    store.record_shot_start(start, source_mainline1_artifact_id="a" * 64)
    result = prepare(coordinator(Remote()), store, policy=policy, artifact="old")
    assert result.session_id == f"managed-{NEXT_BIRTH}"
    assert store.list_shots(BINDING.session_id) == (start,)
    assert (
        store.get_shot_ownership(BINDING.session_id, start.shot_id).source_mainline1_artifact_id
        == "a" * 64
    )


@pytest.mark.parametrize("damage", ["gap", "wrong_session", "wrong_artifact", "earlier_unsettled"])
def test_ambiguous_shot_recovery_refuses_before_end_or_http(tmp_path, damage):
    from tests.unit.edge.route_assistance.test_turn_ownership import shot

    store = store_at(tmp_path / "field.db")
    start = replace(
        shot(), session_id=BINDING.session_id, media=replace(shot().media, artifact_id="old")
    )
    store.record_shot_start(start)
    with sqlite3.connect(store.path) as db:
        payload = json.loads(db.execute("SELECT payload_json FROM field_shots").fetchone()[0])
        if damage == "gap":
            payload["sequence"] = 2
        if damage == "wrong_session":
            payload["session_id"] = "unrelated"
        if damage == "wrong_artifact":
            payload["media"]["artifact_id"] = "other"
        db.execute("UPDATE field_shots SET payload_json=?", (json.dumps(payload),))
    if damage == "earlier_unsettled":
        next_shot = replace(start, shot_id="shot-0002", sequence=2, started_ms=200)
        # Insert deliberately contradictory persisted recovery evidence.
        with sqlite3.connect(store.path) as db:
            db.execute(
                "INSERT INTO field_shots VALUES(?,?,?,?)",
                (BINDING.session_id, "shot-0002", 2, json.dumps(asdict(next_shot))),
            )
    remote = Remote()
    with pytest.raises(ValueError):
        prepare(coordinator(remote), store)
    assert remote.calls == []
    assert store.load_session(BINDING.session_id).status is FieldSessionStatus.OPEN


def test_usb_remote_ended_replay_really_closes_old_scene_before_new_birth(tmp_path):
    store = store_at(tmp_path / "field.db")
    remote = Remote()
    remote.bindings[REQUEST] = replace(BINDING, status="ended")
    result = prepare(coordinator(remote), store, policy="usb_resume", artifact="old")
    assert result.session_id == f"managed-{NEXT_BIRTH}"
    assert store.load_session(BINDING.session_id).status is FieldSessionStatus.ENDED
