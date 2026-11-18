"""Real signed offline peer/store controls; no server, model, camera or native run."""

import threading

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.field_assistance.server import _BackgroundFieldCommandWorker
from scripts.algorithm.offline_managed_transport import OfflineManagedRemote
from tests.unit.scripts.test_offline_managed_transport import opened, runtime


@pytest.mark.parametrize("manual_end", [False, True])
def test_fullgame_three_actual_setup_commands_and_sealed_close(tmp_path, manual_end):
    remote = OfflineManagedRemote(observer_mode="fullgame-v1")
    cloud = runtime(remote)
    store = SQLiteFieldRecordStore(tmp_path / "field.sqlite3")
    session = cloud.make_managed_startup().prepare(
        store, artifact_id="offline-artifact", rule_profile_id=None, policy="blackbox_fresh"
    )
    worker = _BackgroundFieldCommandWorker(
        gateway=cloud.gateway, store=store, match_lock=threading.RLock()
    )
    specs = [
        ("offline-red", "claim_role", {"role": "red"}),
        ("offline-blue", "claim_role", {"role": "blue"}),
        (
            "offline-red",
            "set_breaker",
            dict(
                role="red",
                mode="assistance",
                format="single",
                race_target=None,
                ranking_intent=False,
                settings_revision=1,
            ),
        ),
    ]
    for actor, kind, payload in specs:
        command = remote.issue(
            session_id=session.session_id,
            actor_user_id=actor,
            command_type=kind,
            payload=payload,
            expected_revision=store.load_session(session.session_id).match.revision,
        )
        assert worker.claim_and_apply_once(raise_errors=True)
        assert remote.completion(command).outcome == "applied"
    assert store.load_session(session.session_id).status.value == "open"
    if manual_end:
        with pytest.raises(ValueError, match="ISSUE_CAPACITY"):
            remote.issue(
                session_id=session.session_id,
                actor_user_id="offline-red",
                command_type="end_session",
                payload={},
                expected_revision=3,
            )
        assert remote.snapshot()["issued_count"] == 3
        return
    binding = cloud.gateway.open_managed_session(
        open_request_id=store.get_managed_open_request_for_session(
            cloud.make_managed_startup().scope, session.session_id
        ).open_request_id,
        expected_table_id="offline-table",
    )
    close = cloud.gateway.close_managed_session(binding=binding)
    assert (close.coverage, close.closed_target, close.contiguous_ack_sequence) == ("sealed", 3, 3)
    with pytest.raises(ValueError, match="ISSUE_AFTER_CLOSE"):
        remote.issue(
            session_id=session.session_id,
            actor_user_id="offline-red",
            command_type="end_session",
            payload={},
            expected_revision=3,
        )


@pytest.mark.parametrize(
    "actor,kind,payload",
    [
        ("offline-blue", "claim_role", {"role": "blue"}),
        ("offline-red", "end_session", {}),
    ],
)
def test_fullgame_remote_forbids_wrong_first_actor_and_manual_end(actor, kind, payload):
    remote = OfflineManagedRemote(observer_mode="fullgame-v1")
    _, _, binding = opened(remote)
    with pytest.raises(ValueError, match="ISSUE_ORDER"):
        remote.issue(
            session_id=binding.session_id,
            actor_user_id=actor,
            command_type=kind,
            payload=payload,
            expected_revision=0,
        )
    assert remote.snapshot()["issued_count"] == 0


def test_unknown_observer_version_is_rejected():
    with pytest.raises(ValueError, match="unknown offline observer mode"):
        OfflineManagedRemote(observer_mode="future")
