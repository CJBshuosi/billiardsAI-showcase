"""Frozen ownership must control selection, replay and publication."""

import json
import sqlite3

import pytest

from tests.unit.edge.route_assistance.test_persistence import (
    NOW,
    _available_clip,
    _postgame_store,
    _save_clip,
    _store,
)


def private_store(tmp_path):
    store = _postgame_store(tmp_path / "field.sqlite3")
    return store


def mutate(store, table, transform):
    column = "payload_json" if table != "field_clip_save_requests" else "receipt_json"
    with sqlite3.connect(store.path) as db:
        value = json.loads(db.execute(f"SELECT {column} FROM {table}").fetchone()[0])
        transform(value)
        db.execute(
            f"UPDATE {table} SET {column}=?",
            (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False),),
        )


def test_saved_owner_survives_roster_change_delayed_generation_and_restart(tmp_path):
    store = private_store(tmp_path)
    with sqlite3.connect(store.path) as db:
        raw = json.loads(db.execute("SELECT payload_json FROM field_match_state").fetchone()[0])
        raw["red_user_id"] = "replacement-C"
        db.execute("UPDATE field_match_state SET payload_json=?", (json.dumps(raw),))
    receipt = _save_clip(store)
    assert receipt.schema_version == "2.0"
    assert receipt.owner_user_id == "manual-red"
    assert receipt.actor_role == "red"
    _available_clip(store, tmp_path)
    restarted = _store(store.path)
    item = restarted.claim_due_saved_media(NOW)
    descriptor = json.loads(item.descriptor_json)
    assert descriptor["schema_version"] == "2.0"
    assert descriptor["owner_user_id"] == "manual-red"
    assert descriptor["ownership_receipt"]["actor_user_id"] == "manual-red"
    assert descriptor["ownership_receipt_sha256"] == receipt.ownership_receipt_sha256
    clip = restarted.list_postgame_clips("session-1")[0]
    assert (clip.sequence, clip.actor_role, clip.owner_user_id, clip.can_save) == (
        1,
        "red",
        "manual-red",
        True,
    )
    assert _save_clip(restarted) == receipt


@pytest.mark.parametrize(
    "table,changes",
    [
        ("field_shot_ownership", {"source_epoch_id": "wrong"}),
        ("field_shot_ownership", {"source_started_ms": 3001}),
        ("field_shot_ownership", {"sequence": 2}),
        ("field_shot_ownership", {"match_revision": 2}),
        (
            "field_shot_ownership",
            {"evidence_status": "unknown", "actor_user_id": None, "actor_role": None},
        ),
        ("field_shot_ownership", {"revision": True}),
        ("field_shot_ownership", {"extra": 1}),
        ("field_live_shot_media", {"job_id": "wrong"}),
        ("field_live_shot_media", {"match_revision": 2}),
    ],
)
def test_invalid_ownership_disables_and_rejects_save_but_keeps_replay(tmp_path, table, changes):
    store = private_store(tmp_path)
    _available_clip(store, tmp_path)
    mutate(store, table, lambda value: value.update(changes))
    clip = store.list_postgame_clips("session-1")[0]
    assert clip.can_save is False
    assert clip.ownership_unavailable_reason == "SHOT_OWNERSHIP_UNAVAILABLE"
    assert clip.local_path is not None
    with pytest.raises(ValueError, match="ownership"):
        _save_clip(store)


@pytest.mark.parametrize("table", ["field_shot_ownership", "field_live_shot_media"])
def test_missing_provenance_fails_closed(tmp_path, table):
    store = private_store(tmp_path)
    with sqlite3.connect(store.path) as db:
        db.execute(f"DELETE FROM {table}")
    with pytest.raises(ValueError, match="ownership"):
        _save_clip(store)


@pytest.mark.parametrize("target", ["field_shot_ownership", "field_clip_save_requests"])
def test_tampered_owner_blocks_existing_outbox_and_request_replay(tmp_path, target):
    store = private_store(tmp_path)
    _save_clip(store)
    _available_clip(store, tmp_path)
    key = "actor_user_id" if target == "field_shot_ownership" else "owner_user_id"
    mutate(store, target, lambda value: value.update({key: "attacker"}))
    with pytest.raises(ValueError, match="ownership"):
        _save_clip(store)
    assert store.claim_due_saved_media(NOW) is None
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT status FROM field_saved_media_outbox").fetchone()[0] == "blocked"


def test_legacy_receipt_replays_without_rewrite_but_cannot_upload(tmp_path):
    store = private_store(tmp_path)
    _save_clip(store)
    _available_clip(store, tmp_path)

    def legacy(value):
        for key in ("schema_version", "owner_user_id", "actor_role", "ownership_receipt_sha256"):
            value.pop(key, None)

    mutate(store, "field_clip_save_requests", legacy)
    with sqlite3.connect(store.path) as db:
        original = db.execute("SELECT receipt_json FROM field_clip_save_requests").fetchone()[0]
    assert _save_clip(store).schema_version == "1.0"
    assert store.claim_due_saved_media(NOW) is None
    with sqlite3.connect(store.path) as db:
        assert (
            db.execute("SELECT receipt_json FROM field_clip_save_requests").fetchone()[0]
            == original
        )


def test_v2_fixture_matches_python_descriptor_and_recursive_utf8_digest():
    import hashlib
    from pathlib import Path

    from billiards_ai.edge.route_assistance.postgame import SavedMediaOutboxItem
    from billiards_ai.edge.route_assistance.records import ownership_receipt_from_payload

    fixture = json.loads(
        (
            Path(__file__).parents[3]
            / "fixtures/synthetic/miniprogram/saved-postgame-clip-e2e-v2.json"
        ).read_text()
    )
    descriptor = fixture["descriptor"]
    ownership = ownership_receipt_from_payload(descriptor["ownership_receipt"])
    item = SavedMediaOutboxItem(
        **{
            **{
                key: value
                for key, value in descriptor.items()
                if key not in {"schema_version", "publication_kind", "ownership_receipt"}
            },
            "ownership_receipt": ownership,
            "local_path": Path("/synthetic/clip.mp4"),
            "status": "pending",
            "attempts": 0,
            "claim_token": None,
            "lease_until_utc": None,
            "next_attempt_at_utc": "2026-10-29T00:00:00Z",
            "last_error_code": None,
        }
    )
    assert item.descriptor_json == fixture["descriptor_json"]
    assert (
        hashlib.sha256(item.descriptor_json.encode("utf-8")).hexdigest()
        == fixture["descriptor_sha256"]
    )


def test_roster_revision_preserves_physical_evidence_but_fences_stale_rule_application(tmp_path):
    from dataclasses import replace

    from billiards_ai.edge.route_assistance.eight_ball_rules import decide_eight_ball
    from billiards_ai.edge.route_assistance.records import FieldShotRecord, ShotMediaRange
    from tests.unit.edge.route_assistance.test_rule_persistence import _facts, _match, _store

    store = _store(tmp_path)
    session = store.open_or_resume_session(artifact_id="usb-rules")
    match = _match()
    store.save_match(session.session_id, match, expected_revision=0)
    started = FieldShotRecord(
        session.session_id,
        "shot-0001",
        1,
        match.active_role,
        1000,
        None,
        None,
        None,
        None,
        ShotMediaRange("usb-rules", "usb:epoch-rules", "usb_live", 1000, 1001, "epoch-rules"),
    )
    store.record_shot_start(started, source_start_event_id="start-1")
    changed = replace(match, red_user_id="replacement-C", revision=2)
    store.save_match(session.session_id, changed, expected_revision=1)
    settled = replace(
        started,
        settled_ms=2000,
        media=replace(started.media, end_ms=2000),
        live_job_id="job-1",
        live_evidence_digest="e" * 64,
    )
    store.record_shot_settle(settled, source_epoch_id="epoch-rules")
    assert store.list_shots(session.session_id)[0].settled_ms == 2000
    with pytest.raises(ValueError, match="stale"):
        store.apply_automatic_rule_decision(
            _facts(), decide_eight_ball(match, _facts()), elapsed_ms=1.0
        )
    assert store.load_session(session.session_id).match == changed
    assert store.get_shot_ownership(session.session_id, "shot-0001").actor_user_id == "user-red"


def test_v2_fixture_has_identical_canonical_json_and_hash_in_javascript():
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("JavaScript runtime unavailable")
    fixture = (
        Path(__file__).parents[3] / "fixtures/synthetic/miniprogram/saved-postgame-clip-e2e-v2.json"
    )
    script = r"""
const fs=require('node:fs'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const f=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const sorted=x=>Array.isArray(x)?x.map(sorted):x&&typeof x==='object'?
 Object.fromEntries(Object.keys(x).sort().map(k=>[k,sorted(x[k])])):x;
const canonical=x=>JSON.stringify(sorted(x));
const sha=x=>crypto.createHash('sha256').update(x,'utf8').digest('hex');
assert.equal(canonical(f.descriptor.ownership_receipt),f.ownership_receipt_json);
assert.equal(sha(f.ownership_receipt_json),f.descriptor.ownership_receipt_sha256);
assert.equal(canonical(f.descriptor),f.descriptor_json);
assert.equal(sha(f.descriptor_json),f.descriptor_sha256);
assert.equal(canonical(f.save_receipt),f.save_receipt_json);
assert.equal(sha(f.save_receipt_json),f.descriptor.save_receipt_sha256);
"""
    result = subprocess.run([node, "-e", script, str(fixture)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "changes",
    [
        {"sequence": 9007199254740992},
        {"match_revision": True},
        {"evidence_reason": {"number": 9007199254740992}},
        {"actor_user_id": " "},
        {"actor_user_id": "owner\nforged"},
        {"source_start_event_id": ""},
    ],
)
def test_malformed_receipt_cannot_project_save_authority(tmp_path, changes):
    store = private_store(tmp_path)
    mutate(store, "field_shot_ownership", lambda value: value.update(changes))
    assert store.list_postgame_clips("session-1")[0].can_save is False
    with pytest.raises(ValueError, match="ownership"):
        _save_clip(store)


def test_blue_owner_is_frozen_independently_from_red(tmp_path):
    from dataclasses import replace

    from billiards_ai.edge.route_assistance.control import WristbandRole
    from billiards_ai.edge.route_assistance.records import LiveShotMediaReference, ShotMediaRange
    from tests.unit.edge.route_assistance.test_persistence import _match, _shot

    store = _store(tmp_path / "field.sqlite3")
    store.open_or_resume_session(artifact_id="artifact-1")
    store.save_match("session-1", _match(1, role=WristbandRole.BLUE), expected_revision=0)
    started = replace(
        _shot(),
        actor_role=WristbandRole.BLUE,
        media=ShotMediaRange("artifact-1", "usb:epoch-1", "usb_live", 3000, 3001, "epoch-1"),
    )
    store.record_shot_start(started, source_start_event_id="blue-start")
    store.record_shot_settle(
        replace(
            started,
            settled_ms=5000,
            media=replace(started.media, end_ms=5000),
            live_job_id="job-1",
            live_evidence_digest="e" * 64,
        ),
        source_epoch_id="epoch-1",
    )
    store.attach_live_shot_media(
        LiveShotMediaReference(
            "session-1", "shot-0001", "epoch-1", 1, "job-1", str(tmp_path / "usb"), "d" * 64
        )
    )
    store.end_session("session-1")
    receipt = _save_clip(store)
    assert (receipt.owner_user_id, receipt.actor_role) == ("manual-blue", "blue")
    _available_clip(store, tmp_path)
    assert (
        json.loads(store.claim_due_saved_media(NOW).descriptor_json)["owner_user_id"]
        == "manual-blue"
    )


def test_usb_fact_rejects_settle_roster_user(tmp_path):
    from dataclasses import replace

    from billiards_ai.edge.route_assistance.records import LiveShotFact

    store = private_store(tmp_path)
    shot = store.list_shots("session-1")[0]
    fact = LiveShotFact.create(
        session_id="session-1",
        shot_id="shot-0001",
        sequence=1,
        actor_role=shot.actor_role,
        actor_user_id="replacement-C",
        start_frame_id=1,
        settle_frame_id=2,
        mode="observation",
        observed_outcome="unknown",
        rule_revision=2,
        recording_reference="session-recording:epoch-1:1-2",
    )
    with pytest.raises(ValueError, match="ownership"):
        store.record_shot_settle(replace(shot), source_epoch_id="epoch-1", live_fact=fact)
    assert store.get_live_shot_fact("session-1", "shot-0001") is None


def test_late_materialization_cannot_enqueue_changed_destination(tmp_path):
    store = private_store(tmp_path)
    _save_clip(store)
    mutate(store, "field_shot_ownership", lambda value: value.update(actor_user_id="replacement-C"))
    with pytest.raises(ValueError, match="ownership"):
        _available_clip(store, tmp_path)
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM field_saved_media_outbox").fetchone()[0] == 0


@pytest.mark.parametrize("bad_payload", ["[]", "null", "not-json"])
def test_corrupt_outbox_shot_is_quarantined_without_crashing_claim(tmp_path, bad_payload):
    store = private_store(tmp_path)
    _save_clip(store)
    _available_clip(store, tmp_path)
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE field_shots SET payload_json=?", (bad_payload,))
    assert store.claim_due_saved_media(NOW) is None
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT status FROM field_saved_media_outbox").fetchone()[0] == "blocked"


def test_non_text_outbox_metadata_is_blocked_instead_of_crashing_claim(tmp_path):
    store = private_store(tmp_path)
    _save_clip(store)
    _available_clip(store, tmp_path)
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE field_postgame_clip_availability SET content_sha256=?", (b"a" * 64,))
    assert store.claim_due_saved_media(NOW) is None
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT status FROM field_saved_media_outbox").fetchone()[0] == "blocked"


@pytest.mark.parametrize("field", ["session_id", "evidence_reason"])
@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"], ids=["high", "low"])
def test_ownership_receipt_rejects_non_utf8_required_and_optional_text(tmp_path, field, surrogate):
    from dataclasses import asdict, replace

    from billiards_ai.edge.route_assistance.records import (
        ownership_receipt_from_payload,
        ownership_receipt_json,
    )

    store = private_store(tmp_path)
    receipt = store.get_shot_ownership("session-1", "shot-0001")
    malformed = replace(receipt, **{field: "invalid-" + surrogate})
    with pytest.raises(ValueError, match="ownership"):
        ownership_receipt_from_payload(asdict(malformed))
    with pytest.raises(ValueError, match="ownership"):
        ownership_receipt_json(malformed)


@pytest.mark.parametrize("field", ["actor_user_id", "source_start_event_id", "evidence_reason"])
@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"], ids=["high", "low"])
def test_escaped_surrogate_receipt_disables_save_and_retains_local_replay(
    tmp_path, field, surrogate
):
    store = private_store(tmp_path)
    _available_clip(store, tmp_path)
    with sqlite3.connect(store.path) as db:
        payload = json.loads(
            db.execute("SELECT payload_json FROM field_shot_ownership").fetchone()[0]
        )
        payload[field] = "invalid-" + surrogate
        # JSON escapes are valid UTF-8 storage bytes; decoded text still must be rejected.
        db.execute(
            "UPDATE field_shot_ownership SET payload_json=?",
            (json.dumps(payload, ensure_ascii=True),),
        )
    clip = store.list_postgame_clips("session-1")[0]
    assert clip.can_save is False
    assert clip.ownership_unavailable_reason == "SHOT_OWNERSHIP_UNAVAILABLE"
    assert clip.local_path is not None
    with pytest.raises(ValueError, match="ownership"):
        _save_clip(store)
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM field_clip_save_requests").fetchone()[0] == 0
