"""Pin actual local selections to the descriptors consumed by signed cloud E2E.

Changing save/outbox ownership to the final roster, upgrading a pending receipt,
or filling unknown ownership from a later roster must break this acceptance.
Only synthetic bytes and controlled source/control snapshots are used here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope, reduce_field_command
from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import (
    FieldShotRecord,
    LiveShotFact,
    LiveShotMediaReference,
    ShotMediaRange,
)
from billiards_ai.edge.route_assistance.sync import FieldSyncClient
from scripts.algorithm.serve_mainline1_usb_field import _BackgroundSavedMediaSyncWorker
from tests.integration.lab.field_assistance.test_postgame_screen_saved_clips import (
    _CloudContractTransport,
    _Credentials,
)

NOW = datetime(2026, 10, 29, 10, tzinfo=UTC)
FIXTURE_PATH = (
    Path(__file__).parents[3] / "fixtures/synthetic/miniprogram/private-shot-clips-v2.json"
)
MP4 = bytes.fromhex("000000186674797069736f6d0000020069736f6d69736f32")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def build_private_fixture(tmp_path, *, unknown_red=False):
    """Generate portable evidence through real start, settle, media and save APIs."""
    database = tmp_path / "field.sqlite3"
    store = SQLiteFieldRecordStore(
        database, now_utc=lambda: NOW, session_id_factory=lambda: "session-private"
    )
    session = store.open_or_resume_session(artifact_id="field-artifact-private")
    authority = {"field_session_commands": {}, "session_role_binding_history": {}}

    def authority_row(command_id, role, user, revision, command_type="claim_role"):
        history = {
            "session_id": session.session_id,
            "role": role,
            "user_id": user,
            "resulting_revision": revision,
            "command_id": command_id,
            "table_id": "table-1",
            "device_id": "device-1",
        }
        authority["session_role_binding_history"][
            sha(f"{session.session_id}:{role}:{revision}")
        ] = history
        authority["field_session_commands"][command_id] = {
            "command_id": command_id,
            "session_id": session.session_id,
            "table_id": "table-1",
            "claimed_by_device_id": "device-1",
            "command_type": command_type,
            "actor_user_id": user or "user-a",
            "validated_payload": {"role": role} if command_type == "claim_role" else {},
            "applied_role": role,
            "status": "applied",
            "resulting_revision": revision,
        }

    for sequence, (role, user) in enumerate((("red", "user-a"), ("blue", "user-b")), 1):
        command = FieldCommandEnvelope(
            command_id=f"claim-{role}",
            session_id=session.session_id,
            actor_user_id=user,
            command_type="claim_role",
            payload={"role": role},
            expected_revision=sequence - 1,
            server_sequence=sequence,
        )
        result = store.apply_field_command(command, reduce_field_command)
        assert result.outcome == "applied"
        snapshot = store.load_session(session.session_id).match
        bound_user = snapshot.red_user_id if role == "red" else snapshot.blue_user_id
        authority_row(command.command_id, role, bound_user, result.resulting_revision)

    snapshot = store.load_session(session.session_id).match
    store.save_match(
        session.session_id,
        replace(snapshot, active_role=WristbandRole.RED, revision=3),
        expected_revision=2,
    )
    for sequence, role in ((1, WristbandRole.RED), (2, WristbandRole.BLUE)):
        if sequence == 2:
            snapshot = store.load_session(session.session_id).match
            store.save_match(
                session.session_id,
                replace(snapshot, active_role=role, revision=4),
                expected_revision=3,
            )
        match = store.load_session(session.session_id).match
        start = 3000 * sequence
        shot = FieldShotRecord(
            session_id=session.session_id,
            shot_id=f"shot-{sequence:04d}",
            sequence=sequence,
            actor_role=role,
            started_ms=start,
            settled_ms=None,
            recommendation_id=None,
            power_band=None,
            power_percent=None,
            media=ShotMediaRange(
                session.artifact_id,
                f"usb:epoch-{sequence}",
                "usb_live",
                start,
                start + 1,
                f"epoch-{sequence}",
            ),
        )
        store.record_shot_start(
            shot,
            source_start_event_id=f"start-击球-{sequence}",
            ownership_ambiguous=unknown_red and sequence == 1,
        )
        owner = store.get_shot_ownership(session.session_id, shot.shot_id)
        fact = (
            None
            if owner.actor_user_id is None
            else LiveShotFact.create(
                session_id=session.session_id,
                shot_id=shot.shot_id,
                sequence=sequence,
                actor_role=role,
                actor_user_id=owner.actor_user_id,
                start_frame_id=sequence * 100,
                settle_frame_id=sequence * 100 + 60,
                mode="observation",
                observed_outcome="unknown",
                rule_revision=match.revision,
                recording_reference=f"session-recording:epoch-{sequence}:100-160",
            )
        )
        store.record_shot_settle(
            replace(
                shot,
                settled_ms=start + 2000,
                media=replace(shot.media, end_ms=start + 2000),
                live_job_id=f"job-{sequence}",
                live_evidence_digest=str(sequence) * 64,
            ),
            source_epoch_id=f"epoch-{sequence}",
            live_fact=fact,
        )
        store.attach_live_shot_media(
            LiveShotMediaReference(
                session.session_id,
                shot.shot_id,
                f"epoch-{sequence}",
                match.revision,
                f"job-{sequence}",
                str(tmp_path / "raw" / f"job-{sequence}"),
                "d" * 64,
            )
        )

    # Deliberate authoritative roster replacement, not a claim allowed by the
    # live reducer after play started. Later history must not decide old shots.
    snapshot = store.load_session(session.session_id).match
    released = replace(snapshot, red_user_id=None, revision=5)
    store.save_match(session.session_id, released, expected_revision=4)
    authority_row("release-red", "red", released.red_user_id, released.revision, "release_role")
    replacement = replace(released, red_user_id="user-c", revision=6)
    store.save_match(session.session_id, replacement, expected_revision=5)
    authority_row("claim-replacement-red", "red", replacement.red_user_id, replacement.revision)
    store.end_session(session.session_id)
    manifest = store.get_postgame_manifest(session.session_id)

    def select(sequence):
        return store.save_postgame_clip(
            request_id=f"screen-save-{sequence}",
            session_id=session.session_id,
            shot_id=f"shot-{sequence:04d}",
            postgame_manifest_sha256=manifest.manifest_sha256,
        )

    def available(sequence):
        media = tmp_path / "postgame" / session.session_id / f"clip-{sequence}.mp4"
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(MP4)
        store.record_postgame_clip_available(
            session_id=session.session_id,
            shot_id=f"shot-{sequence:04d}",
            job_id=f"job-{sequence}",
            artifact_id=str(sequence + 2) * 64,
            local_path=media,
            content_sha256=hashlib.sha256(MP4).hexdigest(),
            byte_size=len(MP4),
            postgame_manifest_sha256=manifest.manifest_sha256,
        )

    if unknown_red:
        available(1)
        assert not store.list_postgame_clips(session.session_id)[0].can_save
        with pytest.raises(ValueError, match="ownership"):
            select(1)
        assert store.list_saved_media_outbox() == ()
        assert store.get_shot_ownership(session.session_id, "shot-0001").actor_user_id is None
        return store, None

    red = select(1)
    assert red.status == "SAVE_PENDING" and red.clip_id is None
    assert store.list_saved_media_outbox() == ()
    with sqlite3.connect(database) as db:
        original_red_bytes = db.execute(
            "SELECT receipt_json FROM field_clip_save_requests WHERE request_id='screen-save-1'"
        ).fetchone()[0]
    available(1)
    available(2)
    blue = select(2)
    assert blue.status == "SAVED" and blue.clip_id is not None
    store = SQLiteFieldRecordStore(database, now_utc=lambda: NOW)
    assert select(1) == red and select(2) == blue
    with sqlite3.connect(database) as db:
        assert (
            db.execute(
                "SELECT receipt_json FROM field_clip_save_requests WHERE request_id='screen-save-1'"
            ).fetchone()[0]
            == original_red_bytes
        )
    clips = store.list_postgame_clips(session.session_id)
    assert [(row.actor_role, row.owner_user_id, row.can_save) for row in clips] == [
        ("red", "user-a", True),
        ("blue", "user-b", True),
    ]
    assert manifest.expected_jobs == ("job-1", "job-2")
    selections = []
    for item in sorted(store.list_saved_media_outbox(), key=lambda row: row.shot_id):
        receipt = {red.shot_id: red, blue.shot_id: blue}[item.shot_id]
        descriptor = json.loads(item.descriptor_json)
        selections.append(
            {
                "descriptor": descriptor,
                "descriptor_json": item.descriptor_json,
                "descriptor_sha256": sha(item.descriptor_json),
                "save_receipt": asdict(receipt),
                "save_receipt_json": canonical(asdict(receipt)),
                "ownership_receipt_json": canonical(descriptor["ownership_receipt"]),
            }
        )
        assert descriptor["save_receipt_sha256"] == sha(canonical(asdict(receipt)))
        assert descriptor["job_id"] == f"job-{descriptor['ownership_receipt']['sequence']}"
        media = store.get_live_shot_media(session.session_id, item.shot_id)
        assert media.job_id == item.job_id
        assert media.source_epoch_id == item.ownership_receipt.source_epoch_id
        assert media.match_revision == item.ownership_receipt.match_revision
    final = store.load_session(session.session_id).match
    fixture = {
        "schema_version": "2.0",
        "mp4_hex": MP4.hex(),
        "selections": selections,
        "source_facts": [
            asdict(store.get_live_shot_fact(session.session_id, shot.shot_id))
            for shot in store.list_shots(session.session_id)
        ],
        "authority_seed": authority,
        "published_session": {
            "schema_version": "1.0",
            "session_id": session.session_id,
            "artifact_id": session.artifact_id,
            "status": "ended",
            "sync_status": "published",
            "evidence_source_kind": "usb_live",
            "postgame_manifest_sha256": manifest.manifest_sha256,
            "participants": [
                {"user_id": final.red_user_id, "role": "red"},
                {"user_id": final.blue_user_id, "role": "blue"},
            ],
            "shots": [
                {
                    "shot_id": shot.shot_id,
                    "sequence": shot.sequence,
                    "started_ms": shot.started_ms,
                    "settled_ms": shot.settled_ms,
                    "live_job_id": shot.live_job_id,
                    "actor_role": shot.actor_role,
                }
                for shot in store.list_shots(session.session_id)
            ],
            "_sync_publication": {
                "table_id": "table-1",
                "device_id": "device-1",
                "content_sha256": "c" * 64,
            },
        },
    }
    return store, json.loads(canonical(fixture))


def test_actual_start_save_outbox_matches_two_owner_cloud_fixture(tmp_path):
    store, generated = build_private_fixture(tmp_path)
    assert generated == json.loads(FIXTURE_PATH.read_text())
    assert [fact["actor_user_id"] for fact in generated["source_facts"]] == ["user-a", "user-b"]
    assert "user-c" not in canonical(generated["selections"])
    peers = {item.shot_id: _CloudContractTransport() for item in store.list_saved_media_outbox()}
    for peer in peers.values():
        peer.session_published = True

    class Transport:
        def post_json(self, path, headers, body):
            if path == "/v1/sync/saved-clips":
                self.peer = peers[json.loads(body)["shot_id"]]
            return self.peer.post_json(path, headers, body)

        def put_bytes(self, *args):
            return self.peer.put_bytes(*args)

    client = FieldSyncClient(credentials=_Credentials(), transport=Transport())
    for attempt in range(4):
        store = SQLiteFieldRecordStore(
            store.path, now_utc=lambda attempt=attempt: NOW + timedelta(minutes=attempt + 1)
        )
        worker = _BackgroundSavedMediaSyncWorker(
            store=store,
            client=client,
            trusted_postgame_root=tmp_path / "postgame",
            wake_event=threading.Event(),
            now_utc=lambda attempt=attempt: NOW + timedelta(minutes=attempt + 1),
        )
        assert worker.run_once()
    assert all(row.status == "acknowledged" for row in store.list_saved_media_outbox())
    for selection in generated["selections"]:
        descriptor = selection["descriptor"]
        peer = peers[descriptor["shot_id"]]
        assert peer.saved_descriptor == descriptor
        assert peer.saved_document["owner_user_id"] == descriptor["owner_user_id"]
        assert peer.uploaded_bytes == MP4
        assert (peer.create_count, peer.upload_count, peer.complete_count) == (2, 1, 1)


def test_unknown_start_cannot_be_upgraded_by_final_roster_or_restart(tmp_path):
    store, _ = build_private_fixture(tmp_path, unknown_red=True)
    restarted = SQLiteFieldRecordStore(store.path)
    assert restarted.get_shot_ownership("session-private", "shot-0001").evidence_status == "unknown"
    assert restarted.claim_due_saved_media(NOW + timedelta(days=1)) is None
    with pytest.raises(ValueError, match="ownership"):
        restarted.save_postgame_clip(
            request_id="retry-unknown",
            session_id="session-private",
            shot_id="shot-0001",
            postgame_manifest_sha256=restarted.get_postgame_manifest(
                "session-private"
            ).manifest_sha256,
        )
