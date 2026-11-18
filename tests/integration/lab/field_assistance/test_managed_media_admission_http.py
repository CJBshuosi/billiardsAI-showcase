"""Real signed clients/routes/repos and immutable SQLite save flow.

Only platform SDK identity, DB transactions and shot/media input are synthetic.
No head, ownership, admission or successful provider result is seeded. This
does not claim raw-camera/inference, real WeChat, native DB or provider proof.
"""

import hashlib
import json
import threading
from dataclasses import asdict, replace

from billiards_ai.edge.route_assistance.commands import reduce_field_command
from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.fact_report import (
    compose_field_fact_report,
    field_sync_envelope,
)
from billiards_ai.edge.route_assistance.records import (
    FieldShotRecord,
    LiveShotMediaReference,
    ShotMediaRange,
)
from billiards_ai.edge.route_assistance.sync import FieldSyncClient, UrllibSyncHttpTransport
from billiards_ai.shared.models.miniprogram import canonical_sync_hash
from scripts.algorithm.serve_mainline1_usb_field import _BackgroundSavedMediaSyncWorker
from tests.integration.lab.field_assistance.test_managed_command_http import (
    REQUEST,
    SCOPE,
    Credentials,
    gateway,
    materialize,
    synthetic_resources,
)
from tests.integration.lab.field_assistance.test_managed_command_http import bridge as bridge
from tests.integration.lab.field_assistance.test_managed_shared_startup_http import phone


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_actual_signed_roles_settings_ended_facts_and_screen_save_stay_pending(bridge, tmp_path):
    base, control = bridge
    cloud = gateway(base)
    resource = synthetic_resources(tmp_path)
    records, binding = materialize(tmp_path, cloud, resource_spec=resource)
    records.bind_artifact_path(resource.artifact_id, tmp_path)
    records.register_managed_capture_epoch(SCOPE, REQUEST, "admission-epoch")
    session_id = binding.session_id
    alice = phone(control, "auth.me")["user_id"]
    bob = phone(control, "auth.me", who="b")["user_id"]
    assert alice != bob

    def apply(kind, payload, who="a"):
        revision = records.load_session(session_id).match.revision
        issued = phone(
            control,
            "field.command.submit",
            who=who,
            session_id=session_id,
            command_type=kind,
            payload=payload,
            expected_revision=revision,
            idempotency_key=f"admission-http-intent-{revision:016d}",
        )
        command = cloud.claim_field_command()
        assert command.command_id == issued["command_id"]
        result = records.apply_field_command(command, reduce_field_command)
        assert result.outcome == "applied"
        assert cloud.complete_field_command(result) == result
        return result

    for who, role in (("a", "red"), ("b", "blue")):
        joined = phone(
            control, "join.request", who=who, table_code="12345678901234567890123456789012"
        )
        assert joined["session_id"] == session_id and joined["role"] is None
        apply("claim_role", {"role": role}, who)
    proposed = phone(
        control,
        "ranking.match.settings.propose",
        session_id=session_id,
        mode="observation",
        format="single",
        race_target=None,
        first_breaker="red",
        ranking_intent=False,
    )
    assert proposed["status"] == "awaiting_settings_confirmations"
    for who in ("a", "b"):
        phone(control, "ranking.match.settings.confirm", who=who, session_id=session_id)
    apply("set_breaker", {"role": "red"})
    match = records.load_session(session_id).match
    assert match.settings_frozen and match.red_user_id == alice and match.blue_user_id == bob

    # Controlled input enters the production shot/start ownership and settle APIs.
    shot = FieldShotRecord(
        session_id=session_id,
        shot_id="shot-0001",
        sequence=1,
        actor_role=WristbandRole.RED,
        started_ms=3000,
        settled_ms=None,
        recommendation_id=None,
        power_band=None,
        power_percent=None,
        media=ShotMediaRange(
            resource.artifact_id, "usb:admission-epoch", "usb_live", 3000, 3001, "admission-epoch"
        ),
    )
    records.record_shot_start(shot, source_start_event_id="admission-shot-start")
    owner = records.get_shot_ownership(session_id, shot.shot_id)
    assert owner.actor_user_id == alice and owner.match_revision == match.revision
    records.record_shot_settle(
        replace(
            shot,
            settled_ms=5000,
            media=replace(shot.media, end_ms=5000),
            live_job_id="shot-job",
            live_evidence_digest="a" * 64,
        ),
        source_epoch_id="admission-epoch",
    )
    records.attach_live_shot_media(
        LiveShotMediaReference(
            session_id,
            shot.shot_id,
            "admission-epoch",
            match.revision,
            "shot-job",
            str(tmp_path / "raw"),
            "d" * 64,
        )
    )
    records.end_session(session_id)
    sealed = cloud.close_managed_session(binding=binding)
    assert sealed.coverage == "sealed" and sealed.contiguous_ack_sequence == 3
    manifest = records.get_postgame_manifest(session_id)
    receipt = records.save_postgame_clip(
        request_id="screen-save-http",
        session_id=session_id,
        shot_id=shot.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    assert receipt.status == "SAVE_PENDING"
    payload = bytes.fromhex("000000186674797069736f6d0000020069736f6d69736f32")
    root = tmp_path / "postgame"
    media = root / session_id / "clip.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(payload)
    records.record_postgame_clip_available(
        session_id=session_id,
        shot_id=shot.shot_id,
        job_id="shot-job",
        artifact_id="d" * 64,
        local_path=media,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    selected = records.list_saved_media_outbox()[0]
    original_descriptor = selected.descriptor_json
    descriptor = json.loads(original_descriptor)
    assert descriptor["ownership_receipt"] == json.loads(canonical(asdict(owner)))
    assert (
        records.save_postgame_clip(
            request_id=receipt.request_id,
            session_id=session_id,
            shot_id=shot.shot_id,
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        == receipt
    )

    report = compose_field_fact_report(
        records.load_session(session_id), records.list_shots(session_id)
    )
    envelope = field_sync_envelope(
        report,
        (),
        postgame_manifest_sha256=manifest.manifest_sha256,
        evidence_source_kind="usb_live",
    )
    job = records.enqueue_sync_job(
        session_id=session_id,
        manifest_hash=canonical_sync_hash(envelope),
        manifest_json=canonical(envelope.model_dump(mode="json", exclude_none=False)),
        media=(),
    )
    client = FieldSyncClient(credentials=Credentials(), transport=UrllibSyncHttpTransport(base))
    synced = client.run_once(job)
    assert synced.outcome == "complete", (synced, control("snapshot")["requests"][-1])
    published = control("snapshot")["rows"]["sessions"][session_id]
    original_facts = json.loads(job.manifest_json)
    assert published["created_at_utc"] == original_facts["session"]["created_at_utc"]
    assert published["ended_at_utc"] == original_facts["session"]["ended_at_utc"]
    assert "." not in published["ended_at_utc"]
    assert (
        published["_sync_publication"]["content_sha256"]
        == hashlib.sha256(job.manifest_json.encode()).hexdigest()
    )
    worker = _BackgroundSavedMediaSyncWorker(
        store=records,
        client=client,
        trusted_postgame_root=root,
        wake_event=threading.Event(),
        managed_scope=SCOPE,
    )
    assert worker.run_once()
    retry = records.list_saved_media_outbox()[0]
    assert retry.status == "pending" and retry.last_error_code == "SYNC_STORAGE_UNAVAILABLE"
    assert retry.descriptor_json == original_descriptor
    assert records.get_shot_ownership(session_id, shot.shot_id) == owner

    snapshot = control("snapshot")
    rows = snapshot["rows"]
    raw = next(iter(rows["saved_clip_sync_jobs"].values()))
    root_row = rows["clip_storage_authorities"][selected.clip_id]
    assert raw["status"] == "pending" and root_row["high_water"] == 0
    assert raw["storage_admission"]["epoch"] == root_row["epoch"]
    assert len(raw["storage_admission"]) == 4
    assert raw["descriptor"] == descriptor
    assert (
        rows["managed_session_births"][session_id]["media_admission_contract"]
        == "managed_clip_admission_v1"
    )
    assert rows["session_command_authority"][session_id]["coverage"] == "sealed"
    assert snapshot["providerCalls"] == {"authorize": 0, "verify": 0}
    assert not rows.get("clips")
    assert all(attempt["count"] <= 90 for attempt in snapshot["attempts"])
    saved_requests = [
        request for request in snapshot["requests"] if "/saved-clips" in request["path"]
    ]
    assert [request["status"] for request in saved_requests] == [200, 503]
    # Direct signed completion is also pending; no provider or fake publication.
    result = client._post(f"/v1/sync/saved-clips/{raw['job_id']}/complete", b"{}")
    assert result.status_code == 503
    assert json.loads(result.body) == {"ok": False, "error": {"code": "SYNC_STORAGE_UNAVAILABLE"}}
    replay = client.run_saved_media_once(selected, trusted_postgame_root=root)
    assert (replay.outcome, replay.error_code) == ("retry", "SYNC_STORAGE_UNAVAILABLE")
    after = control("snapshot")
    assert after["rows"]["saved_clip_sync_jobs"] == rows["saved_clip_sync_jobs"]
    assert after["rows"]["clip_storage_authorities"] == rows["clip_storage_authorities"]
    assert after["providerCalls"] == snapshot["providerCalls"]

    # Exercise the real signed generic handler boundary, including replay disguise.
    for managed, kind in ((False, "usb_live"), (True, "usb_live"), (True, "development_replay")):
        forbidden = json.loads(job.manifest_json)
        forbidden["session"]["session_id"] = session_id if managed else "ordinary-session"
        forbidden["session"]["evidence_source_kind"] = kind
        if kind == "development_replay":
            forbidden["session"]["limitations"].append("development_replay_media")
        forbidden["media"] = [
            {
                "clip_id": "unsaved",
                "session_id": forbidden["session"]["session_id"],
                "shot_id": shot.shot_id,
                "artifact_id": resource.artifact_id,
                "relative_path": "clip.mp4",
                "source_kind": kind,
                "content_sha256": "a" * 64,
                "byte_size": 24,
                "media_type": "video/mp4",
            }
        ]
        body = canonical(forbidden).encode()
        headers = client._headers("/v1/sync/jobs", body)
        headers.update(
            {"x-idempotency-key": f"unsaved-{managed}-{kind}", "x-manifest-sha256": "f" * 64}
        )
        response = UrllibSyncHttpTransport(base).post_json("/v1/sync/jobs", headers, body)
        assert response.status_code == 409
        assert json.loads(response.body) == {"ok": False, "error": {"code": "SYNC_CONFLICT"}}
    refused = control("snapshot")
    assert refused["providerCalls"] == {"authorize": 0, "verify": 0}
    assert refused["rows"]["sync_jobs"] == after["rows"]["sync_jobs"]
    assert not refused["rows"].get("clips")
