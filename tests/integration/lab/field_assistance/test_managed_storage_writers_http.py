"""Signed software composition; synthetic SDK/DB/shot input, no native proof."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from urllib.parse import urlsplit
from urllib.request import urlopen

import pytest

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
from tests.integration.lab.field_assistance.test_managed_command_http import (
    REQUEST,
    SCOPE,
    Credentials,
    gateway,
    materialize,
)
from tests.integration.lab.field_assistance.test_managed_command_http import bridge as bridge
from tests.integration.lab.field_assistance.test_managed_media_admission_http import canonical
from tests.integration.lab.field_assistance.test_managed_shared_startup_http import phone


class ObjectTransport(UrllibSyncHttpTransport):
    """Map only the known synthetic HTTPS SDK host to the isolated HTTP store."""

    def __init__(self, base, *, before_put=None):
        super().__init__(base, timeout_seconds=5, allow_redirects=False)
        self.base = base
        self.before_put = before_put
        self.grants = []

    def post_json(self, path, headers, body):
        result = super().post_json(path, headers, body)
        if path.endswith("/media-authorization") and result.status_code == 200:
            self.grants.append(json.loads(result.body)["data"])
        return result

    def put_bytes(self, url, payload, content_type, headers):
        parsed = urlsplit(url)
        assert parsed.scheme == "https" and parsed.netloc == "synthetic.objects.invalid"
        assert not parsed.query and not parsed.fragment
        if self.before_put:
            callback, self.before_put = self.before_put, None
            callback()
        return super().put_bytes(self.base + parsed.path, payload, content_type, headers)


@pytest.fixture
def saved_match(bridge, tmp_path, request):
    base, control = bridge
    cloud = gateway(base)
    resource_factory = getattr(request, "param", None)
    resource = None if resource_factory is None else resource_factory(tmp_path)
    records, binding = materialize(tmp_path, cloud, resource_spec=resource)
    session_id = binding.session_id
    artifact_id = records.load_session(session_id).artifact_id
    if resource is not None:
        records.bind_artifact_path(artifact_id, tmp_path)
        records.register_managed_capture_epoch(SCOPE, REQUEST, "writer-epoch")
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
            idempotency_key=f"writer-http-intent-{revision:016d}",
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

    owners = []
    prepared_media = {}
    for sequence, role in ((1, WristbandRole.RED), (2, WristbandRole.BLUE)):
        if sequence == 2:
            apply("switch_turn", {})
            match = records.load_session(session_id).match
        prepare_media = getattr(resource_factory, "prepare_media", None)
        prepared_media[sequence] = (
            {} if prepare_media is None else prepare_media(resource, binding, sequence)
        )
        prepared = prepared_media[sequence]
        shot = FieldShotRecord(
            session_id=session_id,
            shot_id=f"shot-{sequence:04d}",
            sequence=sequence,
            actor_role=role,
            started_ms=sequence * 3000,
            settled_ms=None,
            recommendation_id=None,
            power_band=None,
            power_percent=None,
            media=ShotMediaRange(
                artifact_id,
                "usb:writer-epoch",
                "usb_live",
                sequence * 3000,
                sequence * 3000 + 1,
                "writer-epoch",
            ),
        )
        records.record_shot_start(shot, source_start_event_id=f"writer-start-{sequence}")
        owner = records.get_shot_ownership(session_id, shot.shot_id)
        assert owner.actor_user_id == (alice if sequence == 1 else bob)
        assert owner.match_revision == match.revision
        owners.append(owner)
        records.record_shot_settle(
            replace(
                shot,
                settled_ms=sequence * 3000 + 2000,
                media=replace(shot.media, end_ms=sequence * 3000 + 2000),
                live_job_id=f"shot-job-{sequence}",
                live_evidence_digest="a" * 64,
            ),
            source_epoch_id="writer-epoch",
        )
        records.attach_live_shot_media(
            LiveShotMediaReference(
                session_id,
                shot.shot_id,
                "writer-epoch",
                match.revision,
                f"shot-job-{sequence}",
                prepared.get("media_path", str(tmp_path / "raw")),
                prepared.get("media_sha256", "d" * 64),
            )
        )
    records.end_session(session_id)
    sealed = cloud.close_managed_session(binding=binding)
    assert sealed.coverage == "sealed" and sealed.contiguous_ack_sequence == 4
    manifest = records.get_postgame_manifest(session_id)
    root = tmp_path / "postgame"
    receipts = []
    payloads = []
    for sequence in (1, 2):
        shot_id = f"shot-{sequence:04d}"
        receipt = records.save_postgame_clip(
            request_id=f"screen-save-{sequence}",
            session_id=session_id,
            shot_id=shot_id,
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        assert receipt.status == "SAVE_PENDING"
        assert len(records.list_saved_media_outbox()) == sequence - 1
        payload = bytes.fromhex("000000186674797069736f6d0000020069736f6d69736f32") + bytes(
            [sequence]
        )
        prepared = prepared_media[sequence]
        path = prepared.get("local_path", root / session_id / f"clip-{sequence}.mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records.record_postgame_clip_available(
            session_id=session_id,
            shot_id=shot_id,
            job_id=f"shot-job-{sequence}",
            artifact_id=prepared.get("artifact_id", "d" * 64),
            local_path=path,
            content_sha256=hashlib.sha256(payload).hexdigest(),
            byte_size=len(payload),
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
        assert (
            records.save_postgame_clip(
                request_id=receipt.request_id,
                session_id=session_id,
                shot_id=shot_id,
                postgame_manifest_sha256=manifest.manifest_sha256,
            )
            == receipt
        )
        receipts.append(receipt)
        payloads.append(payload)
    selected = sorted(records.list_saved_media_outbox(), key=lambda item: item.shot_id)
    descriptors = [item.descriptor_json for item in selected]
    for item, owner in zip(selected, owners, strict=True):
        assert json.loads(item.descriptor_json)["ownership_receipt"] == json.loads(
            canonical(asdict(owner))
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
    client = FieldSyncClient(credentials=Credentials(), transport=ObjectTransport(base))
    assert client.run_once(job).outcome == "complete"
    yield dict(
        base=base,
        control=control,
        records=records,
        root=root,
        selected=selected,
        payloads=payloads,
        session=session_id,
        client=client,
    )
    assert [
        item.descriptor_json
        for item in sorted(records.list_saved_media_outbox(), key=lambda i: i.shot_id)
    ] == descriptors
    assert records.get_postgame_manifest(session_id) == manifest
    for item, owner, receipt in zip(selected, owners, receipts, strict=True):
        assert records.get_shot_ownership(session_id, item.shot_id) == owner
        if prepare_media is not None and records.get_owner_delete(item.clip_id) is not None:
            with pytest.raises(ValueError, match="^OWNER_CLIP_DELETE_REQUESTED$"):
                records.save_postgame_clip(
                    request_id=receipt.request_id,
                    session_id=session_id,
                    shot_id=item.shot_id,
                    postgame_manifest_sha256=manifest.manifest_sha256,
                )
        else:
            assert (
                records.save_postgame_clip(
                    request_id=receipt.request_id,
                    session_id=session_id,
                    shot_id=item.shot_id,
                    postgame_manifest_sha256=manifest.manifest_sha256,
                )
                == receipt
            )
    snapshot = control("snapshot")
    published = snapshot["rows"]["sessions"][session_id]
    assert (
        published["_sync_publication"]["content_sha256"]
        == hashlib.sha256(job.manifest_json.encode()).hexdigest()
    )
    assert published["ended_at_utc"] == json.loads(job.manifest_json)["session"]["ended_at_utc"]
    assert all(attempt["count"] <= 90 for attempt in snapshot["attempts"])
    assert all(
        call["method"] in {"get", "set", "update", "remove"}
        for attempt in snapshot["attempts"]
        for call in attempt["calls"]
    )


def assert_views(case):
    control = case["control"]
    for index, who in enumerate(("a", "b")):
        own = case["selected"][index]
        other = case["selected"][1 - index]
        listed = phone(control, "clips.list", who=who, session_id=case["session"])
        assert [item["clip_id"] for item in listed] == [own.clip_id]
        playback = phone(control, "clips.getPlayback", who=who, clip_id=own.clip_id)
        assert playback["clip"]["clip_id"] == own.clip_id
        parsed = urlsplit(playback["url"])
        assert parsed.scheme == "https" and parsed.netloc == "synthetic.objects.invalid"
        assert not parsed.query and not parsed.fragment
        with urlopen(case["base"] + parsed.path, timeout=5) as response:
            assert response.read() == case["payloads"][index]
        for action, extra in (
            ("clips.getPlayback", {}),
            ("clips.delete", {"request_id": f"foreign-{who}"}),
        ):
            refused = control(
                "phone",
                action=action,
                openid=f"synthetic-phone-{who}",
                payload=dict(clip_id=other.clip_id, **extra),
            )
            assert refused == {"ok": False, "error": {"code": "MEDIA_NOT_AVAILABLE"}}


@pytest.mark.parametrize("bridge", ["writers"], indirect=True)
@pytest.mark.parametrize("scenario", ["overlap", "request-loss", "response-loss", "late-delete"])
def test_actual_signed_registered_writers_publish_exclusive_bytes_and_retain_late_results(
    saved_match, scenario
):
    case = saved_match
    base, control, client = case["base"], case["control"], case["client"]
    first, second = case["selected"]
    transport = ObjectTransport(base)
    writer = FieldSyncClient(credentials=Credentials(), transport=transport)
    if scenario == "overlap":

        def later_grant():
            job = next(iter(control("snapshot")["rows"]["saved_clip_sync_jobs"].values()))
            assert (
                client._post(
                    f"/v1/sync/saved-clips/{job['job_id']}/media-authorization", b"{}"
                ).status_code
                == 200
            )

        transport.before_put = later_grant
    if scenario in {"request-loss", "response-loss"}:
        control("fault", kind=dict(path="/complete", kind=scenario))
    if scenario == "late-delete":
        control("hold-upload")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                writer.run_saved_media_once, first, trusted_postgame_root=case["root"]
            )
            try:
                deadline = time.monotonic() + 5
                while not control("snapshot")["heldUpload"]:
                    assert time.monotonic() < deadline, (
                        "SDK upload did not reach controlled return boundary"
                    )
                    time.sleep(0.01)
                assert (
                    client.run_saved_media_once(first, trusted_postgame_root=case["root"]).outcome
                    == "complete"
                )
                assert (
                    client.run_saved_media_once(second, trusted_postgame_root=case["root"]).outcome
                    == "complete"
                )
                assert_views(case)
                winner = control("snapshot")["rows"]["clips"][first.clip_id]
                deleted = phone(
                    control, "clips.delete", clip_id=first.clip_id, request_id="owner-delete-1"
                )
                assert deleted["clip_id"] == first.clip_id
            finally:
                if control("snapshot")["heldUpload"]:
                    control("release-upload")
            assert pending.result(timeout=5).error_code == "SYNC_CONFLICT"
        assert phone(control, "clips.list", session_id=case["session"]) == []
        refused = control("phone", action="clips.getPlayback", payload={"clip_id": first.clip_id})
        assert refused["error"]["code"] == "MEDIA_NOT_AVAILABLE"
        assert (
            client.run_saved_media_once(first, trusted_postgame_root=case["root"]).error_code
            == "SYNC_CONFLICT"
        )
    else:
        result = writer.run_saved_media_once(first, trusted_postgame_root=case["root"])
        if scenario.endswith("loss"):
            assert (result.outcome, result.error_code) == ("retry", "NETWORK_UNAVAILABLE")
            before = control("snapshot")
            assert (
                writer.run_saved_media_once(first, trusted_postgame_root=case["root"]).outcome
                == "complete"
            )
            if scenario == "response-loss":
                assert control("snapshot")["sdkCalls"] == before["sdkCalls"]
        else:
            assert result.outcome == "complete", result
        assert (
            client.run_saved_media_once(second, trusted_postgame_root=case["root"]).outcome
            == "complete"
        )
        assert_views(case)
        before = control("snapshot")
        assert (
            writer.run_saved_media_once(first, trusted_postgame_root=case["root"]).outcome
            == "complete"
        )
        assert control("snapshot")["sdkCalls"] == before["sdkCalls"]
        job = next(
            job
            for job in before["rows"]["saved_clip_sync_jobs"].values()
            if job["clip_id"] == first.clip_id
        )
        for body in (
            b"{}",
            canonical({"upload_attempt": transport.grants[0]["upload_attempt"]}).encode(),
        ):
            assert (
                writer._post(f"/v1/sync/saved-clips/{job['job_id']}/complete", body).status_code
                == 200
            )
        replayed = control("snapshot")
        assert replayed["sdkCalls"] == before["sdkCalls"]
        for collection in (
            "media_storage_operations",
            "media_storage_objects",
            "clip_storage_authorities",
            "clips",
        ):
            assert replayed["rows"][collection] == before["rows"][collection]
    snapshot = control("snapshot")
    assert snapshot["providerCalls"] == {"authorize": 0, "verify": 0}
    assert not snapshot["spoolEntries"]
    entries = [
        entry
        for page in snapshot["rows"]["media_storage_operations"].values()
        for entry in page["entries"]
    ]
    operations = [entry["operation"] for entry in entries]
    assert len(operations) == (
        6 if scenario == "late-delete" else 5 if scenario in {"overlap", "request-loss"} else 4
    )
    observed = [obj for operation in operations for obj in operation["objects"]]
    assert {obj["cloud_path"] for obj in observed} == set(snapshot["objects"])
    assert len({obj["cloud_path"] for obj in observed}) == len(observed)
    assert all(
        operation["outcome"] in {"external_issued", "server_succeeded"} for operation in operations
    )
    assert all(
        obj["cloud_path"].startswith(("managed-clips-v1/", "published/managed-clips-v1/"))
        for obj in observed
    )
    assert {obj["file_id"] for obj in observed} == {
        obj["fileID"] for obj in snapshot["objects"].values()
    }
    assert all(call["kind"] in {"grant", "read", "copy"} for call in snapshot["sdkCalls"])
    for index, item in enumerate(case["selected"]):
        catalog = control("storage-snapshot", clipId=item.clip_id)
        assert catalog["erasable"] is False and catalog["complete"] is False
        assert catalog["status"] == "legacy_authority_incomplete"
        assert all(reason.startswith("unresolved_operation:") for reason in catalog["reasons"])
        assert len(catalog["reasons"]) == sum(
            op["kind"] == "upload_authorization" for op in catalog["operations"]
        )
        clips = snapshot["rows"]["clips"]
        clip = clips[item.clip_id]
        descriptor = json.loads(item.descriptor_json)
        for key in (
            "ownership_receipt",
            "ownership_receipt_sha256",
            "save_request_id",
            "save_receipt_sha256",
            "postgame_manifest_sha256",
            "content_sha256",
            "byte_size",
        ):
            assert clip[key] == descriptor[key]
        for obj in catalog["objects"]:
            body = snapshot["objects"][obj["cloud_path"]]["hex"]
            assert body is None or body == case["payloads"][index].hex()
        referenced = [obj for obj in catalog["objects"] if obj["references"]]
        assert len(referenced) == 2
        assert {obj["kind"] for obj in referenced} == {"staging", "published"}
        assert all(
            obj["references"]
            == [{"clip_id": item.clip_id, "identity_sha256": catalog["identity_sha256"]}]
            for obj in referenced
        )
        assert clip["storage_file_id"] == next(
            obj["file_id"] for obj in referenced if obj["kind"] == "published"
        )
    rows_json = canonical(snapshot["rows"])
    assert not any(
        secret in rows_json
        for secret in ("synthetic-token", "synthetic-signature", "synthetic.objects.invalid")
    )
    for grant in transport.grants:
        assert set(grant["upload_attempt"]) == {"contract", "operation_id", "sequence"}
    if scenario == "overlap":
        completed = [
            r
            for r in snapshot["requests"]
            if "/saved-clips/" in r["path"] and r["path"].endswith("/complete")
        ]
        assert completed[0]["body"]["upload_attempt"] == transport.grants[0]["upload_attempt"]
        unused = [obj for obj in snapshot["objects"].values() if obj["hex"] is None]
        assert len(unused) == 1
        selected_upload = next(
            op
            for op in operations
            if op["operation_id"] == transport.grants[0]["upload_attempt"]["operation_id"]
        )
        copy = next(
            op
            for op in operations
            if op["clip_id"] == first.clip_id and op["kind"] == "server_copy"
        )
        assert copy["input_object_id"] == selected_upload["target"]["object_id"]
        assert selected_upload["sequence"] == 1 and copy["sequence"] == 3
    if scenario == "late-delete":
        assert snapshot["rows"]["clip_storage_authorities"][first.clip_id]["admission"] == "closed"
        assert snapshot["rows"].get("owner_clip_tombstones", {}).get(first.clip_id)
        assert sum(op["outcome"] == "server_succeeded" for op in operations) == 3
        final_clip = snapshot["rows"]["clips"][first.clip_id]
        for key in ("storage_file_id", "storage_objects", "published_at"):
            assert final_clip[key] == winner[key]
        late = [
            entry
            for entry in entries
            if entry["operation"]["clip_id"] == first.clip_id
            and entry["operation"]["kind"] == "server_copy"
            and entry["object"]["file_id"] != winner["storage_file_id"]
        ]
        assert len(late) == 1 and late[0]["object"]["references"] == []
    attempts = control("snapshot")["attempts"]
    route_max = {
        suffix: max(
            (
                count
                for request in snapshot["requests"]
                if "/saved-clips" in request["path"] and request["path"].endswith(suffix)
                for count in request["attempts"]
            ),
            default=0,
        )
        for suffix in ("/saved-clips", "/media-authorization", "/complete")
    }
    print(
        f"{scenario}: route_max={route_max} callbacks={len(attempts)} "
        f"maximum={max(a['count'] for a in attempts)} "
        f"operations={len(operations)} sdk={len(snapshot['sdkCalls'])}"
    )
