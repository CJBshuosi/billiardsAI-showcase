"""Actual signed cloud-to-original-device erasure; controlled synthetic inputs only."""
# ruff: noqa: F811

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import pytest

from billiards_ai.edge.route_assistance.local_media_cleanup import process_local_owner_delete_once
from billiards_ai.edge.route_assistance.owner_device_delivery import OwnerDeletionReceiptOutbox
from billiards_ai.edge.route_assistance.owner_device_transport import (
    OwnerDeviceMetadataHttpTransport,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.sync import FieldSyncClient, UrllibSyncHttpTransport
from billiards_ai.lab.field_assistance.cloud_gateway import (
    DevelopmentAttestedFieldCloudGateway,
    SignedFieldCloudGateway,
)
from billiards_ai.lab.field_assistance.cloud_runtime import (
    _BackgroundLocalMediaCleanupWorker,
    _BackgroundOwnerDeviceDeletionWorker,
)
from billiards_ai.lab.field_assistance.managed_resource_recovery import (
    ManagedRuntimeResourceResolver,
)
from billiards_ai.lab.mainline1_shot_jobs import ShotAnalysisJob
from tests.integration.lab.field_assistance.test_durable_upload_requests_http import (
    Scene,
    clip_rows,
)
from tests.integration.lab.field_assistance.test_managed_command_http import (
    SCOPE,
    Credentials,
    bridge,  # noqa: F401
    synthetic_resources,
)
from tests.integration.lab.field_assistance.test_managed_shared_startup_http import phone
from tests.integration.lab.field_assistance.test_managed_storage_writers_http import (
    saved_match,  # noqa: F401
)
from tests.integration.lab.field_assistance.test_upload_terminal_receipts_http import (
    ControlledExit,
    drain,
)
from tests.unit.edge.route_assistance.test_managed_session_persistence import (
    snapshot as local_snapshot,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def device_resources(root):
    resource = synthetic_resources(root)
    Path(resource.paths["ring_root"]).mkdir()
    return resource


def prepare_media(resource, binding, sequence):
    """Controlled upstream job/interval/MP4; never creates deletion authority."""
    session = binding.session_id
    jobs = Path(resource.paths["jobs_root"])
    raw = jobs / "media" / session / f"shot-job-{sequence}"
    raw.mkdir(parents=True)
    segment = b"synthetic-copied-raw-" + str(sequence).encode()
    (raw / "segment-1.mjpeg").write_bytes(segment)
    identity = dict(
        schema_version="1.0",
        source_epoch_id="writer-epoch",
        width=640,
        height=480,
        start_frame_id=1,
        end_frame_id=1,
        frame_count=1,
    )
    identity["segments"] = [{"name": "segment-1.mjpeg", "sha256": sha(segment)}]
    digest = sha(canonical(identity).encode())
    interval = {k: v for k, v in identity.items() if k != "segments"}
    interval.update(media_sha256=digest, segment_sha256={"segment-1.mjpeg": sha(segment)})
    (raw / "interval.json").write_text(canonical(interval))
    job = ShotAnalysisJob(
        f"shot-job-{sequence}",
        session,
        f"shot-{sequence:04d}",
        "writer-epoch",
        1,
        1,
        raw,
        digest,
        resource.capture_setup_id,
        resource.calibration_artifact_id,
        resource.config_sha256,
        resource.model_package_id,
        resource.identity_training_package_id,
        resource.identity_training_package_sha256,
    )
    artifact = str(sequence) * 64
    payload = bytes.fromhex("000000186674797069736f6d0000020069736f6d69736f32") + bytes([sequence])
    output = Path(resource.paths["artifact_root"]) / "postgame" / session / artifact
    output.mkdir(parents=True)
    overlay = output / "overlay.mp4"
    overlay.write_bytes(payload)
    manifest = dict(job.identity())
    manifest.update(
        kind="mainline1_shot_local_video",
        artifact_role="usb_postgame_shot",
        source_kind="usb_live",
        live_shot_id=job.shot_id,
        artifact_id=artifact,
        inventory={
            "overlay.mp4": {"bytes": len(payload), "sha256": sha(payload)},
            "summary.json": {"bytes": 2, "sha256": sha(b"{}")},
        },
    )
    (output / "manifest.json").write_text(canonical(manifest))
    (output / "summary.json").write_text("{}")
    encoded = canonical(job.identity())
    with sqlite3.connect(jobs / "analysis-jobs.sqlite3") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS shot_analysis_jobs("
            "job_id, payload_json, payload_sha256, status, artifact_id)"
        )
        db.execute(
            "INSERT INTO shot_analysis_jobs VALUES(?,?,?,?,?)",
            (job.job_id, encoded, sha(encoded.encode()), "available", artifact),
        )
    return dict(media_path=str(raw), media_sha256=digest, local_path=overlay, artifact_id=artifact)


device_resources.prepare_media = prepare_media


pytestmark = [
    pytest.mark.parametrize("bridge", ["owner-device-delivery"], indirect=True),
    pytest.mark.parametrize("saved_match", [device_resources], indirect=True),
]


def owner_client(scene):
    transport = OwnerDeviceMetadataHttpTransport(scene.case["base"], timeout_seconds=5)
    gateway = SignedFieldCloudGateway(credentials=Credentials(), transport=transport)
    attested = DevelopmentAttestedFieldCloudGateway(
        delegate=gateway, expected_cloudbase_environment_id="synthetic"
    )
    return FieldSyncClient(
        credentials=Credentials(),
        transport=transport,
        before_mutation=attested.runtime_attestation,
        owner_scope=SCOPE,
    )


class Chain:
    def __init__(self, case, *, publish=True):
        self.scene = Scene(case)
        self.case, self.control = case, case["control"]
        if publish:
            self.scene.step("acknowledged")
            drain(self.scene)
        self.scene.clock = datetime.now(UTC) - timedelta(seconds=1)
        self.client = owner_client(self.scene)
        self.metadata = self.new_metadata()
        self.diagnostics = {}

    @property
    def store(self):
        return self.scene.store

    def new_metadata(self):
        return _BackgroundOwnerDeviceDeletionWorker(
            store=self.store,
            client=self.client,
            managed_scope=SCOPE,
            now_utc=lambda: self.scene.clock,
        )

    def advance(self, milliseconds=5001):
        self.control("device-advance", advanceMs=milliseconds)

    def delete(self):
        target = self.scene.target
        denied = self.control(
            "phone",
            action="clips.delete",
            openid=f"synthetic-phone-{self.scene.other_who}",
            payload={"clip_id": target.clip_id, "request_id": "foreign-delete"},
        )
        assert denied["ok"] is False
        deleted = phone(
            self.control,
            "clips.delete",
            who=self.scene.target_who,
            clip_id=target.clip_id,
            request_id="actual-device-chain",
        )
        self.job = deleted["deletion_job_id"]
        self.before = self.control("snapshot")
        self.original = local_snapshot(self.store.path)
        self.original_outbox = self.store.list_saved_media_outbox()
        spec = self.store.get_managed_resource_spec_for_session(SCOPE, self.case["session"]).spec
        raw = Path(spec.paths["jobs_root"]) / "media" / self.case["session"]
        self.original_files = {}
        for item in self.case["selected"]:
            paths = (
                item.local_path,
                item.local_path.with_name("manifest.json"),
                item.local_path.with_name("summary.json"),
                raw / item.job_id / "interval.json",
                raw / item.job_id / "segment-1.mjpeg",
            )
            self.original_files.update((path, path.read_bytes()) for path in paths)
        waiting = self.before["rows"]["owner_device_deletion_deliveries"][self.job]
        assert waiting["status"] == "waiting_cloud" and waiting["task_sha256"] is None
        assert self.job not in self.before["rows"].get("owner_device_deletion_tasks", {})
        assert self.store.get_owner_delete(target.clip_id) is None

    def receive(self, *, expect_rollback=False):
        rolled_back = False
        for _ in range(40):
            before = self.control("snapshot")
            self.metadata.run_once()
            after = self.control("snapshot")
            assert (
                len(after["ownerErasure"]["nativeCalls"])
                - len(before["ownerErasure"]["nativeCalls"])
                <= 1
            )
            state = self.store.get_owner_delete(self.scene.target.clip_id)
            if self.metadata.last_error_code:
                assert expect_rollback
                assert state is None
                for name in (
                    "owner_cloud_erasure_receipts",
                    "owner_cloud_erasure_witnesses",
                    "owner_device_deletion_tasks",
                ):
                    assert self.job not in after["rows"].get(name, {})
                execution = after["rows"]["owner_cloud_erasure_executions"][self.job]
                previous = before["rows"]["owner_cloud_erasure_executions"][self.job]
                assert execution["index"] == len(execution["targets"])
                for key in (
                    "contract",
                    "job_id",
                    "scope_sha256",
                    "inputs_sha256",
                    "assessment_sha256",
                    "targets",
                    "admission",
                ):
                    assert execution[key] == previous[key]
                assert after["objects"] == before["objects"]
                rolled_back = True
                self.metadata = self.new_metadata()
            if state is not None:
                receipt = after["rows"]["owner_cloud_erasure_receipts"][self.job]
                witness = after["rows"]["owner_cloud_erasure_witnesses"][self.job]
                issued = after["rows"]["owner_device_deletion_tasks"][self.job]
                assert len(json.loads(state.task.canonical_json())) == 19
                assert issued["task"] == json.loads(state.task.canonical_json())
                assert (
                    state.task.cloud_erase_receipt_sha256
                    == witness["receipt_sha256"]
                    == receipt["receipt_sha256"]
                )
                assert issued["task_sha256"] == state.task_sha256
                assert (
                    after["rows"]["owner_device_deletion_deliveries"][self.job]["status"]
                    == "leased"
                )
                # Exact stored witness, receipt, task and ready metadata share finalization.
                writes = {
                    "owner_cloud_erasure_witnesses",
                    "owner_cloud_erasure_receipts",
                    "owner_device_deletion_tasks",
                    "owner_device_deletion_deliveries",
                }
                assert any(
                    writes <= {c["collection"] for c in a["calls"] if c["method"] != "get"}
                    for a in after["attempts"]
                )
                assert rolled_back == expect_rollback
                return state
            assert self.job not in after["rows"].get("owner_device_deletion_tasks", {})
            self.advance()
        pytest.fail("actual signed bounded poll failed to deliver actual retained-proof task")

    def restart(self):
        self.store_path = self.store.path
        self.scene.store = SQLiteFieldRecordStore(self.store_path, now_utc=lambda: self.scene.clock)
        self.metadata = self.new_metadata()

    def cleanup(self):
        current = self.store.path.parent / "unrelated-current-runtime"
        worker = _BackgroundLocalMediaCleanupWorker(
            store=self.store,
            jobs_db=current / "analysis-jobs.sqlite3",
            artifact_root=current,
            jobs_root=current,
            managed_scope=SCOPE,
            resource_resolver=ManagedRuntimeResourceResolver(self.store, SCOPE, self.diagnostics),
            recovery_diagnostics=self.diagnostics,
        )
        return worker.run_once()

    def execute(self):
        spec = self.store.get_managed_resource_spec_for_session(SCOPE, self.case["session"]).spec
        return process_local_owner_delete_once(
            self.store,
            Path(spec.paths["jobs_root"]) / "analysis-jobs.sqlite3",
            Path(spec.paths["artifact_root"]),
            Path(spec.paths["jobs_root"]),
            now_utc=self.scene.clock,
        )

    def settle(self):
        for _ in range(4):
            self.metadata.run_once()
            rows = queue(self.store)
            if rows and rows[0]["status"] == "acknowledged":
                return rows[0]
        pytest.fail("actual metadata worker failed to settle actual signed completion ACK")

    def close(self):
        self.metadata.stop()


def queue(store):
    with sqlite3.connect(store.path) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT * FROM field_owner_completion_outbox")]


def audit(chain, label):
    scene, after, before = chain.scene, chain.control("snapshot"), chain.before
    rows = after["rows"]
    local = chain.store.get_owner_delete(scene.target.clip_id)
    assert local.status == "completed"
    outbox = queue(chain.store)
    assert len(outbox) == 1 and outbox[0]["status"] == "acknowledged"
    receipt = json.loads(local.receipt_json)
    assert len(receipt) == 7 and receipt["reclaimed_bytes"] > scene.target.byte_size
    assert (
        receipt["cloud_erase_receipt_sha256"]
        == rows["owner_cloud_erasure_receipts"][chain.job]["receipt_sha256"]
    )
    assert receipt["journal_sha256"] == sha(local.progress_json.encode())
    assert all(obj["state"] == "deleted" for obj in json.loads(local.progress_json)["objects"])
    assert outbox[0]["local_receipt_json"] == local.receipt_json
    ack = json.loads(outbox[0]["ack_json"])
    assert len(ack) == 7 and ack == rows["owner_device_deletion_receipts"][chain.job]["ack"]
    changed = {
        "sync_nonces",
        "owner_cloud_erasure_executions",
        "owner_cloud_erasure_receipts",
        "owner_cloud_erasure_witnesses",
        "owner_device_deletion_tasks",
        "owner_device_deletion_deliveries",
        "owner_device_deletion_repair_cursors",
        "owner_device_deletion_receipts",
    }
    assert set(rows) - set(before["rows"]) <= changed
    for name, value in before["rows"].items():
        if name not in changed:
            assert rows[name] == value, name
    for name in ("sdkCalls", "objectPuts", "providerCalls", "spoolEntries"):
        assert after[name] == before[name]
    cloud_receipt = rows["owner_cloud_erasure_receipts"][chain.job]
    target_keys = {t["cloud_path"] for t in cloud_receipt["targets"]}
    assert set(after["objects"]) == set(before["objects"]) - target_keys
    assert all(
        after["objects"][k] == value
        for k, value in before["objects"].items()
        if k not in target_keys
    )
    mutable = {
        "field_owner_media_deletions",
        "field_owner_completion_outbox",
        "field_local_media_schedule_state",
        "field_saved_media_outbox",
    }
    now = local_snapshot(chain.store.path)
    assert {k: v for k, v in chain.original.items() if k not in mutable} == {
        k: v for k, v in now.items() if k not in mutable
    }
    expected_outbox = tuple(
        replace(
            item,
            status="blocked",
            claim_token=None,
            lease_until_utc=None,
            last_error_code="OWNER_CLIP_DELETE_REQUESTED",
        )
        if item.clip_id == scene.target.clip_id
        else item
        for item in chain.original_outbox
    )
    assert tuple(chain.store.list_saved_media_outbox()) == expected_outbox
    assert not scene.target.local_path.exists()
    assert scene.other.local_path.read_bytes() == chain.case["payloads"][1 - scene.target_index]
    spec = chain.store.get_managed_resource_spec_for_session(SCOPE, chain.case["session"]).spec
    raw = Path(spec.paths["jobs_root"]) / "media" / chain.case["session"]
    assert not (raw / scene.target.job_id / "segment-1.mjpeg").exists()
    removed = {scene.target.local_path, raw / scene.target.job_id / "segment-1.mjpeg"}
    assert all(not path.exists() for path in removed)
    for path, original_bytes in chain.original_files.items():
        if path not in removed:
            assert path.read_bytes() == original_bytes, path.name
    assert not (chain.store.path.parent / "unrelated-current-runtime").exists()
    assert (
        phone(chain.control, "clips.list", who=scene.target_who, session_id=chain.case["session"])
        == []
    )
    other = phone(
        chain.control, "clips.list", who=scene.other_who, session_id=chain.case["session"]
    )
    assert [row["clip_id"] for row in other] == [scene.other.clip_id]
    for who in (scene.target_who, scene.other_who):
        refused = chain.control(
            "phone",
            action="clips.getPlayback",
            openid=f"synthetic-phone-{who}",
            payload={"clip_id": scene.target.clip_id},
        )
        assert refused["error"]["code"] == "MEDIA_NOT_AVAILABLE"
    playback = phone(
        chain.control, "clips.getPlayback", who=scene.other_who, clip_id=scene.other.clip_id
    )
    with urlopen(chain.case["base"] + urlsplit(playback["url"]).path, timeout=5) as response:
        assert response.read() == chain.case["payloads"][1 - scene.target_index]
    callbacks = after["attempts"]
    assert all(a["count"] <= 90 for a in callbacks)
    assert all(
        c["method"] in {"get", "set", "update", "remove"} for a in callbacks for c in a["calls"]
    )
    owner_requests = []
    for i, request in enumerate(after["requests"]):
        if not request["path"].startswith("/v1/owner-media-deletions/"):
            continue
        owner_requests.append(request)
        assert after["requests"][i - 1]["path"] == "/v1/runtime-attestation"
        assert after["requests"][i - 1]["status"] == 200
        assert request["bodyRaw"] == canonical(request["body"])
        assert request["bodySha256"] == sha(request["bodyRaw"].encode())
        if request["path"].endswith("/claim"):
            assert request["nativeSteps"] <= 1
        if request["path"].endswith("/complete"):
            assert request["bodyRaw"] == outbox[0]["completion_json"]
            assert len(request["body"]) == 4 and len(request["bodyRaw"].encode()) <= 16384
            for forbidden in (
                str(scene.target.local_path),
                "delivery_token",
                "lease_until",
                "progress_json",
            ):
                assert forbidden not in request["bodyRaw"]
    native = after["ownerErasure"]["nativeCalls"]
    assert all(call["key"] is None or call["key"] in target_keys for call in native)
    print(
        f"{label}: callbacks={len(callbacks)} maximum={max(a['count'] for a in callbacks)} "
        f"owner_callback_max={max(n for r in owner_requests for n in r['attempts'])} "
        f"polls={sum(r['path'].endswith('/claim') for r in owner_requests)} "
        f"completes={sum(r['path'].endswith('/complete') for r in owner_requests)} "
        f"attestations={sum(r['path'] == '/v1/runtime-attestation' for r in after['requests'])} "
        f"native={len(native)} native_deletes={sum(c['method'] == 'DELETE' for c in native)} "
        f"targets={len(target_keys)} puts={len(after['objectPuts'])} sdk={len(after['sdkCalls'])}"
    )
    evidence = chain.store.path.parent / "owner-device-delivery-evidence.json"
    evidence.write_text(
        canonical(
            dict(
                label=label,
                callbacks=callbacks,
                requests=owner_requests,
                task=json.loads(local.task.canonical_json()),
                cloud_receipt=cloud_receipt,
                local_receipt=receipt,
                ack=ack,
                native=native,
            )
        )
    )


@pytest.mark.parametrize("cloud_rollback", [False, True])
def test_actual_signed_cloud_to_device_original_unlink_outbox_and_ack(saved_match, cloud_rollback):
    chain = Chain(saved_match)
    try:
        chain.delete()
        if cloud_rollback:
            chain.control("device-fail-write", collection="owner_device_deletion_tasks")
        chain.receive(expect_rollback=cloud_rollback)
        assert chain.cleanup()
        assert chain.settle()["attempts"] == 1
        audit(chain, f"whole-chain-cloud-rollback={cloud_rollback}")
    finally:
        chain.close()


def test_expired_remote_lease_offline_unlink_restart_and_lost_ack_stable_replay(saved_match):
    chain = Chain(saved_match)
    try:
        chain.delete()
        chain.receive()
        chain.advance(31001)
        before = chain.control("snapshot")
        # No network operation is involved in actual original-resource local cleanup.
        assert chain.cleanup()
        assert chain.control("snapshot") == before
        original = chain.store.get_owner_delete(chain.scene.target.clip_id)
        pending = queue(chain.store)[0]
        assert pending["status"] == "pending"
        chain.control("fault", kind=dict(path="/runtime-attestation", kind="request-loss"))
        box = OwnerDeletionReceiptOutbox(chain.store, SCOPE)
        item = box.claim_due(chain.scene.clock)
        result = chain.client.complete_owner_delete_once(item)
        assert (result.outcome, result.error_code) == ("retry", "attestation_failed")
        assert box.retry(item, result.error_code, chain.scene.clock)
        chain.restart()
        chain.control("fault", kind=dict(path="/complete", kind="response-loss"))
        box = OwnerDeletionReceiptOutbox(chain.store, SCOPE)
        sent = box.claim_due(chain.scene.clock)
        result = chain.client.complete_owner_delete_once(sent)
        assert result.outcome == "retry"
        committed = chain.control("snapshot")
        cloud_ack = committed["rows"]["owner_device_deletion_receipts"][chain.job]["ack"]
        delivery = before["rows"]["owner_device_deletion_deliveries"][chain.job]
        assert cloud_ack["received_at_utc"] > delivery["lease_until_utc"]
        assert queue(chain.store)[0]["status"] == "leased"
        # Abrupt local restart retains the independent send lease; expire only that lease.
        chain.scene.clock += timedelta(seconds=61)
        chain.restart()
        newer = OwnerDeletionReceiptOutbox(chain.store, SCOPE).claim_due(chain.scene.clock)
        assert newer.claim_token != sent.claim_token
        replay = chain.client.complete_owner_delete_once(newer)
        assert replay.outcome == "ack" and replay.ack == cloud_ack
        box = OwnerDeletionReceiptOutbox(chain.store, SCOPE)
        assert box.acknowledge(sent, replay.ack) is False
        assert box.acknowledge(newer, replay.ack) is True
        assert box.acknowledge(newer, replay.ack) is True
        assert chain.store.get_owner_delete(chain.scene.target.clip_id) == original
        after = chain.control("snapshot")
        for name in (
            "owner_device_deletion_receipts",
            "owner_cloud_erasure_receipts",
            "owner_cloud_erasure_witnesses",
            "owner_device_deletion_tasks",
            "owner_device_deletion_deliveries",
        ):
            assert after["rows"][name] == committed["rows"][name]
        assert after["ownerErasure"] == committed["ownerErasure"]
        assert queue(chain.store)[0]["completion_json"] == pending["completion_json"]
        audit(chain, "expired-offline-restart-ack-loss-stale-token")
    finally:
        chain.close()


def test_atomic_local_queue_abort_keeps_terminal_journal_and_no_cloud_completion(saved_match):
    chain = Chain(saved_match)
    try:
        chain.delete()
        chain.receive()
        with sqlite3.connect(chain.store.path) as db:
            db.execute(
                "CREATE TRIGGER reject_owner_queue BEFORE INSERT ON field_owner_completion_outbox "
                "BEGIN SELECT RAISE(ABORT,'synthetic queue rollback'); END"
            )
        failed = chain.execute()
        assert failed.outcome == "blocked", failed
        state = chain.store.get_owner_delete(chain.scene.target.clip_id)
        assert state.receipt_json is None and queue(chain.store) == []
        assert not chain.scene.target.local_path.exists()
        assert all(o["state"] == "deleted" for o in json.loads(state.progress_json)["objects"])
        # A trigger abort is permanent/blocked by policy; the SQLITE_FULL recovery
        # branch is separately exercised by the original executor unit gate.
        assert state.status == "blocked"
        with sqlite3.connect(chain.store.path) as db:
            db.execute("DROP TRIGGER reject_owner_queue")
        assert chain.client.poll_owner_delete_once(SCOPE).outcome == "idle"
        assert chain.control("snapshot")["rows"].get("owner_device_deletion_receipts", {}) == {}
    finally:
        chain.close()


def test_missing_original_resources_never_uses_current_root_and_recovers_after_restart(saved_match):
    chain = Chain(saved_match)
    try:
        chain.delete()
        chain.receive()
        original = chain.store.get_owner_delete(chain.scene.target.clip_id)
        spec = chain.store.get_managed_resource_spec_for_session(SCOPE, chain.case["session"]).spec
        jobs = Path(spec.paths["jobs_root"])
        withheld = jobs.with_name("original-jobs-withheld")
        jobs.rename(withheld)
        try:
            assert chain.cleanup() is False
            assert chain.store.get_owner_delete(chain.scene.target.clip_id) == original
            assert chain.scene.target.local_path.exists()
            assert chain.diagnostics == {
                chain.case["session"]: "LOCAL_MEDIA_CLEANUP_ORIGINAL_CONTEXT_UNAVAILABLE"
            }
            assert queue(chain.store) == []
        finally:
            withheld.rename(jobs)
        chain.restart()
        assert chain.cleanup()
        chain.settle()
        audit(chain, "missing-original-no-current-fallback-restart")
    finally:
        chain.close()


def test_more_than_one_bad_remote_page_cannot_starve_actual_original_task(saved_match):
    chain = Chain(saved_match)
    try:
        chain.delete()
        chain.control("device-bad-deliveries", jobId=chain.job)
        bad_before = chain.control("snapshot")["rows"]["owner_device_deletion_deliveries"]
        chain.receive()
        after = chain.control("snapshot")["rows"]
        for i in range(5):
            key = f"bad-delivery-{i}"
            assert after["owner_device_deletion_deliveries"][key] == {
                **bad_before[key],
                "status": "blocked",
                "diagnostic": "invalid_metadata",
                "revision": 2,
            }
            assert key not in after["owner_device_deletion_tasks"]
        assert chain.cleanup()
        chain.settle()
        audit(chain, "five-bad-remote-deliveries-then-actual-task")
    finally:
        chain.close()


def test_completion_callback_rollback_keeps_queue_and_retries_actual_bytes(
    saved_match, monkeypatch
):
    chain = Chain(saved_match)
    try:
        chain.delete()
        chain.receive()
        assert chain.cleanup()
        original = chain.store.get_owner_delete(chain.scene.target.clip_id)
        immutable_completion = queue(chain.store)[0]["completion_json"]
        observed = []
        send_leases = []
        transport = chain.client._transport
        post_json = transport.post_json

        def observe(path, headers, body):
            if path.endswith("/complete"):
                send_leases.append(queue(chain.store)[0])
            result = post_json(path, headers, body)
            if path.endswith("/complete"):
                payload = json.loads(result.body)
                observed.append((result.status_code, payload.get("error", {}).get("code")))
            return result

        # Observe the actual owned transport response without replacing its bytes.
        monkeypatch.setattr(transport, "post_json", observe)
        chain.control("device-fail-write", collection="owner_device_deletion_deliveries")
        before = chain.control("snapshot")
        for _ in range(2):
            chain.metadata.run_once()
            if observed:
                break
        after = chain.control("snapshot")
        failed = queue(chain.store)[0]
        print(
            canonical(
                dict(
                    completion_http=observed,
                    queue_status=failed["status"],
                    queue_error=failed["last_error_code"],
                    worker_error=chain.metadata.last_error_code,
                    attempts=failed["attempts"],
                    token_cleared=failed["claim_token"] is None,
                    lease_cleared=failed["lease_until_utc"] is None,
                    next_attempt_at_utc=failed["next_attempt_at_utc"],
                )
            )
        )
        assert not after["rows"].get("owner_device_deletion_receipts")
        assert (
            after["rows"]["owner_device_deletion_deliveries"]
            == before["rows"]["owner_device_deletion_deliveries"]
        )
        assert after["ownerErasure"] == before["ownerErasure"]
        assert failed["status"] == "pending"
        assert failed["attempts"] == 1 and failed["claim_token"] is None
        assert failed["lease_until_utc"] is None
        assert send_leases[0]["status"] == "leased" and send_leases[0]["claim_token"]
        due = datetime.fromisoformat(failed["next_attempt_at_utc"].replace("Z", "+00:00"))
        assert due == (chain.scene.clock + timedelta(seconds=1)).replace(microsecond=0)
        failed_request = [r for r in after["requests"] if r["path"].endswith("/complete")][-1]
        assert failed_request["status"] == 503
        assert failed_request["response"] == {
            "ok": False,
            "error": {"code": "SYNC_STORAGE_UNAVAILABLE"},
        }
        assert (
            failed["last_error_code"] == chain.metadata.last_error_code == "transport_unavailable"
        )
        assert observed == [(503, "SYNC_STORAGE_UNAVAILABLE")]
        assert failed["completion_json"] == immutable_completion
        chain.scene.clock += timedelta(seconds=2)
        settled = chain.settle()
        assert settled["attempts"] == 2 and settled["completion_json"] == immutable_completion
        assert settled["last_error_code"] is None
        assert send_leases[1]["status"] == "leased" and send_leases[1]["attempts"] == 2
        assert send_leases[0]["claim_token"] != send_leases[1]["claim_token"]
        assert settled["claim_token"] == send_leases[1]["claim_token"]
        assert settled["lease_until_utc"] is None
        assert observed == [(503, "SYNC_STORAGE_UNAVAILABLE"), (200, None)]
        assert chain.store.get_owner_delete(chain.scene.target.clip_id) == original
        audit(chain, "completion-callback-rollback-worker-retry")
    finally:
        chain.close()


def test_actual_unlink_sqlite_full_queue_fault_restart_reconciles_without_reunlink(
    saved_match, monkeypatch
):
    from billiards_ai.edge.route_assistance import owner_device_delivery as delivery

    chain = Chain(saved_match)
    try:
        chain.delete()
        chain.receive()
        enqueue = delivery.enqueue_owner_completion_in

        def storage_full(*args, **kwargs):
            error = sqlite3.OperationalError("controlled queue full")
            error.sqlite_errorcode = sqlite3.SQLITE_FULL
            raise error

        # Fault only the post-unlink queue insertion. The original receiver,
        # executor, journal, SQLite transactions and subsequent enqueue all run.
        with monkeypatch.context() as fault:
            fault.setattr(delivery, "enqueue_owner_completion_in", storage_full)
            assert chain.execute().outcome == "retry"
        assert delivery.enqueue_owner_completion_in is enqueue
        original = chain.store.get_owner_delete(chain.scene.target.clip_id)
        assert original.status == "retry" and original.receipt_json is None
        assert queue(chain.store) == [] and not chain.scene.target.local_path.exists()
        assert all(o["state"] == "deleted" for o in json.loads(original.progress_json)["objects"])
        chain.scene.clock += timedelta(seconds=4)
        chain.restart()
        assert chain.execute().outcome == "completed"
        assert (
            chain.store.get_owner_delete(chain.scene.target.clip_id).progress_json
            == original.progress_json
        )
        chain.settle()
        audit(chain, "actual-unlink-full-queue-restart-recovery")
    finally:
        chain.close()


def test_unknown_started_writer_signed_poll_never_issues_device_authority(saved_match):
    chain = Chain(saved_match, publish=False)
    try:
        transport = chain.scene.worker._client._transport
        original = transport.put_bytes_once

        def interrupted(*args, **kwargs):
            original(*args, **kwargs)
            raise ControlledExit()

        transport.put_bytes_once = interrupted
        # The upload due clock is the existing Scene's future clock. Local task
        # acceptance clock is irrelevant because this scene must never accept one.
        chain.scene.clock += timedelta(seconds=3)
        with pytest.raises(ControlledExit):
            chain.scene.worker.run_once()
        started = clip_rows(chain.store, chain.scene.target)[0]
        assert started["state"] == "started"
        signed = FieldSyncClient(
            credentials=Credentials(), transport=UrllibSyncHttpTransport(chain.case["base"])
        )
        assert (
            signed._post(
                f"/v1/sync/saved-clips/{started['cloud_job_id']}/complete",
                canonical({"upload_attempt": json.loads(started["upload_attempt_json"])}).encode(),
            ).status_code
            == 200
        )
        chain.scene.restart()
        chain.scene.step("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
        drain(chain.scene)
        chain.delete()
        before = chain.control("snapshot")
        for _ in range(5):
            result = chain.client.poll_owner_delete_once(SCOPE)
            assert result.outcome == "idle" and result.task is None
            chain.advance()
        after = chain.control("snapshot")
        for name in (
            "owner_cloud_erasure_receipts",
            "owner_cloud_erasure_witnesses",
            "owner_device_deletion_tasks",
            "owner_device_deletion_receipts",
        ):
            assert not after["rows"].get(name)
        assert chain.store.get_owner_delete(chain.scene.target.clip_id) is None
        assert queue(chain.store) == []
        assert after["ownerErasure"]["nativeCalls"] == []
        assert after["ownerErasure"]["credentials"] == 0
        for name in ("objects", "sdkCalls", "objectPuts", "providerCalls"):
            assert after[name] == before[name]
        assert all(
            item.local_path.read_bytes() == data
            for item, data in zip(chain.case["selected"], chain.case["payloads"], strict=True)
        )
        print(
            f"unknown-writer: polls=5 native=0 credentials=0 "
            f"max={max(a['count'] for a in after['attempts'])} "
            "device_tasks=0 local_tasks=0 completions=0"
        )
    finally:
        chain.close()
