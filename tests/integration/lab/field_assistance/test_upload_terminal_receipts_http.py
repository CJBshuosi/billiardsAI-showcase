"""Actual signed receipt composition; controlled synthetic DB/SDK/media only."""
# Imported pytest fixtures are injected by their public names.
# ruff: noqa: F811

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import timedelta
from urllib.parse import urlsplit
from urllib.request import urlopen

import pytest

from billiards_ai.edge.route_assistance.sync import FieldSyncClient
from billiards_ai.edge.route_assistance.upload_receipt_delivery import (
    UploadReceiptOutbox,
    UploadReceiptOutboxItem,
)
from billiards_ai.lab.field_assistance.cloud_gateway import DevelopmentAttestedFieldCloudGateway
from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundUploadReceiptSyncWorker
from tests.integration.lab.field_assistance.test_durable_upload_requests_http import (
    Scene,
    clip_rows,
    ledger_rows,
)
from tests.integration.lab.field_assistance.test_managed_command_http import (
    SCOPE,
    Credentials,
    bridge,  # noqa: F401
    gateway,
    synthetic_resources,
)
from tests.integration.lab.field_assistance.test_managed_media_admission_http import canonical
from tests.integration.lab.field_assistance.test_managed_shared_startup_http import phone
from tests.integration.lab.field_assistance.test_managed_storage_writers_http import (
    assert_views,
    saved_match,  # noqa: F401
)
from tests.unit.edge.route_assistance.test_managed_session_persistence import (
    snapshot as local_snapshot,
)

pytestmark = [
    pytest.mark.parametrize("bridge", ["writers"], indirect=True),
    pytest.mark.parametrize("saved_match", [synthetic_resources], indirect=True),
]


def queue(scene):
    with sqlite3.connect(scene.store.path) as db:
        db.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in db.execute("SELECT * FROM field_upload_receipt_outbox ORDER BY request_id")
        ]


def receipt_requests(snapshot):
    return [r for r in snapshot["requests"] if r["path"].endswith("/upload-receipt")]


def client(scene):
    # The exact production hook performs a separate signed deployment-level POST.
    attested = DevelopmentAttestedFieldCloudGateway(
        delegate=gateway(scene.case["base"]), expected_cloudbase_environment_id="synthetic"
    )
    return FieldSyncClient(
        credentials=Credentials(),
        transport=scene.worker._client._transport,
        before_mutation=attested.runtime_attestation,
    )


def receipt_worker(scene):
    return _BackgroundUploadReceiptSyncWorker(
        store=scene.store,
        client=client(scene),
        managed_scope=SCOPE,
        now_utc=lambda: scene.clock,
        lease_seconds=1,
    )


def drain(scene):
    worker = receipt_worker(scene)
    for _ in range(4):
        if not worker.run_once():
            break
    assert all(row["status"] == "acknowledged" for row in queue(scene))


def stable_provider(before, after):
    for key in ("sdkCalls", "objectPuts", "objects", "providerCalls", "spoolEntries"):
        assert after[key] == before[key]
    for key in ("managed_upload_requests", "media_storage_operations", "clip_storage_authorities"):
        assert after["rows"][key] == before["rows"][key]
    mutable = {"sync_nonces", "upload_terminal_receipts"}
    assert {k: v for k, v in before["rows"].items() if k not in mutable} == {
        k: v for k, v in after["rows"].items() if k not in mutable
    }


def audit(scene, label, *, removed=False):
    case = scene.case
    snapshot = case["control"]("snapshot")
    callbacks = snapshot["attempts"]
    assert all(a["count"] <= 90 for a in callbacks)
    assert all(
        c["method"] in {"get", "set", "update", "remove"} for a in callbacks for c in a["calls"]
    )
    pages = snapshot["rows"].get("upload_terminal_receipts", {})
    entries = [entry for page in pages.values() for entry in page["entries"] if entry]
    assert len({entry["request_binding"]["request_id"] for entry in entries}) == len(entries)
    for page in pages.values():
        assert len(page["entries"]) == 16 and page["page_index"] < 4
        assert len(canonical(page).encode()) <= 128 * 1024
    for entry in entries:
        original = next(
            r for r in queue(scene) if r["request_id"] == entry["request_binding"]["request_id"]
        )
        assert canonical(entry["receipt"]) == original["receipt_json"]
        assert entry["receipt_sha256"] == original["receipt_sha256"]
    operations = [
        entry
        for page in snapshot["rows"]["media_storage_operations"].values()
        for entry in page["entries"]
        if entry
    ]
    targets = [obj for entry in operations for obj in entry["operation"]["objects"]]
    assert len(targets) == len({obj["cloud_path"] for obj in targets})
    assert {obj["cloud_path"] for obj in targets} == set(snapshot["objects"])
    assert {obj["file_id"] for obj in targets} == {
        obj["fileID"] for obj in snapshot["objects"].values()
    }
    assert all(
        len([e for e in operations if e["operation"]["clip_id"] == item.clip_id]) <= 64
        for item in case["selected"]
    )
    for index, item in enumerate(case["selected"]):
        descriptor = json.loads(item.descriptor_json)
        clip = snapshot["rows"]["clips"].get(item.clip_id)
        if clip is None:
            assert scene.item(item).status != "acknowledged"
            clip = snapshot["rows"]["clip_storage_authorities"][item.clip_id]["identity"]
        for key in (
            "ownership_receipt",
            "ownership_receipt_sha256",
            "save_request_id",
            "save_receipt_sha256",
            "postgame_manifest_sha256",
            "content_sha256",
            "byte_size",
        ):
            if key in clip:
                assert clip[key] == descriptor[key]
            else:
                assert key in {"ownership_receipt", "save_request_id"}
        inventory = case["control"]("storage-snapshot", clipId=item.clip_id)
        assert inventory["erasable"] is False and inventory["complete"] is False
        assert all(
            snapshot["objects"][obj["cloud_path"]]["hex"] in (None, case["payloads"][index].hex())
            for obj in inventory["objects"]
        )
        if not (removed and item.clip_id == scene.target.clip_id):
            assert item.local_path.read_bytes() == case["payloads"][index]
    for put in snapshot["objectPuts"]:
        body = bytes.fromhex(snapshot["objects"][put["cloudPath"]]["hex"])
        assert put["sha256"] == hashlib.sha256(body).hexdigest()
        assert put["byteSize"] == len(body)
    after = local_snapshot(scene.store.path)
    mutable = {
        "field_saved_media_outbox",
        "field_saved_upload_requests",
        "field_upload_receipt_outbox",
    }
    assert {k: v for k, v in scene.original.items() if k not in mutable} == {
        k: v for k, v in after.items() if k not in mutable
    }
    for index, request in enumerate(snapshot["requests"]):
        if not request["path"].endswith("/upload-receipt"):
            continue
        assert request["bodyRaw"] == canonical(request["body"])
        assert request["bodySha256"] == hashlib.sha256(request["bodyRaw"].encode()).hexdigest()
        # Every actual receipt send is immediately preceded by actual signed attestation.
        assert snapshot["requests"][index - 1]["path"] == "/v1/runtime-attestation"
        assert snapshot["requests"][index - 1]["status"] == 200
        if request["status"] == 200:
            request_id = request["body"]["upload_request"]["request_id"]
            original = next(r for r in queue(scene) if r["request_id"] == request_id)
            ack = request["response"]["data"]
            assert ack["receipt_sha256"] == original["receipt_sha256"]
            assert ack["erasable"] is False
            page_hash = hashlib.sha256(original["clip_id"].encode()).hexdigest()
            expected_reads = {f"page-{page_hash}-{i}" for i in range(4)}
            receipt_callbacks = []
            for calls in request["calls"]:
                reads = [
                    c["id"]
                    for c in calls
                    if c["collection"] == "upload_terminal_receipts" and c["method"] == "get"
                ]
                if reads:
                    receipt_callbacks.append(reads)
                    assert len(reads) == 4 and set(reads) == expected_reads
                else:
                    assert {c["collection"] for c in calls} == {"sync_nonces"}
            assert len(receipt_callbacks) == 1
    encoded = canonical({k: v for k, v in snapshot["rows"].items() if k != "sync_devices"})
    for secret in (
        "synthetic-only-secret",
        "synthetic-token",
        "synthetic-signature",
        "synthetic.objects.invalid",
        str(scene.target.local_path),
    ):
        assert secret not in encoded
    sends = receipt_requests(snapshot)
    print(
        f"{label}: callbacks={len(callbacks)} maximum={max(a['count'] for a in callbacks)} "
        f"receipt_max={max((n for r in sends for n in r['attempts']), default=0)} "
        f"receipts={len(sends)} attestations="
        f"{sum(r['path'] == '/v1/runtime-attestation' for r in snapshot['requests'])} "
        f"pages={len(pages)} entries={len(entries)} targets={len(targets)} "
        f"puts={len(snapshot['objectPuts'])} sdk={len(snapshot['sdkCalls'])}"
    )


@pytest.mark.parametrize("lose_ack", [False, True])
def test_first_late_receipt_after_owner_delete_compaction_and_source_removal(saved_match, lose_ack):
    case = saved_match
    scene = Scene(case)
    scene.step("acknowledged")
    assert scene.worker.run_once() is False
    assert_views(case)
    original_queue = queue(scene)
    assert len(original_queue) == 2
    # Settle only the unrelated owner, ensuring target's first receipt is late.
    outbox = UploadReceiptOutbox(scene.store, SCOPE)
    for _ in range(2):
        item = outbox.claim_due(scene.clock, lease_seconds=1)
        if item.clip_id == scene.target.clip_id:
            outbox.retry(
                item,
                "receipt_unavailable",
                (scene.clock + timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
            )
        else:
            result = client(scene).run_upload_receipt_once(item)
            assert result.outcome == "complete"
            assert outbox.acknowledge(item, result.receipt_ack)
    target = next(r for r in original_queue if r["clip_id"] == scene.target.clip_id)
    assert not any(
        r["path"].endswith(f"/{target['cloud_job_id']}/upload-receipt")
        for r in case["control"]("snapshot")["requests"]
    )
    deleted = phone(
        case["control"],
        "clips.delete",
        who=scene.target_who,
        clip_id=scene.target.clip_id,
        request_id="late-receipt-owner-delete",
    )
    assert deleted["clip_id"] == scene.target.clip_id
    # Explicit test I/O: remove only mutable job, then only owned synthetic source.
    case["control"]("compact-saved-job", jobId=target["cloud_job_id"])
    scene.target.local_path.unlink()
    scene.clock += timedelta(seconds=11)
    before = case["control"]("snapshot")
    assert target["cloud_job_id"] not in before["rows"]["saved_clip_sync_jobs"]
    assert before["rows"]["clip_storage_authorities"][scene.target.clip_id]["admission"] == "closed"
    if lose_ack:
        case["control"]("fault", kind=dict(path="/upload-receipt", kind="response-loss"))
    assert receipt_worker(scene).run_once()
    committed = case["control"]("snapshot")
    sent = receipt_requests(committed)[-1]
    assert sent["status"] == 200 and sent["response"]["data"]["erasable"] is False
    assert sent["bodyRaw"] == target["receipt_json"]
    if lose_ack:
        assert (
            next(r for r in queue(scene) if r["request_id"] == target["request_id"])["status"]
            == "pending"
        )
        scene.restart()
        assert receipt_worker(scene).run_once()
        replayed = case["control"]("snapshot")
        replay = receipt_requests(replayed)[-1]
        assert (replay["bodyRaw"], replay["bodySha256"], replay["response"]) == (
            sent["bodyRaw"],
            sent["bodySha256"],
            sent["response"],
        )
        assert (
            replayed["rows"]["upload_terminal_receipts"]
            == committed["rows"]["upload_terminal_receipts"]
        )
        assert len(replayed["requests"]) == len(committed["requests"]) + 2
    after = case["control"]("snapshot")
    stable_provider(before, after)
    assert all(r["status"] == "acknowledged" for r in queue(scene))
    assert scene.worker.run_once() is False
    assert (
        phone(case["control"], "clips.list", who=scene.target_who, session_id=case["session"]) == []
    )
    own = phone(case["control"], "clips.list", who=scene.other_who, session_id=case["session"])
    assert [c["clip_id"] for c in own] == [scene.other.clip_id]
    playback = phone(
        case["control"], "clips.getPlayback", who=scene.other_who, clip_id=scene.other.clip_id
    )
    with urlopen(case["base"] + urlsplit(playback["url"]).path, timeout=5) as result:
        assert result.read() == case["payloads"][1 - scene.target_index]
    for who in (scene.target_who, scene.other_who):
        refused = case["control"](
            "phone",
            action="clips.getPlayback",
            openid=f"synthetic-phone-{who}",
            payload={"clip_id": scene.target.clip_id},
        )
        assert refused["error"]["code"] == "MEDIA_NOT_AVAILABLE"
    audit(scene, f"late-delete-ack-loss={lose_ack}", removed=True)


def test_lost_authorization_retires_original_then_receipt_without_new_authorization(saved_match):
    case = saved_match
    scene = Scene(case)
    case["control"]("fault", kind=dict(path="/media-authorization", kind="response-loss"))
    scene.step("pending", "NETWORK_UNAVAILABLE")
    scene.restart()
    scene.step("pending", "SYNC_UPLOAD_REQUEST_RETIRED")
    retired = clip_rows(scene.store, scene.target)[0]
    assert retired["state"] == "retired" and retired["upload_attempt_json"]
    before = case["control"]("snapshot")
    drain(scene)
    after = case["control"]("snapshot")
    stable_provider(before, after)
    assert all(
        r["path"].endswith(("/upload-receipt", "/runtime-attestation"))
        for r in after["requests"][len(before["requests"]) :]
    )
    sent = next(
        r
        for r in receipt_requests(after)
        if r["body"]["upload_request"]["request_id"] == retired["request_id"]
    )
    assert sent["body"]["evidence"]["kind"] == "retired_before_start"
    assert sent["body"]["upload_attempt"] == json.loads(retired["upload_attempt_json"])
    audit(scene, "authorization-loss-retired")


class ControlledExit(BaseException):
    """Test-only abrupt interruption after actual PUT, before local HTTP result."""


def test_started_then_unknown_never_queues_or_sends_terminal(saved_match):
    scene = Scene(saved_match)
    transport = scene.worker._client._transport
    original = transport.put_bytes_once

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise ControlledExit()

    transport.put_bytes_once = interrupted
    with pytest.raises(ControlledExit):
        scene.worker.run_once()
    assert clip_rows(scene.store, scene.target)[0]["state"] == "started"
    assert not any(r["clip_id"] == scene.target.clip_id for r in queue(scene))
    drain(scene)
    scene.restart()
    scene.step("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
    assert clip_rows(scene.store, scene.target)[0]["state"] == "unknown"
    before = saved_match["control"]("snapshot")
    assert receipt_worker(scene).run_once() is False
    assert saved_match["control"]("snapshot") == before
    assert not any(r["clip_id"] == scene.target.clip_id for r in queue(scene))
    assert not any(
        r["path"].endswith(
            f"/{clip_rows(scene.store, scene.target)[0]['cloud_job_id']}/upload-receipt"
        )
        for r in before["requests"]
    )
    assert len(ledger_rows(scene.store)) == 2
    audit(scene, "started-unknown")


def test_signed_foreign_job_and_conflicting_evidence_refused_without_provider(saved_match):
    scene = Scene(saved_match)
    scene.step("acknowledged")
    drain(scene)
    rows = queue(scene)
    original = rows[0]
    # Reclaim from retained immutable fields solely for direct signed negative stimuli.
    item = UploadReceiptOutboxItem(
        request_id=original["request_id"],
        cloud_job_id=rows[1]["cloud_job_id"],
        clip_id=original["clip_id"],
        receipt_json=original["receipt_json"],
        receipt_sha256=original["receipt_sha256"],
        claim_token="55555555-5555-4555-8555-555555555555",
        scope=SCOPE,
        attempts=1,
        lease_until_utc="2099-01-01T00:00:00.000000Z",
        next_attempt_at_utc=None,
        last_error_code=None,
    )
    before = saved_match["control"]("snapshot")
    foreign = client(scene).run_upload_receipt_once(item)
    assert foreign.outcome == "blocked" and foreign.error_code == "receipt_conflict"
    body = json.loads(original["receipt_json"])
    body["evidence"]["local_binding_sha256"] = "0" * 64
    altered = canonical(body)
    conflict = client(scene).run_upload_receipt_once(
        replace(
            item,
            cloud_job_id=original["cloud_job_id"],
            receipt_json=altered,
            receipt_sha256=hashlib.sha256(altered.encode()).hexdigest(),
        )
    )
    assert conflict.outcome == "blocked" and conflict.error_code == "receipt_conflict"
    after = saved_match["control"]("snapshot")
    stable_provider(before, after)
    assert after["rows"]["upload_terminal_receipts"] == before["rows"]["upload_terminal_receipts"]
    audit(scene, "scope-and-conflict")
