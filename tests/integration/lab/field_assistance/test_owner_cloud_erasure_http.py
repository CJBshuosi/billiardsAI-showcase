"""Signed cloud erase composition; controlled DB/identity/SDK/native admission only."""
# Imported pytest fixtures are injected by their public names.
# ruff: noqa: F811

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import urlopen

import pytest

from billiards_ai.edge.route_assistance.owner_media_deletion import OwnerDeleteDeviceTask
from billiards_ai.edge.route_assistance.sync import FieldSyncClient, UrllibSyncHttpTransport
from tests.integration.lab.field_assistance.test_durable_upload_requests_http import (
    Scene,
    clip_rows,
)
from tests.integration.lab.field_assistance.test_managed_command_http import (
    Credentials,
    bridge,  # noqa: F401
    synthetic_resources,
)
from tests.integration.lab.field_assistance.test_managed_media_admission_http import canonical
from tests.integration.lab.field_assistance.test_managed_shared_startup_http import phone
from tests.integration.lab.field_assistance.test_managed_storage_writers_http import (
    assert_views,
    saved_match,  # noqa: F401
)
from tests.integration.lab.field_assistance.test_upload_terminal_receipts_http import (
    ControlledExit,
    client,
    drain,
    queue,
    receipt_requests,
)
from tests.unit.edge.route_assistance.test_managed_session_persistence import (
    snapshot as local_snapshot,
)

pytestmark = [
    pytest.mark.parametrize("bridge", ["owner-erasure"], indirect=True),
    pytest.mark.parametrize("saved_match", [synthetic_resources], indirect=True),
]
BUCKET = "synthetic-media-123456"


def wait_for(control, name):
    deadline = time.monotonic() + 5
    while not control("snapshot")["ownerErasure"][name]:
        assert time.monotonic() < deadline, f"controlled {name} boundary not reached"
        time.sleep(0.01)


def delete(scene):
    result = phone(
        scene.case["control"],
        "clips.delete",
        who=scene.target_who,
        clip_id=scene.target.clip_id,
        request_id="signed-cloud-owner-delete",
    )
    assert result["status"] == "deletion_requested"
    return result["deletion_job_id"]


def complete(scene, job_id):
    control = scene.case["control"]
    for _ in range(80):
        before = control("snapshot")["ownerErasure"]["nativeCalls"]
        result = control("erase-once", jobId=job_id)
        after = control("snapshot")["ownerErasure"]["nativeCalls"]
        assert len(after) - len(before) <= 1
        if result["status"] == "complete":
            return result["receipt"]
    pytest.fail("actual internal worker did not establish complete exact-target absence")


def closed(scene):
    scene.step("acknowledged")
    assert_views(scene.case)
    job_id = delete(scene)
    drain(scene)  # Actual attested signed terminal delivery, first target receipt after delete.
    return job_id


def assert_device_issuance(before, rows, receipt):
    """Assert exact intentional finalization additions; never ignore unknown rows."""
    job = receipt["deletion_job_id"]

    def digest(value):
        return hashlib.sha256(canonical(value).encode()).hexdigest()

    supplied = dict(
        contract="owner_delete_device_v1",
        deletion_job_id=job,
        environment_id=receipt["environment_id"],
        cloud_erase_receipt_sha256=receipt["receipt_sha256"],
    )
    task = {
        f.name: supplied[f.name] if f.name in supplied else receipt["scope"][f.name]
        for f in fields(OwnerDeleteDeviceTask)
    }
    parsed = OwnerDeleteDeviceTask.from_payload(task)
    issued = rows["owner_device_deletion_tasks"][job]
    stamp = issued["issued_at_utc"]
    assert (
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
        == stamp
    )
    expected_issue = dict(
        contract="owner_device_deletion_issuance_v1",
        deletion_job_id=job,
        environment_id=task["environment_id"],
        device_id=task["device_id"],
        table_id=task["table_id"],
        clip_id=task["clip_id"],
        task=task,
        task_sha256=parsed.task_sha256,
        cloud_erase_receipt_sha256=receipt["receipt_sha256"],
        issued_at_utc=stamp,
    )
    page_key = hashlib.sha256(task["clip_id"].encode()).hexdigest()
    pages = [
        before["rows"].get("upload_terminal_receipts", {}).get(f"page-{page_key}-{i}")
        for i in range(4)
    ]
    entries = [entry for page in pages if page for entry in page["entries"] if entry]
    witness = dict(
        contract="owner_erasure_closure_witness_v1",
        job_id=job,
        receipt_sha256=receipt["receipt_sha256"],
        scope_sha256=receipt["scope_sha256"],
        inputs_sha256=receipt["inputs_sha256"],
        receipt_sequences=[entry["sequence"] for entry in entries],
        receipt_page_sha256=[digest(page) for page in pages],
        birth_sha256=digest(before["rows"]["managed_session_births"][task["session_id"]]),
    )
    waiting = before["rows"]["owner_device_deletion_deliveries"][job]
    ready = {
        **waiting,
        "environment_id": task["environment_id"],
        "status": "ready",
        "task_sha256": parsed.task_sha256,
        "due_at_utc": "1970-01-01T00:00:00.000Z",
        "revision": waiting["revision"] + 1,
    }
    for name, value in (
        ("owner_cloud_erasure_witnesses", witness),
        ("owner_device_deletion_tasks", expected_issue),
        ("owner_device_deletion_deliveries", ready),
    ):
        assert rows[name] == {**before["rows"].get(name, {}), job: value}


def audit(scene, job_id, before, receipt, label):
    case, control = scene.case, scene.case["control"]
    after = control("snapshot")
    rows = after["rows"]
    assert rows["owner_cloud_erasure_receipts"][job_id] == receipt
    assert receipt["contract"] == "owner_cloud_erase_receipt_v1"
    assert receipt["scope"] == rows["owner_clip_tombstones"][scene.target.clip_id]
    assert receipt["deletion_job_id"] == job_id
    assert receipt["environment_id"] == "synthetic"
    assert (
        receipt["authority_epoch"]
        == rows["clip_storage_authorities"][scene.target.clip_id]["epoch"]
    )
    descriptor = json.loads(scene.target.descriptor_json)
    for key in (
        "ownership_receipt",
        "ownership_receipt_sha256",
        "save_request_id",
        "save_receipt_sha256",
        "postgame_manifest_sha256",
        "content_sha256",
        "byte_size",
    ):
        assert receipt["scope"][key] == descriptor[key]
    assert receipt["scope"]["owner_user_id"] == descriptor["owner_user_id"]
    assert receipt["scope"]["session_id"] == case["session"]
    assert receipt["scope"]["device_id"] == "device-1"
    assert receipt["scope"]["table_id"] == "table-1"
    assert (
        receipt["scope_sha256"] == hashlib.sha256(canonical(receipt["scope"]).encode()).hexdigest()
    )
    assert (
        receipt["targets_sha256"]
        == hashlib.sha256(canonical(receipt["targets"]).encode()).hexdigest()
    )
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    assert receipt["receipt_sha256"] == hashlib.sha256(canonical(unsigned).encode()).hexdigest()
    native = receipt["native_admission"]
    assert (native["scope_sha256"], native["inputs_sha256"], native["targets_sha256"]) == (
        receipt["scope_sha256"],
        receipt["inputs_sha256"],
        receipt["targets_sha256"],
    )
    assert (native["bucket"], native["region"], native["provider_policy"]) == (
        BUCKET,
        "ap-guangzhou",
        "cos_exact_target_absence_v1",
    )
    inventory = control("storage-snapshot", clipId=scene.target.clip_id)
    assert {t["cloud_path"] for t in receipt["targets"]} == {
        op["target"]["cloud_path"] for op in inventory["operations"]
    }
    assert all(r["status"] == "absent" for r in receipt["results"])
    assert [r["target_sha256"] for r in receipt["results"]] == [
        hashlib.sha256(canonical(t).encode()).hexdigest() for t in receipt["targets"]
    ]
    target_keys = {t["cloud_path"] for t in receipt["targets"]}
    assert all(
        t["file_id"] == f"cloud://synthetic.{BUCKET}/{t['cloud_path']}" for t in receipt["targets"]
    )
    assert target_keys <= set(before["objects"])
    assert set(after["objects"]) == set(before["objects"]) - target_keys
    assert all(
        after["objects"][k] == v for k, v in before["objects"].items() if k not in target_keys
    )
    assert not any(key in target_keys for key in after["ownerErasure"]["versions"])
    for key in ("sdkCalls", "objectPuts", "providerCalls", "spoolEntries"):
        assert after[key] == before[key]
    assert_device_issuance(before, rows, receipt)
    intentional = {
        "owner_cloud_erasure_executions",
        "owner_cloud_erasure_receipts",
        "owner_cloud_erasure_witnesses",
        "owner_device_deletion_tasks",
        "owner_device_deletion_deliveries",
    }
    assert set(rows) - set(before["rows"]) <= intentional
    # All original facts, owner job/tombstone and closed admission stay byte-exact.
    for name, value in before["rows"].items():
        if name not in intentional:
            assert rows[name] == value, name
    assert after["requests"] == before["requests"]
    # Exact retained evidence remains reviewable outside the bounded console log.
    evidence = {
        "scenario": label,
        "receipt": receipt,
        "native_calls": after["ownerErasure"]["nativeCalls"],
        "versions_before": before["ownerErasure"]["versions"],
        "objects_before": before["objects"],
        "objects_after": after["objects"],
        "callbacks": after["attempts"],
        "original_cloud_sha256": {
            name: hashlib.sha256(canonical(value).encode()).hexdigest()
            for name, value in before["rows"].items()
            if name not in {"owner_cloud_erasure_executions", "owner_cloud_erasure_receipts"}
        },
    }
    evidence_path = scene.store.path.parent / "owner-cloud-erasure-evidence.json"
    evidence_path.write_text(canonical(evidence), encoding="utf-8")
    mutable = {
        "field_saved_media_outbox",
        "field_saved_upload_requests",
        "field_upload_receipt_outbox",
    }
    after_local = local_snapshot(scene.store.path)
    assert {k: v for k, v in scene.original.items() if k not in mutable} == {
        k: v for k, v in after_local.items() if k not in mutable
    }
    assert [item.local_path.read_bytes() for item in case["selected"]] == case["payloads"]
    assert phone(control, "clips.list", who=scene.target_who, session_id=case["session"]) == []
    assert [
        c["clip_id"]
        for c in phone(control, "clips.list", who=scene.other_who, session_id=case["session"])
    ] == [scene.other.clip_id]
    for who in (scene.target_who, scene.other_who):
        refused = control(
            "phone",
            action="clips.getPlayback",
            openid=f"synthetic-phone-{who}",
            payload={"clip_id": scene.target.clip_id},
        )
        assert refused["error"]["code"] == "MEDIA_NOT_AVAILABLE"
    playback = phone(control, "clips.getPlayback", who=scene.other_who, clip_id=scene.other.clip_id)
    with urlopen(case["base"] + urlsplit(playback["url"]).path, timeout=5) as response:
        assert response.read() == case["payloads"][1 - scene.target_index]
    callbacks = after["attempts"]
    assert all(a["count"] <= 90 for a in callbacks)
    assert all(
        c["method"] in {"get", "set", "update", "remove"} for a in callbacks for c in a["calls"]
    )
    erase_callbacks = [
        a
        for a in callbacks
        if any(c["collection"].startswith("owner_cloud_erasure_") for c in a["calls"])
    ]
    assert erase_callbacks
    for attempt in erase_callbacks:
        assert (
            sum(
                c["collection"] == "upload_terminal_receipts" and c["method"] == "get"
                for c in attempt["calls"]
            )
            == 4
        )
        assert not any(c["collection"] == "managed_upload_requests" for c in attempt["calls"])
        assert all(c["collection"] in intentional for c in attempt["calls"] if c["method"] != "get")
    calls = after["ownerErasure"]["nativeCalls"]
    assert {c["key"] for c in calls if c["method"] == "DELETE"} == target_keys
    assert all(c["key"] is None or c["key"] in target_keys for c in calls)
    assert after["ownerErasure"]["credentials"] == len(calls)
    assert not after["ownerErasure"]["heldNative"]
    retained = canonical({k: v for k, v in rows.items() if k != "sync_devices"})
    for secret in (
        "synthetic-only-secret",
        "controlled-secret",
        "controlled-id",
        "synthetic-signature",
    ):
        assert secret not in retained
    print(
        f"{label}: targets={len(target_keys)} native={len(calls)} "
        f"callbacks={len(callbacks)} max={max(a['count'] for a in callbacks)} "
        f"erase_callbacks={len(erase_callbacks)} "
        f"erase_max={max(a['count'] for a in erase_callbacks)} "
        f"sdk={len(after['sdkCalls'])} puts={len(after['objectPuts'])} "
        f"evidence={evidence_path}"
    )


@pytest.mark.parametrize("mode", ["NeverEnabled", "Enabled", "Suspended"])
def test_phone_delete_late_signed_receipt_orphan_and_winner_all_erased(saved_match, mode):
    scene, control = Scene(saved_match), saved_match["control"]
    control("hold-upload")
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(scene.worker.run_once)
        try:
            deadline = time.monotonic() + 5
            while not control("snapshot")["heldUpload"]:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            uploaded = clip_rows(scene.store, scene.target)[0]
            assert uploaded["state"] == "succeeded"
            assert (
                client(scene)
                ._post(
                    f"/v1/sync/saved-clips/{uploaded['cloud_job_id']}/complete",
                    canonical(
                        {"upload_attempt": json.loads(uploaded["upload_attempt_json"])}
                    ).encode(),
                )
                .status_code
                == 200
            )
            assert_views(saved_match)
            job_id = delete(scene)
            assert not any(
                r["body"]["upload_request"]["request_id"] == uploaded["request_id"]
                for r in receipt_requests(control("snapshot"))
            )
        finally:
            if control("snapshot")["heldUpload"]:
                control("release-upload")
        assert pending.result(timeout=5)
    # No receipt/unknown application closure can be turned into provider absence.
    assert control("erase-once", jobId=job_id) == {"status": "pending", "receipt": None}
    assert control("snapshot")["ownerErasure"]["nativeCalls"] == []
    drain(scene)
    sent = next(
        r
        for r in receipt_requests(control("snapshot"))
        if r["body"]["upload_request"]["request_id"] == uploaded["request_id"]
    )
    terminal = next(r for r in queue(scene) if r["request_id"] == uploaded["request_id"])
    assert sent["bodyRaw"] == terminal["receipt_json"]
    assert sent["bodySha256"] == terminal["receipt_sha256"]
    assert sent["response"]["data"]["erasable"] is False
    uploaded_snapshot = control("snapshot")
    for obj in control("storage-snapshot", clipId=scene.target.clip_id)["objects"]:
        assert uploaded_snapshot["objects"][obj["cloud_path"]]["hex"] == (
            saved_match["payloads"][scene.target_index].hex()
        )
    control("erase-versioning", mode=mode, clipId=scene.target.clip_id)
    before = control("snapshot")
    receipt = complete(scene, job_id)
    assert [t["writer_state"] for t in receipt["targets"]] == [
        "upload_succeeded",
        "copy_succeeded",
        "copy_succeeded",
    ]
    assert len(receipt["scope"]["storage_objects"]) == 2
    assert (
        len(
            {t["file_id"] for t in receipt["targets"]}
            - {t["file_id"] for t in receipt["scope"]["storage_objects"]}
        )
        == 1
    )
    if mode != "NeverEnabled":
        native = control("snapshot")["ownerErasure"]["nativeCalls"]
        deletes = [c for c in native if c["method"] == "DELETE"]
        assert len(deletes) == 9
        assert {c["query"]["versionId"] for c in deletes} == {"current", "historical", "marker"}
        assert all(c["status"] == 404 for c in native if c["probe"])
        # The controlled current view begins at marker404 with retained media;
        # the real provider must list/delete all histories before any final proof.
        assert all(len(v) == 3 for v in before["ownerErasure"]["versions"].values())
        for key, versions in before["ownerErasure"]["versions"].items():
            assert before["objects"][key]["hex"] is None  # current delete marker
            assert versions[-1] == {"id": "marker", "marker": True, "hex": None}
            assert all(
                v["hex"] == saved_match["payloads"][scene.target_index].hex() for v in versions[:-1]
            )
        assert all(c["method"] == "DELETE" or not c["probe"] for c in native[:5])
    audit(scene, job_id, before, receipt, f"late-orphan-{mode}")


def test_unknown_started_writer_refuses_erase_without_any_native_request(saved_match):
    scene, control = Scene(saved_match), saved_match["control"]
    transport = scene.worker._client._transport
    original = transport.put_bytes_once

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise ControlledExit()

    transport.put_bytes_once = interrupt
    with pytest.raises(ControlledExit):
        scene.worker.run_once()
    started = clip_rows(scene.store, scene.target)[0]
    assert started["state"] == "started"
    signed = FieldSyncClient(
        credentials=Credentials(), transport=UrllibSyncHttpTransport(saved_match["base"])
    )
    assert (
        signed._post(
            f"/v1/sync/saved-clips/{started['cloud_job_id']}/complete",
            canonical({"upload_attempt": json.loads(started["upload_attempt_json"])}).encode(),
        ).status_code
        == 200
    )
    job_id = delete(scene)
    scene.restart()
    scene.step("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
    drain(scene)
    before = control("snapshot")
    assert control("erase-once", jobId=job_id) == {"status": "pending", "receipt": None}
    after = control("snapshot")
    assert after["objects"] == before["objects"]
    assert after["ownerErasure"]["nativeCalls"] == []
    assert after["ownerErasure"]["credentials"] == 0
    assert not after["rows"].get("owner_cloud_erasure_receipts")
    assert after["sdkCalls"] == before["sdkCalls"]
    assert not any(r["clip_id"] == scene.target.clip_id for r in queue(scene))
    print(
        f"unknown-started: native=0 credentials=0 "
        f"callbacks={len(after['attempts'])} max={max(a['count'] for a in after['attempts'])}"
    )


def test_checkpoint_crash_restart_and_lost_checkpoint_and_final_ack_replay(saved_match):
    scene, control = Scene(saved_match), saved_match["control"]
    job_id = closed(scene)
    before = control("snapshot")
    assert control("erase-once", jobId=job_id)["status"] == "pending"  # versioning checkpoint
    checkpoint = control("snapshot")["rows"]["owner_cloud_erasure_executions"][job_id]
    control("erase-crash-after-delete")
    with pytest.raises(HTTPError) as error:
        control("erase-once", jobId=job_id)
    assert error.value.code == 500
    crashed = control("snapshot")
    assert (
        crashed["rows"]["owner_cloud_erasure_executions"][job_id]["checkpoint"]
        == checkpoint["checkpoint"]
    )
    assert len(crashed["objects"]) == len(before["objects"]) - 1
    # This is a fresh factory/worker invocation retaining controlled DB/map;
    # no claim is made about OS reboot or real CloudBase persistence.
    control("erase-restart", advanceMs=30001)
    with pytest.raises(OSError):
        control("erase-once", jobId=job_id, loseAck=True)
    replayed = control("snapshot")
    deletes = [c for c in replayed["ownerErasure"]["nativeCalls"] if c["method"] == "DELETE"]
    assert len(deletes) == 2 and deletes[0]["key"] == deletes[1]["key"]
    assert (
        replayed["rows"]["owner_cloud_erasure_executions"][job_id]["checkpoint"]["phase"]
        == "simple_probe"
    )
    control("erase-restart")
    # Lose only the invocation response after actual final receipt commit.
    control("erase-lose-final-ack")
    with pytest.raises(OSError):
        complete(scene, job_id)
    committed = control("snapshot")
    receipt = committed["rows"]["owner_cloud_erasure_receipts"][job_id]
    native_count = len(committed["ownerErasure"]["nativeCalls"])
    control("erase-restart", advanceMs=30001)
    assert control("erase-once", jobId=job_id) == {"status": "complete", "receipt": receipt}
    assert len(control("snapshot")["ownerErasure"]["nativeCalls"]) == native_count
    assert (
        control("snapshot")["rows"]["owner_cloud_erasure_receipts"]
        == committed["rows"]["owner_cloud_erasure_receipts"]
    )
    audit(scene, job_id, before, receipt, "crash-checkpoint-final-ack-loss")


def test_native_result_after_lease_takeover_cannot_checkpoint_or_finalize(saved_match):
    scene, control = Scene(saved_match), saved_match["control"]
    job_id = closed(scene)
    before = control("snapshot")
    control("erase-once", jobId=job_id)  # actual versioning response/checkpoint
    control("erase-hold-native")
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(control, "erase-once", jobId=job_id)
        try:
            wait_for(control, "heldNative")
            held = control("snapshot")
            assert held["ownerErasure"]["nativeCalls"][-1]["method"] == "DELETE"
            control("erase-restart", advanceMs=30001)
            control("erase-once", jobId=job_id)
            fresh = control("snapshot")["rows"]["owner_cloud_erasure_executions"][job_id]
        finally:
            if control("snapshot")["ownerErasure"]["heldNative"]:
                control("erase-release-native")
        assert pending.result(timeout=5) == {"status": "pending", "receipt": None}
    assert control("snapshot")["rows"]["owner_cloud_erasure_executions"][job_id] == fresh
    assert not control("snapshot")["rows"].get("owner_cloud_erasure_receipts")
    receipt = complete(scene, job_id)
    audit(scene, job_id, before, receipt, "stale-native-result")
