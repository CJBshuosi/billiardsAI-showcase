"""Signed durable recovery; synthetic SDK/DB/model/shot boundaries, no native proof."""
# Imported pytest fixtures are injected by their public names.
# ruff: noqa: F811

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import urlopen

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.sync import FieldSyncClient, UrllibSyncHttpTransport
from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundSavedMediaSyncWorker
from tests.integration.lab.field_assistance.test_managed_command_http import (
    ROOT,
    SCOPE,
    Credentials,
    bridge,  # noqa: F401
    synthetic_resources,
)
from tests.integration.lab.field_assistance.test_managed_media_admission_http import canonical
from tests.integration.lab.field_assistance.test_managed_shared_startup_http import phone
from tests.integration.lab.field_assistance.test_managed_storage_writers_http import (
    ObjectTransport,
    assert_views,
    saved_match,  # noqa: F401
)
from tests.unit.edge.route_assistance.test_managed_session_persistence import (
    snapshot as local_snapshot,
)


def ledger_rows(store):
    with sqlite3.connect(store.path) as db:
        db.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in db.execute(
                "SELECT * FROM field_saved_upload_requests ORDER BY created_at_utc,request_id"
            )
        ]


def clip_rows(store, item):
    return [row for row in ledger_rows(store) if row["clip_id"] == item.clip_id]


class SingleObjectTransport(UrllibSyncHttpTransport):
    """Only translate the known synthetic SDK host; run production once PUT."""

    def __init__(self, base, store, *, exit_after_put=False):
        super().__init__(base, timeout_seconds=5, allow_redirects=False)
        self.base, self.store = base, store
        self.exit_after_put = exit_after_put

    def post_json(self, path, headers, body):
        if path.endswith("/media-authorization"):
            request_id = json.loads(body)["upload_request"]["request_id"]
            row = next(row for row in ledger_rows(self.store) if row["request_id"] == request_id)
            assert row["state"] == "requested"  # committed before signed network I/O
        elif "/saved-clips/" in path and path.endswith("/complete"):
            attempt = json.loads(body)["upload_attempt"]
            row = next(
                row
                for row in ledger_rows(self.store)
                if row["upload_attempt_json"] == canonical(attempt)
            )
            assert row["state"] == "succeeded" and row["terminal_sha256"]
        return super().post_json(path, headers, body)

    def put_bytes_once(self, url, body, content_type, headers):
        parsed = urlsplit(url)
        assert parsed.scheme == "https" and parsed.netloc == "synthetic.objects.invalid"
        assert not parsed.query and not parsed.fragment
        started = [row for row in ledger_rows(self.store) if row["state"] == "started"]
        assert len(started) == 1  # committed before the one physical transport call
        result = super().put_bytes_once(self.base + parsed.path, body, content_type, headers)
        if self.exit_after_put:
            # Real process exit after delivery, before client can record HTTP200.
            os._exit(73)
        return result

    def put_bytes(self, *_):
        pytest.fail("managed worker must use the production single-use PUT capability")


class Scene:
    def __init__(self, case):
        self.case = case
        self.clock = datetime.now(UTC) + timedelta(seconds=1)
        self.store = case["records"]
        self.store._now_utc = lambda: self.clock
        self.original = local_snapshot(self.store.path)
        self.worker = self.new_worker()
        # The real outbox orders by permanent clip ID, not shot number. Publish
        # whichever original owner is first, then fault the other owner's clip.
        assert self.worker.run_once()
        acknowledged = [
            item for item in self.store.list_saved_media_outbox() if item.status == "acknowledged"
        ]
        assert len(acknowledged) == 1
        self.other = acknowledged[0]
        self.target = next(item for item in case["selected"] if item.clip_id != self.other.clip_id)
        self.target_index = case["selected"].index(self.target)
        self.target_who = ("a", "b")[self.target_index]
        self.other_who = ("b", "a")[self.target_index]

    def new_worker(self):
        client = FieldSyncClient(
            credentials=Credentials(),
            transport=SingleObjectTransport(self.case["base"], self.store),
        )
        return _BackgroundSavedMediaSyncWorker(
            store=self.store,
            client=client,
            # Deliberately different: original persisted root must win.
            trusted_postgame_root=self.case["root"] / "successor-root",
            wake_event=threading.Event(),
            now_utc=lambda: self.clock,
            lease_seconds=1,
            managed_scope=SCOPE,
        )

    def item(self, selected=None):
        selected = self.target if selected is None else selected
        return next(
            row for row in self.store.list_saved_media_outbox() if row.clip_id == selected.clip_id
        )

    def restart(self):
        self.clock += timedelta(seconds=10)
        self.store = SQLiteFieldRecordStore(self.store.path, now_utc=lambda: self.clock)
        self.worker = self.new_worker()

    def step(self, status, error=None):
        assert self.worker.run_once()
        item = self.item()
        assert (item.status, item.last_error_code) == (status, error)


def authorize_requests(case, item):
    job = next(
        job
        for job in case["control"]("snapshot")["rows"]["saved_clip_sync_jobs"].values()
        if job["clip_id"] == item.clip_id
    )
    path = f"/v1/sync/saved-clips/{job['job_id']}/media-authorization"
    return [r for r in case["control"]("snapshot")["requests"] if r["path"] == path]


def audit(case, scene, scenario, *, operations, puts, requests):
    snapshot = case["control"]("snapshot")
    entries = [
        entry
        for page in snapshot["rows"]["media_storage_operations"].values()
        for entry in page["entries"]
    ]
    targets = [obj for entry in entries for obj in entry["operation"]["objects"]]
    assert len(entries) == operations
    assert len(targets) == len({obj["cloud_path"] for obj in targets}) == len(snapshot["objects"])
    assert {obj["cloud_path"] for obj in targets} == set(snapshot["objects"])
    assert {obj["file_id"] for obj in targets} == {
        obj["fileID"] for obj in snapshot["objects"].values()
    }
    assert len(snapshot["objectPuts"]) == puts
    assert len({put["cloudPath"] for put in snapshot["objectPuts"]}) == puts
    for put in snapshot["objectPuts"]:
        body = bytes.fromhex(snapshot["objects"][put["cloudPath"]]["hex"])
        assert put["sha256"] == hashlib.sha256(body).hexdigest()
        assert put["byteSize"] == len(body)
    assert len(snapshot["rows"]["managed_upload_requests"]) == requests
    for index, item in enumerate(case["selected"]):
        catalog = case["control"]("storage-snapshot", clipId=item.clip_id)
        assert catalog["erasable"] is False and catalog["complete"] is False
        descriptor = json.loads(item.descriptor_json)
        clip = snapshot["rows"]["clips"][item.clip_id]
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
            assert snapshot["objects"][obj["cloud_path"]]["hex"] in (
                None,
                case["payloads"][index].hex(),
            )
    retained = {key: value for key, value in snapshot["rows"].items() if key != "sync_devices"}
    encoded = canonical(retained) + canonical(ledger_rows(scene.store))
    for secret in (
        "synthetic-token",
        "synthetic-signature",
        "synthetic.objects.invalid",
        "synthetic-only-secret",
        str(scene.target.local_path),
    ):
        assert secret not in encoded
    # Only outbox lifecycle and permanent upload evidence may change. Original
    # source/birth/resource/epoch/fact/ownership/save/descriptor rows are exact.
    after = local_snapshot(scene.store.path)
    mutable = {
        "field_saved_media_outbox",
        "field_saved_upload_requests",
        "field_upload_receipt_outbox",
    }
    assert {key: value for key, value in scene.original.items() if key not in mutable} == {
        key: value for key, value in after.items() if key not in mutable
    }
    assert [item.local_path.read_bytes() for item in case["selected"]] == case["payloads"]
    assert snapshot["providerCalls"] == {"authorize": 0, "verify": 0}
    assert not snapshot["spoolEntries"]
    assert all(call["kind"] in {"grant", "read", "copy"} for call in snapshot["sdkCalls"])
    callbacks = case["control"]("snapshot")["attempts"]
    assert all(
        call["method"] in {"get", "set", "update", "remove"}
        for attempt in callbacks
        for call in attempt["calls"]
    )
    assert all(attempt["count"] <= 90 for attempt in callbacks)
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
        f"{scenario}: callbacks={len(callbacks)} maximum={max(a['count'] for a in callbacks)} "
        f"route_max={route_max} operations={operations} targets={len(targets)} "
        f"requests={requests} puts={puts} sdk={len(snapshot['sdkCalls'])}"
    )


@pytest.mark.parametrize("bridge", ["writers"], indirect=True)
@pytest.mark.parametrize("saved_match", [synthetic_resources], indirect=True)
def test_lost_authorization_reuses_permanent_request_then_retires_before_fresh_target(saved_match):
    case = saved_match
    scene = Scene(case)
    case["control"]("fault", kind=dict(path="/media-authorization", kind="response-loss"))
    scene.step("pending", "NETWORK_UNAVAILABLE")
    requested = clip_rows(scene.store, scene.target)
    assert [row["state"] for row in requested] == ["requested"]
    before = case["control"]("snapshot")
    scene.restart()
    scene.step("pending", "SYNC_UPLOAD_REQUEST_RETIRED")
    retired = clip_rows(scene.store, scene.target)
    assert [row["state"] for row in retired] == ["retired"]
    assert retired[0]["request_id"] == requested[0]["request_id"]
    assert case["control"]("snapshot")["sdkCalls"] == before["sdkCalls"]
    assert case["control"]("snapshot")["objectPuts"] == before["objectPuts"]
    scene.restart()
    scene.step("acknowledged")
    history = clip_rows(scene.store, scene.target)
    assert [row["state"] for row in history] == ["retired", "succeeded"]
    assert history[0] == retired[0]
    attempts = [json.loads(row["upload_attempt_json"]) for row in history]
    assert attempts[0]["operation_id"] != attempts[1]["operation_id"]
    calls = authorize_requests(case, scene.target)
    assert len(calls) == 3
    assert calls[0]["body"] == calls[1]["body"]
    assert calls[2]["body"] != calls[1]["body"]
    assert_views(case)
    audit(case, scene, "authorize-loss", operations=5, puts=2, requests=3)


@pytest.mark.parametrize("bridge", ["writers"], indirect=True)
@pytest.mark.parametrize("saved_match", [synthetic_resources], indirect=True)
def test_success_receipt_survives_complete_ack_loss_and_replays_exact_complete_without_put(
    saved_match,
):
    case = saved_match
    scene = Scene(case)
    case["control"]("fault", kind=dict(path="/complete", kind="response-loss"))
    scene.step("pending", "NETWORK_UNAVAILABLE")
    receipt = clip_rows(scene.store, scene.target)
    assert len(receipt) == 1 and receipt[0]["state"] == "succeeded"
    assert receipt[0]["terminal_json"] and receipt[0]["terminal_sha256"]
    before = case["control"]("snapshot")
    assert before["rows"]["clips"][scene.target.clip_id]["published_at"]
    scene.restart()
    # Completion replay must not need source bytes or a reconstructed URL.
    withheld = scene.target.local_path.with_suffix(".withheld")
    scene.target.local_path.rename(withheld)
    try:
        scene.step("acknowledged")
    finally:
        withheld.rename(scene.target.local_path)
    after = case["control"]("snapshot")
    assert clip_rows(scene.store, scene.target) == receipt
    assert after["sdkCalls"] == before["sdkCalls"]
    assert after["objectPuts"] == before["objectPuts"]
    assert len(after["requests"]) == len(before["requests"]) + 1
    assert after["requests"][-1]["path"].endswith("/complete")
    assert after["requests"][-1]["body"] == before["requests"][-1]["body"]
    assert_views(case)
    audit(case, scene, "complete-ack-loss", operations=4, puts=2, requests=2)


@pytest.mark.parametrize("bridge", ["writers"], indirect=True)
@pytest.mark.parametrize("saved_match", [synthetic_resources], indirect=True)
def test_process_exit_after_started_stays_unknown_even_when_signed_remote_complete_succeeds(
    saved_match,
):
    case = saved_match
    scene = Scene(case)
    child = """
import sys, threading
from datetime import datetime
from pathlib import Path
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.sync import FieldSyncClient
from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundSavedMediaSyncWorker
from tests.integration.lab.field_assistance.test_managed_command_http import Credentials, SCOPE
from tests.integration.lab.field_assistance.test_durable_upload_requests_http import (
    SingleObjectTransport,
)
store = SQLiteFieldRecordStore(Path(sys.argv[2]),
    now_utc=lambda: datetime.fromisoformat(sys.argv[3]))
worker = _BackgroundSavedMediaSyncWorker(store=store,
    client=FieldSyncClient(credentials=Credentials(),
        transport=SingleObjectTransport(sys.argv[1], store, exit_after_put=True)),
    trusted_postgame_root=store.path.parent / 'wrong-root',
    wake_event=threading.Event(), now_utc=lambda: datetime.fromisoformat(sys.argv[3]),
    lease_seconds=1, managed_scope=SCOPE)
worker.run_once()
raise SystemExit(99)
"""
    process = subprocess.run(
        [sys.executable, "-c", child, case["base"], str(scene.store.path), scene.clock.isoformat()],
        cwd=ROOT,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode == 73, process.stdout + process.stderr
    started = clip_rows(scene.store, scene.target)
    assert len(started) == 1 and started[0]["state"] == "started"
    # Controlled remote stimulus: a separate signed caller publishes the original
    # delivered target. The recovering worker must still refuse all network I/O.
    client = FieldSyncClient(
        credentials=Credentials(), transport=UrllibSyncHttpTransport(case["base"])
    )
    assert (
        client._post(
            f"/v1/sync/saved-clips/{started[0]['cloud_job_id']}/complete",
            canonical({"upload_attempt": json.loads(started[0]["upload_attempt_json"])}).encode(),
        ).status_code
        == 200
    )
    before = case["control"]("snapshot")
    assert before["rows"]["clips"][scene.target.clip_id]["published_at"]
    scene.restart()
    scene.step("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
    unknown = clip_rows(scene.store, scene.target)
    assert unknown[0]["state"] == "unknown"
    assert unknown[0]["request_id"] == started[0]["request_id"]
    assert unknown[0]["terminal_json"] is None
    assert case["control"]("snapshot") == before
    scene.restart()
    assert scene.worker.run_once() is False
    assert case["control"]("snapshot") == before
    assert_views(case)
    audit(case, scene, "started-process-exit", operations=4, puts=2, requests=2)


@pytest.mark.parametrize("bridge", ["writers"], indirect=True)
@pytest.mark.parametrize("saved_match", [synthetic_resources], indirect=True)
def test_delayed_original_issuer_after_owner_close_retains_observation_without_delivery(
    saved_match,
):
    case = saved_match
    scene = Scene(case)
    try:
        case["control"]("hold-grant")
        available = True
    except HTTPError:
        available = False
    assert available, "bridge needs a controlled SDK getUploadMetadata return barrier"
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(scene.worker.run_once)
        try:
            deadline = time.monotonic() + 5
            while not case["control"]("snapshot")["heldGrant"]:
                assert time.monotonic() < deadline, (
                    "SDK grant did not reach controlled return boundary"
                )
                time.sleep(0.01)
            requested = clip_rows(scene.store, scene.target)[0]
            assert requested["state"] == "requested"
            # Controlled competing old-client stimulus, through the real signed
            # compatibility client and registered provider (no published-row fake).
            # The phone can delete only after an actual publication exists.
            competitor = FieldSyncClient(
                credentials=Credentials(), transport=ObjectTransport(case["base"])
            )
            assert (
                competitor.run_saved_media_once(
                    scene.target, trusted_postgame_root=case["root"]
                ).outcome
                == "complete"
            )
            deleted = phone(
                case["control"],
                "clips.delete",
                who=scene.target_who,
                clip_id=scene.target.clip_id,
                request_id="owner-close-before-grant-return",
            )
            assert deleted["clip_id"] == scene.target.clip_id
        finally:
            if case["control"]("snapshot")["heldGrant"]:
                case["control"]("release-grant")
        assert pending.result(timeout=5)
    assert scene.item().status == "blocked" and scene.item().last_error_code == "SYNC_CONFLICT"
    retained = clip_rows(scene.store, scene.target)
    assert retained == [requested]
    snapshot = case["control"]("snapshot")
    assert (
        snapshot["rows"]["clip_storage_authorities"][scene.target.clip_id]["admission"] == "closed"
    )
    entries = [
        entry
        for page in snapshot["rows"]["media_storage_operations"].values()
        for entry in page["entries"]
        if entry["operation"]["clip_id"] == scene.target.clip_id
    ]
    assert len(entries) == 3
    late = [
        entry
        for entry in entries
        if snapshot["objects"][entry["object"]["cloud_path"]]["hex"] is None
    ]
    assert len(late) == 1 and late[0]["operation"]["outcome"] == "external_issued"
    assert late[0]["object"]["references"] == []
    assert all(
        put["cloudPath"] != late[0]["object"]["cloud_path"] for put in snapshot["objectPuts"]
    )
    # Permanent reservation recovers after close, with neither credentials nor reopen.
    client = FieldSyncClient(
        credentials=Credentials(), transport=UrllibSyncHttpTransport(case["base"])
    )
    response = client._post(
        f"/v1/sync/saved-clips/{requested['cloud_job_id']}/media-authorization",
        canonical(
            {
                "upload_request": {
                    "contract": "managed_upload_request_v1",
                    "request_id": requested["request_id"],
                }
            }
        ).encode(),
    )
    assert response.status_code == 200
    assert set(json.loads(response.body)["data"]) == {
        "upload_request",
        "upload_attempt",
        "request_status",
    }
    assert json.loads(response.body)["data"]["request_status"] == "reserved"
    for who in ("a", "b"):
        refused = case["control"](
            "phone",
            action="clips.getPlayback",
            openid=f"synthetic-phone-{who}",
            payload={"clip_id": scene.target.clip_id},
        )
        assert refused["error"]["code"] == "MEDIA_NOT_AVAILABLE"
    # The other original owner's clip remains private and byte-exact after close.
    listed = phone(case["control"], "clips.list", who=scene.other_who, session_id=case["session"])
    assert [clip["clip_id"] for clip in listed] == [scene.other.clip_id]
    assert (
        phone(case["control"], "clips.list", who=scene.target_who, session_id=case["session"]) == []
    )
    playback = phone(
        case["control"], "clips.getPlayback", who=scene.other_who, clip_id=scene.other.clip_id
    )
    with urlopen(case["base"] + urlsplit(playback["url"]).path, timeout=5) as result:
        assert result.read() == case["payloads"][1 - scene.target_index]
    scene.restart()
    before = case["control"]("snapshot")
    assert scene.worker.run_once() is False
    assert case["control"]("snapshot") == before
    assert clip_rows(scene.store, scene.target) == retained
    audit(case, scene, "late-issuer-close", operations=5, puts=2, requests=2)
