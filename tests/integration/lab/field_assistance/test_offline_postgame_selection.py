"""Conditional kiosk component coverage with a controlled producer, never a native episode."""

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import timedelta

import pytest

from billiards_ai.edge.route_assistance.control import MatchControlSnapshot, WristbandRole
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundLocalMediaCleanupWorker
from billiards_ai.lab.mainline1_session_analysis import (
    DevelopmentSessionAnalysisPublisher,
    SQLiteSessionAnalysisOutbox,
)
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from scripts.algorithm import offline_managed_transport as protocol
from scripts.algorithm import run_offline_managed_blackbox as driver
from tests.integration.lab.field_assistance.test_postgame_screen_saved_clips import (
    _media_request,
    _review_request,
    _running_review_server,
)
from tests.unit.lab.test_mainline1_shot_worker import NOW, _record_ended_usb_session, _worker

SESSION = "session-1"
SHOT = "shot-0001"
REVIEW = "/api/postgame-review"


@contextmanager
def scene(root, *, count=2, processed=None, owner="offline-red"):
    database, artifacts, jobs_root = root / "field.sqlite3", root / "artifacts", root / "jobs"
    records = SQLiteFieldRecordStore(
        database, now_utc=lambda: NOW, session_id_factory=lambda: SESSION
    )
    artifact_id = "usb-field-" + hashlib.sha256(str(artifacts.resolve()).encode()).hexdigest()
    records.open_or_resume_session(artifact_id=artifact_id)
    records.save_match(
        SESSION,
        MatchControlSnapshot(red_user_id=owner, active_role=WristbandRole.RED, revision=1),
        expected_revision=0,
    )
    jobs_db = jobs_root / "analysis-jobs.sqlite3"
    jobs = SQLiteShotAnalysisJobStore(jobs_db)
    for n in range(1, count + 1):
        _record_ended_usb_session(
            records,
            jobs,
            root / f"capture-{n}",
            session_id=SESSION,
            shot_id=f"shot-{n:04d}",
            job_id=f"job-{n}",
            epoch=f"epoch-{n}",
            artifact_id=artifact_id,
            sequence=n,
            end_session=False,
            media_root=jobs_root / "media" / SESSION / f"job-{n}",
        )
    records.end_session(SESSION)
    records.bind_artifact_path(artifact_id, artifacts)
    publisher = DevelopmentSessionAnalysisPublisher(
        records=records,
        jobs=jobs,
        outbox=SQLiteSessionAnalysisOutbox(jobs_db),
        artifact_root=artifacts,
    )
    worker = _worker(records, jobs, artifacts / "postgame" / SESSION, analysis_publisher=publisher)
    for n in range(count if processed is None else processed):
        assert worker.process_one(f"analysis-{n}", SESSION, NOW + timedelta(seconds=n))
    cleanup = _BackgroundLocalMediaCleanupWorker(
        store=records,
        jobs_db=jobs_db,
        artifact_root=artifacts,
        jobs_root=jobs_root,
        poll_seconds=0.02,
    )
    with _running_review_server(root, database, artifacts) as server:
        actor = driver._Actor(
            protocol.OfflineManagedRemote(),
            {"boundary": {"shot_id": SHOT}, "postgame": {}, "provider_delivery": "NOT_ASSERTED"},
            threading.Event(),
            (1, 1, 1, 4),
            None,
        )
        actor.store, actor.session, actor.token, actor.server = records, SESSION, "secret", server
        cleanup.start()
        try:
            yield actor
        finally:
            cleanup.stop()


@pytest.mark.parametrize("count", [1, 2])
def test_original_selection_and_real_disposal(tmp_path, count):
    with scene(tmp_path, count=count) as actor:
        store = actor.store
        clips = store.list_postgame_clips(SESSION)
        original = clips[0].local_path.read_bytes()
        review = _review_request(actor.server)[1]
        posts = []
        request = actor.request
        transient = True

        def observe(path, timeout, **kwargs):
            nonlocal transient
            if transient and path == REVIEW:
                transient = False
                raise ConnectionResetError
            if kwargs.get("payload") is not None:
                posts.append((path, kwargs["payload"].copy(), kwargs["nonce"]))
            return request(path, timeout, **kwargs)

        actor.request = observe
        actor.review()
        assert [path for path, _, _ in posts] == [REVIEW + "/save"] * 2 + [REVIEW + "/complete"]
        assert posts[0] == posts[1]
        rows = store.list_saved_media_outbox()
        assert len(rows) == 1 and rows[0].shot_id == SHOT
        assert rows[0].owner_user_id == "offline-red" and rows[0].status == "pending"
        assert clips[0].local_path.read_bytes() == original
        with sqlite3.connect(store.path) as db:
            receipts = db.execute("SELECT receipt_json FROM field_clip_save_requests").fetchall()
            assert len(receipts) == 1 and json.loads(receipts[0][0])["schema_version"] == "2.0"
            assert db.execute(
                "SELECT status FROM field_postgame_reviews WHERE session_id=?", (SESSION,)
            ).fetchone() == ("completed",)
        for item, clip in zip(review["clips"][1:], clips[1:], strict=True):
            assert not clip.local_path.exists()
            assert store.is_postgame_discarded(SESSION, clip.shot_id)
            assert _media_request(actor.server, item["media_url"])[0] == 404
        result = actor.report["postgame"]["selection"]
        assert [result[k] for k in ("save", "idempotency", "completion")] == ["VERIFIED"] * 3
        assert result["cleanup"] == ("VERIFIED" if count > 1 else "NOT_EXERCISED")
        assert result["unsaved_count"] == count - 1
        assert result["unsaved_clip_ids"] == [c.clip_id for c in clips[1:]]
        assert actor.report["provider_delivery"] == "NOT_ASSERTED"
        retained = json.dumps(actor.report)
        assert review["kiosk_nonce"] not in retained and str(tmp_path) not in retained
        (tmp_path / "component-selection.json").write_text(retained)


@pytest.mark.parametrize(
    "fault,posts,saves,completed",
    [
        ("zero", 0, 0, False),
        ("processing", 0, 0, False),
        ("became_processing", 0, 0, False),
        ("missing", 0, 0, False),
        ("ownerless", 0, 0, False),
        ("wrong_owner", 0, 0, False),
        ("wrong_session", 0, 0, False),
        ("wrong_manifest", 0, 0, False),
        ("rejected_save", 1, 0, False),
        ("lost_save", 1, 1, False),
        ("bad_receipt", 1, 1, False),
        ("bad_owner", 1, 1, False),
        ("lost_replay", 2, 1, False),
        ("changed_manifest", 1, 1, False),
        ("missing_sibling", 2, 1, False),
        ("incomplete", 2, 1, False),
        ("rejected_complete", 3, 1, False),
        ("lost_complete", 3, 1, True),
    ],
)
def test_uncertain_or_invalid_selection_stops_writes(tmp_path, fault, posts, saves, completed):
    owner = {"ownerless": None, "wrong_owner": "offline-blue"}.get(fault, "offline-red")
    with scene(
        tmp_path,
        owner=owner,
        count=0 if fault == "zero" else 2,
        processed={"processing": 0, "incomplete": 1}.get(fault),
    ) as actor:
        if fault == "missing":
            actor.report["boundary"]["shot_id"] = "shot-missing"
        actor.budgets = (1, 1, 1, 0.65)
        calls = []
        header_checked = False
        request = actor.request

        def inject(path, timeout, **kwargs):
            nonlocal header_checked
            if kwargs.get("media") is True:
                header_checked = True
            if kwargs.get("payload") is not None:
                calls.append((path, kwargs["payload"].copy()))
                if fault == "rejected_save" or (fault == "rejected_complete" and len(calls) == 3):
                    kwargs["nonce"] = "invalid"
                result = request(path, timeout, **kwargs)
                if (fault, len(calls)) in {
                    ("lost_save", 1),
                    ("lost_replay", 2),
                    ("lost_complete", 3),
                }:
                    raise ConnectionResetError("sensitive body")
                if fault in {"bad_owner", "bad_receipt"}:
                    result["owner_user_id" if fault == "bad_owner" else "clip_id"] = "invalid"
                return result
            result = request(path, timeout, **kwargs)
            if path == REVIEW:
                if fault == "became_processing" and header_checked:
                    result.update(phase="POSTGAME_PROCESSING", analysis_status="PROCESSING")
                if fault == "wrong_session":
                    result["review_session_id"] = "other"
                if fault == "wrong_manifest" or (fault == "changed_manifest" and len(calls) == 1):
                    result["postgame_manifest_sha256"] = "0" * 64
                if fault == "missing_sibling" and len(calls) >= 2:
                    result["clips"].pop()
            return result

        actor.request = inject
        with pytest.raises(driver._EpisodeFailure) as failure:
            actor.review()
        if fault == "missing_sibling":
            assert failure.value.args == ("POSTGAME_INCOMPLETE_INVENTORY",)
        assert len(calls) == posts
        if posts >= 2:
            assert calls[0] == calls[1]
        assert len(actor.store.list_saved_media_outbox()) == saves
        with sqlite3.connect(actor.store.path) as db:
            assert db.execute("SELECT COUNT(*) FROM field_clip_save_requests").fetchone() == (
                saves,
            )
            assert db.execute(
                "SELECT status FROM field_postgame_reviews WHERE session_id=?", (SESSION,)
            ).fetchone() == ("completed" if completed else "active",)
        assert "FAILED" in actor.report["postgame"]["selection"].values()
        assert actor.report["postgame"]["selection"]["cleanup"] != "VERIFIED"
        assert "sensitive" not in json.dumps(actor.report)
