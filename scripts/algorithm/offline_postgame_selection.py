"""Observe a single disposable scene's actual kiosk writes and local disposal."""

import json
import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from http.client import HTTPException

from billiards_ai.edge.route_assistance.postgame import (
    ClipSaveReceipt,
    ClipSaveStatus,
    clip_save_receipt_from_json,
)
from billiards_ai.edge.route_assistance.records import ownership_receipt_sha256
from billiards_ai.lab.field_assistance.server import _open_review_media

REVIEW = "/api/postgame-review"


class PostgameSelectionFailure(Exception):
    """Only locally defined codes cross the episode reporting boundary."""


def _require(condition, code):
    if not condition:
        raise PostgameSelectionFailure(code)


@contextmanager
def _stage(stages, name):
    stages[name] = "FAILED"
    try:
        yield
    except (KeyError, TypeError, AttributeError, ValueError, OSError, HTTPException, sqlite3.Error):
        raise PostgameSelectionFailure("POSTGAME_" + name.upper() + "_FAILED") from None
    stages[name] = "VERIFIED"


def _rows(store, sql, session):
    with closing(sqlite3.connect(store.path.as_uri() + "?mode=ro", uri=True, timeout=0.5)) as db:
        return db.execute(sql, (session,)).fetchall()


def _verify_save(store, expected, owner, clip):
    rows = _rows(
        store,
        "SELECT receipt_json FROM field_clip_save_requests WHERE session_id=?",
        expected.session_id,
    )
    _require(
        len(rows) == 1 and clip_save_receipt_from_json(rows[0][0]) == expected, "POSTGAME_SAVE_ROWS"
    )
    items = [
        row for row in store.list_saved_media_outbox() if row.session_id == expected.session_id
    ]
    _require(len(items) == 1, "POSTGAME_OUTBOX_MISMATCH")
    item = items[0]
    _require(
        item.shot_id == expected.shot_id
        and item.clip_id == expected.clip_id
        and item.save_request_id == expected.request_id
        and item.postgame_manifest_sha256 == expected.postgame_manifest_sha256
        and item.ownership_receipt == owner
        and item.ownership_receipt_sha256 == expected.ownership_receipt_sha256
        and item.status == "pending"
        and item.attempts == 0
        and item.local_path == clip.local_path
        and item.content_sha256 == clip.content_sha256
        and item.byte_size == clip.byte_size,
        "POSTGAME_OUTBOX_MISMATCH",
    )


def _media_url(clip):
    _require(
        isinstance(clip.clip_id, str) and re.fullmatch(r"clip-[a-f0-9]{64}", clip.clip_id),
        "POSTGAME_CLIP_MISMATCH",
    )
    return f"/media/postgame/{clip.clip_id}.mp4?manifest={clip.postgame_manifest_sha256}"


def run_postgame_selection(*, session_id, shot_id, store, read, post, wait, report) -> None:
    """Callbacks share the actor's postgame deadline. Only read probes participate in waits."""
    report["playable_clip_ids"] = []
    stages = report["selection"] = dict.fromkeys(
        ("save", "idempotency", "completion", "cleanup"), "NOT_ATTEMPTED"
    )

    def review():
        data = read(REVIEW)
        if data is None:
            return None
        _require(
            data["review_session_id"] == session_id and data["postgame_manifest_sha256"] == digest,
            "POSTGAME_SCOPE_MISMATCH",
        )
        _require(
            data["phase"] in {"POSTGAME_REVIEW", "POSTGAME_PROCESSING", "FAILED_RETRYING"}
            and data["analysis_status"] in {"PROCESSING", "PARTIAL", "AVAILABLE", "FAILED_RETRYING"}
            and type(data["expected_count"]) is int
            and data["expected_count"] == len(manifest.expected_jobs)
            and type(data["available_count"]) is int
            and 0 <= data["available_count"] <= len(manifest.expected_jobs)
            and isinstance(data["kiosk_nonce"], str)
            and data["kiosk_nonce"],
            "POSTGAME_REVIEW_MISMATCH",
        )
        report.update(
            review_session_id=session_id,
            manifest_sha256=digest,
            **{
                k: data[k]
                for k in ("phase", "analysis_status", "expected_count", "available_count")
            },
        )
        return data

    def target(data, status):
        if (
            not data
            or data["phase"] != "POSTGAME_REVIEW"
            or data["analysis_status"] not in {"AVAILABLE", "PARTIAL"}
            or data["available_count"] == 0
        ):
            return None
        matches = [c for c in data["clips"] if c["shot_id"] == shot_id]
        if not matches or not matches[0]["clip_id"]:
            return None
        _require(len(matches) == 1, "POSTGAME_CLIP_MISMATCH")
        rows = [c for c in store.list_postgame_clips(session_id) if c.shot_id == shot_id]
        _require(len(rows) == 1, "POSTGAME_CLIP_MISMATCH")
        clip, item = rows[0], matches[0]
        _require(
            clip.job_id in manifest.expected_jobs
            and clip.session_id == session_id
            and clip.postgame_manifest_sha256 == digest
            and clip.clip_id == item["clip_id"]
            and clip.save_status == item["save_status"] == status
            and clip.can_save
            and item["can_save"] is True
            and item["actor_role"] == clip.actor_role
            and item["media_url"] == _media_url(clip),
            "POSTGAME_CLIP_MISMATCH",
        )
        return clip

    with _stage(stages, "save"):
        manifest = store.get_postgame_manifest(session_id)
        _require(
            manifest is not None and manifest.session_id == session_id, "POSTGAME_SCOPE_MISMATCH"
        )
        digest = manifest.manifest_sha256

        def playable():
            data = review()
            clip = target(data, ClipSaveStatus.NOT_SAVED)
            return clip if clip and read(_media_url(clip), media=True) is True else None

        selected = wait(playable)
        report["playable_clip_ids"] = [selected.clip_id]
        owner = store.get_shot_ownership(session_id, shot_id)
        _require(
            owner
            and owner.session_id == session_id
            and owner.shot_id == shot_id
            and owner.evidence_status == "bound_at_start"
            and owner.source_kind == "usb_live"
            and owner.actor_role == selected.actor_role
            and owner.actor_role in {"red", "blue"}
            and owner.actor_user_id == selected.owner_user_id == f"offline-{owner.actor_role}",
            "POSTGAME_OWNER_MISMATCH",
        )
        scope = dict(session_id=session_id, postgame_manifest_sha256=digest)
        payload = dict(**scope, shot_id=shot_id, request_id=str(uuid.uuid4()))
        expected = ClipSaveReceipt(
            **payload,
            status=ClipSaveStatus.SAVED,
            clip_id=selected.clip_id,
            schema_version="2.0",
            owner_user_id=owner.actor_user_id,
            actor_role=owner.actor_role,
            ownership_receipt_sha256=ownership_receipt_sha256(owner),
        )

        def save_once(data):
            raw = post(REVIEW + "/save", payload, nonce=data["kiosk_nonce"])
            _require(
                clip_save_receipt_from_json(json.dumps(raw)) == expected,
                "POSTGAME_RECEIPT_MISMATCH",
            )
            _verify_save(store, expected, owner, selected)

        fresh = wait(review)
        _require(target(fresh, ClipSaveStatus.NOT_SAVED) == selected, "POSTGAME_CLIP_MISMATCH")
        save_nonce = fresh["kiosk_nonce"]
        save_once(fresh)
        stages.update(
            clip_id=selected.clip_id,
            shot_id=shot_id,
            owner=owner.actor_user_id,
            actor_role=owner.actor_role,
            ownership_receipt_sha256=expected.ownership_receipt_sha256,
            receipt_schema=expected.schema_version,
            request_id=expected.request_id,
            content_sha256=selected.content_sha256,
            outbox_count=1,
        )

    with _stage(stages, "idempotency"):
        fresh = wait(review)
        saved = target(fresh, ClipSaveStatus.SAVED)
        _require(
            saved and saved.clip_id == selected.clip_id and fresh["kiosk_nonce"] == save_nonce,
            "POSTGAME_CLIP_MISMATCH",
        )
        save_once(fresh)

    with _stage(stages, "completion"):

        def completable():
            data = review()
            return data if data and data["can_complete"] is True else None

        data = wait(completable)
        clips = store.list_postgame_clips(session_id)
        _require(
            data["phase"] == "POSTGAME_REVIEW"
            and data["analysis_status"] == "AVAILABLE"
            and len(clips)
            == len(manifest.expected_jobs)
            == len(data["clips"])
            == data["available_count"]
            and {c.job_id for c in clips} == set(manifest.expected_jobs),
            "POSTGAME_INCOMPLETE_INVENTORY",
        )
        inventory = {c["shot_id"]: c for c in data["clips"]}
        for clip in clips:
            item = inventory[clip.shot_id]
            _require(
                clip.session_id == session_id
                and clip.postgame_manifest_sha256 == digest
                and item["media_url"] == _media_url(clip)
                and item["clip_id"] == clip.clip_id
                and clip.local_path is not None
                and clip.local_path.is_file()
                and clip.save_status
                == item["save_status"]
                == ("SAVED" if clip.shot_id == shot_id else "NOT_SAVED"),
                "POSTGAME_INCOMPLETE_INVENTORY",
            )
        siblings = [c for c in clips if c.shot_id != shot_id]
        _require(
            saved in clips and len(siblings) == len(clips) - 1, "POSTGAME_INCOMPLETE_INVENTORY"
        )
        stages.update(unsaved_count=len(siblings), unsaved_clip_ids=[c.clip_id for c in siblings])
        result = post(REVIEW + "/complete", scope, nonce=data["kiosk_nonce"])
        _require(
            result["phase"] == "REVIEW_COMPLETE"
            and result["review_session_id"] is None
            and _rows(
                store, "SELECT status FROM field_postgame_reviews WHERE session_id=?", session_id
            )
            == [("completed",)],
            "POSTGAME_COMPLETION_MISMATCH",
        )

    with _stage(stages, "cleanup"):

        def disposed():
            rows = _rows(
                store,
                "SELECT shot_id,clip_id,postgame_manifest_sha256,status "
                "FROM field_local_media_discards WHERE session_id=?",
                session_id,
            )
            _require(
                {r[:3] for r in rows} == {(c.shot_id, c.clip_id, digest) for c in siblings},
                "POSTGAME_DISCARD_MISMATCH",
            )
            if any(r[3] != "completed" for r in rows):
                _require(
                    all(r[3] in {"pending", "leased", "completed"} for r in rows),
                    "POSTGAME_DISCARD_FAILED",
                )
                return False
            for clip in siblings:
                _require(
                    store.is_postgame_discarded(session_id, clip.shot_id)
                    and not clip.local_path.exists()
                    and not clip.local_path.is_symlink(),
                    "POSTGAME_DISCARD_MISMATCH",
                )
                if read(_media_url(clip), media="denied") is not True:
                    return False
            return True

        wait(disposed)

        def retained():
            _require(
                not store.is_postgame_discarded(session_id, shot_id), "POSTGAME_SELECTED_DISCARDED"
            )
            _verify_save(store, expected, owner, selected)
            # The product verifier checks the pinned file's full size and digest.
            with _open_review_media(store.artifact_path_for_session(session_id), selected):
                return True

        wait(retained)
    if not siblings:
        stages["cleanup"] = "NOT_EXERCISED"
