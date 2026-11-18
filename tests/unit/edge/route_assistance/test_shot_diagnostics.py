"""Only exact persisted capture-loss diagnostics may exempt a real job."""

import importlib
import importlib.util
from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import FieldShotRecord, ShotMediaRange


@pytest.fixture
def shot():
    return FieldShotRecord(
        session_id="session-1",
        shot_id="shot-1",
        sequence=1,
        actor_role=None,
        started_ms=80,
        settled_ms=240,
        recommendation_id=None,
        power_band=None,
        power_percent=None,
        media=ShotMediaRange("artifact", "usb:epoch-1", "usb_live", 0, 320, "epoch-1"),
        live_evidence_digest="e" * 64,
        live_job_id="abstained:capture_interrupted:epoch-1:shot-1",
    )


def classify(shot):
    name = "billiards_ai.edge.route_assistance.shot_diagnostics"
    assert importlib.util.find_spec(name), "shared diagnostic classifier is missing"
    return importlib.import_module(name).is_capture_interruption(shot)


@pytest.mark.parametrize(
    "marker",
    [
        "abstained:capture_interrupted:epoch-1:shot-1",
        "capture-interrupted:epoch-1:shot-1",
        "abstained:incomplete_freeze:epoch-1:shot-1",
    ],
)
def test_exact_current_and_historical_capture_loss(shot, marker):
    assert classify(replace(shot, live_job_id=marker))


@pytest.mark.parametrize(
    "marker",
    [
        "job-1",
        "abstained:unknown:epoch-1:shot-1",
        "abstained:incomplete_enqueue:epoch-1:shot-1",
        "abstained:incomplete_freeze:wrong:shot-1",
        "abstained:incomplete_freeze:epoch-1:wrong",
        "abstained:incomplete_freeze:epoch-1:shot-1:extra",
        "abstained:capture_interrupted:wrong:shot-1",
        "abstained:capture_interrupted:epoch-1:wrong",
        "capture-interrupted:wrong:shot-1",
        "capture-interrupted:epoch-1:wrong",
        "capture-interrupted:epoch-1:shot-1:extra",
        "abstained:capture_interrupted:epoch-1:shot-1:extra",
    ],
)
def test_lookalike_is_still_real_expected_work(shot, marker):
    assert not classify(replace(shot, live_job_id=marker))


@pytest.mark.parametrize("state", ["unsettled", "no_evidence", "replay"])
def test_requires_settled_usb_evidence(shot, state):
    changes = dict(live_evidence_digest=None, live_job_id=None)
    if state == "unsettled":
        changes["settled_ms"] = None
    elif state == "replay":
        changes["media"] = ShotMediaRange(
            "artifact", "presentation.mp4", "development_replay", 0, 320
        )
    assert not classify(replace(shot, **changes))


@pytest.mark.parametrize(
    "marker,expected",
    [
        ("abstained:capture_interrupted:epoch-1:shot-1", ()),
        ("capture-interrupted:epoch-1:shot-1", ()),
        ("abstained:incomplete_freeze:epoch-1:shot-1", ()),
        ("abstained:unknown:epoch-1:shot-1", ("abstained:unknown:epoch-1:shot-1",)),
        ("capture-interrupted:epoch-1:wrong", ("capture-interrupted:epoch-1:wrong",)),
        (
            "abstained:incomplete_enqueue:epoch-1:shot-1",
            ("abstained:incomplete_enqueue:epoch-1:shot-1",),
        ),
    ],
)
def test_freeze_exempts_only_exact_diagnostic_and_preserves_head(tmp_path, shot, marker, expected):
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite", session_id_factory=lambda: "session-1"
    )
    store.open_or_resume_session(artifact_id="artifact")
    store.record_shot_start(
        replace(shot, settled_ms=None, live_job_id=None, live_evidence_digest=None),
        source_start_event_id="start",
    )
    store.record_shot_settle(replace(shot, live_job_id=marker), source_epoch_id="epoch-1")
    store.end_session("session-1")
    manifest = store.get_postgame_manifest("session-1")
    assert manifest.expected_jobs == expected
    store.end_session("session-1")
    assert store.get_postgame_manifest("session-1") == manifest
