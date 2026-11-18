"""Processing position must bind replay origin and immutable admitted media."""

import importlib
import importlib.util

import pytest

from billiards_ai.lab.mainline1_session_audit import ShotBoundaryLedger


def _journal(tmp_path, *, config="c" * 64, initial=0):
    name = "billiards_ai.lab.mainline1_recording_processing"
    assert importlib.util.find_spec(name) is not None, "persistent processing journal is missing"
    cls = importlib.import_module(name).RecordingProcessingJournal
    path = tmp_path / "boundaries.sqlite3"
    ShotBoundaryLedger(path, session_id="session", source_epoch_id="epoch")
    return cls(
        path=path,
        session_id="session",
        source_epoch_id="epoch",
        config_sha256=config,
        initial_shot_number=initial,
    )


def _segment(index):
    return dict(
        segment_index=index,
        last_frame_id=59 + index * 60,
        last_captured_at_ns=983_333_333 + index * 1_000_000_000,
        prefix_sha256=("a" if index == 0 else "b") * 64,
    )


def test_restart_keeps_original_shot_number_and_distinguishes_admitted_from_processed(tmp_path):
    journal = _journal(tmp_path)
    journal.admit_segment(_segment(0))
    assert journal.state["checkpoint"] is None
    journal.checkpoint(_segment(0))
    journal.admit_segment(_segment(1))
    restarted = _journal(tmp_path, initial=9)
    state = restarted.state
    assert state["initial_shot_number"] == 0
    assert state["input_prefix"] == _segment(1)
    assert state["checkpoint"] == _segment(0)
    restarted.checkpoint(_segment(1))
    assert restarted.state["checkpoint"]["last_frame_id"] == 119
    with pytest.raises(ValueError, match="backward"):
        restarted.checkpoint(_segment(0))


def test_checkpoint_cannot_get_ahead_of_events_or_change_config_and_input_identity(tmp_path):
    journal = _journal(tmp_path)
    with pytest.raises(ValueError, match="admitted"):
        journal.checkpoint(_segment(0))
    journal.admit_segment(_segment(0))
    with pytest.raises(ValueError, match="processed"):
        journal.admit_segment(_segment(1))
    with pytest.raises(ValueError, match="conflict"):
        journal.admit_segment(_segment(0) | {"prefix_sha256": "d" * 64})
    before = journal.state
    with pytest.raises(ValueError, match="config"):
        _journal(tmp_path, config="e" * 64)
    assert journal.state == before
