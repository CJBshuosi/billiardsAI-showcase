from __future__ import annotations

from pathlib import Path

from tests.support.miniprogram_sync import run_fixture_sync


def test_ended_fixture_sync_maps_shot_and_clip_lineage(tmp_path: Path) -> None:
    result = run_fixture_sync(tmp_path, fixture="session-facts-v1.json")
    assert result.session_id == "session-0001"
    assert result.duplicate_session_count == 0
    assert {(clip.session_id, clip.shot_id) for clip in result.clips} == {
        ("session-0001", "shot-0001")
    }
    assert result.sync_status == "complete"
