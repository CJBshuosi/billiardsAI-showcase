"""Run the page's real projection validator, including unknown final counts."""

import json
import subprocess

import pytest

from billiards_ai.lab.field_assistance.pages import render_screen_page


def _validate(value):
    page = render_screen_page()
    start = page.index("function validate(value){")
    end = page.index("  function mediaUrl", start)
    source = page[start:end]
    code = source + (
        "\ntry {process.stdout.write(JSON.stringify(validate(JSON.parse(process.argv[1]))));}"
        " catch(e) {process.exit(3);}"
    )
    return subprocess.run(["node", "-e", code, json.dumps(value)], capture_output=True, text=True)


def _processing():
    return dict(
        schema_version="recording-review/v2",
        phase="RECORDING_PROCESSING",
        review_session_id="original-scene",
        postgame_manifest_sha256="a" * 64,
        expected_count=None,
        discovered_count=2,
        available_count=1,
        can_complete=False,
        processing=dict(
            status="PROCESSING", processed_frame_id=299, cutoff_frame_id=500, remaining_frames=201
        ),
        clips=[
            dict(shot_id="shot-1", save_status="NOT_SAVED"),
            dict(shot_id="shot-2", save_status="NOT_SAVED"),
        ],
    )


def test_page_accepts_growing_inventory_without_turning_unknown_final_count_into_zero():
    result = _validate(_processing())
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["expected_count"] is None
    assert value["available_count"] == 1 and not value["can_complete"]


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_count": 0},
        {"discovered_count": 0},
        {"available_count": 3},
        {"can_complete": True},
    ],
)
def test_page_rejects_malformed_or_prematurely_complete_growing_inventory(changes):
    assert _validate(_processing() | changes).returncode == 3


def test_interrupted_prefix_has_its_own_status_and_cannot_authorize_completion():
    value = _processing() | dict(
        phase="RECORDING_INTERRUPTED",
        expected_count=2,
        recording_capture=dict(end_reason="PROCESS_INTERRUPTED", tail_completeness="UNKNOWN"),
    )
    assert _validate(value).returncode == 0
    assert _validate(value | dict(can_complete=True)).returncode == 3
    assert (
        _validate(
            value | dict(recording_capture=dict(end_reason="USER_ENDED_RECORDING"))
        ).returncode
        == 3
    )
