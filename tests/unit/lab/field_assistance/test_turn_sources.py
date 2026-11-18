"""Synthetic video runs through the existing Mainline 1 producer, not MP4 playback inference."""

import hashlib
import json
from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.recording import ShotTimelineIndex
from billiards_ai.lab.field_assistance import server as field_server
from billiards_ai.lab.field_assistance.live_state import load_replay_timeline
from billiards_ai.lab.field_assistance.mainline1_replay import load_mainline1_replay_inputs
from billiards_ai.lab.field_assistance.replay import (
    run_field_route_replay,
    verify_field_route_replay_artifact,
)
from billiards_ai.lab.mainline1_shot_local_video import run_mainline1_shot_local_video
from billiards_ai.lab.planning.rules import EightBallContext
from tests.unit.lab.field_assistance.test_replay import _config as replay_config
from tests.unit.lab.test_mainline1_shot_local_video import (
    RecordingBackend,
    RecordingClassifier,
    _calibration,
    _config,
    _synthetic_video,
)


@pytest.fixture
def source_pair(tmp_path):
    video = _synthetic_video(tmp_path / "synthetic.avi")
    source = run_mainline1_shot_local_video(
        video=video,
        calibration=_calibration(),
        vision_backend=RecordingBackend(),
        identity_classifier=RecordingClassifier(),
        config=_config(),
        output_root=tmp_path / "mainline1",
    )
    inputs = load_mainline1_replay_inputs(artifact_path=source.path, video_path=video)
    replay = run_field_route_replay(
        video=video,
        observations_path=inputs.observations_path,
        calibration=inputs.calibration,
        context=EightBallContext(group="solid"),
        output_root=tmp_path / "replay",
        shot_windows_path=inputs.shot_windows_path,
        events_path=inputs.events_path,
        source_mainline1_artifact_id=inputs.artifact_id,
        config=replay_config(),
    )
    timeline = load_replay_timeline(replay.path / "timeline.json", duration_ms=500)
    return inputs, verify_field_route_replay_artifact(replay.path), timeline


def test_verify_source_start_mapping_without_promoting_observed_to_reviewed(source_pair):
    inputs, manifest, timeline = source_pair
    index = ShotTimelineIndex.from_timeline(timeline)
    assert len(index.boundaries) == 2
    events, reason = field_server._verified_start_events(inputs, manifest, timeline, index)
    assert dict(events) == {1: "shot-0001-event-shot-start"}
    assert reason == "SOURCE_START_VERIFIED"


def test_missing_source_never_claims_authoritative_start(source_pair):
    _, manifest, timeline = source_pair
    events, reason = field_server._verified_start_events(
        None, manifest, timeline, ShotTimelineIndex.from_timeline(timeline)
    )
    assert dict(events) == {} and reason == "SOURCE_INPUTS_MISSING"


@pytest.mark.parametrize(
    "field",
    ["source_mainline1_artifact_id", "video_sha256", "events_sha256", "shot_windows_sha256"],
)
def test_foreign_source_link_rejected_before_server_starts(source_pair, field):
    inputs, manifest, timeline = source_pair
    events, reason = field_server._verified_start_events(
        inputs, {**manifest, field: "f" * 64}, timeline, ShotTimelineIndex.from_timeline(timeline)
    )
    assert dict(events) == {} and reason == "SOURCE_LINK_MISMATCH"


def test_window_fallback_boundary_cannot_be_certified_as_observed_start(source_pair):
    inputs, manifest, timeline = source_pair
    index = ShotTimelineIndex.from_timeline(timeline)
    wrong_start = replace(index.boundaries[0], timestamp_ms=0)
    events, reason = field_server._verified_start_events(
        inputs, manifest, timeline, replace(index, boundaries=(wrong_start, index.boundaries[1]))
    )
    assert dict(events) == {} and reason == "SOURCE_START_UNVERIFIED"


@pytest.mark.parametrize("change", ["abstained", "missing", "wrong_shot", "duplicate"])
def test_non_observed_or_cross_shot_start_stays_unknown(source_pair, change):
    inputs, manifest, timeline = source_pair
    rows = json.loads(inputs.events_path.read_text())
    event = next(row for row in rows["events"] if row["event_type"] == "shot_start")
    if change == "abstained":
        event["status"] = "abstained"
    elif change == "wrong_shot":
        event["shot_id"] = "shot-9999"
    elif change == "duplicate":
        rows["events"].append(dict(event))
    else:
        rows["events"].remove(event)
    inputs.events_path.write_text(json.dumps(rows))
    # Synthetic, inventory-consistent negative fixtures. Not recorded-video evidence.
    source_manifest_path = inputs.artifact_path / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    digest = hashlib.sha256(inputs.events_path.read_bytes()).hexdigest()
    source_manifest["inventory"]["world-events.json"] = {
        "bytes": inputs.events_path.stat().st_size,
        "sha256": digest,
    }
    source_manifest_path.write_text(json.dumps(source_manifest))
    events, _ = field_server._verified_start_events(
        inputs,
        {**manifest, "events_sha256": digest},
        timeline,
        ShotTimelineIndex.from_timeline(timeline),
    )
    assert dict(events) == {}


@pytest.mark.parametrize("change", ["missing_file", "tampered", "foreign_video"])
def test_unusable_optional_source_closes_attribution_not_base_replay(source_pair, change):
    inputs, manifest, timeline = source_pair
    if change == "missing_file":
        inputs = replace(inputs, artifact_path=inputs.artifact_path / "missing")
    elif change == "tampered":
        inputs.events_path.write_text("{}")
    else:
        foreign = inputs.video_path.with_name("foreign.avi")
        foreign.write_bytes(b"foreign")
        inputs = replace(inputs, video_path=foreign)
    events, reason = field_server._verified_start_events(
        inputs, manifest, timeline, ShotTimelineIndex.from_timeline(timeline)
    )
    assert dict(events) == {} and reason == "SOURCE_CONTENT_INVALID"


def test_repeated_source_load_is_deterministic_immutable_snapshot(source_pair):
    inputs, manifest, timeline = source_pair
    index = ShotTimelineIndex.from_timeline(timeline)
    first = field_server._verified_start_events(inputs, manifest, timeline, index)
    second = field_server._verified_start_events(inputs, manifest, timeline, index)
    assert first == second
    inputs.events_path.write_text("{}")
    assert dict(first[0]) == {1: "shot-0001-event-shot-start"}


@pytest.mark.parametrize("source_mode", ["explicit", "missing", "invalid"])
def test_server_builder_uses_optional_snapshot_without_breaking_base_replay(
    source_pair, tmp_path, source_mode
):
    from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope
    from tests.unit.edge.route_assistance.test_turn_ownership import match
    from tests.unit.lab.field_assistance.test_live_state import FakeClock

    inputs, manifest, _ = source_pair
    original = inputs
    if source_mode == "missing":
        inputs = None
    elif source_mode == "invalid":
        inputs = replace(inputs, artifact_path=tmp_path / "absent")
    now = FakeClock()
    server = field_server.build_field_assistance_server(
        artifact_path=tmp_path / "replay" / manifest["artifact_id"],
        host="127.0.0.1",
        port=0,
        control_token="local-test",
        monotonic=now,
        duration_probe=lambda _: 500,
        mainline1_inputs=inputs,
    )
    try:
        state = server.field_state
        state.record_store.save_match(state.session_id, match(), expected_revision=0)
        state.match = match()
        state.replay_clock.start()
        now.advance(0.401)
        state.before_field_command(
            FieldCommandEnvelope("probe", state.session_id, "user-red", "continue_turn", {}, 1, 1)
        )
        receipt = state.record_store.get_shot_ownership(state.session_id, "shot-0001")
        assert receipt.source_mainline1_artifact_id == original.artifact_id
        assert receipt.evidence_status == (
            "bound_at_start" if source_mode == "explicit" else "unknown"
        )
        pending = state.record_store.get_turn_confirmation(state.session_id, "shot-0001")
        assert pending.action == "abstain" and pending.status == "pending"
        assert state.match.revision == 1  # no automatic switch or continue
        assert state.record_store.list_shots(state.session_id)[0].settled_ms == 400
    finally:
        server.server_close()
