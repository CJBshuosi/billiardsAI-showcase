from __future__ import annotations

from types import SimpleNamespace

import billiards_ai.lab.mainline1_shot_local_video as legacy
from billiards_ai.lab.mainline1_shot_product import build_mainline1_shot_product


def test_realtime_provenance_payload_is_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(legacy, "_detections_payload", lambda _: {})
    monkeypatch.setattr(legacy, "_window_payload", lambda _: {})
    monkeypatch.setattr(legacy, "_anchors_payload", lambda _: {})
    monkeypatch.setattr(legacy, "_cue_payload", lambda _: {})
    monkeypatch.setattr(legacy, "_fragment_payload", lambda _: {})
    monkeypatch.setattr(legacy, "_active_tracks_payload", lambda _: {})
    monkeypatch.setattr(legacy, "_canonical_event_timeline_payload", lambda *_: {})
    monkeypatch.setattr(legacy, "_world_events_payload", lambda _: {})
    monkeypatch.setattr(legacy, "_table_state_payload", lambda *_, **__: {})
    monkeypatch.setattr(legacy, "_summary_payload", lambda **_: {})
    common = {
        "decoded_frames": (SimpleNamespace(frame_id=0),),
        "observations": (),
        "results": (),
        "analyses": (),
        "calibration": SimpleNamespace(),
        "config": SimpleNamespace(event=SimpleNamespace(cushion_distance_mm=90)),
        "fps": 60.0,
    }

    file_product = build_mainline1_shot_product(**common)
    live_product = build_mainline1_shot_product(
        **common,
        realtime_perception={"capture_fps": 60.0, "anchor_fps": 8.0},
    )

    assert "realtime-perception.json" not in file_product.payloads
    assert live_product.payloads["realtime-perception.json"] == {
        "capture_fps": 60.0,
        "anchor_fps": 8.0,
    }
