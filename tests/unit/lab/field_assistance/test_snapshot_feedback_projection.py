"""Original shot feedback uses the same calibrated camera overlay as advice."""

from types import SimpleNamespace

import pytest

from billiards_ai.lab.field_assistance.live_source import (
    USBLiveProjectionSource,
    table_to_pixel,
)
from tests.unit.lab.field_assistance.test_snapshot_guidance import settled_feedback


@pytest.mark.parametrize("actual", [{"x_mm": 1000, "y_mm": 510}, None])
def test_previous_region_is_projected_without_old_or_hidden_next_route(
    tmp_path, monkeypatch, actual
):
    service, _, _, _ = settled_feedback(tmp_path, monkeypatch, actual=actual)
    role = SimpleNamespace(value="red")
    service._context = "open"
    service._context_key = (1, role)
    source = SimpleNamespace(
        runtime=SimpleNamespace(snapshot_guidance=service),
        _snapshot_projection_failure=None,
        _ended=False,
    )
    snapshot = SimpleNamespace(
        match=SimpleNamespace(
            red_group=None, blue_group=None, active_role=role, revision=1, mode=None
        )
    )
    try:
        value = USBLiveProjectionSource._snapshot_guidance_payload(
            source, snapshot, None, SimpleNamespace(width=320, height=240), service.calibration
        )
        assert value["phase"] == "reviewing"
        assert value["recommendation"] is None
        overlay = value["overlay"]
        assert overlay["width"] == 320 and overlay["height"] == 240
        assert overlay["cue_path"] == overlay["object_path"] == []
        assert len(overlay["region"]) == 32
        assert overlay["region"][0] == pytest.approx(table_to_pixel(service.calibration, 1100, 500))
        assert value["feedback"]["actual_cue"] == actual
        source._ended = True
        ended = USBLiveProjectionSource._snapshot_guidance_payload(
            source, snapshot, None, SimpleNamespace(width=320, height=240), service.calibration
        )
        assert ended["feedback"] is None
        assert ended.get("overlay") is None
    finally:
        service.close()
