"""Headless real-engine smoke and geometry mapping for Story 5.2."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from types import SimpleNamespace

import pytest

from billiards_ai.shared.models.ideal_path import (
    IdealPathEvent,
    IdealPathEventKind,
    IdealPathPocket,
    IdealPathPoint,
    IdealPathRequest,
)


def _adapter_module():
    assert importlib.util.find_spec("billiards_ai.cloud.engine.pooltool_adapter") is not None, (
        "Story 5.2 pooltool adapter is missing"
    )
    return importlib.import_module("billiards_ai.cloud.engine.pooltool_adapter")


def test_s1_preparation_rejects_an_already_compiled_engine():
    module = _adapter_module()
    from billiards_ai.cloud.engine.ideal_path import EngineUnavailableError

    with pytest.raises(EngineUnavailableError, match="Python kernels"):
        module.PooltoolIdealPathAdapter.prepare_numeric_runtime()


@pytest.fixture
def ideal_request() -> IdealPathRequest:
    return IdealPathRequest(
        cue_ball={"ball_id": 0, "position": {"x_mm": 500, "y_mm": 400}},
        object_balls=({"ball_id": 3, "position": {"x_mm": 1800, "y_mm": 900}},),
        target_ball_id=3,
        target_pocket="TOP_RIGHT",
        intent="DIRECT_POT",
        target_ball_reviewed=True,
        target_pocket_reviewed=True,
        intent_reviewed=True,
        calibration_status="ACTIVE",
        table={
            "configuration_version": "chinese-eight-ball-9ft-v1",
            "length_mm": 2540,
            "width_mm": 1270,
            "ball_diameter_mm": 57,
            "corner_pocket_mouth_mm": 86,
            "side_pocket_mouth_mm": 92,
            "cushion_nose_width_mm": 40,
            "cushion_nose_height_mm": 36,
        },
    )


def test_real_adapter_maps_mm_geometry_without_gui_or_network(ideal_request):
    module = _adapter_module()
    adapter = module.PooltoolIdealPathAdapter()

    system = adapter._build_system(ideal_request)
    specs = adapter._table_specs(ideal_request)
    mapping = adapter._coordinate_mapping(system.table, ideal_request)

    assert importlib.metadata.version("pooltool-billiards") == "0.6.0"
    assert system.table.l == pytest.approx(2.540)
    assert system.table.w == pytest.approx(1.270)
    assert system.balls["0"].params.R == pytest.approx(0.0285)
    assert specs.cushion_width == pytest.approx(0.040)
    assert specs.corner_pocket_width == pytest.approx(0.086)
    assert specs.side_pocket_width == pytest.approx(0.092)
    assert system.table.cushion_segments.linear["3"].p1[2] == pytest.approx(0.036)
    assert adapter._physics_engine().resolver.version == 9

    pocket_anchors = {
        IdealPathPocket.BOTTOM_LEFT: ("lb", IdealPathPoint(x_mm=0, y_mm=0)),
        IdealPathPocket.BOTTOM_CENTER: ("lc", IdealPathPoint(x_mm=1270, y_mm=0)),
        IdealPathPocket.BOTTOM_RIGHT: ("lt", IdealPathPoint(x_mm=2540, y_mm=0)),
        IdealPathPocket.TOP_LEFT: ("rb", IdealPathPoint(x_mm=0, y_mm=1270)),
        IdealPathPocket.TOP_CENTER: ("rc", IdealPathPoint(x_mm=1270, y_mm=1270)),
        IdealPathPocket.TOP_RIGHT: ("rt", IdealPathPoint(x_mm=2540, y_mm=1270)),
    }
    for _pocket, (pocket_id, business_point) in pocket_anchors.items():
        actual_center = system.table.pockets[pocket_id].center
        assert tuple(actual_center[:2]) != pytest.approx(
            (business_point.y_mm / 1000, business_point.x_mm / 1000)
        )
        assert mapping.to_pooltool_xy(business_point) == pytest.approx(tuple(actual_center[:2]))
        assert mapping.to_business_point(actual_center) == business_point

    for point in (
        ideal_request.cue_ball.position,
        ideal_request.object_balls[0].position,
        IdealPathPoint(x_mm=1270, y_mm=635),
    ):
        assert mapping.to_business_point(mapping.to_pooltool_xy(point)) == point

    cue_xy = mapping.to_pooltool_xy(ideal_request.cue_ball.position)
    assert tuple(system.balls["0"].state.rvw[0][:2]) == pytest.approx(cue_xy)


def test_coordinate_mapping_rejects_material_out_of_bounds_values(ideal_request):
    module = _adapter_module()
    adapter = module.PooltoolIdealPathAdapter()
    system = adapter._build_system(ideal_request)
    mapping = adapter._coordinate_mapping(system.table, ideal_request)
    lb = system.table.pockets["lb"].center
    lt = system.table.pockets["lt"].center
    rb = system.table.pockets["rb"].center
    middle_short_axis = (float(lb[0]) + float(rb[0])) / 2
    long_axis_per_mm = (float(lt[1]) - float(lb[1])) / ideal_request.table.length_mm

    tiny_roundoff = (middle_short_axis, float(lb[1]) - 0.25 * long_axis_per_mm, 0.0)
    materially_outside = (middle_short_axis, float(lb[1]) - 2 * long_axis_per_mm, 0.0)

    assert mapping.to_business_point(tiny_roundoff).x_mm == 0
    with pytest.raises(ValueError, match="outside"):
        mapping.to_business_point(materially_outside)


def test_adapter_rejects_target_ball_entering_wrong_pocket(ideal_request, monkeypatch):
    module = _adapter_module()
    adapter = module.PooltoolIdealPathAdapter()
    system = SimpleNamespace(
        balls={"0": object(), "3": object()},
        table=SimpleNamespace(pockets={"rt": object()}),
        strike=lambda **_kwargs: None,
    )
    wrong_pocket_events = (
        IdealPathEvent(
            sequence=1,
            kind="CUE_STRIKE",
            ball_id=0,
            position=ideal_request.cue_ball.position,
        ),
        IdealPathEvent(
            sequence=2,
            kind="BALL_CONTACT",
            ball_id=0,
            other_ball_id=3,
            position=ideal_request.object_balls[0].position,
        ),
        IdealPathEvent(
            sequence=3,
            kind="POCKETED",
            ball_id=3,
            pocket="BOTTOM_LEFT",
            position={"x_mm": 0, "y_mm": 0},
        ),
    )
    fake_pt = SimpleNamespace(
        pot=SimpleNamespace(
            PottingConfig=SimpleNamespace(
                default=lambda: SimpleNamespace(calculate_angle=lambda *_args: 0.0)
            )
        ),
        simulate=lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(module, "pt", fake_pt)
    monkeypatch.setattr(adapter, "_require_engine", lambda: None)
    monkeypatch.setattr(adapter, "_build_system", lambda _request: system)
    monkeypatch.setattr(adapter, "_physics_engine", lambda: object())
    monkeypatch.setattr(adapter, "_business_events", lambda *_args: wrong_pocket_events)

    with pytest.raises(RuntimeError, match="target pocket"):
        adapter.calculate(ideal_request)


def test_headless_pt_simulate_fixture_is_repeatable_and_pots_explicit_target(ideal_request):
    module = _adapter_module()
    adapter = module.PooltoolIdealPathAdapter()

    first = adapter.calculate(ideal_request)
    repeated = adapter.calculate(ideal_request)

    assert first == repeated
    assert first
    assert first[0].kind is IdealPathEventKind.CUE_STRIKE
    assert any(event.kind is IdealPathEventKind.BALL_CONTACT for event in first)
    assert any(
        event.kind is IdealPathEventKind.POCKETED
        and event.ball_id == 3
        and event.pocket is IdealPathPocket.TOP_RIGHT
        for event in first
    )
    assert first[0].position == ideal_request.cue_ball.position
