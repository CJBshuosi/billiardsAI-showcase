from types import SimpleNamespace

import numpy as np
import pytest

from billiards_ai.lab.adapters.pooltool_physics import PooltoolPhysicsBackend
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import (
    GuidanceStatus,
    PhysicsStatus,
    PlannedShot,
    PocketId,
    PowerBand,
)
from billiards_ai.shared.models.table_state import (
    BallKind,
    CanonicalBall,
    CanonicalTableState,
    MotionStatus,
)


class _FakeState:
    def __init__(self, xy: tuple[float, float]) -> None:
        self.rvw = np.asarray([[xy[0], xy[1], 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


class _FakeBall:
    def __init__(
        self,
        ball_id: str,
        xy: tuple[float, float],
        *,
        iterable_history: bool = False,
    ) -> None:
        self.id = ball_id
        self.xy = xy
        if iterable_history:
            self.history_cts = (
                _FakeState(xy),
                _FakeState((xy[0] + 0.1, xy[1])),
            )
        else:
            self.history_cts = SimpleNamespace(
                rvw=np.asarray(
                    [
                        [[xy[0], xy[1], 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [
                            [xy[0] + 0.1, xy[1], 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                    ]
                )
            )


class _FakePooltool:
    __version__ = "0.6.0"

    def __init__(self, *, iterable_history: bool = False) -> None:
        self.created_balls: dict[str, _FakeBall] = {}
        self.cue_state: dict[str, float] = {}
        owner = self

        class Ball:
            @staticmethod
            def create(ball_id: str, *, xy: tuple[float, float]) -> _FakeBall:
                ball = _FakeBall(ball_id, xy, iterable_history=iterable_history)
                owner.created_balls[ball_id] = ball
                return ball

        class Table:
            @staticmethod
            def default() -> object:
                return object()

        class Cue:
            def __init__(self, cue_ball_id: str) -> None:
                self.cue_ball_id = cue_ball_id

            def set_state(self, **kwargs: float) -> None:
                owner.cue_state.update(kwargs)

        class System:
            def __init__(self, *, cue: object, table: object, balls: dict[str, _FakeBall]) -> None:
                self.cue = cue
                self.table = table
                self.balls = balls
                self.events = [
                    SimpleNamespace(event_type="stick-ball"),
                    SimpleNamespace(event_type="ball-ball"),
                ]

        self.Ball = Ball
        self.Table = Table
        self.Cue = Cue
        self.System = System

    def simulate(self, system: object, *, continuous: bool, dt: float) -> object:
        assert continuous is True
        assert dt == pytest.approx(1 / 60)
        return system


def _ball(track: str, identity: str, kind: BallKind, x: int, y: int) -> CanonicalBall:
    return CanonicalBall(
        track_id=track,
        identity=identity,
        kind=kind,
        identity_confidence=0.9,
        center=Point(x_mm=x, y_mm=y),
        radius_mm=28.5,
        detection_confidence=0.95,
    )


def _state() -> CanonicalTableState:
    return CanonicalTableState(
        frame_id=1,
        timestamp_ms=0,
        calibration_id="cal-1",
        motion_status=MotionStatus.STATIONARY,
        balls=(
            _ball("cue-track", "cue", BallKind.CUE, 500, 600),
            _ball("target", "1", BallKind.SOLID, 1500, 600),
        ),
        source_refs=("frame:1",),
    )


def _candidate() -> PlannedShot:
    return PlannedShot(
        candidate_id="candidate-1",
        target_track_id="target",
        target_identity="1",
        target_kind=BallKind.SOLID,
        pocket=PocketId.TOP_RIGHT,
        status=GuidanceStatus.PLANNED,
        ghost_ball_center=Point(x_mm=1443, y_mm=600),
        cue_path=(Point(x_mm=500, y_mm=600), Point(x_mm=1443, y_mm=600)),
        object_path=(Point(x_mm=1500, y_mm=600), Point(x_mm=2540, y_mm=1270)),
        cut_angle_deg=0,
        cue_travel_mm=943,
        object_travel_mm=1200,
        evidence_confidence=0.9,
        evidence_refs=("frame:1",),
    )


def test_adapter_converts_mm_to_metres_and_sets_cue_state() -> None:
    fake = _FakePooltool()
    result = PooltoolPhysicsBackend(module=fake).simulate(_state(), _candidate(), PowerBand.MEDIUM)
    assert fake.created_balls["cue"].xy == pytest.approx((0.50, 0.60))
    assert fake.cue_state["V0"] == 1.5
    assert fake.cue_state["phi"] == pytest.approx(0.0)
    assert result.status is PhysicsStatus.SIMULATED
    assert result.cue_path[-1] == Point(x_mm=600, y_mm=600)
    assert "stick-ball" in result.events


def test_adapter_extracts_pooltool_state_sequence_history() -> None:
    fake = _FakePooltool(iterable_history=True)
    result = PooltoolPhysicsBackend(module=fake).simulate(_state(), _candidate(), PowerBand.LIGHT)
    assert result.status is PhysicsStatus.SIMULATED
    assert result.cue_path == (
        Point(x_mm=500, y_mm=600),
        Point(x_mm=600, y_mm=600),
    )


def test_missing_pooltool_returns_typed_unavailable_result() -> None:
    result = PooltoolPhysicsBackend(module=None).simulate(_state(), _candidate(), PowerBand.LIGHT)
    assert result.status is PhysicsStatus.UNAVAILABLE
    assert result.reason == "pooltool_not_installed"


def test_backend_exception_is_a_typed_failed_result() -> None:
    fake = _FakePooltool()
    fake.simulate = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    result = PooltoolPhysicsBackend(module=fake).simulate(_state(), _candidate(), PowerBand.FIRM)
    assert result.status is PhysicsStatus.FAILED
    assert result.reason == "simulation_failed:RuntimeError"
