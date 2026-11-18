"""Project-owned optional boundary around pooltool physics simulation."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from math import atan2, degrees, hypot
from typing import Any, Protocol

import numpy as np

from billiards_ai.shared.constants import TABLE_LENGTH_MM, TABLE_WIDTH_MM
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import (
    PhysicsResult,
    PhysicsStatus,
    PlannedShot,
    PowerBand,
)
from billiards_ai.shared.models.table_state import BallKind, CanonicalTableState

AUTO_IMPORT = object()
POWER_SPEEDS = {PowerBand.LIGHT: 0.75, PowerBand.MEDIUM: 1.5, PowerBand.FIRM: 2.25}


class PhysicsBackend(Protocol):
    version: str

    def simulate(
        self,
        state: CanonicalTableState,
        candidate: PlannedShot,
        power: PowerBand,
    ) -> PhysicsResult: ...


def _import_pooltool() -> object | None:
    try:
        return importlib.import_module("pooltool")
    except ImportError:
        return None


def _event_name(event: object) -> str:
    value = getattr(event, "event_type", type(event).__name__)
    return str(getattr(value, "value", value))


def _history_points(ball: object, *, limit: int = 600) -> tuple[Point, ...]:
    history = getattr(ball, "history_cts", None)
    if history is None:
        history = getattr(ball, "history", None)
    if history is None:
        return ()
    rvw = getattr(history, "rvw", None)
    if rvw is None:
        try:
            states = tuple(history)
        except TypeError:
            return ()
        rvw = tuple(getattr(state, "rvw", None) for state in states)
        if not rvw or any(value is None for value in rvw):
            return ()
    array = np.asarray(rvw, dtype=float)
    if array.ndim < 3 or array.shape[1] < 1 or array.shape[2] < 2:
        return ()
    if len(array) > limit:
        indexes = np.linspace(0, len(array) - 1, limit, dtype=int)
        array = array[indexes]
    points = []
    for state in array:
        x_m, y_m = state[0, 0], state[0, 1]
        if np.isfinite((x_m, y_m)).all():
            points.append(Point(x_mm=round(float(x_m) * 1000), y_mm=round(float(y_m) * 1000)))
    return tuple(points)


def _cue_control_score(path: Sequence[Point]) -> float:
    if not path:
        return 0.0
    endpoint = path[-1]
    distance = hypot(
        endpoint.x_mm - TABLE_LENGTH_MM / 2,
        endpoint.y_mm - TABLE_WIDTH_MM / 2,
    )
    diagonal = hypot(TABLE_LENGTH_MM, TABLE_WIDTH_MM)
    return max(0.0, min(1.0, 1.0 - distance / diagonal))


class PooltoolPhysicsBackend:
    version = "pooltool/0.6.0"

    def __init__(self, module: object | None = AUTO_IMPORT) -> None:
        self._pt = _import_pooltool() if module is AUTO_IMPORT else module

    def _unavailable(self, candidate: PlannedShot) -> PhysicsResult:
        return PhysicsResult(
            candidate_id=candidate.candidate_id,
            status=PhysicsStatus.UNAVAILABLE,
            backend_version=self.version,
            reason="pooltool_not_installed",
        )

    def simulate(
        self,
        state: CanonicalTableState,
        candidate: PlannedShot,
        power: PowerBand,
    ) -> PhysicsResult:
        if self._pt is None:
            return self._unavailable(candidate)
        state.require_plannable()
        pt = self._pt
        try:
            balls: dict[str, Any] = {}
            for ball in state.balls:
                simulation_id = "cue" if ball.kind is BallKind.CUE else ball.track_id
                balls[simulation_id] = pt.Ball.create(
                    simulation_id,
                    xy=(ball.center.x_mm / 1000, ball.center.y_mm / 1000),
                )
            cue = pt.Cue(cue_ball_id="cue")
            start, end = candidate.cue_path[0], candidate.cue_path[-1]
            phi = degrees(atan2(end.y_mm - start.y_mm, end.x_mm - start.x_mm))
            cue.set_state(V0=POWER_SPEEDS[power], phi=phi)
            system = pt.System(cue=cue, table=pt.Table.default(), balls=balls)
            simulated = pt.simulate(system, continuous=True, dt=1 / 60)
            if simulated is not None:
                system = simulated
            cue_path = _history_points(system.balls["cue"])
            object_paths = {
                ball_id: path
                for ball_id, ball in sorted(system.balls.items())
                if ball_id != "cue" and (path := _history_points(ball))
            }
            events = tuple(_event_name(event) for event in list(system.events)[:256])
            return PhysicsResult(
                candidate_id=candidate.candidate_id,
                status=PhysicsStatus.SIMULATED,
                backend_version=self.version,
                cue_path=cue_path,
                object_paths=object_paths,
                events=events,
                cue_ball_control_score=_cue_control_score(cue_path),
            )
        except Exception as exc:
            return PhysicsResult(
                candidate_id=candidate.candidate_id,
                status=PhysicsStatus.FAILED,
                backend_version=self.version,
                reason=f"simulation_failed:{type(exc).__name__}",
            )
