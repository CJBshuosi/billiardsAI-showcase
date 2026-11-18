"""The only adapter boundary allowed to import pooltool."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

try:
    import pooltool as pt
except Exception:  # pragma: no cover - exercised on unsupported deployments
    pt = None

from billiards_ai.cloud.engine.ideal_path import EngineUnavailableError
from billiards_ai.shared.models.ideal_path import (
    IdealPathEvent,
    IdealPathEventKind,
    IdealPathPocket,
    IdealPathPoint,
    IdealPathRequest,
)

_POCKET_IDS = {
    IdealPathPocket.BOTTOM_LEFT: "lb",
    IdealPathPocket.BOTTOM_CENTER: "lc",
    IdealPathPocket.BOTTOM_RIGHT: "lt",
    IdealPathPocket.TOP_LEFT: "rb",
    IdealPathPocket.TOP_CENTER: "rc",
    IdealPathPocket.TOP_RIGHT: "rt",
}
_POCKETS_BY_ID = {pocket_id: pocket for pocket, pocket_id in _POCKET_IDS.items()}
_BUSINESS_POCKET_FACTORS = {
    "lb": (0.0, 0.0),
    "lc": (0.5, 0.0),
    "lt": (1.0, 0.0),
    "rb": (0.0, 1.0),
    "rc": (0.5, 1.0),
    "rt": (1.0, 1.0),
}
_AFFINE_TRIANGLES = (
    ("lb", "lc", "rb"),
    ("lc", "rc", "rb"),
    ("lc", "lt", "rc"),
    ("lt", "rt", "rc"),
)
_AFFINE_TOLERANCE = 1e-9
_BOUNDARY_TOLERANCE_MM = 0.5


@dataclass(frozen=True, slots=True)
class _PooltoolCoordinateMapping:
    """Six-anchor piecewise-affine mapping between engine and business planes."""

    length_mm: int
    width_mm: int
    pocket_centers: tuple[tuple[str, tuple[float, float]], ...]

    @classmethod
    def from_table(cls, table: object, request: IdealPathRequest) -> _PooltoolCoordinateMapping:
        centers = tuple(
            (
                pocket_id,
                (
                    float(table.pockets[pocket_id].center[0]),  # type: ignore[attr-defined]
                    float(table.pockets[pocket_id].center[1]),  # type: ignore[attr-defined]
                ),
            )
            for pocket_id in _BUSINESS_POCKET_FACTORS
        )
        mapping = cls(
            length_mm=request.table.length_mm,
            width_mm=request.table.width_mm,
            pocket_centers=centers,
        )
        mapping._validate_affine_geometry()
        return mapping

    def _business_anchor_xy(self, pocket_id: str) -> tuple[float, float]:
        x_factor, y_factor = _BUSINESS_POCKET_FACTORS[pocket_id]
        return self.length_mm * x_factor, self.width_mm * y_factor

    def _business_anchor(self, pocket_id: str) -> IdealPathPoint:
        x_mm, y_mm = self._business_anchor_xy(pocket_id)
        return IdealPathPoint(x_mm=round(x_mm), y_mm=round(y_mm))

    @staticmethod
    def _barycentric_weights(
        point: tuple[float, float],
        triangle: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ) -> tuple[float, float, float]:
        (ax, ay), (bx, by), (cx, cy) = triangle
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denominator) <= _AFFINE_TOLERANCE:
            raise ValueError("coordinate mapping contains a degenerate triangle")
        first = ((by - cy) * (point[0] - cx) + (cx - bx) * (point[1] - cy)) / denominator
        second = ((cy - ay) * (point[0] - cx) + (ax - cx) * (point[1] - cy)) / denominator
        return first, second, 1.0 - first - second

    @classmethod
    def _map_point(
        cls,
        point: tuple[float, float],
        source: dict[str, tuple[float, float]],
        destination: dict[str, tuple[float, float]],
    ) -> tuple[float, float]:
        candidates: list[tuple[float, tuple[float, float, float], tuple[str, str, str]]] = []
        for triangle_ids in _AFFINE_TRIANGLES:
            weights = cls._barycentric_weights(
                point,
                tuple(source[pocket_id] for pocket_id in triangle_ids),  # type: ignore[arg-type]
            )
            candidate = (min(weights), weights, triangle_ids)
            if candidate[0] >= -_AFFINE_TOLERANCE:
                candidates = [candidate]
                break
            candidates.append(candidate)
        _, weights, triangle_ids = max(candidates, key=lambda candidate: candidate[0])
        return tuple(
            sum(
                weight * destination[pocket_id][axis]
                for weight, pocket_id in zip(weights, triangle_ids, strict=True)
            )
            for axis in (0, 1)
        )

    def _validate_affine_geometry(self) -> None:
        business = {
            pocket_id: self._business_anchor_xy(pocket_id) for pocket_id in _BUSINESS_POCKET_FACTORS
        }
        pooltool = dict(self.pocket_centers)
        for triangle_ids in _AFFINE_TRIANGLES:
            for anchors in (business, pooltool):
                self._barycentric_weights(
                    anchors[triangle_ids[0]],
                    tuple(anchors[pocket_id] for pocket_id in triangle_ids),  # type: ignore[arg-type]
                )

    def to_pooltool_xy(self, position: IdealPathPoint) -> tuple[float, float]:
        return self._map_point(
            (float(position.x_mm), float(position.y_mm)),
            {
                pocket_id: self._business_anchor_xy(pocket_id)
                for pocket_id in _BUSINESS_POCKET_FACTORS
            },
            dict(self.pocket_centers),
        )

    def to_business_point(self, xyz: object) -> IdealPathPoint:
        pool_x = float(xyz[0])  # type: ignore[index]
        pool_y = float(xyz[1])  # type: ignore[index]
        x_mm, y_mm = self._map_point(
            (pool_x, pool_y),
            dict(self.pocket_centers),
            {
                pocket_id: self._business_anchor_xy(pocket_id)
                for pocket_id in _BUSINESS_POCKET_FACTORS
            },
        )
        if not (
            -_BOUNDARY_TOLERANCE_MM <= x_mm <= self.length_mm + _BOUNDARY_TOLERANCE_MM
            and -_BOUNDARY_TOLERANCE_MM <= y_mm <= self.width_mm + _BOUNDARY_TOLERANCE_MM
        ):
            raise ValueError("pooltool point is outside the versioned table plane")
        return IdealPathPoint(x_mm=round(x_mm), y_mm=round(y_mm))

    def pocket_to_business_point(self, pocket_id: str) -> IdealPathPoint:
        if pocket_id not in _BUSINESS_POCKET_FACTORS:
            raise ValueError("unknown pooltool pocket id")
        return self._business_anchor(pocket_id)


class PooltoolIdealPathAdapter:
    """Map strict millimetre business requests to one explicit pooltool shot."""

    adapter_version = "1.0"
    engine_name = "pooltool-billiards"
    engine_version = "0.6.0"

    @staticmethod
    def _require_engine() -> None:
        if pt is None:
            raise EngineUnavailableError("pooltool import unavailable")
        try:
            installed = version("pooltool-billiards")
        except PackageNotFoundError as exc:
            raise EngineUnavailableError("pooltool distribution unavailable") from exc
        if installed != PooltoolIdealPathAdapter.engine_version:
            raise EngineUnavailableError("unsupported pooltool version")

    @staticmethod
    def _table_specs(request: IdealPathRequest):
        PooltoolIdealPathAdapter._require_engine()
        table = request.table
        return pt.objects.PocketTableSpecs(  # type: ignore[union-attr]
            l=table.length_mm / 1000,
            w=table.width_mm / 1000,
            cushion_width=table.cushion_nose_width_mm / 1000,
            cushion_height=table.cushion_nose_height_mm / 1000,
            corner_pocket_width=table.corner_pocket_mouth_mm / 1000,
            side_pocket_width=table.side_pocket_mouth_mm / 1000,
        )

    @staticmethod
    def _physics_engine():
        """Use pooltool's immutable v9 defaults, never mutable user config."""

        PooltoolIdealPathAdapter._require_engine()
        resolver = pt.physics.resolve.resolver.default_resolver()  # type: ignore[union-attr]
        return pt.physics.PhysicsEngine(resolver=resolver)  # type: ignore[union-attr]

    @staticmethod
    def prepare_numeric_runtime() -> None:
        """Validate the original Python kernels loaded by the S1 preparer.

        The fresh owned worker selects Python functions during import. Do not
        mix previously compiled kernels into that runtime or compile a shot.
        """
        PooltoolIdealPathAdapter._require_engine()
        import sys
        from types import ModuleType

        from numba.core.registry import CPUDispatcher

        for name, module in tuple(sys.modules.items()):
            if (name == "pooltool" or name.startswith("pooltool.")) and isinstance(
                module, ModuleType
            ):
                if any(isinstance(value, CPUDispatcher) for value in tuple(vars(module).values())):
                    raise EngineUnavailableError("S1 requires fresh Pooltool Python kernels")

    @staticmethod
    def _coordinate_mapping(table: object, request: IdealPathRequest) -> _PooltoolCoordinateMapping:
        return _PooltoolCoordinateMapping.from_table(table, request)

    def _build_system(self, request: IdealPathRequest):
        self._require_engine()
        radius_m = request.table.ball_diameter_mm / 2000
        table = pt.Table.from_table_specs(self._table_specs(request))  # type: ignore[union-attr]
        mapping = self._coordinate_mapping(table, request)
        business_balls = (request.cue_ball, *request.object_balls)
        balls = {
            str(ball.ball_id): pt.Ball.create(  # type: ignore[union-attr]
                str(ball.ball_id),
                xy=mapping.to_pooltool_xy(ball.position),
                R=radius_m,
            )
            for ball in business_balls
        }
        cue = pt.Cue.default()  # type: ignore[union-attr]
        cue.cue_ball_id = "0"
        return pt.System(cue=cue, table=table, balls=balls)  # type: ignore[union-attr]

    def _event_position(
        self,
        event: object,
        ball_id: str,
        mapping: _PooltoolCoordinateMapping,
    ) -> IdealPathPoint:
        if str(event.event_type) == "ball_pocket":  # type: ignore[attr-defined]
            pocket_id = event.ids[1]  # type: ignore[attr-defined]
            return mapping.pocket_to_business_point(pocket_id)
        ball = event.get_ball(ball_id, initial=True)  # type: ignore[attr-defined]
        return mapping.to_business_point(ball.state.rvw[0])

    def _business_events(
        self,
        simulation: object,
        request: IdealPathRequest,
    ) -> tuple[IdealPathEvent, ...]:
        converted: list[IdealPathEvent] = []
        mapping = self._coordinate_mapping(simulation.table, request)  # type: ignore[attr-defined]
        for event in simulation.events:  # type: ignore[attr-defined]
            event_type = str(event.event_type)
            other_ball_id = None
            pocket = None
            if event_type == "stick_ball":
                kind = IdealPathEventKind.CUE_STRIKE
                ball_id = "0"
            elif event_type == "ball_ball":
                kind = IdealPathEventKind.BALL_CONTACT
                ball_id = event.ids[0]
                other_ball_id = int(event.ids[1])
            elif event_type in {"ball_linear_cushion", "ball_circular_cushion"}:
                kind = IdealPathEventKind.CUSHION_CONTACT
                ball_id = event.ids[0]
            elif event_type == "ball_pocket":
                kind = IdealPathEventKind.POCKETED
                ball_id = event.ids[0]
                pocket = _POCKETS_BY_ID.get(event.ids[1])
                if pocket is None:
                    raise ValueError("unknown pooltool pocket event")
            elif event_type in {"rolling_stationary", "spinning_stationary"}:
                kind = IdealPathEventKind.STOP
                ball_id = event.ids[0]
            else:
                continue
            converted.append(
                IdealPathEvent(
                    sequence=len(converted) + 1,
                    kind=kind,
                    ball_id=int(ball_id),
                    other_ball_id=other_ball_id,
                    pocket=pocket,
                    position=self._event_position(event, ball_id, mapping),
                )
            )
        return tuple(converted)

    def calculate(self, request: IdealPathRequest) -> tuple[IdealPathEvent, ...]:
        """Run one fixed, headless direct-pot simulation for the reviewed intent."""

        self._require_engine()
        if request.target_ball_id is None or request.target_pocket is None:
            raise ValueError("adapter requires an explicit target ball and pocket")
        system = self._build_system(request)
        target_id = str(request.target_ball_id)
        pocket_id = _POCKET_IDS[request.target_pocket]
        angle = pt.pot.PottingConfig.default().calculate_angle(  # type: ignore[union-attr]
            system.balls["0"],
            system.balls[target_id],
            system.table,
            system.table.pockets[pocket_id],
        )
        system.strike(V0=2.5, phi=angle, theta=0.0, a=0.0, b=0.0)
        simulation = pt.simulate(  # type: ignore[union-attr]
            system,
            engine=self._physics_engine(),
            max_events=100,
        )
        events = self._business_events(simulation, request)
        if not any(
            event.kind is IdealPathEventKind.POCKETED
            and event.ball_id == request.target_ball_id
            and event.pocket is request.target_pocket
            for event in events
        ):
            raise RuntimeError("explicit target pocket was not reached")
        return events
