import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/algorithm/pooltool_native_smoke.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_pooltool_native_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pooltool:
    __version__ = "0.6.0"

    class Ball:
        @staticmethod
        def create(ball_id: str, *, xy: tuple[float, float]):
            return SimpleNamespace(id=ball_id, xy=xy)

    class Cue:
        def __init__(self, cue_ball_id: str) -> None:
            self.cue_ball_id = cue_ball_id

        def set_state(self, **kwargs: float) -> None:
            self.state = kwargs

    class Table:
        @staticmethod
        def default():
            return object()

    class System:
        def __init__(self, *, cue, table, balls) -> None:
            self.cue = cue
            self.table = table
            self.balls = balls
            self.events = [SimpleNamespace(event_type="stick-ball")]

    @staticmethod
    def simulate(system, *, continuous: bool, dt: float):
        assert continuous is True
        assert dt == 1 / 60
        return system


def test_native_smoke_exercises_real_api_shape_without_project_dependencies() -> None:
    result = _load().run_native_smoke(_Pooltool())
    assert result == {
        "schema_version": "1.0",
        "status": "PASS",
        "pooltool_version": "0.6.0",
        "ball_count": 2,
        "event_count": 1,
        "synthetic_only": True,
    }
