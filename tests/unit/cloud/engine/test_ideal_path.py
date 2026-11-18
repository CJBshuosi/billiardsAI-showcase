"""Business orchestration tests for Story 5.2's replaceable adapter."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest


def _modules():
    assert importlib.util.find_spec("billiards_ai.shared.models.ideal_path") is not None, (
        "Story 5.2 ideal-path contract is missing"
    )
    assert importlib.util.find_spec("billiards_ai.cloud.engine.ideal_path") is not None, (
        "Story 5.2 ideal-path service is missing"
    )
    return (
        importlib.import_module("billiards_ai.shared.models.ideal_path"),
        importlib.import_module("billiards_ai.cloud.engine.ideal_path"),
    )


def _request(models, **updates):
    payload = {
        "cue_ball": {"ball_id": 0, "position": {"x_mm": 500, "y_mm": 635}},
        "object_balls": ({"ball_id": 3, "position": {"x_mm": 1500, "y_mm": 635}},),
        "target_ball_id": 3,
        "target_pocket": "TOP_RIGHT",
        "intent": "DIRECT_POT",
        "target_ball_reviewed": True,
        "target_pocket_reviewed": True,
        "intent_reviewed": True,
        "calibration_status": "ACTIVE",
        "table": {
            "configuration_version": "chinese-eight-ball-9ft-v1",
            "length_mm": 2540,
            "width_mm": 1270,
            "ball_diameter_mm": 57,
            "corner_pocket_mouth_mm": 86,
            "side_pocket_mouth_mm": 92,
            "cushion_nose_width_mm": 40,
        },
    }
    payload.update(updates)
    return models.IdealPathRequest(**payload)


class _FakeAdapter:
    adapter_version = "fake-1"
    engine_name = "fake-engine"
    engine_version = "test"

    def __init__(self, events=(), failure: Exception | None = None):
        self.events = events
        self.failure = failure
        self.calls = []

    def calculate(self, request):
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure
        return self.events


def _successful_events(models, *, first_contact=3, pocket="TOP_RIGHT"):
    return (
        models.IdealPathEvent(
            sequence=1,
            kind="CUE_STRIKE",
            ball_id=0,
            position={"x_mm": 500, "y_mm": 635},
        ),
        models.IdealPathEvent(
            sequence=2,
            kind="BALL_CONTACT",
            ball_id=0,
            other_ball_id=first_contact,
            position={"x_mm": 1500, "y_mm": 635},
        ),
        models.IdealPathEvent(
            sequence=3,
            kind="POCKETED",
            ball_id=3,
            pocket=pocket,
            position={"x_mm": 2540, "y_mm": 1270},
        ),
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"target_ball_id": None},
        {"target_pocket": None},
        {"intent": None},
        {"target_ball_reviewed": False},
        {"target_pocket_reviewed": False},
        {"intent_reviewed": False},
        {"calibration_status": "STALE"},
        {"calibration_status": "INVALID"},
    ],
)
def test_unknown_or_unreliable_intent_abstains_without_calling_adapter(updates):
    models, service = _modules()
    adapter = _FakeAdapter()

    result = service.calculate_ideal_path(_request(models, **updates), adapter)

    assert result.status is models.IdealPathStatus.UNSUPPORTED_INTENT
    assert result.events == ()
    assert adapter.calls == []


def test_target_ball_must_exist_and_cue_ball_cannot_be_the_target():
    models, service = _modules()
    adapter = _FakeAdapter()

    missing = service.calculate_ideal_path(_request(models, target_ball_id=7), adapter)
    cue = service.calculate_ideal_path(_request(models, target_ball_id=0), adapter)

    assert missing.status is models.IdealPathStatus.UNSUPPORTED_INTENT
    assert cue.status is models.IdealPathStatus.UNSUPPORTED_INTENT
    assert adapter.calls == []


def test_engine_unavailable_and_unexpected_failures_are_typed_without_traceback():
    models, service = _modules()
    unavailable = _FakeAdapter(failure=service.EngineUnavailableError("secret import path"))
    broken = _FakeAdapter(failure=RuntimeError("third-party traceback details"))

    unavailable_result = service.calculate_ideal_path(_request(models), unavailable)
    broken_result = service.calculate_ideal_path(_request(models), broken)

    assert unavailable_result.status is models.IdealPathStatus.ENGINE_UNAVAILABLE
    assert broken_result.status is models.IdealPathStatus.SIMULATION_FAILED
    assert "secret" not in unavailable_result.model_dump_json()
    assert "traceback" not in broken_result.model_dump_json()
    assert unavailable_result.events == broken_result.events == ()


def test_success_is_deterministic_and_adapter_objects_cannot_leak():
    models, service = _modules()
    events = _successful_events(models)
    adapter = _FakeAdapter(events=events)
    request = _request(models)

    first = service.calculate_ideal_path(request, adapter)
    repeated = service.calculate_ideal_path(request, adapter)
    leaked = service.calculate_ideal_path(request, _FakeAdapter(events=(object(),)))

    assert first == repeated
    assert first.model_dump_json() == repeated.model_dump_json()
    assert first.status is models.IdealPathStatus.SUCCEEDED
    assert leaked.status is models.IdealPathStatus.SIMULATION_FAILED
    assert leaked.events == ()


@pytest.mark.parametrize(
    "events_factory",
    [
        lambda models: (
            models.IdealPathEvent(
                sequence=1,
                kind="STOP",
                ball_id=0,
                position={"x_mm": 500, "y_mm": 635},
            ),
        ),
        lambda models: _successful_events(models)[1:],
        lambda models: _successful_events(models, first_contact=8),
        lambda models: _successful_events(models)[:-1],
        lambda models: _successful_events(models, pocket="BOTTOM_LEFT"),
    ],
    ids=[
        "stop-only",
        "missing-cue-strike",
        "wrong-first-cue-contact",
        "missing-target-pocket-event",
        "target-entered-wrong-pocket",
    ],
)
def test_service_rejects_semantically_incomplete_adapter_success(events_factory):
    models, service = _modules()
    adapter = _FakeAdapter(events=events_factory(models))

    result = service.calculate_ideal_path(_request(models), adapter)

    assert result.status is models.IdealPathStatus.SIMULATION_FAILED
    assert result.reason is models.IdealPathReason.ADAPTER_FAILURE
    assert result.events == ()


def test_adapter_metadata_is_snapshotted_safely_and_never_escapes():
    models, service = _modules()

    class ExplodingMetadataAdapter:
        @property
        def adapter_version(self):
            raise RuntimeError("unstable metadata")

        @property
        def engine_name(self):
            raise RuntimeError("unstable metadata")

        @property
        def engine_version(self):
            raise RuntimeError("unstable metadata")

        def calculate(self, request):
            raise AssertionError("metadata failure must stop before simulation")

    first = service.calculate_ideal_path(_request(models), ExplodingMetadataAdapter())
    repeated = service.calculate_ideal_path(_request(models), ExplodingMetadataAdapter())

    assert first == repeated
    assert first.status is models.IdealPathStatus.SIMULATION_FAILED
    assert first.events == ()
    assert first.adapter_version
    assert first.engine_name
    assert first.engine_version


def test_only_pooltool_adapter_module_may_import_pooltool():
    source_root = Path(__file__).parents[4] / "src" / "billiards_ai"
    violations = []
    for path in source_root.rglob("*.py"):
        if path.name == "pooltool_adapter.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "pooltool" or name.startswith("pooltool.") for name in names):
                violations.append(str(path.relative_to(source_root)))

    assert violations == []
