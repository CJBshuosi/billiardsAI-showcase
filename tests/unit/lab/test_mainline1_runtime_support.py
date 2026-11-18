"""Public runtime assembly is independent of product CLI and training execution."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def support():
    name = "billiards_ai.lab.mainline1_runtime_support"
    assert importlib.util.find_spec(name) is not None, "canonical runtime support is missing"
    return importlib.import_module(name)


@pytest.mark.parametrize(
    "name", ["_detector", "_session", "_capture", "_preflight_capture", "_select_table_corners"]
)
def test_existing_product_cli_helpers_reexport_the_same_callable(name):
    helpers = support()
    old = importlib.import_module("scripts.algorithm.run_mainline1_product_realtime")
    assert getattr(old, name) is getattr(helpers, name)


def test_existing_preflight_defaults_and_keyword_interfaces_are_preserved():
    helpers = support()
    expected = {
        "_detector": {"device": "cuda"},
        "_preflight_capture": {"sample_count": 12},
        "_select_table_corners": {
            "minimum_confirmations": 3,
            "drift_threshold_px": 12.0,
            "consensus_threshold_px": 5.0,
        },
    }
    for name, defaults in expected.items():
        parameters = inspect.signature(getattr(helpers, name)).parameters
        for parameter, default in defaults.items():
            assert parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[parameter].default == default


class PreflightSource:
    def __init__(self, count):
        self.count = count
        self.started = self.cancelled = self.frames_closed = False
        self.stop = None

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def frames(self, stop):
        self.stop = stop
        try:
            yield from range(self.count)
        finally:
            self.frames_closed = True


@pytest.mark.parametrize("count", [2, 5])
def test_preflight_capture_cancels_and_closes_after_real_iterator_consumption(count):
    source = PreflightSource(count)
    assert support()._preflight_capture(source, sample_count=2) == (0, 1)
    assert source.started and source.cancelled and source.frames_closed
    assert source.stop.is_set()


def test_preflight_failure_cancels_and_closes_the_iterator():
    source = PreflightSource(1)
    with pytest.raises(RuntimeError, match="generator raised StopIteration"):
        support()._preflight_capture(source, sample_count=2)
    assert source.started and source.cancelled and source.frames_closed
    assert source.stop.is_set()


def test_invalid_preflight_count_does_not_open_the_source():
    source = PreflightSource(5)
    with pytest.raises(ValueError, match="at least two frames"):
        support()._preflight_capture(source, sample_count=1)
    assert not source.started and not source.cancelled


def test_isolated_core_prepares_recording_and_media_without_cli_training_or_models(tmp_path):
    code = r"""
import importlib.abc, sys
from pathlib import Path
root, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / "src"))
blocked = ("scripts", "billiards_ai.lab.training", "torch", "rfdetr", "onnxruntime", "transformers")
class NoTrainingOrModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            raise ImportError("unexpected runtime dependency: " + fullname)
sys.meta_path.insert(0, NoTrainingOrModels())
from billiards_ai.lab import mainline1_runtime_support as helpers
from billiards_ai.lab.mainline1_field_runtime import build_live_runtime_factory
from billiards_ai.lab.field_assistance.runtime_resources import inspect_managed_runtime_resources
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.shared.calibration.homography import calibrate
calls = []
def forbidden(*args, **kwargs):
    calls.append("model")
    raise AssertionError("ordinary recording/media must not construct semantic models")
helpers._detector = helpers._session = forbidden
helpers.Dinov2IdentityFeatureBackend.__init__ = forbidden
config = root / "configs/algorithm/shot-local.yaml"
factory = build_live_runtime_factory(
    capture_factory=forbidden, capture_config=helpers.RealtimeCaptureConfig(),
    calibration=calibrate("test",[(0,1079),(1919,1079),(1919,0),(0,0)],
                          table_length_mm=2540, table_width_mm=1270),
    detector=None, identity_classifier=None,
    shot_config=helpers.load_mainline1_shot_local_config(config),
    capture_setup_id="test", calibration_artifact_id="test-cal", config_path=config,
    ring_root=output / "ring", jobs_root=output / "jobs", session_policy="recording-v1")
assert callable(factory)
records = SQLiteFieldRecordStore(output / "state.sqlite3")
session = records.open_or_resume_session(
    artifact_id="recording", session_policy="recording-v1")
jobs = SQLiteShotAnalysisJobStore(output / "media-jobs.sqlite3")
worker = Mainline1RecordingMediaWorker(
    records=records, jobs=jobs, artifact_root=output / "artifacts")
assert worker.schedule(session.session_id) == ()
assert calls == []
assert not any(any(n == p or n.startswith(p + ".") for p in blocked) for n in sys.modules)
print("ordinary and media preparation succeeded without CLI/training/models")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "ordinary and media preparation succeeded" in result.stdout
