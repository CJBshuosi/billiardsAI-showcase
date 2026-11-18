"""Real spawn lifecycle checks; controlled observers are not model acceptance."""

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.lab.field_assistance.snapshot_guidance import SnapshotConfig, SnapshotGuidance
from billiards_ai.shared.calibration.homography import calibrate
from tests.unit.lab.field_assistance.test_snapshot_guidance import frame

BACKEND_AUTOLOAD = "TORCH_DEVICE_BACKEND_AUTOLOAD"


@dataclass
class BackendPolicyPreparer:
    root: Path

    def __call__(self):
        (self.root / "resource-backend-env.json").write_text(
            json.dumps({"flag": os.environ.get(BACKEND_AUTOLOAD), "pid": os.getpid()})
        )


@dataclass
class BackendPolicyFactory:
    root: Path

    def __call__(self):
        (self.root / "model-backend-env.json").write_text(
            json.dumps({"flag": os.environ.get(BACKEND_AUTOLOAD), "pid": os.getpid()})
        )
        return CountingObserver(self.root)


@dataclass
class CountingFactory:
    root: Path
    wait_for_release: bool = False
    fail: bool = False
    block_observation: bool = False
    wait_for_numeric: bool = False
    wait_for_observation_release: bool = False

    def __call__(self):
        with (self.root / "loads.txt").open("a") as stream:
            stream.write(str(os.getpid()) + "\n")
        if self.wait_for_numeric:
            deadline = time.monotonic() + 1
            while not (self.root / "prepared.txt").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("independent numeric preparation did not start")
                threading.Event().wait(0.01)
        if self.fail:
            raise ValueError("controlled asset failure")
        if self.wait_for_release:
            while not (self.root / "release").exists():
                threading.Event().wait(0.01)
        return CountingObserver(
            self.root, self.block_observation, self.wait_for_observation_release
        )


@dataclass
class CountingObserver:
    root: Path
    block: bool = False
    wait_for_release: bool = False

    def observe(self, value, calibration):
        with (self.root / "observed.txt").open("a") as stream:
            stream.write(str(value.capture_frame_id) + "\n")
        if self.block:
            threading.Event().wait()
        if self.wait_for_release:
            while not (self.root / "observation-release").exists():
                threading.Event().wait(0.01)
        # Controlled low-confidence evidence exercises real calculation and
        # honest planner abstention, never an accepted recommendation/demo.
        from billiards_ai.lab.shot_local.models import (
            BallGroup,
            CalibratedBallObservation,
            ObservationFrame,
        )

        return ObservationFrame(
            value.capture_frame_id,
            value.timestamp_ms,
            (CalibratedBallObservation("cue", 500, 500, 0.8, BallGroup.CUE, 0.99),),
        )


@dataclass
class RecordingPlanner:
    root: Path

    def __call__(self, state, context):
        from billiards_ai.lab.planning.s1 import recommend_s1

        with (self.root / "planned.txt").open("a") as stream:
            stream.write(str(state.frame_id) + "\n")
        return recommend_s1(state, context)


@dataclass
class ResourcePreparation:
    root: Path
    wait: bool = False
    fail: bool = False
    delay_seconds: float = 0

    def __call__(self):
        with (self.root / "prepared.txt").open("a") as stream:
            stream.write(str(os.getpid()) + "\n")
        if self.fail:
            raise RuntimeError("controlled numerical preparation failure")
        if self.delay_seconds:
            threading.Event().wait(self.delay_seconds)
        if self.wait:
            while not (self.root / "numeric-release").exists():
                threading.Event().wait(0.01)


def service(tmp_path, *, resource_preparer=None, config=None, planner=None, **factory_options):
    return SnapshotGuidance(
        hub=None,
        session_id="test",
        source_epoch_id="epoch1",
        calibration=calibrate("test", [(0, 239), (319, 239), (319, 0), (0, 0)]),
        observer_factory=CountingFactory(tmp_path, **factory_options),
        journal_path=tmp_path / "guidance.json",
        config=config
        or SnapshotConfig(sample_ms=100, stable_ms=200, stale_seconds=5, task_seconds=0.15),
        **({"resource_preparer": resource_preparer} if resource_preparer else {}),
        **({"planner": planner} if planner else {}),
    )


def wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "the owned worker did not reach the required lifecycle state"


def test_real_worker_feedback_advances_on_fresh_stable_pixels_without_an_extra_request(tmp_path):
    # Controlled prior advice tests scheduling only. The real child observes
    # low-confidence pixels and honestly abstains, never a model/pot acceptance.
    runner = service(tmp_path)
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] == "ready")
        runner.observe(frame(0), context="solid")
        previous = dict(
            source_frame_id=0,
            region=dict(center=dict(x_mm=1000, y_mm=500), radius_mm=100),
        )
        with runner._lock:
            runner._recommendation = previous
            runner._phase = "ready"
        runner.observe(frame(1, moving=True), context="solid")
        seen_at = None
        for number in range(2, 100):
            runner.observe(frame(number, moving=True), context="solid")
            value = runner.projection()
            if value["phase"] == "reviewing":
                seen_at = time.monotonic()
                assert value["feedback"]["recommendation"] == previous
                assert value["feedback"]["actual_cue"] is None
                assert value["feedback"]["region_result"] == "unknown"
                assert value["recommendation"] is None
                break
            time.sleep(0.03)
        assert seen_at is not None, runner.projection()
        first_worker = runner.projection()["worker_pid"]
        first_tasks = runner.projection()["tasks_started"]
        next_frame = number + 1
        for number in range(next_frame, next_frame + 99):
            runner.observe(frame(number, moving=True), context="solid")
            value = runner.projection()
            if value["phase"] == "abstained":
                break
            assert value["phase"] == "reviewing"
            assert value["recommendation"] is None
            time.sleep(0.03)
        assert value["phase"] == "abstained"
        assert time.monotonic() - seen_at >= 0.95
        assert value["feedback"] is None
        assert value["worker_pid"] == first_worker
        assert value["tasks_started"] == first_tasks == value["tasks_completed"] == 1
        assert len((tmp_path / "observed.txt").read_text().splitlines()) == 1
        assert len((tmp_path / "loads.txt").read_text().splitlines()) == 1
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_cold_preparation_may_pass_61_seconds_without_becoming_a_product_failure(tmp_path):
    runner = service(tmp_path, resource_preparer=ResourcePreparation(tmp_path, wait=True))
    runner.start()
    try:
        wait_for(lambda: (tmp_path / "prepared.txt").exists())
        with runner._lock:
            runner._model_started_at -= 61
            runner._poll_preparation()
            assert runner.projection()["model_state"] == "preparing"
            assert runner.projection()["error"] is None
        (tmp_path / "numeric-release").touch()
        wait_for(lambda: runner.projection()["model_state"] == "ready")
        assert runner.projection()["resource_preparation_seconds"] >= 61
        assert runner.projection()["tasks_started"] == 0
        assert len((tmp_path / "prepared.txt").read_text().splitlines()) == 1
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_moving_during_inference_discards_it_then_reuses_same_environment_for_latest_scene(
    tmp_path,
):
    runner = service(
        tmp_path,
        resource_preparer=ResourcePreparation(tmp_path),
        wait_for_observation_release=True,
        planner=RecordingPlanner(tmp_path),
    )
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] == "ready")
        pid = runner.projection()["worker_pid"]
        for number in range(4):
            runner.observe(frame(number), context="solid")
        wait_for(lambda: (tmp_path / "observed.txt").exists())
        for number in range(4, 9):
            runner.observe(frame(number, moving=number % 2 == 0), context="solid")
        value = runner.projection()
        assert value["recommendation"] is None and value["phase"] == "moving"
        assert value["worker_pid"] == pid and runner._process.is_alive()
        assert value["tasks_cancelled"] == 1 and value["model_loads"] == 1
        assert value["tasks_started"] == 1
        assert not (tmp_path / "planned.txt").exists()
        (tmp_path / "observation-release").touch()
        wait_for(lambda: not runner._busy)
        assert not (tmp_path / "planned.txt").exists()
        for number in range(9, 22):
            runner.observe(frame(number), context="solid")
            if runner.projection()["tasks_started"] == 2:
                break
        wait_for(lambda: (tmp_path / "planned.txt").exists())
        wait_for(lambda: not runner._busy)
        runner.observe(frame(number + 1), context="solid")
        value = runner.projection()
        assert value["tasks_completed"] == 1 and value["tasks_discarded"] == 1
        assert value["worker_pid"] == pid and value["model_loads"] == 1
        assert len((tmp_path / "loads.txt").read_text().splitlines()) == 1
        assert len((tmp_path / "prepared.txt").read_text().splitlines()) == 1
        assert len((tmp_path / "planned.txt").read_text().splitlines()) == 1
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_hot_deadline_discards_result_without_reloading_or_retrying_same_scene(tmp_path):
    runner = service(
        tmp_path, wait_for_observation_release=True, planner=RecordingPlanner(tmp_path)
    )
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] == "ready")
        pid = runner.projection()["worker_pid"]
        for number in range(4):
            runner.observe(frame(number), context="solid")
        wait_for(lambda: (tmp_path / "observed.txt").exists())
        time.sleep(0.2)
        runner.observe(frame(4), context="solid")
        value = runner.projection()
        assert value["error"] == "guidance_task_timeout"
        assert value["worker_pid"] == pid and runner._process.is_alive()
        assert value["recommendation"] is None
        (tmp_path / "observation-release").touch()
        wait_for(lambda: not runner._busy)
        for number in range(5, 12):
            runner.observe(frame(number), context="solid")
        assert runner.projection()["tasks_started"] == 1
        assert runner.projection()["model_loads"] == 1
        assert not (tmp_path / "planned.txt").exists()
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_obsolete_task_deadline_cannot_clear_the_new_scenes_pending_review(tmp_path):
    runner = service(tmp_path, block_observation=True)
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] == "ready")
        for number in range(4):
            runner.observe(frame(number), context="solid")
        wait_for(lambda: (tmp_path / "observed.txt").exists())
        with runner._lock:
            runner._invalidate()
            runner._pending_review = 7
            runner._phase = "moving"
            runner._started_at -= 1
            runner._poll_task()
            assert runner._pending_review == 7
            assert runner.projection()["phase"] == "moving"
            assert runner.projection()["error"] is None
            assert runner.projection()["tasks_cancelled"] == 1
    finally:
        runner.close()


def test_only_hard_task_fault_reaps_and_recovers_worker_once(tmp_path):
    config = SnapshotConfig(
        sample_ms=100, stable_ms=200, stale_seconds=5, task_seconds=0.1, hard_task_seconds=0.3
    )
    runner = service(tmp_path, block_observation=True, config=config)
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] == "ready")
        pid = runner.projection()["worker_pid"]
        for number in range(4):
            runner.observe(frame(number), context="solid")
        wait_for(lambda: (tmp_path / "observed.txt").exists())
        wait_for(
            lambda: (
                runner.projection()["model_loads"] == 2
                and runner.projection()["model_state"] == "ready"
            )
        )
        assert runner.projection()["error"] == "guidance_hard_task_timeout"
        assert runner.projection()["worker_pid"] != pid
        assert runner.projection()["tasks_started"] == 1
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        # A second genuine crash exhausts the single recovery allowance.
        runner._process.terminate()
        wait_for(lambda: runner.projection()["model_state"] == "failed")
        assert runner.projection()["model_loads"] == 2
    finally:
        runner.close()
    assert runner.cleanup_complete


@pytest.mark.parametrize("initial", [None, "0", "1"])
@pytest.mark.parametrize("failure_stage", [None, "resource", "model"])
def test_preparation_scopes_builtin_torch_policy_and_restores_exact_environment(
    monkeypatch, initial, failure_stage
):
    from billiards_ai.lab.field_assistance.snapshot_guidance import _prepare_resources

    if initial is None:
        monkeypatch.delenv(BACKEND_AUTOLOAD, raising=False)
    else:
        monkeypatch.setenv(BACKEND_AUTOLOAD, initial)
    calls = []
    observer = object()
    failure = ValueError("original preparation failure")

    def resource():
        calls.append(("resource", os.environ.get(BACKEND_AUTOLOAD)))
        if failure_stage == "resource":
            raise failure

    def factory():
        calls.append(("model", os.environ.get(BACKEND_AUTOLOAD)))
        if failure_stage == "model":
            raise failure
        return observer

    if failure_stage:
        with pytest.raises(ValueError) as caught:
            _prepare_resources(factory, resource)
        assert caught.value is failure
    else:
        actual, timings = _prepare_resources(factory, resource)
        assert actual is observer
        assert 0 <= timings["model_preparation_seconds"] <= timings["resource_preparation_seconds"]
    expected = ["resource"] if failure_stage == "resource" else ["resource", "model"]
    assert calls == [(name, "0") for name in expected]
    assert os.environ.get(BACKEND_AUTOLOAD) == initial
    assert (BACKEND_AUTOLOAD in os.environ) is (initial is not None)


def test_real_spawn_preparation_policy_leaves_parent_environment_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv(BACKEND_AUTOLOAD, "1")
    runner = service(tmp_path, resource_preparer=BackendPolicyPreparer(tmp_path))
    runner.factory = BackendPolicyFactory(tmp_path)
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] in {"ready", "failed"})
        assert runner.projection()["model_state"] == "ready"
        assert os.environ[BACKEND_AUTOLOAD] == "1"
        resource = json.loads((tmp_path / "resource-backend-env.json").read_bytes())
        model = json.loads((tmp_path / "model-backend-env.json").read_bytes())
        assert resource["flag"] == model["flag"] == "0"
        assert resource["pid"] == model["pid"] != os.getpid()
        assert runner.projection()["tasks_started"] == 0
        assert not (tmp_path / "observed.txt").exists()
    finally:
        runner.close()
    assert runner.cleanup_complete and os.environ[BACKEND_AUTOLOAD] == "1"


def test_owned_component_probe_reuses_the_same_preparation_policy(tmp_path, monkeypatch):
    from billiards_ai.lab.field_assistance.snapshot_guidance import _calculate
    from billiards_ai.lab.planning.s1 import recommend_s1
    from scripts.development.s1_snapshot_probe import calculate_component

    monkeypatch.setenv(BACKEND_AUTOLOAD, "1")
    result, timings = calculate_component(
        _calculate,
        BackendPolicyFactory(tmp_path),
        frame(1),
        calibrate("test", [(0, 239), (319, 239), (319, 0), (0, 0)]),
        "solid",
        "epoch1",
        recommend_s1,
        mode="prepared",
        resource_preparer=BackendPolicyPreparer(tmp_path),
    )
    assert json.loads((tmp_path / "resource-backend-env.json").read_bytes())["flag"] == "0"
    assert json.loads((tmp_path / "model-backend-env.json").read_bytes())["flag"] == "0"
    assert os.environ[BACKEND_AUTOLOAD] == "1"
    assert result["route"] is None  # Original low-confidence observer honestly abstains.
    assert timings["within_snapshot_budget"] is True


def test_numeric_preparation_gates_ready_without_observing_motion(tmp_path):
    runner = service(tmp_path, resource_preparer=ResourcePreparation(tmp_path, wait=True))
    runner.start()
    try:
        wait_for(lambda: (tmp_path / "prepared.txt").exists())
        for number in range(10):
            runner.observe(frame(number, moving=number % 2 == 0), context="solid")
        assert runner.projection()["model_state"] == "preparing"
        assert runner.projection()["tasks_started"] == 0
        assert not (tmp_path / "observed.txt").exists()
        assert not (tmp_path / "loads.txt").exists()
        (tmp_path / "numeric-release").touch()
        wait_for(lambda: runner.projection()["model_state"] == "ready")
        for number in range(10, 15):
            runner.observe(frame(number), context="solid")
        wait_for(lambda: (tmp_path / "observed.txt").exists())
        assert len((tmp_path / "prepared.txt").read_text().splitlines()) == 1
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_numeric_preparation_finishes_before_models_start(tmp_path):
    runner = service(
        tmp_path, wait_for_numeric=True, resource_preparer=ResourcePreparation(tmp_path)
    )
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] in {"ready", "failed"})
        assert runner.projection()["model_state"] == "ready"
        assert not (tmp_path / "observed.txt").exists()
        assert len((tmp_path / "loads.txt").read_text().splitlines()) == 1
        assert len((tmp_path / "prepared.txt").read_text().splitlines()) == 1
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_numeric_failure_stops_without_starting_models(tmp_path):
    runner = service(
        tmp_path,
        wait_for_release=True,
        resource_preparer=ResourcePreparation(tmp_path, fail=True),
    )
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] == "failed")
        assert "controlled numerical preparation failure" in runner.projection()["error"]
        assert not (tmp_path / "observed.txt").exists()
        assert runner._process is None and runner._connection is None
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_model_failure_after_numeric_preparation_reaps_the_worker(tmp_path):
    runner = service(
        tmp_path,
        fail=True,
        wait_for_numeric=True,
        resource_preparer=ResourcePreparation(tmp_path, wait=True),
    )
    runner.start()
    try:
        wait_for(lambda: (tmp_path / "prepared.txt").exists())
        assert not (tmp_path / "loads.txt").exists()
        (tmp_path / "numeric-release").touch()
        wait_for(lambda: runner.projection()["model_state"] == "failed")
        assert "controlled asset failure" in runner.projection()["error"]
        assert runner._process is None and runner._connection is None
        assert not (tmp_path / "observed.txt").exists()
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_numeric_preparation_failure_is_terminal_without_observation_or_retry(tmp_path):
    runner = service(tmp_path, resource_preparer=ResourcePreparation(tmp_path, fail=True))
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] == "failed")
        for number in range(10):
            runner.observe(frame(number), context="solid")
        assert runner.projection()["tasks_started"] == 0
        assert not (tmp_path / "observed.txt").exists()
        assert len((tmp_path / "prepared.txt").read_text().splitlines()) == 1
        assert not (tmp_path / "loads.txt").exists()
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_close_reaps_a_worker_blocked_in_numeric_preparation(tmp_path):
    runner = service(tmp_path, resource_preparer=ResourcePreparation(tmp_path, wait=True))
    runner.start()
    wait_for(lambda: (tmp_path / "prepared.txt").exists())
    process = runner._process
    runner.close()
    assert runner.cleanup_complete
    # multiprocessing.close rejects a live process, and then releases its
    # native handles; querying that released object must report it as closed.
    with pytest.raises(ValueError, match="closed"):
        process.is_alive()
    assert not (tmp_path / "observed.txt").exists()


def test_real_worker_keeps_model_load_time_separate_from_numeric_preparation(tmp_path):
    runner = service(tmp_path, resource_preparer=ResourcePreparation(tmp_path, delay_seconds=0.15))
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] == "ready")
        value = runner.projection()
        assert value["model_preparation_seconds"] >= 0
        assert value["resource_preparation_seconds"] - value["model_preparation_seconds"] >= 0.15
        assert not (tmp_path / "observed.txt").exists()
    finally:
        runner.close()
    assert runner.cleanup_complete


@pytest.mark.parametrize("bad", [None, True, -1, float("nan"), float("inf"), 10**400, 61])
def test_invalid_ready_timing_cannot_be_published(tmp_path, monkeypatch, bad):
    runner = service(tmp_path)
    runner._process = SimpleNamespace(is_alive=lambda: True, pid=0)
    runner._connection = SimpleNamespace(
        poll=lambda: True,
        recv=lambda: {
            "kind": "ready",
            "model_preparation_seconds": bad,
            "resource_preparation_seconds": 2.0,
        },
    )
    runner._model_started_at = time.monotonic()
    failures = []
    monkeypatch.setattr(runner, "_fail_preparation", failures.append)
    try:
        runner._poll_preparation()
        assert failures == ["guidance_preparation_protocol"]
        assert runner.projection()["model_preparation_seconds"] is None
        assert runner.projection()["resource_preparation_seconds"] is None
    finally:
        runner._process = runner._connection = None
        runner.close()


def test_completed_stable_requests_reuse_loaded_models_without_idle_inference(tmp_path):
    runner = service(tmp_path)
    runner.start()
    try:
        wait_for(lambda: runner.projection().get("model_state") == "ready")
        assert len((tmp_path / "loads.txt").read_text().splitlines()) == 1
        assert not (tmp_path / "observed.txt").exists()
        for number in range(4):
            runner.observe(frame(number), context="solid")
        wait_for(lambda: (tmp_path / "observed.txt").exists())
        wait_for(lambda: runner.active_pid is None or runner._connection.poll())
        runner.observe(frame(4), context="solid")
        assert runner.projection()["phase"] == "abstained"
        assert runner.projection()["tasks_started"] == 1
        assert runner.active_pid is None
        for number in range(5, 9):
            runner.observe(frame(number, moving=number % 2 == 0), context="solid")
        assert (tmp_path / "observed.txt").read_text().splitlines() == ["2"]
        for number in range(9, 20):
            runner.observe(frame(number), context="solid")
            if runner.projection()["tasks_started"] == 2:
                break
        wait_for(lambda: len((tmp_path / "observed.txt").read_text().splitlines()) == 2)
        wait_for(lambda: runner.active_pid is None or runner._connection.poll())
        runner.observe(frame(number + 1), context="solid")
        assert runner.projection()["tasks_completed"] == 2
        assert len((tmp_path / "loads.txt").read_text().splitlines()) == 1
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_model_preparation_never_observes_moving_frames_or_blocks_motion(tmp_path):
    runner = service(tmp_path, wait_for_release=True)
    runner.start()
    try:
        wait_for(lambda: (tmp_path / "loads.txt").exists())
        started = time.monotonic()
        for number in range(10):
            runner.observe(frame(number, moving=number % 2 == 0), context="solid")
        assert time.monotonic() - started < 1
        assert runner.projection().get("model_state") == "preparing"
        assert runner.projection()["tasks_started"] == 0
        assert not (tmp_path / "observed.txt").exists()
        (tmp_path / "release").touch()
        wait_for(lambda: runner.projection().get("model_state") == "ready")
        assert not (tmp_path / "observed.txt").exists()
        assert runner.projection()["tasks_started"] == 0
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_failed_model_preparation_is_latched_without_repeated_loads(tmp_path):
    runner = service(tmp_path, fail=True)
    runner.start()
    try:
        wait_for(lambda: runner.projection().get("model_state") == "failed")
        for number in range(20):
            runner.observe(frame(number, moving=number % 2 == 0), context="solid")
        assert len((tmp_path / "loads.txt").read_text().splitlines()) == 1
        assert runner.projection()["tasks_started"] == 0
        assert not (tmp_path / "observed.txt").exists()
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_ready_message_cannot_hide_expired_model_preparation(tmp_path):
    runner = service(tmp_path)
    try:
        runner._start_worker()
        wait_for(lambda: runner._connection.poll())
        # Model ready is queued, but the consumer resumes after its deadline.
        runner._model_started_at = time.monotonic() - runner.config.preparation_seconds - 1
        runner._poll_preparation()
        assert runner.projection()["model_state"] == "failed"
        assert runner.projection()["error"] == "guidance_preparation_timeout"
        assert runner._process is None
        assert not (tmp_path / "observed.txt").exists()
    finally:
        runner.close()


def test_blocked_ready_task_expires_without_reloading_or_retrying_the_same_scene(tmp_path):
    runner = service(tmp_path, block_observation=True)
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] == "ready")
        for number in range(4):
            runner.observe(frame(number), context="solid")
        wait_for(lambda: (tmp_path / "observed.txt").exists())
        pid = runner.active_pid
        assert pid is not None
        time.sleep(0.2)
        runner.observe(frame(4), context="solid")
        assert runner.projection()["error"] == "guidance_task_timeout"
        assert runner.active_pid is None
        for number in range(5, 12):
            runner.observe(frame(number), context="solid")
        assert runner.projection()["tasks_started"] == 1
        assert runner.projection()["recommendation"] is None
        assert (tmp_path / "observed.txt").read_text().splitlines() == ["2"]
        assert runner.projection()["worker_pid"] == pid
        assert runner._process.is_alive()
        assert runner.projection()["model_loads"] == 1
    finally:
        runner.close()
    assert runner.cleanup_complete


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), 10**400])
def test_cold_hard_guard_must_remain_positive_and_finite(seconds):
    with pytest.raises(ValueError, match="timing"):
        SnapshotConfig(preparation_seconds=seconds)


def test_idle_worker_exit_revokes_visible_advice_and_pending_actual_review(tmp_path):
    runner = service(tmp_path)
    runner.start()
    try:
        wait_for(lambda: runner.projection()["model_state"] == "ready")
        with runner._lock:
            runner._recommendation = {"source_frame_id": 0, "route": {}, "region": None}
            runner._phase = "ready"
            runner._pending_review = 0
            runner._process.terminate()
        wait_for(lambda: runner.projection()["model_loads"] == 2)
        wait_for(lambda: runner.projection()["model_state"] == "ready")
        assert runner.projection()["error"] == "guidance_idle_worker_exit"
        assert runner.projection()["recommendation"] is None
        assert runner._pending_review is None
    finally:
        runner.close()
    assert runner.cleanup_complete


def test_failed_cleanup_keeps_exact_owner_but_revokes_old_advice(tmp_path):
    from types import SimpleNamespace

    runner = service(tmp_path)
    alive = True
    process = SimpleNamespace(
        pid=17,
        is_alive=lambda: alive,
        terminate=lambda: None,
        kill=lambda: None,
        join=lambda _=None: None,
        close=lambda: None,
    )
    connection = SimpleNamespace(close=lambda: None)
    runner._process, runner._connection = process, connection
    runner._busy = True
    runner._ticket = "owned-ticket"
    runner._recommendation = {"route": {}, "region": None}
    runner._pending_review = 0
    try:
        with pytest.raises(RuntimeError, match="did not terminate"):
            runner._fail_preparation("controlled cleanup failure")
        assert runner._process is process
        assert runner._connection is connection
        assert runner._ticket == "owned-ticket"
        assert runner.projection()["recommendation"] is None
        assert runner._pending_review is None
        assert not runner.cleanup_complete
    finally:
        alive = False
        runner.close()
    assert runner.cleanup_complete
