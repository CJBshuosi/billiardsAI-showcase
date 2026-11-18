"""Real managed CLI and 1080p pixels. External model assets/outputs are controlled."""

import hashlib
import json
import threading

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_runtime import inspect_development_vision_package
from scripts.algorithm import offline_managed_transport as protocol
from scripts.algorithm import serve_mainline1_blackbox_field as cli
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _EmptyClassifier,
    _EmptyDetector,
    _observe_runtime_publication,
)
from tests.unit.lab.field_assistance import test_runtime_resources as resource_fixtures
from tests.unit.scripts.test_run_offline_managed_blackbox import (
    arguments,
    launcher,
    prepare_file_setup,
)

assets = resource_fixtures.assets
_SHOT_SECONDS = 12


def prepare_inputs(assets, video, *, eof):
    prepare_file_setup(assets)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 60, (1920, 1080))
    try:
        assert writer.isOpened()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Non-EOF scenarios need a live camera throughout the shot budget,
        # including a late diagnostic poll on a slower host. Keep EOF explicit.
        for i in range(101 if eof else 60 * (_SHOT_SECONDS + 3)):
            x = 400 + min(30, max(0, i - 90)) * 16
            frame.fill(0)
            cv2.rectangle(frame, (x - 90, 450), (x + 90, 630), (180, 180, 180), -1)
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture
def late_supervisor(monkeypatch):
    from billiards_ai.lab.field_assistance import cloud_runtime as cloud

    driver = launcher()
    monkeypatch.setattr(driver, "_ACTIVE", threading.Lock())
    monkeypatch.setattr(driver, "_RETAINED", None, raising=False)
    entered, release = threading.Event(), threading.Event()
    references, launches = {}, []
    resolve = cloud.ManagedRuntimeResourceResolver.resolve
    init = cloud._BackgroundPostgameAnalysisWorker.__init__
    build, prepare = cli.build_live_field_assistance_server, cli.prepare_cloud_runtime

    def slow_resolver(owner, session, *, purpose):
        context = resolve(owner, session, purpose=purpose)
        if purpose == "media":
            entered.set()
            if not release.wait(30):
                raise RuntimeError("test resolver timed out")
        return context

    def intercept(cmd, **_):
        launches.append(cmd)
        raise OSError("test process boundary")

    def observe_worker(worker, **kwargs):
        init(worker, **kwargs, process_factory=intercept)
        references["worker"] = worker

    def observe_server(**kwargs):
        references["server"] = build(**kwargs)
        return references["server"]

    def observe_runtime(*args, **kwargs):
        references["runtime"] = prepare(*args, **kwargs)
        return references["runtime"]

    monkeypatch.setattr(cloud.ManagedRuntimeResourceResolver, "resolve", slow_resolver)
    monkeypatch.setattr(cloud._BackgroundPostgameAnalysisWorker, "__init__", observe_worker)
    monkeypatch.setattr(cli, "build_live_field_assistance_server", observe_server)
    monkeypatch.setattr(cli, "prepare_cloud_runtime", observe_runtime)
    try:
        yield entered, release, references, launches
    finally:
        release.set()
        if "worker" in references:
            references["worker"]._thread.join(5)
        if "runtime" in references:
            references["runtime"].stop()
        if "server" in references:
            references["server"].server_close()


@pytest.mark.parametrize("mode", ["settled", "eof", "stale", "late-stop", "delayed-observer"])
def test_real_cli_records_actual_stages_without_injection(
    tmp_path, monkeypatch, assets, mode, request
):
    driver = launcher()
    root = tmp_path.resolve() / "fresh"
    video = root.parent / "test.avi"
    prepare_inputs(assets, video, eof=mode == "eof")
    argv = arguments(root, assets, video)
    model_calls = []

    def features(**_):
        model_calls.append("features")
        return object()

    def detector(path, **_):
        model_calls.append("detector")
        result = _EmptyDetector()
        result.package_identity = inspect_development_vision_package(path).identity
        return result

    def classifier(*_, **kwargs):
        model_calls.append("classifier")
        return _EmptyClassifier()

    monkeypatch.setattr(cli.realtime, "Dinov2IdentityFeatureBackend", features)
    monkeypatch.setattr(cli.realtime, "load_development_vision_backend", detector)
    monkeypatch.setattr(cli.realtime, "load_development_group_classifier", classifier)
    if mode == "late-stop":
        entered, release, references, launches = request.getfixturevalue("late_supervisor")
    if mode == "delayed-observer":
        settled = driver._Actor.settled
        delayed = threading.Event()

        def delayed_probe(actor):
            if not delayed.is_set():
                delayed.set()
                threading.Event().wait(6.5)
            return settled(actor)

        monkeypatch.setattr(driver._Actor, "settled", delayed_probe)
    prepare = cli.prepare_runtime

    def observed_prepare(*args, **kwargs):
        model_calls.append("prepare")
        assert kwargs["resource_spec"] and not kwargs["playback_gate"].is_released()
        return prepare(*args, **kwargs)

    monkeypatch.setattr(cli, "prepare_runtime", observed_prepare)
    if mode == "stale":
        issue = protocol.OfflineManagedRemote.issue

        def issue_stale(self, **kwargs):
            if kwargs["command_type"] == "end_session":
                kwargs["expected_revision"] -= 1
            return issue(self, **kwargs)

        monkeypatch.setattr(protocol.OfflineManagedRemote, "issue", issue_stale)
    events, unauthorized = _observe_runtime_publication(monkeypatch)
    assert (
        driver.run_episode(
            argv,
            output_root=root,
            setup_seconds=12,
            shot_seconds=_SHOT_SECONDS,
            end_seconds=8,
            postgame_seconds=4,
        )
        != 0
    )
    report = json.loads((root / "evidence/episode.json").read_text())
    failure_context = {
        "failure": report["failure"],
        "source_sha256": report["capture"].get("source_sha256"),
        "events": [(event.kind, event.frame_id, event.reason) for event in events],
        "cleanup": report["cleanup"],
    }
    # Keep the complete failure reason: pytest abbreviates dict AssertionErrors.
    print("OFFLINE_EPISODE_CONTEXT " + json.dumps(failure_context, sort_keys=True))
    if mode == "late-stop":
        assert (
            not report["cleanup"]["workers_stopped"] and not report["cleanup"]["server_closed"]
        ), (failure_context, {"resolver_entered": entered.is_set()})
        assert entered.is_set() and references["worker"]._thread.is_alive()
        assert report["cleanup"]["cli_exit"] == 2 and report["cleanup"]["observer_joined"]
        assert driver._RETAINED[:2] == (references["runtime"], references["server"])
        # Reusing an existing root cannot launch another CLI even on the buggy driver.
        with pytest.raises(RuntimeError, match="exclusive main-thread ownership"):
            driver.run_episode(argv, output_root=root)
        release.set()
        references["worker"]._thread.join(3)
        assert not references["worker"]._thread.is_alive()
        assert not launches
        return
    assert model_calls == ["prepare", "features", "detector", "classifier"]
    assert not unauthorized
    assert report["setup"]["gate_released"] and report["setup"]["held_on_ready"]
    assert report["cleanup"]["workers_stopped"] and report["cleanup"]["server_closed"]
    assert report["remote"]["failure"] is None
    assert report["remote"]["birth_count"] == 1
    assert report["capture"]["width"] == 1920
    assert report["capture"]["source_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert report["formal_model_qualified"] is False
    assert report["complete_game_acceptance"] is False
    assert report["provider_delivery"] == "NOT_ASSERTED"
    assert report["cleanup"]["observer_joined"] and report["cleanup"]["cli_exit"] == 0
    if mode == "eof":
        assert report["remote"]["issued_count"] == 3
        assert report["failure"] == "CAPTURE_EOF" and not report["end"]
        assert report["capture"]["session_status"] == "open"
        return
    assert [e.kind for e in events][:3] == [
        "SHOT_STARTED",
        "SHOT_SETTLED",
        "SHOT_ENQUEUED",
    ], failure_context
    assert report["boundary"].get("projection_phase") == "stable", report["failure"]
    assert report["boundary"]["owner"] == "offline-red"
    assert report["boundary"]["rule_status"] == "abstained"
    assert report["media"]["job_id"] and report["remote"]["issued_count"] == 4
    assert report["remote"]["completed_prefix"] == 4
    if mode == "stale":
        assert report["end"]["result_code"] == "STALE_REVISION"
        assert report["capture"]["session_status"] == "open"
    else:
        assert report["end"]["result_code"] == "APPLIED"
        assert report["end"]["session_status"] == "ended"
        assert report["failure"] == "POSTGAME_TIMEOUT"
        assert report["postgame"]["available_count"] == 0
        assert report["postgame"]["review_session_id"] == report["session_id"]
        assert (root / "run/artifacts/postgame" / report["session_id"]).is_dir()
        assert not report["postgame"]["playable_clip_ids"]
