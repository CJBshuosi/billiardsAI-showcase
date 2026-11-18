"""Shutdown revokes authority immediately and retains the same cleanup owner."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.recording import ShotTimelineIndex
from billiards_ai.edge.route_assistance.records import FieldSessionStatus
from billiards_ai.lab.field_assistance import server as service
from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource
from billiards_ai.lab.mainline1_product_realtime_runtime import ProductRealtimeEvent
from billiards_ai.lab.mainline1_shot_jobs import ShotAnalysisJob


def launch(call):
    errors = []

    def run():
        try:
            call()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, errors


def make_state(tmp_path, factory):
    store = SQLiteFieldRecordStore(tmp_path / "field.sqlite")
    session = store.open_or_resume_session(artifact_id="artifact")
    lock = threading.RLock()
    source = USBLiveProjectionSource(
        store=store,
        session_id=session.session_id,
        source_epoch_id="epoch-original",
        calibration_id="cal",
        runtime_factory=factory,
        match_lock=lock,
    )
    state = service._LiveServerState(
        artifact_path=tmp_path,
        artifact_id="artifact",
        control_token="secret",
        replay_clock=None,
        match=session.match,
        match_lock=lock,
        frame_store=source,
        record_store=store,
        session_id=session.session_id,
        shot_index=ShotTimelineIndex(()),
        recommendation_power={},
        duration_ms=0,
        projection_source=source,
    )
    return state


class CancelFailureRuntime:
    def __init__(self, *, runner_failure=None):
        self.running = threading.Event()
        self.cancelled = threading.Event()
        self.runner_failure = runner_failure
        self.close_failure = None

    def run(self):
        self.running.set()
        assert self.cancelled.wait(2)
        if self.runner_failure is not None:
            raise self.runner_failure

    def close(self):
        self.cancelled.set()
        if self.close_failure is not None:
            raise self.close_failure


@pytest.mark.parametrize("native_failure", [False, True])
@pytest.mark.parametrize("queued_boundaries", [False, True])
def test_lock_owning_close_preserves_runner_failure_and_queued_receipt(
    tmp_path,
    monkeypatch,
    native_failure,
    queued_boundaries,
):
    """A runner waiting on the caller's match lock must not deadlock its own join."""
    from billiards_ai.lab.mainline1_native_capture import NativeCaptureProtocolError

    failure = (NativeCaptureProtocolError if native_failure else ValueError)(
        "original runner failure"
    )
    runtime = CancelFailureRuntime(runner_failure=failure)
    state = make_state(tmp_path, lambda *args: runtime)
    source = state.projection_source
    # The immutable enqueue receipt arrives during cancellation, after consumption stops.
    if not queued_boundaries:
        source.coordinator.shot_started(shot_id="shot-1", started_ms=10)
    job = ShotAnalysisJob(
        "job-1",
        state.session_id,
        "shot-1",
        "epoch-original",
        1,
        2,
        tmp_path / "media",
        "a" * 64,
        "setup",
        "cal",
        "b" * 64,
        "model",
    )
    original_close = runtime.close

    def cancel():
        if not runtime.cancelled.is_set():
            if queued_boundaries:
                from billiards_ai.lab.mainline1_product_realtime_runtime import LiveSettledShot

                source.channel.publish(
                    ProductRealtimeEvent(
                        "SHOT_STARTED",
                        shot_id="shot-1",
                        frame_id=1,
                        started_ms=10,
                    )
                )
                source.channel.publish(
                    ProductRealtimeEvent(
                        "SHOT_SETTLED",
                        shot_id="shot-1",
                        settled=LiveSettledShot(
                            "shot-1",
                            1,
                            2,
                            20_000_000,
                            (),
                            False,
                            "unknown",
                            {},
                            "epoch-original",
                            "job-1",
                        ),
                    )
                )
            source.channel.publish(ProductRealtimeEvent("SHOT_ENQUEUED", shot_id="shot-1", job=job))
        original_close()

    runtime.close = cancel
    original_join = source._runner.join
    monkeypatch.setattr(source._runner, "join", lambda timeout: original_join(min(timeout, 0.2)))
    source.start()
    assert runtime.running.wait(1)
    try:
        with state.match_lock:
            source.close()
        assert not source._runner.is_alive()
        assert "original runner failure" in source.failure_reason
        assert source._runtime_failure is failure
        assert source.current().frame.payload["rule_status"] == "DEGRADED"
        assert not source.poll_capture_ready() and not source.ready_to_end
        assert not source.channel.can_start_shots()
        media = state.record_store.get_live_shot_media(state.session_id, "shot-1")
        assert (media.job_id, media.source_epoch_id, media.match_revision) == (
            "job-1",
            "epoch-original",
            0,
        )
        if queued_boundaries:
            shots = state.record_store.list_shots(state.session_id)
            assert len(shots) == 1 and shots[0].settled_ms == 20
        generation = source._generation
        source.close()
        assert source._generation == generation
        assert source.channel.pending_events == 0
    finally:
        runtime.cancelled.set()
        source._runner.join(1)
        source._consumer.join(1)


def test_projection_close_failure_stays_pending_and_preserves_original_group(tmp_path):
    runtime = CancelFailureRuntime()
    state = make_state(tmp_path, lambda *args: runtime)
    source = state.projection_source
    failure = ExceptionGroup(
        "original cleanup failures", [OSError("cancel"), RuntimeError("writer")]
    )
    runtime.close_failure = failure
    with pytest.raises(BaseExceptionGroup) as caught:
        source.close()
    assert caught.value is failure
    assert not source._closed
    assert not source.ready_to_end
    runtime.close_failure = None
    source.close()
    assert source._closed
    assert not source.ready_to_end  # Cleanup retry does not erase the original authority failure.


def test_service_shutdown_retries_same_source_outside_match_lock_and_revokes_commands(tmp_path):
    state = make_state(tmp_path, lambda *args: CancelFailureRuntime())
    source = state.projection_source
    entered, release = threading.Event(), threading.Event()
    attempts = []
    failure = RuntimeError("physical close pending")

    def shutdown():
        attempts.append(source)
        entered.set()
        assert release.wait(2)
        if len(attempts) == 1:
            raise failure

    source.shutdown = shutdown
    first, first_errors = launch(state.close_projection)
    assert entered.wait(1)
    second, second_errors = launch(state.close_projection)
    try:
        assert state.match_lock.acquire(timeout=0.2), "blocking shutdown holds the match lock"
        try:
            assert not state.projection_lifecycle_closed
            state.maintain_service()
            state._reset_projection()
            state.maintain_background()
            for command in ("end_session", "emergency_set_breaker"):
                with pytest.raises(ValueError, match="LIVE_AUTHORITY_UNAVAILABLE"):
                    state.mutate({"command": command})
            envelope = FieldCommandEnvelope(
                "cmd-1", state.session_id, "alice", "end_session", {}, 0, 1
            )
            completed = []
            gateway = SimpleNamespace(
                claim_field_command=lambda: envelope, complete_field_command=completed.append
            )
            worker = service._BackgroundFieldCommandWorker(
                gateway=gateway,
                store=state.record_store,
                match_lock=state.match_lock,
                before_application=state.before_field_command,
                command_reducer=state.reduce_live_command,
            )
            assert worker.claim_and_apply_once() is False
            assert completed == []
            assert state.record_store.is_new_ordered_field_command(envelope)
            assert (
                state.record_store.load_session(state.session_id).status is FieldSessionStatus.OPEN
            )
        finally:
            state.match_lock.release()
    finally:
        release.set()
        first.join(1)
        second.join(1)
    assert not first.is_alive() and not second.is_alive()
    assert first_errors == [failure] and second_errors == []
    assert attempts == [source, source]
    assert state.projection_source is source
    assert state.projection_lifecycle_closed


def test_service_pending_close_revokes_source_readiness_and_shot_admission(tmp_path):
    """The gap before blocking shutdown starts cannot grant new live authority."""
    from billiards_ai.edge.route_assistance.control import MatchControlSnapshot, WristbandRole

    state = make_state(tmp_path, lambda *args: CancelFailureRuntime())
    source = state.projection_source
    state.record_store.save_match(
        state.session_id,
        MatchControlSnapshot(
            red_user_id="alice",
            blue_user_id="bob",
            breaker=WristbandRole.RED,
            active_role=WristbandRole.RED,
            revision=1,
        ),
        expected_revision=0,
    )
    assert source.channel.can_start_shots()
    source._validated_ready = True
    failure = RuntimeError("physical cleanup failed before returning")
    source.shutdown = lambda: (_ for _ in ()).throw(failure)
    with pytest.raises(RuntimeError) as caught:
        state.close_projection()
    assert caught.value is failure
    assert state.projection_lifecycle_closing
    assert not state.projection_lifecycle_closed
    assert source._stopped.is_set()
    assert not source.poll_capture_ready()
    assert not source.channel.can_start_shots()
    assert not source.current().playing


def test_shutdown_collects_original_analysis_and_physical_failures_then_retries(tmp_path):
    runtime = CancelFailureRuntime()
    state = make_state(tmp_path, lambda *args: runtime)
    source = state.projection_source
    analysis = OSError("analysis cancellation")
    physical = ExceptionGroup("physical owners", [RuntimeError("decoder"), OSError("writer")])
    runtime.close_failure = analysis
    physical_attempts = []

    def shutdown():
        physical_attempts.append(1)
        if len(physical_attempts) == 1:
            raise physical

    runtime.shutdown = shutdown
    with pytest.raises(BaseExceptionGroup) as caught:
        source.shutdown()
    assert caught.value.exceptions == (analysis, physical)
    assert not source._closed and not source._shutdown_complete
    runtime.close_failure = None
    source.shutdown()
    source.shutdown()
    assert physical_attempts == [1, 1]
    assert source._closed and source._shutdown_complete


@pytest.mark.parametrize("startup", ["cancel", "source_failure", "ready"])
def test_actual_shared_factory_projection_service_owned_retry_retains_session(
    tmp_path,
    monkeypatch,
    startup,
):
    """Real factory/hub/channel/store; only model construction is controlled."""
    from billiards_ai.lab import mainline1_field_runtime as assembly
    from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame, RealtimeCaptureConfig

    entered, first_frame, release_worker = (threading.Event() for _ in range(3))
    models = []
    captures = []

    class RawCapture:
        def __init__(self, epoch):
            self.epoch = epoch
            self.cancelled = threading.Event()
            self.fail = OSError("raw source startup failed")

        def frames(self, stop):
            entered.set()
            assert first_frame.wait(2)
            if startup == "source_failure":
                raise self.fail
            if startup == "ready":
                yield CaptureFrame(self.epoch, 0, 17, 16, 12, b"raw-jpeg")
            assert release_worker.wait(2)

        def cancel(self):
            self.cancelled.set()

    def capture_factory(epoch):
        source = RawCapture(epoch)
        captures.append(source)
        return source

    class ControlledModelRuntime:
        def __init__(self, **kwargs):
            self.source = kwargs["source"]
            models.append(kwargs)

        def run(self):
            for _ in self.source.frames(threading.Event()):
                pass

    monkeypatch.setattr(assembly, "ProductRealtimeRuntime", ControlledModelRuntime)
    config = tmp_path / "config"
    config.write_text("controlled configuration")
    factory = assembly.build_live_runtime_factory(
        capture_factory=capture_factory,
        capture_config=RealtimeCaptureConfig(
            width=16, height=12, fps=2, startup_timeout_seconds=0.3, cancel_timeout_seconds=0.01
        ),
        calibration=None,
        detector=None,
        identity_classifier=None,
        shot_config=None,
        capture_setup_id="setup",
        calibration_artifact_id="cal",
        config_path=config,
        ring_root=tmp_path / "ring",
        jobs_root=tmp_path / "jobs",
    )
    server = service.build_live_field_assistance_server(
        artifact_path=tmp_path / "artifacts",
        state_db_path=tmp_path / "field.sqlite",
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=lambda: "epoch-original",
        calibration_id="cal",
        runtime_factory=factory,
    )
    state = server.field_state
    source = state.projection_source
    session_id = state.session_id
    hub = source.runtime.capture_hub
    assert entered.wait(1)
    try:
        if startup == "cancel":
            source.runtime.close()
            first_frame.set()
        else:
            first_frame.set()
        if startup == "ready":
            assert state.wait_until_capture_ready(timeout_seconds=1) == "epoch-original"
            assert len(models) == 1
            assert models[0]["event_channel"] is source.channel
            assert hub.latest_frame().mjpeg == b"raw-jpeg"
            source.close()
            assert source._closed and not source._shutdown_complete
            assert hub._thread.is_alive()
            assert not captures[0].cancelled.is_set()
        else:
            assert not source.wait_until_ready(timeout_seconds=1)
            assert models == []
            if startup == "source_failure":
                # A failed hub revokes readiness before the runner has necessarily
                # collected the original startup error. Observe that completion
                # before issuing a separate cancellation request.
                assert source._runtime_exited.wait(1)
        if startup != "source_failure":
            with pytest.raises(Exception, match="did not stop"):
                server.server_close()
            assert not state.projection_lifecycle_closed
            assert hub._thread.is_alive()
            assert state.projection_source is source
        release_worker.set()
        server.server_close()
        server.server_close()
        assert state.projection_lifecycle_closed
        assert source._closed and source._shutdown_complete
        assert not hub._thread.is_alive()
        assert state.projection_source is source
        assert (state.session_id, source.session_id, source.source_epoch_id) == (
            session_id,
            session_id,
            "epoch-original",
        )
        assert state.record_store.load_session(session_id).status is FieldSessionStatus.OPEN
        assert state.record_store.list_shots(session_id) == ()
        assert len(captures) == 1
        assert captures[0].cancelled.is_set()
        if startup == "source_failure":
            assert "raw source startup failed" in source.failure_reason
            assert source._runtime_failure.__cause__ is captures[0].fail
    finally:
        first_frame.set()
        release_worker.set()
        server.server_close()


def test_rollover_close_failure_retains_original_service_session_and_source(tmp_path):
    """A failed same-camera close cannot advance the service to another match."""
    runtime = CancelFailureRuntime()
    state = make_state(tmp_path, lambda *args: runtime)
    source = state.projection_source
    session_id, match = state.session_id, state.match
    state.record_store.end_session(session_id)
    state.session_ended_at_monotonic = 0
    state.monotonic = lambda: 10
    runtime.close_failure = OSError("rollover cleanup pending")
    with pytest.raises(OSError, match="rollover cleanup pending"):
        with state.match_lock:
            state._rollover_ended_session(state.record_store.load_session(session_id))
    assert state.session_id == session_id
    assert state.match == match
    assert state.projection_source is source
    assert source.session_id == session_id
    assert source.source_epoch_id == "epoch-original"
    assert not source._closed


@pytest.mark.parametrize("completion", ["maintenance", "http", "shutdown"])
def test_shared_recorder_rollover_retains_service_target_until_handoff_completes(
    tmp_path,
    monkeypatch,
    completion,
):
    """A recorder timeout must not publish half a session or strand HTTP retry."""
    from tests.unit.lab.test_mainline1_field_runtime import (
        _close_test_writers,
        _handoff_factory,
    )

    factory, writers, captures, models = _handoff_factory(tmp_path, monkeypatch)
    state = make_state(tmp_path, factory)
    original = state.projection_source
    hub = original.runtime.capture_hub
    epoch = original.source_epoch_id
    attempts, candidates, allocations = [], [], []
    open_session = state.record_store.open_or_resume_session

    def allocate(**kwargs):
        session = open_session(**kwargs)
        allocations.append(session.session_id)
        return session

    def source_factory(session_id, *, recover_capture=False):
        attempts.append((session_id, recover_capture))
        candidate = USBLiveProjectionSource(
            store=state.record_store,
            session_id=session_id,
            source_epoch_id=epoch,
            calibration_id="cal",
            runtime_factory=factory,
            match_lock=state.match_lock,
        )
        candidates.append(candidate)
        start = candidate.start

        def start_before_publication():
            assert state.session_id == original.session_id
            assert state.projection_source is state.frame_store is original
            start()

        candidate.start = start_before_publication
        return candidate

    monkeypatch.setattr(state.record_store, "open_or_resume_session", allocate)
    state.live_source_factory = source_factory
    state.capture_recovery_attempts = 0
    state.capture_recovery_not_before = 0
    clock = [0]
    state.monotonic = lambda: clock[0]
    command = FieldCommandEnvelope(
        "claim-original",
        state.session_id,
        "alice",
        "claim_role",
        {"role": "red"},
        0,
        1,
    )
    applied = state.record_store.apply_field_command(command, service.reduce_field_command)
    state.match = state.record_store.load_session(state.session_id).match
    # Use the actual local end command and its durable store commit.
    state.mutate({"command": "end_session"})
    original_session, original_match = state.session_id, state.match
    clock[0] = 10
    entered, release = threading.Event(), threading.Event()
    old = writers[0]
    ring_close, writer_close = old._ring.close, old.close
    closes = []

    def blocked_close():
        closes.append(True)
        entered.set()
        assert release.wait(5)
        ring_close()

    monkeypatch.setattr(old._ring, "close", blocked_close)
    monkeypatch.setattr(old, "close", lambda: writer_close(timeout_seconds=0.01))
    try:
        payload = state.payload()
        assert entered.wait(1)
        assert state.session_id == original_session
        assert state.match == original_match
        assert state.projection_source is state.frame_store is original
        assert payload["session"]["session_id"] == payload["frame"]["session_id"]
        assert payload["display"]["phase"] == "DEGRADED"
        assert not getattr(state, "_projection_store_failed", False)
        assert len(allocations) == 1 and len(writers) == 2
        target = allocations[0]
        assert state.record_store.load_session(target).status is FieldSessionStatus.OPEN
        assert hub._recorder is writers[1]
        state.maintain_service()
        assert state.session_id == original_session
        assert len(allocations) == 1 and len(writers) == 2 and closes == [True]
        if completion == "shutdown":
            with pytest.raises(Exception, match="close exceeded timeout"):
                state.close_projection()
            assert not state.projection_lifecycle_closed
        release.set()
        old._thread.join(1)
        assert not old._thread.is_alive()
        if completion == "shutdown":
            state.close_projection()
            assert state.projection_lifecycle_closed
            assert state.session_id == original_session
            assert candidates == []
        else:
            if completion == "maintenance":
                state.maintain_service()
            else:
                payload = state.payload()
                assert payload["session"]["session_id"] == payload["frame"]["session_id"] == target
            replacement = state.projection_source
            assert replacement is state.frame_store is candidates[0]
            assert replacement is not original
            assert replacement.session_id == state.session_id == target
            assert replacement.source_epoch_id == epoch
            assert state.match == state.record_store.load_session(target).match
            assert replacement.runtime.capture_hub is hub
            # No first frame was supplied: publishing the context grants no READY.
            assert not replacement.poll_capture_ready()
            replacement.runtime.runtime_factory(replacement.runtime._analysis_source)
            assert models[-1]["session_id"] == target
            assert models[-1]["boundary_ledger"].session_id == target
            assert models[-1]["boundary_ledger"].source_epoch_id == epoch
            assert models[-1]["evidence_store"].delegate is writers[1]
            state.maintain_service()
            current = state.record_store.load_session(target)
            assert state._rollover_ended_session(current)[1] is False
        assert all(attempt == (target, False) for attempt in attempts)
        assert len(allocations) == 1 and len(writers) == 2 and len(captures) == 1
        assert closes == [True]
        assert state.record_store.list_shots(original_session) == ()
        assert state.record_store.list_shots(target) == ()
        assert state.record_store.get_field_command_application(command) == applied
        with state.record_store._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM field_applied_commands").fetchone()[0]
            assert count == 1
    finally:
        release.set()
        monkeypatch.setattr(old, "close", lambda: writer_close(timeout_seconds=1))
        state.close_projection()
        for candidate in candidates:
            candidate.shutdown()
        _close_test_writers(writers)
        assert hub._thread is None or not hub._thread.is_alive()
        assert all(not writer._thread.is_alive() for writer in writers)
        assert all(
            not item._runner.is_alive() and not item._consumer.is_alive()
            for item in [original, *candidates]
        )


@pytest.mark.parametrize(
    "marker",
    [
        None,
        ("wrong-session", "epoch-original", 0),
        ("target", "wrong-epoch", 0),
        ("target", "epoch-original", 1),
    ],
)
def test_unproven_rollover_construction_failure_is_terminal(tmp_path, marker):
    """Only exact producer proof permits retry; other failures keep their identity."""
    state = make_state(tmp_path, lambda *args: CancelFailureRuntime())
    original = state.projection_source
    state.monotonic = lambda: 0
    state.mutate({"command": "end_session"})
    state.monotonic = lambda: 10
    failure = OSError("constructor failed")
    attempts = []

    def fail(session_id, *, recover_capture=False):
        attempts.append(session_id)
        if marker is not None:
            failure._pending_recorder_handoff_target = (
                session_id if marker[0] == "target" else marker[0],
                *marker[1:],
            )
        raise failure

    state.live_source_factory = fail
    try:
        assert state.payload()["display"]["phase"] == "DEGRADED"
        assert state._projection_store_failed
        assert state._rollover_failure is failure
        assert state.projection_source is original
        assert state.session_id == original.session_id
        state.maintain_service()
        state.payload()
        assert len(attempts) == 1
    finally:
        state.close_projection()


def test_rollover_start_failure_keeps_original_context_and_pending_cleanup(tmp_path):
    """A partially started candidate cannot be published or overwritten by retry."""
    state = make_state(tmp_path, lambda *args: CancelFailureRuntime())
    original = state.projection_source
    state.monotonic = lambda: 0
    state.mutate({"command": "end_session"})
    state.monotonic = lambda: 10
    primary = RuntimeError("start failed after consumer launch")
    cleanup_failure = OSError("candidate cancellation pending")
    candidates = []
    runtime = CancelFailureRuntime()
    runtime.close_failure = cleanup_failure

    def factory(session_id, *, recover_capture=False):
        candidate = USBLiveProjectionSource(
            store=state.record_store,
            session_id=session_id,
            source_epoch_id="epoch-original",
            calibration_id="cal",
            runtime_factory=lambda *args: runtime,
            match_lock=state.match_lock,
        )
        candidates.append(candidate)

        def start():
            # Exercise the real partially started consumer's cleanup.
            candidate._consumer.start()
            raise primary

        candidate.start = start
        return candidate

    state.live_source_factory = factory
    try:
        state.payload()
        candidate = candidates[0]
        assert state.projection_source is state.frame_store is original
        assert state.session_id == original.session_id
        assert state._rollover_failure is primary
        assert state._pending_projection_cleanup is candidate
        assert not candidate.poll_capture_ready()
        state.maintain_service()
        with pytest.raises(RuntimeError) as caught:
            state._reset_projection(session_id=candidate.session_id)
        assert caught.value is primary
        assert len(candidates) == 1
        with pytest.raises(OSError) as caught:
            state.close_projection()
        assert caught.value is cleanup_failure
        assert not state.projection_lifecycle_closed
        assert state._pending_projection_cleanup is candidate
        runtime.close_failure = None
        state.close_projection()
        assert state.projection_lifecycle_closed
        assert state._rollover_failure is primary
    finally:
        runtime.close_failure = None
        state.close_projection()
        for candidate in candidates:
            candidate.shutdown()
        assert all(
            not item._consumer.is_alive() and not item._runner.is_alive()
            for item in [original, *candidates]
        )


@pytest.mark.parametrize("stage", ["construction", "start", "start_and_listener"])
def test_actual_builder_preserves_primary_and_reports_each_cleanup_failure(
    tmp_path,
    monkeypatch,
    capsys,
    stage,
):
    """Builder cleanup must not replace the exception delivered to CLI main."""
    import socket

    primary = RuntimeError(f"original {stage} failure")
    original_cause = LookupError("original primary cause")
    primary.__cause__ = original_cause
    cancellation = OSError("analysis cancellation failed")
    prepared_failure = OSError("prepared cancellation failed")
    prepared_pending = [True]
    attempts, sources, addresses = [], [], []
    runtime = CancelFailureRuntime()
    runtime.close_failure = cancellation
    original_cancel = runtime.close

    def close():
        attempts.append("analysis")
        original_cancel()

    runtime.close = close

    def cancel_prepared():
        attempts.append("prepared")
        if prepared_pending[0]:
            raise prepared_failure

    prepared = SimpleNamespace(cancel=cancel_prepared)
    original_init = service.FieldAssistanceHTTPServer.__init__

    def capture_listener(server, *args, **kwargs):
        try:
            original_init(server, *args, **kwargs)
        finally:
            addresses.append(server.server_address)

    monkeypatch.setattr(service.FieldAssistanceHTTPServer, "__init__", capture_listener)
    if stage == "start_and_listener":
        listener_failure = OSError("listener cleanup failed")
        original_listener_close = service.ThreadingHTTPServer.server_close

        def fail_listener_close(server):
            original_listener_close(server)
            raise listener_failure

        monkeypatch.setattr(service.ThreadingHTTPServer, "server_close", fail_listener_close)
    if stage == "construction":
        original_post_init = service._LiveServerState.__post_init__

        def fail_construction(state):
            original_post_init(state)
            sources.append(state.projection_source)
            raise primary

        monkeypatch.setattr(service._LiveServerState, "__post_init__", fail_construction)
    else:
        original_start = USBLiveProjectionSource.start

        def fail_start(source):
            sources.append(source)
            original_start(source)
            assert runtime.running.wait(1)
            raise primary

        monkeypatch.setattr(USBLiveProjectionSource, "start", fail_start)
    with pytest.raises(RuntimeError) as caught:
        service.build_live_field_assistance_server(
            artifact_path=tmp_path / "artifacts",
            state_db_path=tmp_path / "field.sqlite",
            host="127.0.0.1",
            port=0,
            control_token="secret",
            source_epoch_factory=lambda: "epoch-original",
            calibration_id=None,
            runtime_factory=None,
            runtime_preparer=lambda: ("cal", lambda *args: runtime, prepared),
        )
    assert caught.value is primary
    assert primary.__cause__ is original_cause
    assert "analysis" in attempts
    if stage == "construction":
        assert attempts[-1] == "prepared"
    errors = capsys.readouterr().err
    assert "startup cleanup failed" in errors
    assert "analysis cancellation failed" in errors
    if stage == "start_and_listener":
        assert "listener cleanup failed" in errors
        group = primary._field_cleanup_failures[-1][1]
        assert group.exceptions == (cancellation, listener_failure)
    if stage == "construction":
        assert "prepared cancellation failed" in errors
    retained = primary._field_cleanup_failures
    assert retained
    assert all(isinstance(owner, str) for owner, _ in retained)
    # The listener is released even when both primary and capture cleanup fail.
    with socket.socket() as listener:
        listener.bind(addresses[-1])
    for source in sources:
        assert not source._runner.is_alive() and not source._consumer.is_alive()
    runtime.close_failure = None
    prepared_pending[0] = False
    if stage == "start_and_listener":
        monkeypatch.setattr(service.ThreadingHTTPServer, "server_close", original_listener_close)
    service._retry_failed_startup_cleanup(primary)
    assert primary._field_pending_cleanup == ()
    assert primary.__cause__ is original_cause
    assert all(source._shutdown_complete for source in sources)


def test_builder_failure_with_pending_runner_retains_original_error(tmp_path, monkeypatch):
    """A started worker may outlive failed construction and its first cleanup."""
    release = threading.Event()
    runtime = CancelFailureRuntime()
    sources = []
    primary = RuntimeError("startup failed after runner started")

    def blocked_run():
        runtime.running.set()
        assert release.wait(2)

    runtime.run = blocked_run
    original_start = USBLiveProjectionSource.start

    def fail_start(source):
        sources.append(source)
        original_join = source._runner.join
        monkeypatch.setattr(
            source._runner, "join", lambda timeout: original_join(min(timeout, 0.02))
        )
        original_start(source)
        assert runtime.running.wait(1)
        raise primary

    monkeypatch.setattr(USBLiveProjectionSource, "start", fail_start)
    try:
        with pytest.raises(RuntimeError) as caught:
            service.build_live_field_assistance_server(
                artifact_path=tmp_path / "artifacts",
                state_db_path=tmp_path / "field.sqlite",
                host="127.0.0.1",
                port=0,
                control_token="secret",
                source_epoch_factory=lambda: "epoch-original",
                calibration_id="cal",
                runtime_factory=lambda *args: runtime,
            )
        assert caught.value is primary
        source = sources[0]
        assert source._runner.is_alive()
        assert not source._closed and not source._shutdown_complete
        assert "shutdown timed out" in str(primary._field_cleanup_failures[0][1])
        pending = primary._field_pending_cleanup
        original_errors = primary._field_cleanup_failures
        assert len(pending) == 1  # The partial server owns its source; no duplicate source owner.
        with pytest.raises(RuntimeError, match="shutdown timed out"):
            service._retry_failed_startup_cleanup(primary)
        assert primary._field_pending_cleanup == pending
        assert primary._field_pending_cleanup[0][1] is pending[0][1]
        assert primary._field_cleanup_failures == original_errors
        release.set()
        service._retry_failed_startup_cleanup(primary)
        assert primary._field_pending_cleanup == ()
        assert source._closed and source._shutdown_complete
    finally:
        release.set()
        for source in sources:
            source._runner.join(1)
            source.shutdown()


def test_closed_service_replays_only_exact_durable_command_receipt(tmp_path):
    """An old receipt may be ACKed without replaying projection mutations."""
    from dataclasses import replace

    state = make_state(tmp_path, lambda *args: CancelFailureRuntime())
    envelope = FieldCommandEnvelope(
        "durable-1", state.session_id, "alice", "claim_role", {"role": "red"}, 0, 1
    )
    current = [envelope]
    completed, observations = [], []
    gateway = SimpleNamespace(
        claim_field_command=lambda: current[0], complete_field_command=completed.append
    )
    state.scenario_playback_observer = SimpleNamespace(
        observe=lambda *args: observations.append(args)
    )
    worker = service._BackgroundFieldCommandWorker(
        gateway=gateway,
        store=state.record_store,
        match_lock=state.match_lock,
        before_application=state.before_field_command,
        match_changed=state.set_command_match,
        application_changed=state.handle_command_application,
        command_reducer=state.reduce_live_command,
    )
    assert worker.claim_and_apply_once(raise_errors=True)
    original = completed[0]
    assert original.outcome == "applied" and original.resulting_revision == 1
    assert len(observations) == 1
    state.close_projection()
    assert worker.claim_and_apply_once(raise_errors=True)
    assert completed == [original, original]
    assert len(observations) == 1
    for altered in (
        replace(envelope, payload={"role": "blue"}),
        replace(envelope, command_id="new-2", expected_revision=1, server_sequence=2),
    ):
        current[0] = altered
        assert worker.claim_and_apply_once() is False
        assert completed == [original, original]
    assert state.record_store.load_session(state.session_id).match.revision == 1


@pytest.mark.parametrize("with_reference", [False, True])
def test_receipt_lookup_is_exact_read_only_and_visible_through_wal(
    tmp_path,
    monkeypatch,
    with_reference,
):
    import sqlite3
    from dataclasses import replace

    from billiards_ai.edge.route_assistance import persistence
    from billiards_ai.edge.route_assistance.commands import reduce_field_command
    from billiards_ai.edge.route_assistance.turn_confirmation import TurnConfirmationRef

    store = SQLiteFieldRecordStore(
        tmp_path / "receipt ?#.sqlite", session_id_factory=lambda: "session-ledger"
    )
    store.open_or_resume_session(artifact_id="artifact")
    envelope = FieldCommandEnvelope(
        "command-ledger", "session-ledger", "alice", "claim_role", {"role": "red"}, 0, 1
    )
    reference = TurnConfirmationRef("session-ledger", "shot-1", "d" * 64, 0)
    active_reference = reference if with_reference else None
    original = store.apply_field_command(
        envelope, reduce_field_command, confirmation_ref=active_reference
    )
    connections, statements = [], []
    real_connect = sqlite3.connect
    writer = store._connect()
    writer.execute("BEGIN IMMEDIATE")
    before = tuple(writer.execute("SELECT * FROM field_applied_commands").fetchone())
    expected_digest = (
        "73019aa68f1dfaeaee46d7b008635cba411d9f57f5a70ca658713ae9226bc002"
        if with_reference
        else "c39af9098e5531ccf2129e1317c079a38d39f61940f3a1afe08c82231dec3545"
    )
    assert (
        writer.execute("SELECT payload_digest FROM field_applied_commands").fetchone()[0]
        == expected_digest
    )

    def read_connection(database, *args, **kwargs):
        assert str(database).endswith("?mode=ro") and kwargs.get("uri") is True
        kwargs["timeout"] = 0.02
        connection = real_connect(database, *args, **kwargs)
        connection.set_trace_callback(statements.append)
        connections.append(connection)
        return connection

    try:
        with monkeypatch.context() as patch:
            patch.setattr(persistence.sqlite3, "connect", read_connection)
            assert (
                store.get_field_command_application(envelope, confirmation_ref=active_reference)
                == original
            )
            assert (
                store.get_field_command_application(
                    replace(envelope, command_id="absent"), confirmation_ref=active_reference
                )
                is None
            )
            assert (
                store.get_field_command_application(
                    replace(envelope, payload={"role": "blue"}), confirmation_ref=active_reference
                )
                is None
            )
            assert (
                store.get_field_command_application(
                    envelope, confirmation_ref=replace(reference, evidence_digest="e" * 64)
                )
                is None
            )
            assert (
                store.get_field_command_application(
                    envelope, confirmation_ref=None if with_reference else reference
                )
                is None
            )
        assert statements and all(
            statement.lstrip().upper().startswith("SELECT") for statement in statements
        )
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
        assert tuple(writer.execute("SELECT * FROM field_applied_commands").fetchone()) == before
        assert writer.execute("SELECT COUNT(*) FROM field_applied_commands").fetchone()[0] == 1
        assert store.load_session("session-ledger").match.revision == (0 if with_reference else 1)
    finally:
        writer.rollback()
        writer.close()


def test_pending_cleanup_drain_removes_only_completed_original_workers():
    """A successful phase cannot erase another still-live owner's retry path."""

    class Worker:
        def __init__(self):
            self.release = threading.Event()
            self.thread = threading.Thread(target=self.release.wait, daemon=True)
            self.thread.start()

        def close(self):
            self.thread.join(0.01)
            if self.thread.is_alive():
                raise TimeoutError("owned worker still alive")

    first, second = Worker(), Worker()
    first_close, second_close = first.close, second.close
    primary = RuntimeError("original startup error")
    try:
        service._cleanup_failed_startup(primary, [("first", first_close), ("second", second_close)])
        original_errors = primary._field_cleanup_failures
        assert primary._field_pending_cleanup == (("first", first_close), ("second", second_close))
        second.release.set()
        second.thread.join(1)
        with pytest.raises(TimeoutError, match="owned worker still alive"):
            service._retry_failed_startup_cleanup(primary)
        assert first.thread.is_alive() and not second.thread.is_alive()
        assert primary._field_pending_cleanup == (("first", first_close),)
        assert primary._field_pending_cleanup[0][1] is first_close
        assert primary._field_cleanup_failures == original_errors
        first.release.set()
        first.thread.join(1)
        service._retry_failed_startup_cleanup(primary)
        service._retry_failed_startup_cleanup(primary)
        assert primary._field_pending_cleanup == ()
        assert not first.thread.is_alive()
    finally:
        first.release.set()
        second.release.set()
        first.thread.join(1)
        second.thread.join(1)


def test_runner_failure_revokes_cached_readiness_without_consumer_lock(tmp_path):
    """Deferred failure collection cannot leave a cached READY source eligible."""
    runtime = CancelFailureRuntime(runner_failure=ValueError("runtime failed after READY"))
    state = make_state(tmp_path, lambda *args: runtime)
    source = state.projection_source
    try:
        with state.match_lock:
            source._validated_ready = True
            source.start()
            assert runtime.running.wait(1)
            runtime.cancelled.set()
            assert source._runtime_exited.wait(1)
            assert not source.poll_capture_ready()
            assert "runtime failed after READY" in source.failure_reason
    finally:
        source.close()


@pytest.mark.parametrize("service_revokes_first", [False, True])
def test_waiting_consumer_hands_queued_lineage_to_close_after_failure(
    tmp_path,
    service_revokes_first,
):
    """A consumer past its loop check cannot dequeue failed/revoked lineage."""
    from billiards_ai.lab.mainline1_product_realtime_runtime import LiveSettledShot

    runtime = CancelFailureRuntime(runner_failure=ValueError("original runner failure"))
    state = make_state(tmp_path, lambda *args: runtime)
    source = state.projection_source
    backing_lock = state.match_lock
    waiting, allow_consumer = threading.Event(), threading.Event()

    class HandoffLock:
        def acquire(self, *args, **kwargs):
            if threading.current_thread() is source._consumer:
                waiting.set()
                assert allow_consumer.wait(2)
            return backing_lock.acquire(*args, **kwargs)

        def release(self):
            return backing_lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *args):
            self.release()

    source.lock = state.match_lock = HandoffLock()
    job = ShotAnalysisJob(
        "job-1",
        state.session_id,
        "shot-1",
        "epoch-original",
        1,
        2,
        tmp_path / "media",
        "a" * 64,
        "setup",
        "cal",
        "b" * 64,
        "model",
    )
    original_shutdown = source.shutdown

    def shutdown_after_consumer_resumes():
        assert source._stopped.is_set()
        allow_consumer.set()
        source._consumer.join(1)
        assert not source._consumer.is_alive()
        original_shutdown()

    source.shutdown = shutdown_after_consumer_resumes
    try:
        with state.match_lock:
            source.start()
            assert waiting.wait(1)
            assert runtime.running.wait(1)
            source.channel.publish(
                ProductRealtimeEvent(
                    "SHOT_STARTED",
                    shot_id="shot-1",
                    frame_id=1,
                    started_ms=10,
                )
            )
            source.channel.publish(
                ProductRealtimeEvent(
                    "SHOT_SETTLED",
                    shot_id="shot-1",
                    settled=LiveSettledShot(
                        "shot-1",
                        1,
                        2,
                        20_000_000,
                        (),
                        False,
                        "unknown",
                        {},
                        "epoch-original",
                        "job-1",
                    ),
                )
            )
            source.channel.publish(ProductRealtimeEvent("SHOT_ENQUEUED", shot_id="shot-1", job=job))
            runtime.cancelled.set()
            assert source._runtime_exited.wait(1)
            assert "original runner failure" in source.failure_reason
        if not service_revokes_first:
            allow_consumer.set()
            source._consumer.join(1)
            assert not source._consumer.is_alive()
            assert source.channel.pending_events == 3
        state.close_projection()
        shots = state.record_store.list_shots(state.session_id)
        assert len(shots) == 1 and shots[0].settled_ms == 20
        media = state.record_store.get_live_shot_media(state.session_id, "shot-1")
        assert (media.job_id, media.source_epoch_id, media.match_revision) == (
            "job-1",
            "epoch-original",
            0,
        )
        assert source._runtime_failure is runtime.runner_failure
        assert source.channel.pending_events == 0
        assert state.projection_lifecycle_closed
        assert not source.ready_to_end and not source.poll_capture_ready()
        assert not source.channel.can_start_shots()
        assert source.current().frame.payload["rule_status"] == "DEGRADED"
        assert state.record_store.load_session(state.session_id).status is FieldSessionStatus.OPEN
        generation = source._generation
        state.close_projection()
        assert source._generation == generation
    finally:
        allow_consumer.set()
        runtime.cancelled.set()
        source._consumer.join(1)
        source._runner.join(1)
        source.close()


def test_retained_detector_failure_keeps_shared_capture_owned_until_retry(tmp_path):
    current, retained = CancelFailureRuntime(), CancelFailureRuntime()
    state = make_state(tmp_path, lambda *args: current)
    old = USBLiveProjectionSource(
        store=state.record_store,
        session_id=state.session_id,
        source_epoch_id="old-epoch",
        calibration_id="cal",
        runtime_factory=lambda *args: retained,
        match_lock=state.match_lock,
    )
    state._background_recordings.append(old)
    released = []
    current.shutdown = lambda: released.append("current")
    retained.shutdown = lambda: released.append("retained")
    failure = OSError("current detector cancellation pending")
    current.close_failure = failure
    try:
        with pytest.raises(OSError) as caught:
            state.close_projection()
        assert caught.value is failure
        assert released == [], "shared capture was released while a detector still owned it"
        assert not state.projection_lifecycle_closed
        assert state._background_recordings == [old]
        current.close_failure = None
        state.close_projection()
        assert released == ["current", "retained"]
        assert state.projection_lifecycle_closed
        assert state._background_recordings == []
    finally:
        current.close_failure = None
        state.close_projection()
