"""The whole-match observer must wait beyond a successful capture-stop response."""

import threading
import time
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab import mainline1_product_realtime_runtime as product
from billiards_ai.lab.mainline1_session_recording import ContinuousSessionRecorder
from scripts.algorithm import run_recording_first_fullgame as runner
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import _setup_players
from tests.integration.lab.field_assistance.test_recording_background_end import _end
from tests.integration.lab.field_assistance.test_recording_first_runtime import _recording_server
from tests.unit.lab.test_recording_windows_stat_identity import _windows_stat_views


@pytest.mark.parametrize("windows_stat_clocks", ["native", "raw_segment", "end_reference"])
def test_fullgame_observer_does_not_accept_empty_growing_inventory_as_finished(
    tmp_path, monkeypatch, windows_stat_clocks
):
    held, detect, paused, resume = (threading.Event() for _ in range(4))
    original = product.ProductRealtimeRuntime._handle_visual_boundary

    def delayed(runtime, frame, events):
        if frame.capture_frame_id == 0:
            held.set()
            assert detect.wait(15)
        original(runtime, frame, events)

    monkeypatch.setattr(product.ProductRealtimeRuntime, "_handle_visual_boundary", delayed)
    if windows_stat_clocks == "raw_segment":
        original_read = ContinuousSessionRecorder.read_segment
        installed = threading.Event()

        def first_windows_read(recorder, segment):
            if not installed.is_set():
                installed.set()
                _windows_stat_views(
                    monkeypatch, recorder._root / segment.name, mutate_after_read=False
                )
            return original_read(recorder, segment)

        monkeypatch.setattr(ContinuousSessionRecorder, "read_segment", first_windows_read)
    elif windows_stat_clocks == "end_reference":
        original_validate = SQLiteFieldRecordStore._validate_recording_end_reference
        installed = threading.Event()

        def windows_reference(evidence, raw_manifest_path, manifest, *, deadline):
            if not installed.is_set() and evidence["media"]["reference_path"]:
                installed.set()
                _windows_stat_views(
                    monkeypatch,
                    Path(evidence["media"]["reference_path"]),
                    mutate_after_read=False,
                    path_stat=True,
                )
            return original_validate(evidence, raw_manifest_path, manifest, deadline=deadline)

        monkeypatch.setattr(
            SQLiteFieldRecordStore,
            "_validate_recording_end_reference",
            staticmethod(windows_reference),
        )
    server, release, _ = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=380,
        paused=paused,
        resume=resume,
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    done, results, errors = threading.Event(), [], []
    waiter = None
    report = {}
    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        assert held.wait(3) and paused.wait(7)
        status, reply = _end(server, state.session_id)
        assert status == 200 and reply["recording_end"]["capture_status"] == "STOPPED"
        assert state.record_store.get_recording_end_manifest(state.session_id) is None
        assert state.postgame_review_payload()["clips"] == []
        final_committed, growing_projection = threading.Event(), threading.Event()
        projection_errors = []
        original_finalize = state.record_store.prepare_recording_end
        original_payload = state._postgame_review_payload_for
        original_review = state.postgame_review_payload

        def committed(*args, **kwargs):
            result = original_finalize(*args, **kwargs)
            final_committed.set()
            return result

        def project_across_final_commit(manifest, **kwargs):
            value = original_payload(manifest, **kwargs)
            if (
                manifest is not None
                and value["phase"] == "RECORDING_PROCESSING"
                and not growing_projection.is_set()
            ):
                growing_projection.set()
                assert detect.wait(15)
                assert final_committed.wait(7), "original final manifest did not commit"
            return value

        def observe_review(**kwargs):
            try:
                return original_review(**kwargs)
            except ValueError as exc:
                projection_errors.append(str(exc))
                raise

        monkeypatch.setattr(state.record_store, "prepare_recording_end", committed)
        monkeypatch.setattr(state, "_postgame_review_payload_for", project_across_final_commit)
        monkeypatch.setattr(state, "postgame_review_payload", observe_review)

        def observe():
            try:
                results.append(
                    runner._wait_for_background_completion(
                        server,
                        state,
                        reply["recording_end"],
                        deadline=time.monotonic() + 10,
                        report=report,
                    )
                )
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        waiter = threading.Thread(target=observe, daemon=True)
        waiter.start()
        assert not done.wait(0.25), errors
        detect.set()
        assert done.wait(7), "observer did not follow actual original processing"
        assert not errors, {"observer": errors, "projection": projection_errors}
        frozen, review = results[0]
        assert frozen["cutoff"]["frame_id"] == 380
        assert len(frozen["shots"]) == review["expected_count"] == review["available_count"] == 2
        assert review["processing"]["status"] == "COMPLETE"
        assert all(clip["media_url"] for clip in review["clips"])
        assert not growing_projection.is_set(), "observer must wait for final review admission"
        assert report["processing_seconds"] >= 0.25
    finally:
        detect.set()
        resume.set()
        release.set()
        if waiter is not None:
            waiter.join(3)
        server.shutdown()
        serving.join(2)
        server.server_close()
