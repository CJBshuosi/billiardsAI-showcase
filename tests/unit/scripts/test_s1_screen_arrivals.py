"""Actual frame delivery, not nominal video FPS, defines the display deadline."""

import json
import math

import pytest

from scripts.development import s1_screen_acceptance as screen


def test_arrival_dom_and_worker_evidence_share_high_resolution_clock(tmp_path, monkeypatch):
    # Windows 3.12 monotonic can repeat for adjacent 60fps deliveries. Keep
    # that real coarse-clock behavior; never fabricate increasing timestamps.
    monkeypatch.setattr(screen.time, "monotonic", lambda: 10.0)
    ticks = iter([100.0, 100.0001, 100.0002, 100.0003, 100.0004, 100.0005])
    monkeypatch.setattr(screen.time, "perf_counter", lambda: next(ticks))
    clock = screen.FrameArrivalClock(3)
    for frame in range(3):
        clock.record(frame)
    screen.trace_event(tmp_path / "trace.jsonl", "observe_enter", 2)

    class Browser:
        def evaluate(self, expression):
            return {"result": {"phase": "等待停稳"}}

    dom = screen.browser_state(Browser())
    event = json.loads((tmp_path / "trace.jsonl").read_bytes())
    assert clock.complete() == [100.0, 100.0001, 100.0002]
    assert event["seconds"] == 100.0003
    assert dom["monotonic"] == 100.0004
    assert dom["observed_at"] == 100.0005


def test_actual_delayed_frame_delivery_defines_motion_hiding_without_relaxing_500ms():
    arrivals = [100.0 + frame * 0.1 for frame in range(31)]
    dom = [
        dict(
            monotonic=101.1,
            observed_at=101.2,
            phase="系统自动推荐",
            has_dash=True,
            has_region=True,
            has_halo=True,
        ),
        dict(
            monotonic=102.2,
            observed_at=102.3,
            phase="等待停稳",
            has_dash=False,
            has_region=False,
            has_halo=False,
        ),
    ]
    delays = screen.assess_motion_display(dom, [(20, 29)], arrival_times=arrivals)
    assert delays == pytest.approx([0.3])
    dom[1]["observed_at"] = 102.51
    with pytest.raises(ValueError, match="500ms"):
        screen.assess_motion_display(dom, [(20, 29)], arrival_times=arrivals)


@pytest.mark.parametrize("values", [[1.0], [1.0, 1.0], [2.0, 1.0], [1.0, math.nan]])
def test_missing_or_nonmonotonic_arrivals_cannot_prove_motion(values):
    with pytest.raises(ValueError, match="arrival"):
        screen.motion_wall_windows([(0, 1)], arrival_times=values)


def test_frame_arrival_clock_records_sequential_real_delivery_before_yield():
    clock = screen.FrameArrivalClock(3)
    clock.record(0, at=10.0)
    with pytest.raises(ValueError, match="arrival"):
        clock.record(2, at=11.0)
    clock.record(1, at=10.1)
    with pytest.raises(ValueError, match="arrival"):
        clock.record(2, at=10.1)
    clock.record(2, at=10.2)
    assert clock.complete() == [10.0, 10.1, 10.2]
    with pytest.raises(ValueError, match="arrival"):
        clock.record(3, at=10.3)


def test_partial_capture_clock_is_never_accepted_as_complete():
    clock = screen.FrameArrivalClock(3)
    clock.record(0, at=10.0)
    with pytest.raises(ValueError, match="arrival"):
        clock.complete()


def test_owned_capture_uses_explicit_original_case_allowance_without_renewal():
    # A 8866-frame FILE can still be progressing after its nominal 147.77s.
    # This is a finite functional capture, not evidence of real-time 60fps.
    assert screen.capture_work_seconds(8866, None) == pytest.approx(8866 / 60 + 60)
    assert screen.capture_work_seconds(8866, 240) == 240
    for invalid in (0, -1, 241, float("nan"), True):
        with pytest.raises(ValueError, match="capture allowance"):
            screen.capture_work_seconds(8866, invalid)


def test_last_frame_receipt_enforces_original_deadline_even_when_parent_observes_late(monkeypatch):
    monkeypatch.setattr(screen.time, "perf_counter", lambda: 999.0)
    screen.validate_capture_completion(340.0, 339.99)
    screen.validate_capture_completion(340.0, 340.0)
    for receipt in (None, 340.01, float("nan"), True):
        with pytest.raises((ValueError, TimeoutError), match="capture"):
            screen.validate_capture_completion(340.0, receipt)


def test_failed_progress_write_cannot_skip_real_resource_cleanup(tmp_path):
    from functools import partial

    from scripts.development.s1_probe_diagnostics import ProbeDiagnostics

    blocked = tmp_path / "not-a-directory"
    blocked.write_bytes(b"keep")
    resource = (tmp_path / "resource").open("wb")
    diagnostics = ProbeDiagnostics(tmp_path, "a" * 32)
    checkpoint = partial(screen.screen_stage, {}, blocked / "child", diagnostics)
    with pytest.raises(OSError):
        screen.cleanup_step(checkpoint, "screen_server_close", resource.close)
    assert resource.closed
