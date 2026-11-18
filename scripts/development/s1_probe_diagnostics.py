"""Probe-only bounded stage receipts; detailed one-shot stacks stay machine-local.

No product runtime imports this module. It changes neither owned deadlines nor
component readiness criteria, and never exports raw stack or exception text.
"""

from __future__ import annotations

import faulthandler
import json
import math
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from scripts.development.blackbox_report import atomic_json

STAGES = frozenset(
    {
        "source_before",
        "dependency_metadata",
        "imports",
        "resources_before",
        "capture_setup",
        "calibration",
        "first_frame",
        "model_factory",
        "model_runtime_import",
        "model_vision_loader",
        "model_detector_initialization",
        "model_identity_features_setup",
        "model_identity_weights_preparation",
        "model_detector_execution_preparation",
        "model_group_loader",
        "model_onnx_session_initialization",
        "physics_preparation",
        "detector_inference",
        "identity_inference",
        "planning",
        "resources_after",
        "video_after",
        "source_after",
        "result_write",
        "raw_validation",
        "media_inputs",
        "media_source_after",
        "original_timing",
        "original_encode",
        "original_writer_release",
        "original_fsync",
        "original_verify",
        "screen_capture",
        "screen_end",
        "screen_background",
        "screen_playback",
        "screen_close",
        "screen_verify",
        "screen_clips",
        "screen_assess",
        "screen_sources_after",
        "screen_server_shutdown",
        "screen_server_close",
        "screen_prepared_cancel",
    }
)
MAX_EVENTS = 64
MAX_BYTES = 32 * 1024


class CallStageProxy:
    """Observe one existing backend call without replacing its data or behavior."""

    def __init__(self, target, trace, method, stage):
        self._target, self._trace, self._method, self._stage = target, trace, method, stage

    def __getattr__(self, name):
        value = getattr(self._target, name)
        if name != self._method:
            return value

        def measured(*args, **kwargs):
            with self._trace.stage(self._stage):
                return value(*args, **kwargs)

        return measured


class ProbeDiagnostics:
    def __init__(self, root: Path, run_id: str, *, stack_at: float | None = None):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("invalid diagnostic run identity")
        if stack_at is not None and not math.isfinite(stack_at):
            raise ValueError("invalid stack deadline")
        self.root, self.run_id, self.stack_at = Path(root), run_id, stack_at
        self.events = []
        self._pending_exits = 0
        self.started = time.monotonic()
        self._stack = None
        self._lock = threading.RLock()

    def __enter__(self):
        if (self.root / "local-stages.json").exists():
            raise FileExistsError("diagnostic receipts require a fresh run")
        if self.stack_at is not None:
            self.arm_stack(self.stack_at)
        return self

    def arm_stack(self, stack_at):
        """Arm an absolute deadline after any synchronous stage-entry receipt."""
        if not math.isfinite(stack_at) or self._stack is not None:
            raise ValueError("invalid or already armed stack deadline")
        self._stack = (self.root / "local-stack.txt").open("xb")
        try:
            faulthandler.dump_traceback_later(
                max(0.001, stack_at - time.monotonic()),
                repeat=False,
                file=self._stack,
                exit=False,
            )
        except BaseException:
            self._stack.close()
            self._stack = None
            raise

    def __exit__(self, *_):
        if self._stack is not None:
            faulthandler.cancel_dump_traceback_later()
            self._stack.close()
            self._stack = None

    def _record(self, stage, event, error_type=None):
        row = {"stage": stage, "event": event, "at_seconds": time.monotonic() - self.started}
        if error_type is not None:
            row["error_type"] = (
                error_type
                if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", error_type)
                else "Exception"
            )
        self.events.append(row)
        atomic_json(
            self.root / "local-stages.json",
            {"contract": "s1-probe-stages/v1", "run_id": self.run_id, "events": self.events},
        )

    @contextmanager
    def stage(self, name):
        if name not in STAGES:
            raise ValueError("unknown diagnostic stage")
        with self._lock:
            if len(self.events) + self._pending_exits + 2 > MAX_EVENTS:
                raise ValueError("diagnostic event budget exceeded")
            self._record(name, "enter")
            self._pending_exits += 1
        error_type = None
        try:
            yield
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            with self._lock:
                try:
                    self._record(name, "error" if error_type else "exit", error_type)
                except Exception:
                    if error_type is None:
                        raise
                finally:
                    self._pending_exits -= 1


def read_diagnostic_summary(root: Path, run_id: str):
    """Fail closed on malformed receipts; never include the raw stack or paths."""
    path = Path(root) / "local-stages.json"
    if not path.exists():
        return {"status": "NOT_AVAILABLE"}
    try:
        if path.is_symlink() or path.stat().st_size > MAX_BYTES:
            raise ValueError("diagnostic receipt budget")
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES:
            raise ValueError("diagnostic receipt budget")
        receipt = json.loads(raw)
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"contract", "run_id", "events"}
            or receipt["contract"] != "s1-probe-stages/v1"
            or receipt["run_id"] != run_id
            or not isinstance(receipt["events"], list)
            or not 0 < len(receipt["events"]) <= MAX_EVENTS
        ):
            raise ValueError("invalid diagnostic receipt")
        active, previous = [], 0.0
        preparation = {"model_factory", "physics_preparation"}
        final_receipts = {"source_after", "result_write"}
        preparation_failed = False
        for row in receipt["events"]:
            if not isinstance(row, dict):
                raise ValueError("invalid diagnostic event")
            fields = {"stage", "event", "at_seconds"}
            if row.get("event") == "error":
                fields.add("error_type")
                if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", str(row.get("error_type"))):
                    raise ValueError("invalid diagnostic error")
            if (
                set(row) != fields
                or row["stage"] not in STAGES
                or row["event"] not in {"enter", "exit", "error"}
                or type(row["at_seconds"]) not in {int, float}
                or not previous <= row["at_seconds"] <= 3600
                or not math.isfinite(row["at_seconds"])
            ):
                raise ValueError("invalid diagnostic event")
            if row["event"] == "enter":
                active.append(row["stage"])
            elif active and active[-1] == row["stage"]:
                active.pop()
            elif len(active) == 2 and set(active) == preparation and row["stage"] in active:
                # These two independent boot stages may finish in either
                # order. Other stages retain the original nesting contract.
                active.remove(row["stage"])
            elif (
                preparation_failed
                and row["stage"] in active
                and all(name in preparation | final_receipts for name in active)
                and (
                    row["stage"] in preparation
                    or (
                        row["stage"] in final_receipts
                        and next(
                            (name for name in reversed(active) if name not in preparation), None
                        )
                        == row["stage"]
                    )
                )
            ):
                # First-error fanout does not stop native sibling work. Its
                # terminal event can arrive during the failed child's final
                # source/result receipts; those receipts retain their own LIFO.
                index = len(active) - 1 - active[::-1].index(row["stage"])
                active.pop(index)
            else:
                raise ValueError("invalid diagnostic sequence")
            if row["event"] == "error" and row["stage"] in preparation:
                preparation_failed = True
            previous = row["at_seconds"]
        stack = Path(root) / "local-stack.txt"
        last = receipt["events"][-1]
        return {
            "status": "AVAILABLE",
            "last_stage": last["stage"],
            "last_event": last["event"],
            "events": receipt["events"],
            "stack_capture_available": stack.is_file()
            and not stack.is_symlink()
            and 0 < stack.stat().st_size <= 4 * 1024**2,
        }
    except (ValueError, TypeError, KeyError, OSError):
        return {"status": "INVALID"}
