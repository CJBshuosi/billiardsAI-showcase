from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace


def _module():
    path = Path(__file__).parents[3] / "scripts" / "algorithm" / "serve_field_route_shell.py"
    spec = importlib.util.spec_from_file_location("serve_field_route_shell_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_worker_survives_one_unexpected_store_failure() -> None:
    module = _module()
    recovered = threading.Event()

    class Store:
        def __init__(self) -> None:
            self.attempts = 0

        def claim_due_sync_job(self, _now):
            self.attempts += 1
            if self.attempts == 1:
                raise ValueError("temporary sqlite fault")
            recovered.set()
            return None

    store = Store()
    worker = module._BackgroundSyncWorker(
        store=store,
        client=object(),
        wake=threading.Event(),
    )
    worker.start()
    try:
        assert recovered.wait(timeout=1.0)
    finally:
        worker.stop()

    assert store.attempts >= 2


def test_sync_worker_returns_claimed_job_to_backoff_after_client_exception() -> None:
    module = _module()
    backed_off = threading.Event()
    job = SimpleNamespace(sync_job_id="sync-1", retry_count=0)

    class Store:
        def __init__(self) -> None:
            self.claimed = False

        def claim_due_sync_job(self, _now):
            if self.claimed:
                return None
            self.claimed = True
            return job

        def mark_sync_partial(self, sync_job_id, *, error_code, next_attempt_at):
            assert sync_job_id == "sync-1"
            assert error_code == "SYNC_WORKER_EXCEPTION"
            assert next_attempt_at is not None
            backed_off.set()

    class Client:
        def run_once(self, _job):
            raise ValueError("credential lookup failed")

    worker = module._BackgroundSyncWorker(
        store=Store(),
        client=Client(),
        wake=threading.Event(),
    )
    worker.start()
    try:
        assert backed_off.wait(timeout=1.0)
    finally:
        worker.stop()


def test_sync_worker_recovers_missing_outboxes_without_ui_traffic() -> None:
    module = _module()
    recovered = threading.Event()

    class Store:
        def claim_due_sync_job(self, _now):
            return None

    worker = module._BackgroundSyncWorker(
        store=Store(),
        client=object(),
        wake=threading.Event(),
        recover_missing_jobs=recovered.set,
    )
    worker.start()
    try:
        assert recovered.wait(timeout=1.0)
    finally:
        worker.stop()
