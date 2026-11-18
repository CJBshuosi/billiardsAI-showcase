"""Absolute HTTP budgets and honest elapsed observations for local acceptance."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from scripts.algorithm import run_recording_first_fullgame as runner


def _serve(payload, *, header_delay=0, body_delay=0, drip_headers=False):
    body = json.dumps(payload).encode()

    class Delayed(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if drip_headers:
                    for line in [b"HTTP/1.0 200 OK\r\n", b"X-First: 1\r\n", b"X-Second: 2\r\n"]:
                        self.wfile.write(line)
                        self.wfile.flush()
                        time.sleep(header_delay)
                    self.wfile.write(b"\r\n" + body)
                    self.wfile.flush()
                else:
                    time.sleep(header_delay)
                    self.send_response(200)
                    self.end_headers()
                    time.sleep(body_delay)
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Delayed)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.mark.parametrize("drip_headers", [False, True])
def test_request_total_budget_covers_split_headers_and_body(drip_headers):
    server, thread = _serve({}, header_delay=0.12, body_delay=0.12, drip_headers=drip_headers)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            runner._request(server, "/review", timeout_seconds=0.20)
        assert time.monotonic() - started < 0.45
    finally:
        server.shutdown()
        thread.join(2)
        server.server_close()


def test_completed_inventory_timings_include_http_observation():
    # Stop/final fixtures are immutable adapter boundary inputs; HTTP is real.
    receipt = dict(
        contract="recording-capture-end/v1",
        capture_status="STOPPED",
        session_id="session-1",
        source_epoch_id="epoch-1",
        cutoff={"frame_id": 99},
        accepted_at_utc="2026-11-10T00:00:00Z",
        tail_status="SAVED",
        processing_status="COMPLETE",
    )
    frozen = dict(cutoff=receipt["cutoff"], source_epoch_id="epoch-1", shots=[])
    review = dict(
        review_session_id="session-1",
        phase="RECORDING_ENDED",
        processing={"status": "COMPLETE"},
        expected_count=0,
        discovered_count=0,
        available_count=0,
        clips=[],
    )
    state = SimpleNamespace(
        session_id="session-1",
        projection_source=SimpleNamespace(source_epoch_id="epoch-1"),
        record_store=SimpleNamespace(
            get_recording_end_manifest=lambda _: frozen, get_recording_capture_end=lambda _: receipt
        ),
        media_job_store=SimpleNamespace(
            session_media_progress=lambda _: dict(
                generated_count=0,
                encoding_count=0,
                queued_count=0,
                retrying_count=0,
                failed_count=0,
                items=[],
            )
        ),
    )
    server, thread = _serve(review, body_delay=0.10)
    report = {}
    try:
        runner._wait_for_background_completion(
            server, state, receipt, deadline=time.monotonic() + 2, report=report
        )
        assert report["processing_seconds"] >= 0.09
        assert report["processing_samples"][0]["seconds"] >= 0.09
    finally:
        server.shutdown()
        thread.join(2)
        server.server_close()
