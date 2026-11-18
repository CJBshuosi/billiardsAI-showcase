import http.client
import json
import sqlite3
import threading

import pytest

from scripts.development import blackbox_validation as validation
from scripts.development import reopen_recording_review as review
from tests.unit.lab.test_mainline1_recording_media_worker import NOW, _legacy_two_shot_scene


def test_completed_local_report_has_a_finite_larger_read_budget(tmp_path):
    (tmp_path / "recording").mkdir()
    path = tmp_path / "recording/report.json"
    value = {"evidence": "x" * 373000, "cleanup_confirmed": False}
    path.write_text(json.dumps(value))
    assert validation.read_product_report(tmp_path, "FullgameComplete") == value
    with pytest.raises(validation.ValidationFailure, match="JSON_SIZE_BUDGET"):
        validation.read_product_report(tmp_path, "Fullgame")
    path.write_bytes(b"x" * (1024**2 + 1))
    with pytest.raises(validation.ValidationFailure, match="JSON_SIZE_BUDGET"):
        validation.read_product_report(tmp_path, "FullgameComplete")


def test_readonly_review_uses_real_media_and_refuses_all_mutations(tmp_path, monkeypatch):
    from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker

    records, jobs, _, root, worker = _legacy_two_shot_scene(tmp_path)
    assert worker.process_one("first", "legacy-session", NOW) is not None
    assert worker.process_one("second", "legacy-session", NOW) is not None
    ro_records, ro_jobs = review.readonly_stores(records.path, jobs._database)
    with ro_records._connect() as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM field_sessions")
    with ro_jobs._connect() as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM shot_analysis_jobs")
    monkeypatch.setattr(
        Mainline1RecordingMediaWorker,
        "process_one",
        lambda *a, **kw: pytest.fail("read-only review started an encoder"),
    )
    assert review.probe_http_close(ro_records, ro_jobs, "legacy-session", root) == {
        "idle_connections": 5,
        "cleanup_confirmed": True,
    }
    server = review.build_review_server(ro_records, ro_jobs, "legacy-session", root)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request("GET", "/screen")
        response = connection.getresponse()
        body = response.read().decode()
        assert response.status == 200 and "postgame-player" in body
        assert "setInterval(sync" not in body
        connection.request("GET", "/api/postgame-review")
        response = connection.getresponse()
        projection = json.loads(response.read())
        assert response.status == 200 and "kiosk_nonce" not in projection
        assert projection["playback"]["available_count"] == 2
        clip = projection["clips"][0]
        connection.request("GET", clip["media_url"], headers={"Range": "bytes=0-7"})
        response = connection.getresponse()
        data = response.read()
        assert response.status == 206 and len(data) == 8
        for path in ("/api/control", "/api/postgame-review/save", "/api/postgame-review/complete"):
            connection.request(
                "POST", path, body=b"{}", headers={"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            response.read()
            assert response.status == 405
        assert records.active_postgame_review("legacy-artifact") is not None
        # A real separate writer commits the negative access fence; the viewer
        # must see it through fresh original-DB read-only connections.
        records.complete_postgame_review(
            "legacy-artifact", "legacy-session", projection["postgame_manifest_sha256"]
        )
        connection.request("GET", clip["media_url"])
        response = connection.getresponse()
        response.read()
        assert response.status == 404
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        serving.join(2)
    assert not server._request_connections


def test_review_close_marker_is_bound_to_its_own_run(tmp_path):
    (tmp_path / "close.json").write_text(json.dumps({"run_id": "another"}))
    assert not review.close_requested(tmp_path)
    (tmp_path / "close.json").write_text(json.dumps({"run_id": tmp_path.name}))
    assert review.close_requested(tmp_path)


def test_close_interrupt_is_sticky_and_cleanup_sleep_never_repeats_it(tmp_path):
    (tmp_path / "export").mkdir()
    (tmp_path / "local-logs").mkdir()
    (tmp_path / "close.json").write_text(json.dumps({"run_id": tmp_path.name}))
    clock = [0.0]
    waits = []
    monitor = review.ReviewMonitor(
        tmp_path, {"run_id": tmp_path.name}, clock=lambda: clock[0], sleeper=waits.append
    )
    monitor.sleep(0.05)
    clock[0] = 11
    with pytest.raises(KeyboardInterrupt):
        monitor.sleep(0.05)
    assert monitor.cleaning
    (tmp_path / "export/summary.json").unlink()
    monitor.sleep(0.05)
    monitor.sleep(0.05)
    assert waits == [0.05, 0.05, 0.05] and not (tmp_path / "export/summary.json").exists()


def test_http_start_failure_closes_listener_without_shutdown_wait(tmp_path, monkeypatch):
    from types import SimpleNamespace

    closed = []

    def fail_start(_):
        raise RuntimeError("start failed")

    monkeypatch.setattr(review.threading.Thread, "start", fail_start)
    server = SimpleNamespace(
        serve_forever=lambda: None,
        shutdown=lambda: pytest.fail("shutdown on a never-started service"),
        server_close=lambda: closed.append(True),
    )
    with pytest.raises(RuntimeError, match="start failed"):
        review.serve_review(server, tmp_path, {}, "session")
    assert closed == [True] and not (tmp_path / "ready.json").exists()


@pytest.mark.parametrize(
    "failure,expected", [(None, 1), ("TimeoutError", 3), ("KeyboardInterrupt", 4)]
)
def test_child_retains_large_report_failure_classification(
    tmp_path, monkeypatch, failure, expected
):
    import io
    from types import SimpleNamespace

    (tmp_path / "recording").mkdir()
    (tmp_path / "local-logs").mkdir()
    (tmp_path / "context.json").write_text(json.dumps({"profile": "FullgameComplete"}))
    (tmp_path / "setup-result.json").write_text(
        json.dumps({"identity": {"source_commit": "a" * 40}})
    )
    data = dict(
        source_commit="a" * 40,
        source_sha256=validation.SOURCE_SHA,
        cleanup_confirmed=False,
        padding="x" * 373000,
    )
    if failure:
        data["failure"] = {"type": failure}
    (tmp_path / "recording/report.json").write_text(json.dumps(data))
    monkeypatch.setattr(
        validation.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(("BLACKBOX-RELEASE " + tmp_path.name + "\n").encode())),
    )
    monkeypatch.setattr(validation, "worker", lambda *args: 1)
    assert validation.child_main(tmp_path, "worker") == expected
    result = json.loads((tmp_path / "worker-result.json").read_text())
    assert (
        result["failure_code"]
        == {1: "PRODUCT_FAILED", 3: "PRODUCT_TIMEOUT", 4: "CANCELLED"}[expected]
    )
