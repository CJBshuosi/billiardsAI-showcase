from __future__ import annotations

import hashlib
import http.client
import json
import os
import threading
from pathlib import Path
from urllib.parse import urlencode

import pytest

import billiards_ai.lab.propagation.review_server as review_server
from billiards_ai.lab.propagation.overlays import render_evaluation_overlays
from billiards_ai.lab.propagation.review_server import (
    ReviewConflictError,
    ReviewServerError,
    ReviewSession,
    create_review_server,
)
from tests.integration.lab.propagation.test_overlays import _evaluated_run, _tree_snapshot


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ReviewSession, Path]:
    raw_run, evaluation = _evaluated_run(tmp_path, monkeypatch)
    rendered = render_evaluation_overlays(
        raw_run=raw_run,
        evaluation_artifact=evaluation.evaluation_path,
        review_sample=evaluation.review_sample_path,
        output_root=tmp_path / "renders",
    )
    return (
        ReviewSession.open(
            raw_run=raw_run,
            evaluation_artifact=evaluation.evaluation_path,
            review_sample=evaluation.review_sample_path,
            render_artifact=rendered.path,
            state_path=tmp_path / "working" / "verdict-state.json",
        ),
        raw_run,
    )


def test_review_session_opens_render_media_in_binary_mode_when_platform_requires_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open = os.open
    binary_flag = 0x8000
    monkeypatch.setattr(review_server.os, "O_BINARY", binary_flag, raising=False)

    def windows_open(path: os.PathLike[str] | str, flags: int) -> int:
        if not flags & binary_flag:
            raise OSError("Windows text mode would translate binary media")
        return real_open(path, flags & ~binary_flag)

    monkeypatch.setattr(review_server.os, "open", windows_open)

    session, _ = _session(tmp_path, monkeypatch)

    assert session.current_unit is not None


def test_routine_review_interactions_do_not_repeat_deep_artifact_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _session(tmp_path, monkeypatch)
    unit = session.current_unit
    assert unit is not None

    def reject_deep_verification(*args: object, **kwargs: object) -> None:
        raise AssertionError("routine review interaction repeated deep artifact verification")

    monkeypatch.setattr(review_server, "_load_raw_run", reject_deep_verification)
    monkeypatch.setattr(review_server, "_verify_render", reject_deep_verification)

    assert "第 1 / 51 项" in session.page()
    assert session.current_media_bytes(unit.unit_id).startswith(b"\xff\xd8")
    state = session.submit(
        unit_id=unit.unit_id,
        revision=0,
        acceptable_without_redraw=unit.unit_type == "matched_pair",
        severe_drift_or_wrong_target=False,
    )

    assert state.revision == 1


def test_review_session_resumes_with_optimistic_revisions_and_exports_only_when_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, raw_run = _session(tmp_path, monkeypatch)
    raw_before = _tree_snapshot(raw_run)
    first = session.current_unit
    assert first is not None

    state = session.submit(
        unit_id=first.unit_id,
        revision=0,
        acceptable_without_redraw=first.unit_type == "matched_pair",
        severe_drift_or_wrong_target=False,
        note="合成证据正常",
    )
    assert state.revision == 1
    assert state.completed_count == 1
    assert not list((tmp_path / "working").glob(".pending-*"))
    with pytest.raises(ReviewConflictError, match="revision"):
        session.submit(
            unit_id=first.unit_id,
            revision=0,
            acceptable_without_redraw=False,
            severe_drift_or_wrong_target=False,
        )
    with pytest.raises(ReviewServerError, match="complete"):
        session.export_verdict_artifact(tmp_path / "exports")

    resumed = ReviewSession.open(
        raw_run=session.raw_run,
        evaluation_artifact=session.evaluation_artifact,
        review_sample=session.review_sample,
        render_artifact=session.render_artifact,
        state_path=session.state_path,
    )
    assert resumed.revision == 1
    for unit in resumed.remaining_units:
        resumed.submit(
            unit_id=unit.unit_id,
            revision=resumed.revision,
            acceptable_without_redraw=unit.unit_type == "matched_pair",
            severe_drift_or_wrong_target=False,
        )

    summary = resumed.summary
    assert summary.status == "human_gates_passed"
    assert summary.raw_challenge_sample.acceptable_without_redraw == 48 / 51
    assert summary.raw_challenge_sample.severe_drift_or_wrong_target == 0
    assert summary.population_weighted.acceptable_without_redraw == 48 / 51
    assert summary.population_weighted.severe_drift_or_wrong_target == 0
    assert summary.population_counts == {
        "clean_match": 48,
        "flagged_or_ambiguous_match": 0,
        "miss": 3,
        "false_positive": 0,
    }
    verdict = resumed.export_verdict_artifact(tmp_path / "exports")
    verdict_bytes = (verdict / "verdict.json").read_bytes()
    payload = json.loads(verdict_bytes)
    assert verdict.name == hashlib.sha256(verdict_bytes).hexdigest()
    assert payload["raw_run_hash"] == raw_run.name
    assert (
        payload["qa_identity"]
        == json.loads((resumed.evaluation_artifact / "report.json").read_text())["qa_identity"]
    )
    assert payload["summary"]["status"] == "human_gates_passed"
    assert payload["pilot_status"] == "passed"
    assert len(payload["verdicts"]) == 51
    assert not verdict.stat().st_mode & 0o222
    assert _tree_snapshot(raw_run) == raw_before


def test_verdict_export_rejects_a_source_mutation_racing_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _session(tmp_path, monkeypatch)
    for unit in session.remaining_units:
        session.submit(
            unit_id=unit.unit_id,
            revision=session.revision,
            acceptable_without_redraw=unit.unit_type == "matched_pair",
            severe_drift_or_wrong_target=False,
        )
    report = session.evaluation_artifact / "report.json"
    original_replace = os.replace

    def mutate_source_during_publish(source: Path, destination: Path) -> None:
        if Path(destination).parent.name == "verdicts":
            session.evaluation_artifact.chmod(0o755)
            report.chmod(0o644)
            report.write_text(report.read_text() + " ", encoding="utf-8")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", mutate_source_during_publish)

    with pytest.raises(ReviewServerError, match="source.*changed|publication"):
        session.export_verdict_artifact(tmp_path / "raced-export")
    assert not [path for path in (tmp_path / "raced-export" / "verdicts").glob("*")]


def test_review_session_rejects_a_symlinked_state_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _session(tmp_path, monkeypatch)
    external = tmp_path / "external-state"
    external.mkdir()
    (tmp_path / "linked-state").symlink_to(external, target_is_directory=True)

    with pytest.raises(ReviewServerError, match="state path.*unsafe"):
        ReviewSession.open(
            raw_run=session.raw_run,
            evaluation_artifact=session.evaluation_artifact,
            review_sample=session.review_sample,
            render_artifact=session.render_artifact,
            state_path=tmp_path / "linked-state" / "nested" / "state.json",
        )
    assert not list(external.iterdir())


def test_review_session_rejects_post_open_media_mutation_before_collecting_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _session(tmp_path, monkeypatch)
    unit = session.current_unit
    assert unit is not None
    media = session.render_artifact / "review-units" / f"{unit.unit_id}.jpg"
    original = media.read_bytes()
    media.chmod(0o644)
    media.write_bytes(original + b"tampered")

    with pytest.raises(ReviewServerError, match="source|render|media"):
        session.submit(
            unit_id=unit.unit_id,
            revision=0,
            acceptable_without_redraw=True,
            severe_drift_or_wrong_target=False,
        )
    assert session.revision == 0

    media.write_bytes(original)
    media.chmod(0o444)
    with pytest.raises(ReviewServerError, match="complete"):
        session.export_verdict_artifact(tmp_path / "no-rescued-verdict")


def test_review_page_rejects_swapped_unit_media_after_session_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _session(tmp_path, monkeypatch)
    first, second = session.remaining_units[:2]
    first_media = session.render_artifact / "review-units" / f"{first.unit_id}.jpg"
    second_media = session.render_artifact / "review-units" / f"{second.unit_id}.jpg"
    first_bytes = first_media.read_bytes()
    second_bytes = second_media.read_bytes()
    first_media.chmod(0o644)
    second_media.chmod(0o644)
    first_media.write_bytes(second_bytes)
    second_media.write_bytes(first_bytes)

    with pytest.raises(ReviewServerError, match="source|render|media"):
        session.page()


def test_review_submit_rolls_back_when_media_changes_during_atomic_state_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _session(tmp_path, monkeypatch)
    unit = session.current_unit
    assert unit is not None
    media = session.render_artifact / "review-units" / f"{unit.unit_id}.jpg"
    original_media = media.read_bytes()
    original_replace = os.replace
    raced = False

    def mutate_during_state_replace(source: Path, destination: Path) -> None:
        nonlocal raced
        original_replace(source, destination)
        if Path(destination) == session.state_path and not raced:
            raced = True
            media.chmod(0o644)
            media.write_bytes(original_media + b"raced")

    monkeypatch.setattr(os, "replace", mutate_during_state_replace)

    with pytest.raises(ReviewServerError, match="source|render|media"):
        session.submit(
            unit_id=unit.unit_id,
            revision=0,
            acceptable_without_redraw=True,
            severe_drift_or_wrong_target=False,
        )
    assert session.revision == 0
    persisted = json.loads(session.state_path.read_text())
    assert persisted["revision"] == 0
    assert persisted["verdicts"] == []


@pytest.mark.parametrize("tamper", ["extra", "delete"])
def test_verdict_export_rejects_post_rename_inventory_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    session, _ = _session(tmp_path, monkeypatch)
    for unit in session.remaining_units:
        session.submit(
            unit_id=unit.unit_id,
            revision=session.revision,
            acceptable_without_redraw=unit.unit_type == "matched_pair",
            severe_drift_or_wrong_target=False,
        )
    original_replace = os.replace

    def add_file_after_publish(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        if Path(destination).parent.name == "verdicts":
            Path(destination).chmod(0o755)
            verdict = Path(destination) / "verdict.json"
            verdict.chmod(0o644)
            if tamper == "extra":
                (Path(destination) / "extra.txt").write_text("tampered", encoding="utf-8")
            else:
                verdict.unlink()

    monkeypatch.setattr(os, "replace", add_file_after_publish)

    with pytest.raises(ReviewServerError, match="publication|artifact"):
        session.export_verdict_artifact(tmp_path / "tampered-export")
    assert not [path for path in (tmp_path / "tampered-export" / "verdicts").glob("*")]


def test_review_http_server_is_strictly_loopback_and_serves_only_current_local_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _session(tmp_path, monkeypatch)
    with pytest.raises(ReviewServerError, match="127.0.0.1"):
        create_review_server(session=session, host="0.0.0.0", port=0)

    server = create_review_server(session=session, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/", headers={"Host": "evil.example"})
        response = connection.getresponse()
        assert response.status == 400
        response.read()

        connection.request("GET", "/", headers={"Host": f"127.0.0.1:{server.server_port}"})
        response = connection.getresponse()
        page = response.read().decode()
        assert response.status == 200
        assert response.getheader("Content-Security-Policy") == (
            "default-src 'none'; img-src 'self'; style-src 'self'; "
            "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        assert "第 1 / 51 项" in page

        unit = session.current_unit
        assert unit is not None
        original_read_bytes = Path.read_bytes

        def reject_jpeg_reopen(path: Path) -> bytes:
            if path.suffix == ".jpg":
                raise AssertionError("served media must not be reopened by path")
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", reject_jpeg_reopen)
        connection.request(
            "GET",
            f"/media/{unit.unit_id}.jpg",
            headers={"Host": f"127.0.0.1:{server.server_port}"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.read().startswith(b"\xff\xd8")

        invalid_body = urlencode(
            {
                "unit_id": unit.unit_id,
                "revision": "0",
                "acceptable_without_redraw": "true",
                "severe_drift_or_wrong_target": "true",
                "note": "<b>保留我</b>",
            }
        )
        connection.request(
            "POST",
            "/verdict",
            body=invalid_body,
            headers={
                "Host": f"127.0.0.1:{server.server_port}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(invalid_body.encode())),
            },
        )
        response = connection.getresponse()
        invalid_page = response.read().decode()
        assert response.status == 400
        assert "两个选项不能同时选择“是”" in invalid_page
        assert invalid_page.count('value="true" checked') == 2
        assert "&lt;b&gt;保留我&lt;/b&gt;" in invalid_page
        assert 'name="revision" value="0"' in invalid_page
        assert session.revision == 0

        long_note = "n" * 1001
        invalid_note_body = urlencode(
            {
                "unit_id": unit.unit_id,
                "revision": "0",
                "acceptable_without_redraw": "true",
                "severe_drift_or_wrong_target": "false",
                "note": long_note,
            }
        )
        connection.request(
            "POST",
            "/verdict",
            body=invalid_note_body,
            headers={
                "Host": f"127.0.0.1:{server.server_port}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(invalid_note_body.encode())),
            },
        )
        response = connection.getresponse()
        invalid_note_page = response.read().decode()
        assert response.status == 400
        assert "表单内容无效，请重新确认。" in invalid_note_page
        assert long_note in invalid_note_page
        assert session.revision == 0

        connection.request(
            "GET",
            "/media/../../manifest.json",
            headers={"Host": f"127.0.0.1:{server.server_port}"},
        )
        response = connection.getresponse()
        assert response.status == 404
        response.read()

        body = urlencode(
            {
                "unit_id": unit.unit_id,
                "revision": "0",
                "acceptable_without_redraw": str(unit.unit_type == "matched_pair").lower(),
                "severe_drift_or_wrong_target": "false",
                "note": "",
            }
        )
        connection.request(
            "POST",
            "/verdict",
            body=body,
            headers={
                "Host": f"127.0.0.1:{server.server_port}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        assert response.status == 303
        response.read()
        assert session.revision == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
