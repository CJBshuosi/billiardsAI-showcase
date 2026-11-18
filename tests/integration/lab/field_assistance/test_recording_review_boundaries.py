"""Public projection and model-free original-context recovery boundaries."""

import http.client
import json
import threading
from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.field_assistance.server import FieldAssistanceHTTPServer
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker
from tests.unit.edge.route_assistance.test_recording_selection import _state
from tests.unit.lab import test_mainline1_recording_media_worker as media
from tests.unit.lab.field_assistance.test_managed_startup import BINDING, SCOPE


@pytest.mark.parametrize("kiosk", [False, True])
def test_final_recording_http_projection_never_exposes_private_file_identity(
    tmp_path, kiosk, monkeypatch
):
    records, session, root = media._ordinary_scene(tmp_path)
    private = records.get_recording_end_manifest(session)
    assert private["raw_manifest_path"] and private["raw_segments"][0]["inode"] >= 0
    state = _state(records, session, root, SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3"))
    monkeypatch.setattr(type(state), "maintain_service", lambda self: None)
    monkeypatch.setattr(type(state), "close_projection", lambda self: None)
    server = FieldAssistanceHTTPServer(("127.0.0.1", 0), state)
    # Exercise both public projection variants through an actual unauthenticated GET.
    project = state.postgame_review_payload
    monkeypatch.setattr(
        type(state), "postgame_review_payload", lambda self, **_: project(kiosk=kiosk)
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request("GET", "/api/postgame-review")
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        public = payload["recording_end"]
        assert public["session_id"] == session and public["cutoff"] == private["cutoff"]
        assert public["media_status"] == private["media_status"]
        assert "raw_manifest_path" not in public and "raw_segments" not in public
        encoded = json.dumps(payload)
        assert str(tmp_path) not in encoded
        assert all(
            key not in encoded for key in ('"device"', '"inode"', '"mtime_ns"', '"ctime_ns"')
        )
        assert records.get_recording_end_manifest(session) == private
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        serving.join(3)


@pytest.mark.parametrize(
    "phase", ["ready", "processing", "failed", "missing-db", "missing-db-ready", "foreign"]
)
def test_historical_managed_recording_without_models_keeps_original_queue_or_status(
    tmp_path, monkeypatch, phase
):

    def managed_store(root):
        records = SQLiteFieldRecordStore(root / "state.sqlite3")
        request = records.reserve_managed_open(SCOPE, artifact_id="recording")
        records.record_managed_binding(SCOPE, request.open_request_id, BINDING)
        session = records.materialize_managed_session(
            SCOPE, request.open_request_id, session_policy="recording-v1"
        )
        return records, session.session_id

    # The normal ordinary CLI supplies no semantic resource inspector/spec/epoch.
    monkeypatch.setattr(media, "_store", managed_store)
    records, session, root = media._ordinary_scene(
        tmp_path, production_layout=True, capture_stopped=phase not in {"ready", "missing-db-ready"}
    )
    with pytest.raises(ValueError):
        records.get_managed_resource_spec_for_session(SCOPE, session)
    original = SQLiteShotAnalysisJobStore(tmp_path / "jobs/analysis-jobs.sqlite3")
    if phase == "failed":
        records.set_recording_capture_progress(
            session, tail_status="SAVED", processing_status="FAILED", failure_code="TEST_FAILURE"
        )
    if phase.startswith("missing-db"):
        original._database.rename(tmp_path / "original-retained.sqlite3")
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionCloseState
    from tests.unit.lab.field_assistance.test_managed_startup import NEXT_BIRTH

    original_request = records.get_managed_open_request_for_session(SCOPE, session)
    records.retire_managed_open(
        SCOPE,
        original_request.open_request_id,
        sealed_close=ManagedSessionCloseState(
            BINDING.schema_version,
            session,
            BINDING.birth_id,
            BINDING.birth_contract,
            BINDING.table_id,
            BINDING.device_id,
            "sealed",
            0,
            0,
        ),
    )
    next_request = records.reserve_managed_open(SCOPE, artifact_id="next-recording")
    next_binding = replace(BINDING, birth_id=NEXT_BIRTH, session_id=f"managed-{NEXT_BIRTH}")
    records.record_managed_binding(SCOPE, next_request.open_request_id, next_binding)
    next_session = records.materialize_managed_session(
        SCOPE, next_request.open_request_id, session_policy="recording-v1"
    )
    next_root = tmp_path / "different-artifact"
    next_root.mkdir()
    records.bind_artifact_path("next-recording", next_root)
    current = SQLiteShotAnalysisJobStore(tmp_path / "different-current/analysis-jobs.sqlite3")
    state = _state(
        records, next_session.session_id, next_root, current, artifact_id="next-recording"
    )
    state.managed_session_scope = (
        replace(SCOPE, table_id="foreign") if phase == "foreign" else SCOPE
    )
    if phase == "foreign":
        assert state.postgame_review_payload()["review_session_id"] is None
        return
    jobs = state._media_jobs_for_session(session)
    if phase.startswith("missing-db"):
        assert jobs is None
        assert not original._database.exists()
    else:
        assert jobs._database == original._database
        encoder = Mainline1RecordingMediaWorker(
            records=records, jobs=jobs, artifact_root=root / "postgame" / session
        )
        assert encoder.process_one("historical-ordinary", session, media.NOW) is not None
    review = state.postgame_review_payload(kiosk=True)
    assert review["review_session_id"] == session
    assert review["analysis_status"] == "NOT_REQUESTED"
    assert review["available_count"] == (0 if phase.startswith("missing-db") else 1)
    if phase in {"processing", "failed", "missing-db", "missing-db-ready"}:
        assert review["can_complete"] is False
    if phase.startswith("missing-db"):
        with pytest.raises(ValueError, match="context"):
            state.mutate_postgame_review(
                dict(
                    session_id=session, postgame_manifest_sha256=review["postgame_manifest_sha256"]
                ),
                review["kiosk_nonce"],
                complete=True,
            )
        assert records.active_postgame_review_for_scope(SCOPE).session_id == session
    assert state.media_job_store is current
    assert current.list_session_work_items(session) == ()
