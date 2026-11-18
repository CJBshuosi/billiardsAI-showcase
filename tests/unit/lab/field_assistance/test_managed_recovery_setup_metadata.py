"""Real managed recovery/read-only queue control with resource verification isolated."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
from billiards_ai.lab.field_assistance import managed_resource_recovery as recovery
from billiards_ai.lab.shot_local.models import BallGroup
from tests.unit.scripts.test_offline_fullgame_observer import SID
from tests.unit.scripts.test_offline_fullgame_records import actual_shot, attach, real_records


@pytest.mark.parametrize("fault", [None, "setup_mismatch", "media_path", "noncanonical"])
@pytest.mark.parametrize("purpose", ["analysis", "cleanup"])
@pytest.mark.parametrize("utf8_root", [False, True])
def test_unicode_setup_flows_through_actual_managed_recovery_without_becoming_path(
    tmp_path, monkeypatch, fault, purpose, utf8_root
):
    if utf8_root:
        tmp_path = tmp_path / "工作区"
        tmp_path.mkdir()
    store, spec, jobs, actor, coordinator = real_records(tmp_path, capture_setup_id="台球桌")
    counts = {BallGroup.CUE: 1, BallGroup.EIGHT: 1, BallGroup.SOLID: 7, BallGroup.STRIPE: 7}
    _, job, _ = actual_shot(store, spec, jobs, coordinator, 1, counts)
    attach(store, job)
    store.end_session(SID)
    artifacts = Path(spec.paths["artifact_root"])
    artifacts.mkdir()
    store.bind_artifact_path(spec.artifact_id, artifacts)
    verified = []

    def metadata_boundary(actual, *, purpose):
        # This test qualifies reader/recovery composition, not asset verification.
        # No model/real asset inspection or decoder is invoked.
        assert actual == spec
        verified.append(purpose)

    monkeypatch.setattr(recovery, "verify_managed_runtime_resources", metadata_boundary)
    if fault:
        with sqlite3.connect(actor.database) as db:
            data = json.loads(
                db.execute("select payload_json from shot_analysis_jobs").fetchone()[0]
            )
            if fault != "noncanonical":
                data["capture_setup_id" if fault == "setup_mismatch" else "media_root"] = (
                    "另一张台球桌" if fault == "setup_mismatch" else str(tmp_path / "outside")
                )
            # Keep the real writer encoding for dependency/path negatives so
            # canonical rejection cannot mask those downstream checks.
            encoded = json.dumps(
                data, sort_keys=True, separators=(",", ":"), ensure_ascii=fault == "noncanonical"
            )
            db.execute(
                "update shot_analysis_jobs set payload_json=?,payload_sha256=?",
                (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
            )
    before = actor.database.read_bytes()
    scope = ManagedSessionScope("offline-development", "offline-device", "offline-table")
    if fault:
        with pytest.raises(recovery.ManagedRuntimeResourceUnavailable):
            recovery.resolve_managed_session_resources(store, scope, SID, purpose=purpose)
    else:
        resolved = recovery.resolve_managed_session_resources(store, scope, SID, purpose=purpose)
        assert resolved.spec_record.spec.capture_setup_id == "台球桌"
        assert resolved.jobs_root == Path(spec.paths["jobs_root"])
        assert "台球桌" not in resolved.jobs_root.parts
        assert verified == (["cleanup", "analysis"] if purpose == "analysis" else ["cleanup"])
    assert actor.database.read_bytes() == before
