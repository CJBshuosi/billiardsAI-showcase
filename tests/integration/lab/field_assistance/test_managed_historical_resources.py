"""Retained scope, owner and resource roots across managed scene replacement."""

import hashlib
import sqlite3
import threading
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance.control import MatchControlSnapshot, WristbandRole
from billiards_ai.edge.route_assistance.records import (
    FieldShotRecord,
    LiveShotMediaReference,
    ShotMediaRange,
)
from billiards_ai.lab.field_assistance import cloud_runtime
from billiards_ai.lab.field_assistance import server as service
from tests.integration.lab.field_assistance.test_live_server import review_request
from tests.integration.lab.field_assistance.test_managed_live_startup import (
    artifact_id,
    options_at,
    rows,
    seed,
)
from tests.unit.lab.field_assistance.test_managed_startup import BINDING, SCOPE, Remote, coordinator


def ended_clip(store, session_id, root):
    """Produce real immutable USB lineage and owner receipts, with synthetic clip bytes."""
    session = store.load_session(session_id)
    store.save_match(
        session_id,
        MatchControlSnapshot(
            red_user_id="original-owner",
            blue_user_id="original-blue",
            active_role=WristbandRole.RED,
            revision=1,
        ),
        expected_revision=0,
    )
    start = FieldShotRecord(
        session_id,
        "shot-0001",
        1,
        WristbandRole.RED,
        100,
        None,
        None,
        None,
        None,
        ShotMediaRange(
            session.artifact_id, "usb:old-epoch", "usb_live", 0, 1000, source_epoch_id="old-epoch"
        ),
    )
    store.record_shot_start(start, source_start_event_id="start-1")
    from dataclasses import replace

    store.record_shot_settle(
        replace(start, settled_ms=200, live_job_id="job-1", live_evidence_digest="d" * 64),
        source_epoch_id="old-epoch",
    )
    store.attach_live_shot_media(
        LiveShotMediaReference(
            session_id,
            start.shot_id,
            "old-epoch",
            1,
            "job-1",
            str(root / "raw"),
            "d" * 64,
        )
    )
    owner = store.get_shot_ownership(session_id, start.shot_id)
    store.end_session(session_id)
    manifest = store.get_postgame_manifest(session_id)
    path = root / "postgame" / session_id / "clip.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"original-root-clip")
    store.record_postgame_clip_available(
        session_id=session_id,
        shot_id=start.shot_id,
        postgame_manifest_sha256=manifest.manifest_sha256,
        job_id="job-1",
        artifact_id="c" * 64,
        local_path=path,
        content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        byte_size=path.stat().st_size,
        content_type="video/mp4",
    )
    store.record_postgame_finalization(session_id, '{"clips":[],"report":{}}')
    return manifest, owner, path


def test_old_review_uses_original_root_for_pin_descriptor_save_owner_and_completion(tmp_path):
    options, _ = options_at(tmp_path)
    store = seed(options)
    original_root = options["artifact_path"]
    manifest, original_owner, path = ended_clip(store, BINDING.session_id, original_root)
    options["artifact_path"] = tmp_path / "new-root"
    server = service.build_live_field_assistance_server(
        **options, managed_startup=coordinator(Remote()), fresh_session_on_start=True
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        state = server.field_state
        review = state.postgame_review_payload(kiosk=True)
        assert review["review_session_id"] == BINDING.session_id
        assert review_request(server, review["clips"][0]["media_url"]) == (200, path.read_bytes())
        saved = state.mutate_postgame_review(
            dict(
                request_id="save-original",
                session_id=BINDING.session_id,
                shot_id="shot-0001",
                postgame_manifest_sha256=manifest.manifest_sha256,
            ),
            review["kiosk_nonce"],
            complete=False,
        )
        assert saved["owner_user_id"] == original_owner.actor_user_id == "original-owner"
        assert store.get_shot_ownership(BINDING.session_id, "shot-0001") == original_owner
        state.mutate_postgame_review(
            dict(session_id=BINDING.session_id, postgame_manifest_sha256=manifest.manifest_sha256),
            review["kiosk_nonce"],
            complete=True,
        )
        assert path.exists()
        assert state.postgame_review_payload()["review_session_id"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_analysis_skips_unknown_original_context_without_starving_new_session(tmp_path):
    options, _ = options_at(tmp_path)
    store = seed(options)
    server = service.build_live_field_assistance_server(
        **options, managed_startup=coordinator(Remote()), fresh_session_on_start=True
    )
    try:
        current = server.field_state.session_id
        store.end_session(current)
        commands, diagnostics = [], {}
        worker = cloud_runtime._BackgroundPostgameAnalysisWorker(
            store=store,
            command_factory=lambda sid: [sid],
            process_factory=lambda command, **_: (
                commands.append(command)
                or SimpleNamespace(
                    poll=lambda: None,
                )
            ),
            resource_session_ids=(current,),
            recovery_diagnostics=diagnostics,
        )
        worker.run_once()
        assert commands == [[current]]
        assert diagnostics[BINDING.session_id] == "POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE"
        assert store.get_postgame_manifest(BINDING.session_id) is not None
    finally:
        server.server_close()


def test_cleanup_claim_filter_leaves_historical_work_unleased_and_claims_current(tmp_path):
    options, _ = options_at(tmp_path)
    store = seed(options)
    old_manifest, _, _ = ended_clip(store, BINDING.session_id, options["artifact_path"])
    store.complete_postgame_review(
        store.load_session(BINDING.session_id).artifact_id,
        BINDING.session_id,
        old_manifest.manifest_sha256,
    )
    server = service.build_live_field_assistance_server(
        **options, managed_startup=coordinator(Remote()), fresh_session_on_start=True
    )
    try:
        current = server.field_state.session_id
        manifest, _, _ = ended_clip(store, current, options["artifact_path"])
        store.complete_postgame_review(
            store.load_session(current).artifact_id,
            current,
            manifest.manifest_sha256,
        )
        claim = store.claim_due_local_discard(
            "cleanup",
            datetime.now(UTC),
            session_ids=(current,),
        )
        assert claim.session_id == current
        with sqlite3.connect(store.path) as connection:
            old = connection.execute(
                "SELECT status,attempts,claim_token FROM field_local_media_discards "
                "WHERE session_id=?",
                (BINDING.session_id,),
            ).fetchone()
        assert old == ("pending", 0, None)
    finally:
        server.server_close()


def test_saved_media_uses_original_same_scope_root_and_does_not_claim_foreign(tmp_path):
    options, _ = options_at(tmp_path)
    store = seed(options)
    manifest, _, path = ended_clip(store, BINDING.session_id, options["artifact_path"])
    store.save_postgame_clip(
        request_id="save-old",
        session_id=BINDING.session_id,
        shot_id="shot-0001",
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    from dataclasses import replace

    assert (
        store.claim_due_saved_media(
            datetime.now(UTC),
            managed_scope=replace(SCOPE, table_id="foreign"),
        )
        is None
    )
    roots = []
    worker = cloud_runtime._BackgroundSavedMediaSyncWorker(
        store=store,
        client=SimpleNamespace(
            run_saved_media_once=lambda item, **kw: (
                roots.append(kw["trusted_postgame_root"]) or SimpleNamespace(outcome="complete")
            )
        ),
        trusted_postgame_root=tmp_path / "new-root/postgame",
        wake_event=threading.Event(),
        managed_scope=SCOPE,
    )
    assert worker.run_once()
    assert roots == [path.parents[1]]
    assert store.list_saved_media_outbox()[0].status == "acknowledged"


def test_managed_runtime_wires_current_context_filters_and_preserves_historical_diagnostic(
    tmp_path,
):
    from tests.unit.lab.field_assistance.test_cloud_runtime import _args, _dependencies

    options, _ = options_at(tmp_path)
    store = seed(options)
    foreign = store.open_or_resume_session(artifact_id="foreign-legacy-artifact")
    store.end_session(foreign.session_id)
    server = service.build_live_field_assistance_server(
        **options, managed_startup=coordinator(Remote()), fresh_session_on_start=True
    )
    factories, _, workers = _dependencies()
    runtime = cloud_runtime.prepare_cloud_runtime(_args(tmp_path), factories=factories)
    try:
        runtime.start(server, postgame_command_factory=lambda sid: [sid])
        current = server.field_state.session_id
        assert workers["postgame"].kwargs["resource_session_ids"] == (current,)
        assert workers["cleanup"].kwargs["session_ids"] == (current,)
        assert workers["cleanup"].kwargs["managed_scope"] == SCOPE
        assert workers["cleanup"].kwargs["recovery_diagnostics"] is (
            server.field_state.recovery_diagnostics
        )
        assert workers["saved"].kwargs["managed_scope"] == SCOPE
        assert server.field_state.recovery_diagnostics[BINDING.session_id] == (
            "POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE"
        )
        assert foreign.session_id not in server.field_state.recovery_diagnostics
    finally:
        runtime.stop()
        server.server_close()


@pytest.mark.parametrize("resumed", [False, True], ids=["new-context", "empty-context"])
def test_late_historical_review_completion_refreshes_cleanup_recovery_without_leasing(
    tmp_path,
    resumed,
):
    from dataclasses import replace

    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from tests.unit.lab.field_assistance.test_cloud_runtime import _args, _dependencies

    options, _ = options_at(tmp_path)
    store = seed(options)
    manifest, _, _ = ended_clip(store, BINDING.session_id, options["artifact_path"])
    foreign_scope = replace(SCOPE, table_id="foreign-table")
    foreign_request = "55555555-5555-4555-8555-555555555555"
    foreign_birth = "66666666-6666-4666-8666-666666666666"
    foreign_binding = replace(
        BINDING,
        session_id=f"managed-{foreign_birth}",
        birth_id=foreign_birth,
        table_id=foreign_scope.table_id,
    )
    foreign_store = SQLiteFieldRecordStore(
        store.path,
        managed_open_request_id_factory=lambda: foreign_request,
    )
    foreign_store.reserve_managed_open(foreign_scope, artifact_id="foreign-artifact")
    foreign_store.record_managed_binding(foreign_scope, foreign_request, foreign_binding)
    foreign = foreign_store.materialize_managed_session(foreign_scope, foreign_request)
    foreign_root = tmp_path / "foreign-root"
    store.bind_artifact_path(foreign.artifact_id, foreign_root)
    foreign_manifest, _, _ = ended_clip(store, foreign.session_id, foreign_root)
    assert store.list_ended_sessions_requiring_postgame() == ()
    assert store.list_pending_local_discard_session_ids() == ()
    remote = Remote()
    server = service.build_live_field_assistance_server(
        **options,
        managed_startup=coordinator(remote),
        fresh_session_on_start=True,
    )
    if resumed:
        server.server_close()
        options["source_epoch_factory"] = lambda: "resumed-epoch"
        server = service.build_live_field_assistance_server(
            **options,
            managed_startup=coordinator(remote),
        )
        assert server.field_state.resource_session_ids == ()
    cleanup_workers = []

    def controlled_cleanup(**kwargs):
        worker = cloud_runtime._BackgroundLocalMediaCleanupWorker(**kwargs)
        # Drive the real worker deterministically after the runtime has started.
        worker.start = lambda: None
        cleanup_workers.append(worker)
        return worker

    factories, _, _ = _dependencies()
    runtime = cloud_runtime.prepare_cloud_runtime(
        _args(tmp_path),
        factories=replace(factories, local_cleanup_worker=controlled_cleanup),
    )
    try:
        state = server.field_state
        runtime.start(server, postgame_command_factory=lambda sid: [sid])
        assert state.recovery_diagnostics == {}
        review = state.postgame_review_payload(kiosk=True)
        state.mutate_postgame_review(
            dict(session_id=BINDING.session_id, postgame_manifest_sha256=manifest.manifest_sha256),
            review["kiosk_nonce"],
            complete=True,
        )
        store.complete_postgame_review(
            foreign.artifact_id,
            foreign.session_id,
            foreign_manifest.manifest_sha256,
        )
        before = rows(store)
        assert cleanup_workers[0].run_once() is False
        # Only this scheduler's exact retained-scope cursor rows may change.
        assert cleanup_workers[0]._scheduler.scope == (
            SCOPE.environment_id,
            SCOPE.device_id,
            SCOPE.table_id,
        )
        key = cleanup_workers[0]._scheduler.key
        expected_schedule = {
            (key, "", None, "recording"),
            (key + ":recording", "", None, "owner"),
            (key + ":owner", "", None, "owner"),
            (key + ":discard", BINDING.session_id, BINDING.session_id, "owner"),
        }
        with sqlite3.connect(store.path) as connection:
            assert (
                set(
                    connection.execute(
                        "SELECT state_key,cursor,upper_key,turn "
                        "FROM field_local_media_schedule_state"
                    )
                )
                == expected_schedule
            )
        allowed_insertions = {
            'INSERT INTO "field_local_media_schedule_state" VALUES('
            + ",".join("NULL" if value is None else "'" + value + "'" for value in record)
            + ");"
            for record in expected_schedule
        }
        after = rows(store)
        assert allowed_insertions.issubset(after)
        assert tuple(line for line in after if line not in allowed_insertions) == before
        with sqlite3.connect(store.path) as connection:
            assert connection.execute(
                "SELECT status,attempts,claim_token FROM field_local_media_discards "
                "ORDER BY session_id",
            ).fetchall() == [("pending", 0, None), ("pending", 0, None)]
        payload = state.postgame_review_payload()
        assert payload["phase"] == "REVIEW_COMPLETE"
        assert payload["recovery_diagnostics"] == {
            BINDING.session_id: "LOCAL_MEDIA_CLEANUP_ORIGINAL_CONTEXT_UNAVAILABLE",
        }
        assert foreign.session_id not in str(payload)
    finally:
        runtime.stop()
        server.server_close()


def test_saved_media_missing_original_root_stays_pending_with_diagnostic(tmp_path):
    options, _ = options_at(tmp_path)
    store = seed(options)
    manifest, _, _ = ended_clip(store, BINDING.session_id, options["artifact_path"])
    store.save_postgame_clip(
        request_id="missing-old",
        session_id=BINDING.session_id,
        shot_id="shot-0001",
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    options["artifact_path"].rename(tmp_path / "unavailable-root")
    diagnostics = {}
    worker = cloud_runtime._BackgroundSavedMediaSyncWorker(
        store=store,
        client=SimpleNamespace(
            run_saved_media_once=lambda *a, **kw: pytest.fail(
                "missing original root must not use current media"
            )
        ),
        trusted_postgame_root=tmp_path / "new-root/postgame",
        wake_event=threading.Event(),
        managed_scope=SCOPE,
        recovery_diagnostics=diagnostics,
    )
    assert worker.run_once()
    assert store.list_saved_media_outbox()[0].status == "pending"
    assert diagnostics[BINDING.session_id] == "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE"


def test_unflagged_different_artifact_workers_exclude_managed_and_progress_legacy(tmp_path):
    options, _ = options_at(tmp_path)
    store = seed(options)
    old_manifest, _, _ = ended_clip(store, BINDING.session_id, options["artifact_path"])
    store.complete_postgame_review(
        store.load_session(BINDING.session_id).artifact_id,
        BINDING.session_id,
        old_manifest.manifest_sha256,
    )
    legacy_root = tmp_path / "legacy-root"
    legacy_root.mkdir()
    legacy = store.open_or_resume_session(artifact_id="legacy-artifact")
    store.bind_artifact_path(legacy.artifact_id, legacy_root)
    manifest, _, _ = ended_clip(store, legacy.session_id, legacy_root)
    store.complete_postgame_review(legacy.artifact_id, legacy.session_id, manifest.manifest_sha256)
    pending_legacy = store.open_or_resume_session(artifact_id=legacy.artifact_id)
    store.end_session(pending_legacy.session_id)
    launched = []
    worker = cloud_runtime._BackgroundPostgameAnalysisWorker(
        store=store,
        command_factory=lambda sid: [sid],
        process_factory=lambda command, **_: (
            launched.append(command)
            or SimpleNamespace(
                poll=lambda: None,
            )
        ),
        exclude_managed=True,
    )
    worker.run_once()
    assert launched == [[pending_legacy.session_id]]
    claim = store.claim_due_local_discard("legacy", datetime.now(UTC), exclude_managed=True)
    assert claim.session_id == legacy.session_id
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT status,attempts FROM field_local_media_discards WHERE session_id=?",
            (BINDING.session_id,),
        ).fetchone() == ("pending", 0)


def test_unflagged_saved_claim_excludes_managed_and_unknown_managed_identity(tmp_path):
    options, _ = options_at(tmp_path)
    store = seed(options)
    manifest, _, _ = ended_clip(store, BINDING.session_id, options["artifact_path"])
    store.save_postgame_clip(
        request_id="saved-managed",
        session_id=BINDING.session_id,
        shot_id="shot-0001",
        postgame_manifest_sha256=manifest.manifest_sha256,
    )
    assert store.claim_due_saved_media(datetime.now(UTC), exclude_managed=True) is None
    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM field_managed_open_requests")
    # A managed-looking name without retained binding is not inferred to be legacy.
    assert store.claim_due_saved_media(datetime.now(UTC), exclude_managed=True) is None
    assert store.list_saved_media_outbox()[0].attempts == 0


def test_unflagged_review_does_not_adopt_retired_managed_history_without_scope(tmp_path):
    options, _ = options_at(tmp_path)
    store = seed(options)
    old_root = options["artifact_path"]
    ended_clip(store, BINDING.session_id, old_root)
    options["artifact_path"] = tmp_path / "new-root"
    managed = service.build_live_field_assistance_server(
        **options,
        managed_startup=coordinator(Remote()),
        fresh_session_on_start=True,
    )
    options["artifact_path"] = old_root
    legacy = service.build_live_field_assistance_server(**options)
    try:
        before = rows(store)
        review = legacy.field_state.postgame_review_payload(kiosk=True)
        assert review["phase"] == "RECOVERY_REQUIRED"
        assert review["review_session_id"] is None and "kiosk_nonce" not in review
        assert BINDING.session_id not in str(review)
        assert rows(store) == before
    finally:
        legacy.server_close()
        managed.server_close()


def test_foreign_active_review_blocks_own_pending_with_visible_recovery_and_no_authority(tmp_path):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore

    options, _ = options_at(tmp_path)
    store = SQLiteFieldRecordStore(
        options["state_db_path"], session_id_factory=lambda: "unknown-foreign-session"
    )
    foreign = store.open_or_resume_session(artifact_id=artifact_id(options["artifact_path"]))
    store.end_session(foreign.session_id)
    store = seed(options)
    store.end_session(BINDING.session_id)
    server = service.build_live_field_assistance_server(
        **options,
        managed_startup=coordinator(Remote()),
        fresh_session_on_start=True,
    )
    try:
        before = rows(store)
        state = server.field_state
        review = state.postgame_review_payload(kiosk=True)
        assert review["phase"] == "RECOVERY_REQUIRED"
        assert review["review_session_id"] is None and review["clips"] == []
        assert review["can_complete"] is False and "kiosk_nonce" not in review
        assert review["recovery_diagnostics"][BINDING.session_id] == (
            "POSTGAME_REVIEW_QUEUE_RECOVERY_REQUIRED"
        )
        assert foreign.session_id not in str(review)
        with pytest.raises(PermissionError):
            state.mutate_postgame_review(
                dict(session_id=foreign.session_id, postgame_manifest_sha256="a" * 64),
                "",
                complete=True,
            )
        assert rows(store) == before
    finally:
        server.server_close()
