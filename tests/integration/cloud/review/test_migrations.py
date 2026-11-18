from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPECTED_TABLES = {
    "review_reports",
    "review_result_reports",
    "review_tasks",
    "review_shot_sources",
    "review_shot_decisions",
    "review_corrections",
    "review_shot_results",
    "review_rejections",
    "review_session_ownership",
    "review_publication_lineage",
    "review_metric_authority",
    "approved_evidence_batches",
    "approved_batch_sources",
    "quality_gate_idempotency_receipts",
    "quality_gate_outbox",
}


def _alembic_config(test_database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", test_database_url)
    return config


def _catalog_snapshot(sync_connection) -> dict[str, object]:
    inspector = inspect(sync_connection)
    return {
        "tables": set(inspector.get_table_names()),
        "task_checks": {item["name"] for item in inspector.get_check_constraints("review_tasks")},
        "outbox_indexes": inspector.get_indexes("quality_gate_outbox"),
        "ownership_uniques": inspector.get_unique_constraints("review_session_ownership"),
        "publication_columns": {
            item["name"] for item in inspector.get_columns("review_publication_lineage")
        },
        "metric_columns": {
            item["name"] for item in inspector.get_columns("review_metric_authority")
        },
        "batch_columns": {
            item["name"] for item in inspector.get_columns("approved_evidence_batches")
        },
    }


def _foreign_keys_without_leading_index(sync_connection) -> list[str]:
    inspector = inspect(sync_connection)
    missing: list[str] = []
    for table in sorted(EXPECTED_TABLES):
        candidates: list[tuple[str, ...]] = []
        primary_key = inspector.get_pk_constraint(table).get("constrained_columns") or []
        if primary_key:
            candidates.append(tuple(primary_key))
        candidates.extend(
            tuple(item["column_names"])
            for item in inspector.get_unique_constraints(table)
            if item.get("column_names")
        )
        candidates.extend(
            tuple(item["column_names"])
            for item in inspector.get_indexes(table)
            if item.get("column_names")
        )
        for foreign_key in inspector.get_foreign_keys(table):
            columns = tuple(foreign_key["constrained_columns"])
            if not any(candidate[: len(columns)] == columns for candidate in candidates):
                missing.append(f"{table}.{foreign_key['name']}({','.join(columns)})")
    return missing


@pytest.fixture
def migrated_database(test_database_url: str):
    config = _alembic_config(test_database_url)
    command.downgrade(config, "base")
    yield config
    command.downgrade(config, "base")


def test_alembic_configuration_exists():
    assert (PROJECT_ROOT / "alembic.ini").is_file()


async def test_migration_has_exactly_one_head_and_reaches_it(migrated_database, test_database_url):
    config = migrated_database
    script = ScriptDirectory.from_config(config)

    assert script.get_heads() == ["20260816_03"]

    await asyncio.to_thread(command.upgrade, config, "head")
    engine = create_async_engine(test_database_url)
    try:
        async with engine.connect() as connection:
            current = (
                await connection.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one()
        assert current == script.get_current_head()
    finally:
        await engine.dispose()


async def test_upgrade_creates_the_quality_gate_catalog(migrated_database, test_database_url):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        async with engine.connect() as connection:
            snapshot = await connection.run_sync(_catalog_snapshot)
            trigger_names = set(
                (
                    await connection.execute(
                        text(
                            "SELECT tgname FROM pg_trigger "
                            "WHERE NOT tgisinternal AND tgname LIKE 'trg_review_%'"
                        )
                    )
                ).scalars()
            )
        assert EXPECTED_TABLES <= snapshot["tables"]
        assert {
            "task_status",
            "batch_id",
            "event_id",
            "content_sha256",
            "approval_kind",
            "approval_policy_version",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        } <= snapshot["publication_columns"]
        assert {
            "result_report_id",
            "result_report_revision",
            "session_id",
            "sequence",
            "review_status",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        } <= snapshot["metric_columns"]
        assert {
            "task_id",
            "event_id",
            "approval_kind",
            "approval_policy_version",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        } <= snapshot["batch_columns"]
        assert any(
            item["name"] == "uq_review_session_ownership_authority"
            and item["column_names"] == ["task_id", "session_id", "user_id"]
            for item in snapshot["ownership_uniques"]
        )
        assert {
            "ck_review_tasks_status",
            "ck_review_tasks_claim_all_or_none",
            "ck_review_tasks_terminal_shape",
        } <= snapshot["task_checks"]

        pending_index = next(
            item
            for item in snapshot["outbox_indexes"]
            if item["name"] == "ix_quality_gate_outbox_pending"
        )
        predicate = str(pending_index["dialect_options"]["postgresql_where"])
        assert predicate.strip("()") == "published_at IS NULL"
        assert {
            "trg_review_reports_parent_lineage",
            "trg_review_tasks_terminal_authority",
            "trg_review_decisions_terminal_authority",
            "trg_review_corrections_terminal_authority",
            "trg_review_results_terminal_authority",
            "trg_review_rejections_terminal_authority",
        } <= trigger_names
    finally:
        await engine.dispose()


async def test_migration_round_trip_can_upgrade_a_nonempty_existing_database(
    migrated_database, test_database_url
):
    await asyncio.to_thread(command.upgrade, migrated_database, "20260810_01")
    engine = create_async_engine(test_database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO review_reports "
                    "(report_id, session_id, revision, schema_version, status, generated_at, "
                    "payload, payload_sha256) VALUES "
                    "('00000000-0000-4000-8000-000000000001', "
                    "'00000000-0000-4000-8000-000000000002', 1, '1.0', "
                    "'draft_pending_review', now(), '{}'::jsonb, repeat('0', 64))"
                )
            )
        await asyncio.to_thread(command.upgrade, migrated_database, "head")
        async with engine.connect() as connection:
            snapshot = await connection.run_sync(_catalog_snapshot)
            assert EXPECTED_TABLES <= snapshot["tables"]
            assert (
                await connection.execute(text("SELECT count(*) FROM review_reports"))
            ).scalar_one() == 1

        await engine.dispose()
        await asyncio.to_thread(command.downgrade, migrated_database, "20260810_01")
        engine = create_async_engine(test_database_url)
        async with engine.connect() as connection:
            tables = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
            assert "review_session_ownership" not in tables
            assert (
                await connection.execute(text("SELECT count(*) FROM review_reports"))
            ).scalar_one() == 1

        await engine.dispose()
        await asyncio.to_thread(command.upgrade, migrated_database, "head")
        engine = create_async_engine(test_database_url)
        async with engine.connect() as connection:
            assert (
                await connection.execute(text("SELECT count(*) FROM review_reports"))
            ).scalar_one() == 1
    finally:
        await engine.dispose()


async def test_upgrade_fails_before_alter_when_legacy_publication_rows_exist(
    migrated_database, test_database_url
):
    await asyncio.to_thread(command.upgrade, migrated_database, "20260810_01")
    engine = create_async_engine(test_database_url)
    ids = {
        "report_id": "11000000-0000-4000-8000-000000000001",
        "session_id": "11000000-0000-4000-8000-000000000002",
        "task_id": "11000000-0000-4000-8000-000000000003",
        "user_id": "11000000-0000-4000-8000-000000000004",
    }
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO review_reports "
                    "(report_id, session_id, revision, schema_version, status, generated_at, "
                    "payload, payload_sha256) VALUES "
                    "(:report_id, :session_id, 1, '1.0', 'draft_pending_review', now(), "
                    "'{}'::jsonb, repeat('0', 64))"
                ),
                ids,
            )
            await connection.execute(
                text(
                    "INSERT INTO review_result_reports "
                    "(report_id, revision, source_report_id, source_report_revision, "
                    "parent_report_id, session_id, schema_version, status, generated_at, "
                    "payload, payload_sha256) VALUES "
                    "(:report_id, 1, :report_id, 1, NULL, :session_id, '1.0', 'APPROVED', "
                    "now(), '{}'::jsonb, repeat('0', 64))"
                ),
                ids,
            )
            await connection.execute(
                text(
                    "INSERT INTO review_tasks "
                    "(task_id, session_id, source_report_id, source_report_revision, status, "
                    "lock_version, created_at, updated_at) VALUES "
                    "(:task_id, :session_id, :report_id, 1, 'pending_review', 0, now(), now())"
                ),
                ids,
            )
            await connection.execute(
                text(
                    "INSERT INTO review_publication_lineage "
                    "(task_id, user_id, session_id, report_id, review_revision, approved_at, "
                    "payload, payload_sha256) VALUES "
                    "(:task_id, :user_id, :session_id, :report_id, 1, now(), '{}'::jsonb, "
                    "repeat('0', 64))"
                ),
                ids,
            )

        with pytest.raises(
            DBAPIError,
            match="cannot upgrade 20260810_02: legacy quality-gate publication rows require "
            "explicit migration",
        ):
            await asyncio.to_thread(command.upgrade, migrated_database, "head")

        async with engine.connect() as connection:
            columns = await connection.run_sync(
                lambda sync: {
                    item["name"] for item in inspect(sync).get_columns("review_publication_lineage")
                }
            )
            current = (
                await connection.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one()
        assert "task_status" not in columns
        assert current == "20260810_01"
    finally:
        await engine.dispose()


async def test_migrations_run_only_on_postgresql_17_test_database(
    migrated_database, test_database_url
):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        async with engine.connect() as connection:
            version = int((await connection.execute(text("SHOW server_version_num"))).scalar_one())
            database_name = (
                await connection.execute(text("SELECT current_database()"))
            ).scalar_one()
        assert 170000 <= version < 180000
        assert "test" in database_name.lower()
    finally:
        await engine.dispose()


async def test_every_foreign_key_has_a_leading_index(migrated_database, test_database_url):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        async with engine.connect() as connection:
            missing = await connection.run_sync(_foreign_keys_without_leading_index)
        assert missing == []
    finally:
        await engine.dispose()


async def test_parent_report_must_continue_same_session_by_exactly_one_revision(
    migrated_database, test_database_url
):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    insert = text(
        "INSERT INTO review_reports "
        "(report_id, session_id, revision, parent_report_id, schema_version, status, "
        "generated_at, payload, payload_sha256) VALUES "
        "(:report_id, :session_id, :revision, :parent_report_id, '1.0', "
        "'draft_pending_review', now(), '{}'::jsonb, repeat('0', 64))"
    )
    try:
        async with engine.begin() as connection:
            await connection.execute(
                insert,
                {
                    "report_id": "10000000-0000-4000-8000-000000000001",
                    "session_id": "10000000-0000-4000-8000-000000000002",
                    "revision": 1,
                    "parent_report_id": None,
                },
            )
        for report_id, session_id, revision in (
            (
                "10000000-0000-4000-8000-000000000003",
                "10000000-0000-4000-8000-000000000004",
                2,
            ),
            (
                "10000000-0000-4000-8000-000000000005",
                "10000000-0000-4000-8000-000000000002",
                3,
            ),
        ):
            with pytest.raises(IntegrityError):
                async with engine.begin() as connection:
                    await connection.execute(
                        insert,
                        {
                            "report_id": report_id,
                            "session_id": session_id,
                            "revision": revision,
                            "parent_report_id": "10000000-0000-4000-8000-000000000001",
                        },
                    )
    finally:
        await engine.dispose()


async def test_uuid_authorities_reject_non_v4_values(migrated_database, test_database_url):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        with pytest.raises(IntegrityError):
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "INSERT INTO review_reports "
                        "(report_id, session_id, revision, schema_version, status, generated_at, "
                        "payload, payload_sha256) VALUES "
                        "('10000000-0000-1000-8000-000000000001', "
                        "'10000000-0000-4000-8000-000000000002', 1, '1.0', "
                        "'draft_pending_review', now(), '{}'::jsonb, repeat('0', 64))"
                    )
                )
    finally:
        await engine.dispose()


async def test_terminal_task_defers_then_rejects_incomplete_shot_coverage(
    migrated_database, test_database_url
):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    report_id = "20000000-0000-4000-8000-000000000001"
    session_id = "20000000-0000-4000-8000-000000000002"
    task_id = "20000000-0000-4000-8000-000000000003"
    shot_id = "20000000-0000-4000-8000-000000000004"
    envelope_id = "20000000-0000-4000-8000-000000000005"
    try:
        with pytest.raises(IntegrityError, match="invalid terminal task authority coverage"):
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "INSERT INTO review_reports "
                        "(report_id, session_id, revision, schema_version, status, generated_at, "
                        "payload, payload_sha256) VALUES "
                        "(:report_id, :session_id, 1, '1.0', 'draft_pending_review', now(), "
                        "'{}'::jsonb, repeat('0', 64))"
                    ),
                    {"report_id": report_id, "session_id": session_id},
                )
                await connection.execute(
                    text(
                        "INSERT INTO review_result_reports "
                        "(report_id, revision, source_report_id, source_report_revision, "
                        "parent_report_id, session_id, schema_version, status, generated_at, "
                        "payload, payload_sha256) VALUES "
                        "(:report_id, 1, :report_id, 1, NULL, :session_id, '1.0', 'APPROVED', "
                        "now(), '{}'::jsonb, repeat('0', 64))"
                    ),
                    {"report_id": report_id, "session_id": session_id},
                )
                await connection.execute(
                    text(
                        "INSERT INTO review_tasks "
                        "(task_id, session_id, source_report_id, source_report_revision, status, "
                        "claimed_principal_id, claimed_credential_id, claimed_at, lock_version, "
                        "created_at, updated_at) VALUES "
                        "(:task_id, :session_id, :report_id, 1, 'in_review', 'reviewer', "
                        "'credential', now(), 1, now(), now())"
                    ),
                    {"task_id": task_id, "session_id": session_id, "report_id": report_id},
                )
                await connection.execute(
                    text(
                        "INSERT INTO review_shot_sources "
                        "(task_id, shot_id, session_id, sequence, source_envelope_id, payload, "
                        "payload_sha256) VALUES "
                        "(:task_id, :shot_id, :session_id, 1, :envelope_id, '{}'::jsonb, "
                        "repeat('0', 64))"
                    ),
                    {
                        "task_id": task_id,
                        "shot_id": shot_id,
                        "session_id": session_id,
                        "envelope_id": envelope_id,
                    },
                )
                await connection.execute(
                    text(
                        "UPDATE review_tasks SET status='approved', result_report_id=:report_id, "
                        "result_report_revision=1, completed_at=now(), updated_at=now() "
                        "WHERE task_id=:task_id"
                    ),
                    {"task_id": task_id, "report_id": report_id},
                )
                await connection.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
    finally:
        await engine.dispose()


async def _insert_partial_rejected_terminal(connection) -> dict[str, str]:
    ids = {
        "report_id": "30000000-0000-4000-8000-000000000001",
        "session_id": "30000000-0000-4000-8000-000000000002",
        "task_id": "30000000-0000-4000-8000-000000000003",
        "approved_shot_id": "30000000-0000-4000-8000-000000000004",
        "rejected_shot_id": "30000000-0000-4000-8000-000000000005",
        "approved_envelope_id": "30000000-0000-4000-8000-000000000006",
        "rejected_envelope_id": "30000000-0000-4000-8000-000000000007",
        "user_id": "30000000-0000-4000-8000-000000000008",
    }
    await connection.execute(
        text(
            "INSERT INTO review_reports "
            "(report_id, session_id, revision, schema_version, status, generated_at, "
            "payload, payload_sha256) VALUES "
            "(:report_id, :session_id, 1, '1.0', 'draft_pending_review', now(), "
            "'{}'::jsonb, repeat('0', 64))"
        ),
        ids,
    )
    await connection.execute(
        text(
            "INSERT INTO review_result_reports "
            "(report_id, revision, source_report_id, source_report_revision, parent_report_id, "
            "session_id, schema_version, status, generated_at, payload, payload_sha256) VALUES "
            "(:report_id, 1, :report_id, 1, NULL, :session_id, '1.0', 'APPROVED', now(), "
            "'{}'::jsonb, repeat('0', 64))"
        ),
        ids,
    )
    await connection.execute(
        text(
            "INSERT INTO review_tasks "
            "(task_id, session_id, source_report_id, source_report_revision, status, "
            "claimed_principal_id, claimed_credential_id, claimed_at, lock_version, "
            "created_at, updated_at) VALUES "
            "(:task_id, :session_id, :report_id, 1, 'in_review', 'reviewer', 'credential', "
            "now(), 1, now(), now())"
        ),
        ids,
    )
    for sequence, shot_key, envelope_key in (
        (1, "approved_shot_id", "approved_envelope_id"),
        (2, "rejected_shot_id", "rejected_envelope_id"),
    ):
        await connection.execute(
            text(
                "INSERT INTO review_shot_sources "
                "(task_id, shot_id, session_id, sequence, source_envelope_id, payload, "
                "payload_sha256) VALUES "
                "(:task_id, :shot_id, :session_id, :sequence, :envelope_id, '{}'::jsonb, "
                "repeat('0', 64))"
            ),
            {
                **ids,
                "shot_id": ids[shot_key],
                "sequence": sequence,
                "envelope_id": ids[envelope_key],
            },
        )
    for sequence, shot_key, outcome in (
        (1, "approved_shot_id", "approved"),
        (2, "rejected_shot_id", "rejected"),
    ):
        await connection.execute(
            text(
                "INSERT INTO review_shot_decisions "
                "(task_id, shot_id, outcome, reviewer_principal_id, reviewer_credential_id, "
                "reviewed_at, trajectory_reviewed, payload, payload_sha256) VALUES "
                "(:task_id, :shot_id, :outcome, 'reviewer', 'credential', now(), true, "
                "'{}'::jsonb, repeat('0', 64))"
            ),
            {**ids, "shot_id": ids[shot_key], "outcome": outcome},
        )
        await connection.execute(
            text(
                "INSERT INTO review_shot_results "
                "(task_id, shot_id, result_report_id, result_report_revision, session_id, "
                "sequence, review_status, payload, payload_sha256) VALUES "
                "(:task_id, :shot_id, :report_id, 1, :session_id, :sequence, :outcome, "
                "'{}'::jsonb, repeat('0', 64))"
            ),
            {
                **ids,
                "shot_id": ids[shot_key],
                "sequence": sequence,
                "outcome": outcome,
            },
        )
    await connection.execute(
        text(
            "INSERT INTO review_session_ownership "
            "(task_id, session_id, user_id, bound_at, payload, payload_sha256) VALUES "
            "(:task_id, :session_id, :user_id, now(), '{}'::jsonb, repeat('0', 64))"
        ),
        ids,
    )
    await connection.execute(
        text(
            "UPDATE review_tasks SET status='approved', result_report_id=:report_id, "
            "result_report_revision=1, completed_at=now(), updated_at=now(), lock_version=2 "
            "WHERE task_id=:task_id"
        ),
        ids,
    )
    return ids


async def _insert_exact_publication_graph(connection, ids: dict[str, str]) -> dict[str, str]:
    authority = {
        **ids,
        "batch_id": "30000000-0000-4000-8000-000000000009",
        "event_id": "30000000-0000-4000-8000-00000000000a",
        "calibration_id": "30000000-0000-4000-8000-00000000000b",
        "calibration_revision_id": "30000000-0000-4000-8000-00000000000c",
        "content_sha256": "1" * 64,
    }
    await connection.execute(
        text(
            "INSERT INTO review_publication_lineage "
            "(task_id, user_id, session_id, report_id, review_revision, approved_at, payload, "
            "payload_sha256, task_status, batch_id, event_id, content_sha256, approval_kind, "
            "approval_policy_version, calibration_id, calibration_revision_id, "
            "calibration_revision_number) VALUES "
            "(:task_id, :user_id, :session_id, :report_id, 1, now(), '{}'::jsonb, "
            "repeat('0', 64), 'approved', :batch_id, :event_id, :content_sha256, "
            "'HUMAN_REVIEW', 'human-review-v1', :calibration_id, "
            ":calibration_revision_id, 1)"
        ),
        authority,
    )
    await connection.execute(
        text(
            "INSERT INTO review_metric_authority "
            "(task_id, shot_id, review_revision, payload, payload_sha256, result_report_id, "
            "result_report_revision, session_id, sequence, review_status, calibration_id, "
            "calibration_revision_id, calibration_revision_number) VALUES "
            "(:task_id, :approved_shot_id, 1, '{}'::jsonb, repeat('0', 64), :report_id, 1, "
            ":session_id, 1, 'approved', :calibration_id, :calibration_revision_id, 1)"
        ),
        authority,
    )
    await connection.execute(
        text(
            "INSERT INTO approved_evidence_batches "
            "(batch_id, schema_version, user_id, session_id, report_id, review_revision, "
            "approved_at, content_sha256, payload, payload_sha256, task_id, event_id, "
            "approval_kind, approval_policy_version, calibration_id, calibration_revision_id, "
            "calibration_revision_number) VALUES "
            "(:batch_id, '2.0', :user_id, :session_id, :report_id, 1, now(), "
            ":content_sha256, '{}'::jsonb, repeat('0', 64), :task_id, :event_id, "
            "'HUMAN_REVIEW', 'human-review-v1', :calibration_id, "
            ":calibration_revision_id, 1)"
        ),
        authority,
    )
    await connection.execute(
        text(
            "INSERT INTO approved_batch_sources "
            "(batch_id, task_id, shot_id, sequence, source_envelope_id, report_id, "
            "review_revision, review_status, session_id) VALUES "
            "(:batch_id, :task_id, :approved_shot_id, 1, :approved_envelope_id, :report_id, "
            "1, 'approved', :session_id)"
        ),
        authority,
    )
    await connection.execute(
        text(
            "INSERT INTO quality_gate_outbox "
            "(event_id, schema_version, event_type, aggregate_type, aggregate_id, "
            "aggregate_version, occurred_at, causation_id, correlation_id, idempotency_key, "
            "payload, payload_sha256, event_sha256, available_at, attempt_count) VALUES "
            "(:event_id, '1.0', 'evidence.session.approved.v1', 'review_session', "
            ":session_id, 1, now(), :task_id, :session_id, 'approved:1', '{}'::jsonb, "
            "repeat('0', 64), repeat('2', 64), now(), 0)"
        ),
        authority,
    )
    await connection.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
    return authority


async def test_terminal_guard_allows_partial_rejected_with_one_promotable_shot(
    migrated_database, test_database_url
):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        async with engine.begin() as connection:
            ids = await _insert_partial_rejected_terminal(connection)
            await _insert_exact_publication_graph(connection, ids)
        async with engine.connect() as connection:
            outcomes = tuple(
                (
                    await connection.execute(
                        text(
                            "SELECT outcome FROM review_shot_decisions WHERE task_id=:task_id "
                            "ORDER BY shot_id"
                        ),
                        ids,
                    )
                ).scalars()
            )
        assert outcomes == ("approved", "rejected")
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    "statement",
    (
        "UPDATE review_session_ownership SET user_id="
        "'31000000-0000-4000-8000-000000000001' WHERE task_id=:task_id",
        "DELETE FROM review_session_ownership WHERE task_id=:task_id",
    ),
)
async def test_review_session_ownership_cannot_be_updated_or_deleted(
    migrated_database, test_database_url, statement
):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        async with engine.begin() as connection:
            ids = await _insert_partial_rejected_terminal(connection)
            await _insert_exact_publication_graph(connection, ids)

        with pytest.raises(IntegrityError, match="review session ownership is immutable"):
            async with engine.begin() as connection:
                await connection.execute(text(statement), ids)
    finally:
        await engine.dispose()


async def test_terminal_guard_rejects_result_report_from_another_session(
    migrated_database, test_database_url
):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        with pytest.raises(IntegrityError, match="invalid terminal task result report identity"):
            async with engine.begin() as connection:
                ids = await _insert_partial_rejected_terminal(connection)
                await connection.execute(
                    text(
                        "UPDATE review_result_reports SET session_id="
                        "'32000000-0000-4000-8000-000000000001' "
                        "WHERE report_id=:report_id AND revision=1"
                    ),
                    ids,
                )
                await connection.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
    finally:
        await engine.dispose()


async def test_terminal_guard_rejects_result_shot_with_nonexact_source_sequence(
    migrated_database, test_database_url
):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        with pytest.raises(IntegrityError, match="invalid terminal task result source identity"):
            async with engine.begin() as connection:
                ids = await _insert_partial_rejected_terminal(connection)
                await connection.execute(
                    text(
                        "UPDATE review_shot_results SET sequence=3 "
                        "WHERE task_id=:task_id AND shot_id=:approved_shot_id"
                    ),
                    ids,
                )
                await connection.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
    finally:
        await engine.dispose()


async def test_terminal_guard_rejects_approved_task_with_corrected_shot(
    migrated_database, test_database_url
):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        with pytest.raises(IntegrityError, match="invalid terminal task outcome"):
            async with engine.begin() as connection:
                ids = await _insert_partial_rejected_terminal(connection)
                await connection.execute(
                    text(
                        "UPDATE review_shot_decisions SET outcome='corrected' "
                        "WHERE task_id=:task_id AND shot_id=:approved_shot_id"
                    ),
                    ids,
                )
                await connection.execute(
                    text(
                        "UPDATE review_shot_results SET review_status='corrected' "
                        "WHERE task_id=:task_id AND shot_id=:approved_shot_id"
                    ),
                    ids,
                )
                await connection.execute(
                    text(
                        "INSERT INTO review_corrections "
                        "(task_id, shot_id, field, ai_value, correct_value, note, "
                        "reviewer_principal_id, reviewer_credential_id, reviewed_at, payload, "
                        "payload_sha256) VALUES (:task_id, :approved_shot_id, 'event', "
                        "'null'::jsonb, 'null'::jsonb, 'corrected', 'reviewer', 'credential', "
                        "now(), '{}'::jsonb, repeat('0', 64))"
                    ),
                    ids,
                )
                await connection.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
    finally:
        await engine.dispose()


def _foreign_key_column_signatures(sync_connection, table: str) -> set[tuple[str, ...]]:
    return {
        tuple(item["constrained_columns"])
        for item in inspect(sync_connection).get_foreign_keys(table)
    }


async def test_publication_metric_batch_and_source_have_exact_composite_foreign_keys(
    migrated_database, test_database_url
):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        async with engine.connect() as connection:
            publication_fks = await connection.run_sync(
                _foreign_key_column_signatures, "review_publication_lineage"
            )
            metric_fks = await connection.run_sync(
                _foreign_key_column_signatures, "review_metric_authority"
            )
            batch_fks = await connection.run_sync(
                _foreign_key_column_signatures, "approved_evidence_batches"
            )
            source_fks = await connection.run_sync(
                _foreign_key_column_signatures, "approved_batch_sources"
            )
        assert ("task_id", "session_id", "user_id") in publication_fks
        assert (
            "task_id",
            "session_id",
            "report_id",
            "review_revision",
            "task_status",
        ) in publication_fks
        assert (
            "task_id",
            "shot_id",
            "result_report_id",
            "result_report_revision",
            "review_status",
            "sequence",
            "session_id",
        ) in metric_fks
        assert (
            "task_id",
            "batch_id",
            "user_id",
            "session_id",
            "report_id",
            "review_revision",
            "event_id",
            "content_sha256",
            "approval_kind",
            "approval_policy_version",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        ) in batch_fks
        assert (
            "task_id",
            "shot_id",
            "session_id",
            "sequence",
            "source_envelope_id",
        ) in source_fks
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("statement", "parameters"),
    (
        (
            "UPDATE review_publication_lineage SET user_id=:wrong_id WHERE task_id=:task_id",
            {"wrong_id": "30000000-0000-4000-8000-00000000000d"},
        ),
        (
            "UPDATE review_metric_authority SET sequence=2 "
            "WHERE task_id=:task_id AND shot_id=:approved_shot_id",
            {},
        ),
        (
            "UPDATE approved_evidence_batches SET event_id=:wrong_id WHERE batch_id=:batch_id",
            {"wrong_id": "30000000-0000-4000-8000-00000000000e"},
        ),
        (
            "UPDATE approved_batch_sources SET source_envelope_id=:wrong_id "
            "WHERE batch_id=:batch_id AND shot_id=:approved_shot_id",
            {"wrong_id": "30000000-0000-4000-8000-00000000000f"},
        ),
    ),
)
async def test_direct_sql_cannot_break_exact_publication_authority(
    migrated_database, test_database_url, statement, parameters
):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        async with engine.begin() as connection:
            ids = await _insert_partial_rejected_terminal(connection)
            authority = await _insert_exact_publication_graph(connection, ids)

        with pytest.raises(IntegrityError):
            async with engine.begin() as connection:
                await connection.execute(text(statement), {**authority, **parameters})
                await connection.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))

        async with engine.connect() as connection:
            assert (
                await connection.execute(
                    text("SELECT count(*) FROM approved_batch_sources WHERE batch_id=:batch_id"),
                    authority,
                )
            ).scalar_one() == 1
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("statement", "parameters"),
    (
        (
            "DELETE FROM review_metric_authority "
            "WHERE task_id=:task_id AND shot_id=:approved_shot_id",
            {},
        ),
        (
            "DELETE FROM approved_batch_sources "
            "WHERE task_id=:task_id AND shot_id=:approved_shot_id",
            {},
        ),
        (
            "UPDATE quality_gate_outbox SET event_type='evidence.session.rejected.v1' "
            "WHERE event_id=:event_id",
            {},
        ),
        (
            "INSERT INTO quality_gate_outbox "
            "(event_id, schema_version, event_type, aggregate_type, aggregate_id, "
            "aggregate_version, occurred_at, causation_id, correlation_id, idempotency_key, "
            "payload, payload_sha256, event_sha256, available_at, attempt_count) VALUES "
            "(:extra_event_id, '1.0', 'evidence.session.approved.v1', 'review_session', "
            ":session_id, 2, now(), :task_id, :session_id, 'approved:extra', '{}'::jsonb, "
            "repeat('0', 64), repeat('2', 64), now(), 0)",
            {"extra_event_id": "33000000-0000-4000-8000-000000000001"},
        ),
        (
            "UPDATE review_publication_lineage SET task_status='corrected' WHERE task_id=:task_id",
            {},
        ),
    ),
)
async def test_terminal_publication_graph_rejects_delete_extra_or_wrong_status(
    migrated_database, test_database_url, statement, parameters
):
    await asyncio.to_thread(command.upgrade, migrated_database, "head")
    engine = create_async_engine(test_database_url)
    try:
        async with engine.begin() as connection:
            ids = await _insert_partial_rejected_terminal(connection)
            authority = await _insert_exact_publication_graph(connection, ids)

        with pytest.raises(IntegrityError):
            async with engine.begin() as connection:
                await connection.execute(text(statement), {**authority, **parameters})
                await connection.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
    finally:
        await engine.dispose()
