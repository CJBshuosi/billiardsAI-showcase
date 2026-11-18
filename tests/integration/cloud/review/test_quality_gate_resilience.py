"""Story 11.4 cold-rebuild, race, rollback, and privacy acceptance tests."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID

import pytest
from sqlalchemy import event, text

from billiards_ai.cloud.db import create_database_engine, create_session_factory
from billiards_ai.cloud.review.application import ReviewApplicationError, ReviewErrorCode
from billiards_ai.cloud.review.application_models import (
    ApprovedReviewResult,
    RejectedReviewResult,
)
from billiards_ai.shared.models import ReviewTaskStatus
from tests.integration.cloud.review.quality_gate_support import (
    ReviewScenario,
    authority_counts,
    claim_command,
    compose_runtime,
    finalize_command,
    load_synthetic_fixture,
    reject_command,
    stage_synthetic_graph,
)

FIXTURE = Path(__file__).parents[3] / "fixtures" / "quality_gate" / "v2" / "synthetic.json"


def _runtime(database_url, fixture):
    engine = create_database_engine(database_url)
    sessions = create_session_factory(engine)
    return engine, sessions, compose_runtime(sessions, fixture)


async def _authority_snapshot(session_factory) -> dict[str, tuple[str, ...]]:
    """Read a canonical logical graph through a connection independent of the writer."""

    tables = tuple(await authority_counts(session_factory))
    async with session_factory() as session:
        return {
            table: tuple(
                (
                    await session.execute(
                        text(
                            f"SELECT row_to_json(authority)::text FROM {table} authority "  # noqa: S608
                            "ORDER BY 1"
                        )
                    )
                ).scalars()
            )
            for table in tables
        }


async def test_finalize_replays_after_a_cold_runtime_rebuild(
    test_database_url, migrated_quality_gate
) -> None:
    """A durable finalize receipt must survive destruction of the whole composition root."""

    fixture = load_synthetic_fixture(FIXTURE)
    first_engine, first_sessions, first = _runtime(test_database_url, fixture)
    second_engine = None
    try:
        graph = await stage_synthetic_graph(first, fixture)
        claimed = await first.service.claim(claim_command(graph))
        command = finalize_command(
            graph,
            lock_version=claimed.task.lock_version,
            scenario=ReviewScenario.CORRECT,
        )
        finalized = await first.service.finalize(command)
        before = await authority_counts(first_sessions)

        await first_engine.dispose()
        second_engine, second_sessions, rebuilt = _runtime(test_database_url, fixture)
        replay = await rebuilt.service.finalize(command)
        result = await rebuilt.service.result(command.task)

        assert replay == result == finalized
        assert rebuilt.factories.clock_calls == 0
        assert rebuilt.factories.uuid_calls == 0
        assert rebuilt.factories.subject_calls == 0
        assert rebuilt.factories.calibration_calls == 0
        assert await authority_counts(second_sessions) == before
        assert before["approved_evidence_batches"] == 1
        assert before["quality_gate_outbox"] == 1
        assert before["quality_gate_idempotency_receipts"] == 2
        publication = await rebuilt.publisher.load_publication(replay.batch_id)
        assert publication.task.task.status is ReviewTaskStatus.CORRECTED
    finally:
        if second_engine is not None:
            await second_engine.dispose()
        await first_engine.dispose()


@pytest.mark.parametrize("preferred_winner", ["finalize", "reject"])
async def test_finalize_and_reject_double_miss_commit_one_terminal_graph(
    test_database_url, migrated_quality_gate, monkeypatch, preferred_winner
) -> None:
    """Two independent applications that both miss receipts must leave one terminal graph."""

    fixture = load_synthetic_fixture(FIXTURE)
    first_engine, first_sessions, first = _runtime(test_database_url, fixture)
    second_engine, _second_sessions, second = _runtime(test_database_url, fixture)
    try:
        graph = await stage_synthetic_graph(first, fixture)
        claimed = await first.service.claim(claim_command(graph))
        finalize = finalize_command(
            graph,
            lock_version=claimed.task.lock_version,
            scenario=ReviewScenario.CORRECT,
        )
        reject = reject_command(graph, lock_version=claimed.task.lock_version)

        arrivals = 0
        arrivals_lock = asyncio.Lock()
        both_probed = asyncio.Event()

        async def rendezvous() -> None:
            nonlocal arrivals
            async with arrivals_lock:
                arrivals += 1
                if arrivals == 2:
                    both_probed.set()
            await asyncio.wait_for(both_probed.wait(), timeout=5)

        finalize_probe = first.publisher.probe_finalize_receipt_by_digest
        rejection_probe = second.repository.probe_rejection_receipt

        async def finalize_double_miss(**kwargs):
            replay = await finalize_probe(**kwargs)
            assert replay is None
            await rendezvous()
            if preferred_winner == "reject":
                await asyncio.sleep(0.05)
            return None

        async def reject_double_miss(**kwargs):
            replay = await rejection_probe(**kwargs)
            assert replay is None
            await rendezvous()
            if preferred_winner == "finalize":
                await asyncio.sleep(0.05)
            return None

        monkeypatch.setattr(
            first.publisher, "probe_finalize_receipt_by_digest", finalize_double_miss
        )
        monkeypatch.setattr(second.repository, "probe_rejection_receipt", reject_double_miss)

        outcomes = await asyncio.wait_for(
            asyncio.gather(
                first.service.finalize(finalize),
                second.service.reject(reject),
                return_exceptions=True,
            ),
            timeout=15,
        )

        winners = tuple(item for item in outcomes if not isinstance(item, BaseException))
        failures = tuple(item for item in outcomes if isinstance(item, BaseException))
        assert arrivals == 2
        assert len(winners) == len(failures) == 1
        assert isinstance(failures[0], ReviewApplicationError)
        expected_failure = (
            ReviewErrorCode.REVISION_CONFLICT
            if preferred_winner == "finalize"
            else ReviewErrorCode.REVIEW_RESOURCE_UNAVAILABLE
        )
        assert failures[0].code is expected_failure
        if preferred_winner == "finalize":
            assert isinstance(winners[0], ApprovedReviewResult)
        else:
            assert isinstance(winners[0], RejectedReviewResult)

        counts = await authority_counts(first_sessions)
        assert counts["review_tasks"] == 1
        assert counts["quality_gate_outbox"] == 1
        assert counts["quality_gate_idempotency_receipts"] == 2
        if isinstance(winners[0], ApprovedReviewResult):
            assert counts["approved_evidence_batches"] == 1
            assert counts["review_rejections"] == 0
            publication = await first.publisher.load_publication(winners[0].batch_id)
            assert publication.task.task.status is ReviewTaskStatus.CORRECTED
            assert publication.task.task.id == graph.task.id
            assert publication.batch.batch_id == winners[0].batch_id
            assert await first.outbox.load(publication.event.event_id) == publication.event
        else:
            assert isinstance(winners[0], RejectedReviewResult)
            assert counts["approved_evidence_batches"] == 0
            assert counts["review_rejections"] == 1
            for table in (
                "review_result_reports",
                "review_shot_decisions",
                "review_corrections",
                "review_shot_results",
                "review_publication_lineage",
                "review_metric_authority",
                "approved_evidence_batches",
                "approved_batch_sources",
            ):
                assert counts[table] == 0
            rejected = await second.repository.load_rejected_result(task_id=graph.task.id)
            assert rejected.task.task.status is ReviewTaskStatus.REJECTED
    finally:
        await first_engine.dispose()
        await second_engine.dispose()


async def test_finalize_mid_transaction_failure_is_clean_and_cold_retry_succeeds(
    test_database_url,
    migrated_quality_gate,
) -> None:
    """A mid-publication failure must be invisible to a second connection and retry cleanly."""

    fixture = load_synthetic_fixture(FIXTURE)
    writer_engine, _writer_sessions, writer = _runtime(test_database_url, fixture)
    observer_engine, observer_sessions, _observer = _runtime(test_database_url, fixture)
    retry_engine = None
    graph = await stage_synthetic_graph(writer, fixture)
    claimed = await writer.service.claim(claim_command(graph))
    command = finalize_command(
        graph,
        lock_version=claimed.task.lock_version,
        scenario=ReviewScenario.CORRECT,
    )
    before = await _authority_snapshot(observer_sessions)

    class InjectedFailure(RuntimeError):
        pass

    def fail_after_authority_staging(
        _connection, _cursor, statement, _parameters, _context, _many
    ) -> None:
        if "INSERT INTO approved_evidence_batches" in statement:
            raise InjectedFailure("synthetic finalize mid-transaction failure")

    event.listen(writer_engine.sync_engine, "before_cursor_execute", fail_after_authority_staging)
    try:
        with pytest.raises(ReviewApplicationError) as error:
            await writer.service.finalize(command)
        assert error.value.code is ReviewErrorCode.COMMIT_STATE_UNKNOWN
    finally:
        event.remove(
            writer_engine.sync_engine, "before_cursor_execute", fail_after_authority_staging
        )

    try:
        assert await _authority_snapshot(observer_sessions) == before
        await writer_engine.dispose()
        await observer_engine.dispose()

        retry_engine, retry_sessions, rebuilt = _runtime(test_database_url, fixture)
        finalized = await rebuilt.service.finalize(command)
        result = await rebuilt.service.result(command.task)

        assert finalized == result
        assert finalized.status is ReviewTaskStatus.CORRECTED
        counts = await authority_counts(retry_sessions)
        assert counts["approved_evidence_batches"] == 1
        assert counts["quality_gate_outbox"] == 1
        assert counts["quality_gate_idempotency_receipts"] == 2
    finally:
        if retry_engine is not None:
            await retry_engine.dispose()
        await writer_engine.dispose()
        await observer_engine.dispose()


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_SEPARATOR = re.compile(r"[^a-z0-9]+")


def _key_parts(value: object) -> set[str]:
    separated = _CAMEL_BOUNDARY.sub("_", str(value).strip())
    return {part for part in _KEY_SEPARATOR.split(separated.lower()) if part}


def _assert_privacy_safe(
    value: object,
    *,
    location: str,
    allow_private_reviewer_audit: bool = False,
) -> None:
    forbidden_parts = {
        "openid",
        "raw",
        "media",
        "video",
        "image",
        "file",
        "url",
        "uri",
        "path",
        "locator",
        "token",
        "secret",
        "password",
        "prompt",
        "provider",
    }
    if not allow_private_reviewer_audit:
        forbidden_parts |= {"reviewer", "credential"}

    if isinstance(value, Mapping):
        for key, nested in value.items():
            parts = _key_parts(key)
            assert not (parts & forbidden_parts), f"sensitive key at {location}.{key}"
            assert not ({"open", "id"} <= parts), f"openid key at {location}.{key}"
            _assert_privacy_safe(
                nested,
                location=f"{location}.{key}",
                allow_private_reviewer_audit=allow_private_reviewer_audit,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _assert_privacy_safe(
                nested,
                location=f"{location}[{index}]",
                allow_private_reviewer_audit=allow_private_reviewer_audit,
            )
        return
    if isinstance(value, str):
        assert "://" not in value, f"locator-like value at {location}"
        assert not PurePosixPath(value).is_absolute(), f"absolute path at {location}"
        assert not PureWindowsPath(value).is_absolute(), f"absolute path at {location}"
        assert not value.startswith("~/"), f"home path at {location}"
        if not allow_private_reviewer_audit:
            lowered = value.lower()
            assert "synthetic-reviewer" not in lowered, f"reviewer value at {location}"
            assert "synthetic-credential" not in lowered, f"credential value at {location}"


def _fixture_with_distinct_graph(fixture):
    """Derive a second deterministic graph without duplicating fixture construction rules."""

    identifiers = fixture.identifiers.model_copy(
        update={
            "session_id": UUID("10000000-0000-4000-8000-000000001201"),
            "source_report_id": UUID("30000000-0000-4000-8000-000000001201"),
            "task_id": UUID("40000000-0000-4000-8000-000000001201"),
            "user_id": UUID("50000000-0000-4000-8000-000000001201"),
            "attempt_id": UUID("60000000-0000-4000-8000-000000001201"),
            "asset_id": UUID("70000000-0000-4000-8000-000000001201"),
            "calibration_id": UUID("80000000-0000-4000-8000-000000001201"),
            "calibration_revision_id": UUID("90000000-0000-4000-8000-000000001201"),
        }
    )
    shots = tuple(
        shot.model_copy(
            update={
                "shot_id": UUID(f"20000000-0000-4000-8000-{1200 + shot.sequence:012d}"),
                "source_envelope_id": UUID(f"21000000-0000-4000-8000-{1200 + shot.sequence:012d}"),
            }
        )
        for shot in fixture.shots
    )
    return fixture.model_copy(update={"identifiers": identifiers, "shots": shots})


def _collect_value_paths(value: object, target: str, *, location: str) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            paths |= _collect_value_paths(
                nested,
                target,
                location=f"{location}.{key}",
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            paths |= _collect_value_paths(
                nested,
                target,
                location=f"{location}[{index}]",
            )
    elif str(value) == target:
        paths.add(location)
    return paths


async def test_database_outbox_and_public_projections_are_privacy_safe(
    database_session_factory,
) -> None:
    """Scan actual JSONB, the persisted event, and formal public projections recursively."""

    fixture = load_synthetic_fixture(FIXTURE)
    runtime = compose_runtime(database_session_factory, fixture)
    graph = await stage_synthetic_graph(runtime, fixture)
    claimed = await runtime.service.claim(claim_command(graph))
    command = finalize_command(
        graph,
        lock_version=claimed.task.lock_version,
        scenario=ReviewScenario.CORRECT,
    )
    finalized = await runtime.service.finalize(command)
    result = await runtime.service.result(command.task)
    publication = await runtime.publisher.load_publication(finalized.batch_id)
    outbox_event = await runtime.outbox.load(publication.event.event_id)
    approved_repository = await runtime.approved.load_repository(
        finalized.session_id, finalized.report_id, finalized.report_revision
    )
    approved = approved_repository.get_approved_session_data(
        finalized.session_id, finalized.report_id, finalized.report_revision
    )
    rejected_fixture = _fixture_with_distinct_graph(fixture)
    rejected_runtime = compose_runtime(database_session_factory, rejected_fixture)
    rejected_graph = await stage_synthetic_graph(rejected_runtime, rejected_fixture)
    rejected_claim = await rejected_runtime.service.claim(claim_command(rejected_graph))
    rejected_command = reject_command(
        rejected_graph,
        lock_version=rejected_claim.task.lock_version,
    )
    rejected_result = await rejected_runtime.service.reject(rejected_command)
    rejected_public_result = await rejected_runtime.service.result(rejected_command.task)
    rejected_notification = await rejected_runtime.outbox.load(
        rejected_result.notification_event_id
    )
    assert rejected_public_result == rejected_result

    private_audit_tables = {
        "review_shot_decisions",
        "review_corrections",
        "review_rejections",
    }
    approved_user_id = str(fixture.identifiers.user_id)
    rejected_user_id = str(rejected_fixture.identifiers.user_id)
    approved_user_paths: set[str] = set()
    rejected_user_paths: set[str] = set()
    async with database_session_factory() as session:
        jsonb_tables = tuple(
            (
                await session.execute(
                    text(
                        "SELECT table_name FROM information_schema.columns "
                        "WHERE table_schema=current_schema() AND column_name='payload' "
                        "AND data_type='jsonb' ORDER BY table_name"
                    )
                )
            ).scalars()
        )
        assert "quality_gate_outbox" in jsonb_tables
        assert "approved_evidence_batches" in jsonb_tables
        for table in jsonb_tables:
            payloads = (
                await session.execute(
                    text(f"SELECT payload FROM {table} ORDER BY payload::text")  # noqa: S608
                )
            ).scalars()
            for payload in payloads:
                location = f"db.{table}[*]"
                _assert_privacy_safe(
                    payload,
                    location=f"{location}.payload",
                    allow_private_reviewer_audit=table in private_audit_tables,
                )
                approved_user_paths |= _collect_value_paths(
                    payload,
                    approved_user_id,
                    location=f"{location}.payload",
                )
                rejected_user_paths |= _collect_value_paths(
                    payload,
                    rejected_user_id,
                    location=f"{location}.payload",
                )

    public_values = {
        "application.finalize": finalized.model_dump(mode="json"),
        "application.result": result.model_dump(mode="json"),
        "application.rejected_result": rejected_public_result.model_dump(mode="json"),
        "approved.batch": publication.batch.model_dump(mode="json"),
        "approved.data": approved.model_dump(mode="json"),
        "outbox.event": outbox_event.model_dump(mode="json"),
        "outbox.rejected_notification": rejected_notification.model_dump(mode="json"),
    }
    for name, public_value in public_values.items():
        _assert_privacy_safe(public_value, location=name)
        approved_user_paths |= _collect_value_paths(
            public_value,
            approved_user_id,
            location=name,
        )
        rejected_user_paths |= _collect_value_paths(
            public_value,
            rejected_user_id,
            location=name,
        )

    assert approved_user_paths == {
        "approved.batch.user_id",
        "db.approved_evidence_batches[*].payload.user_id",
        "db.quality_gate_outbox[*].payload.approvedEvidenceBatch.user_id",
        "db.review_publication_lineage[*].payload.user_id",
        "db.review_session_ownership[*].payload.user_id",
        "outbox.event.payload.approvedEvidenceBatch.user_id",
    }
    assert rejected_user_paths == {
        "db.review_session_ownership[*].payload.user_id",
    }
