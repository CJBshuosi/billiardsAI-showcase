from __future__ import annotations

from datetime import UTC, datetime, timedelta

from billiards_ai.edge.route_assistance.postgame import (
    PostgameJobSnapshot,
    PostgameSessionManifest,
    SessionAnalysisStatus,
    project_session_analysis,
    session_scoped_clip_id,
)

NOW = datetime(2026, 9, 6, 12, tzinfo=UTC)


def _manifest(*job_ids: str) -> PostgameSessionManifest:
    return PostgameSessionManifest(
        session_id="session-a",
        ended_at_utc=NOW,
        deadline_at_utc=NOW + timedelta(seconds=120),
        expected_jobs=job_ids,
    )


def test_projection_uses_fixed_precedence_and_requires_finalized_report() -> None:
    """Catches a status reducer that lets input order change the user-visible state."""
    assert project_session_analysis(_manifest(), (), NOW).status is SessionAnalysisStatus.PROCESSING
    assert (
        project_session_analysis(_manifest(), (), NOW, finalized=True).status
        is SessionAnalysisStatus.AVAILABLE
    )
    assert (
        project_session_analysis(
            _manifest("job-1"),
            (PostgameJobSnapshot("job-1", "available", "artifact-1"),),
            NOW,
        ).status
        is SessionAnalysisStatus.PROCESSING
    )
    assert (
        project_session_analysis(
            _manifest("job-1", "job-2"),
            (
                PostgameJobSnapshot("job-1", "available", "artifact-1"),
                PostgameJobSnapshot("job-2", "failed_retrying", None),
            ),
            NOW,
        ).status
        is SessionAnalysisStatus.PARTIAL
    )
    assert (
        project_session_analysis(
            _manifest("job-1"),
            (PostgameJobSnapshot("job-1", "failed_retrying", None),),
            NOW + timedelta(seconds=121),
        ).status
        is SessionAnalysisStatus.FAILED_RETRYING
    )
    assert (
        project_session_analysis(
            _manifest("job-1"),
            (PostgameJobSnapshot("job-1", "pending", None),),
            NOW + timedelta(seconds=121),
        ).status
        is SessionAnalysisStatus.FAILED_RETRYING
    )


def test_clip_id_is_scoped_by_full_session_job_and_artifact_lineage() -> None:
    """Catches restoring the legacy globally-colliding clip-shot identifier."""
    first = session_scoped_clip_id("session-a", "shot-0001", "job-a", "artifact-a")
    second = session_scoped_clip_id("session-b", "shot-0001", "job-b", "artifact-b")
    assert first != second
    assert first.startswith("clip-")
