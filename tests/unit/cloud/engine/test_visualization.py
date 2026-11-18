"""Story 5.3 deterministic overlay generation and failure matrix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import UUID

import pytest

SESSION_ID = UUID("10000000-0000-4000-8000-000000000001")
REPORT_ID = UUID("30000000-0000-4000-8000-000000000001")
SHOT_ID = UUID("50000000-0000-4000-8000-000000000001")
CALIBRATION_ID = UUID("60000000-0000-4000-8000-000000000001")
CALIBRATION_REVISION_ID = UUID("60000000-0000-4000-8000-000000000002")


def _projection(**overrides):
    from billiards_ai.shared.models import (
        ApprovedActualPath,
        ApprovedActualPathStatus,
        ApprovedVisualizationProjection,
        BallPosition,
        CalibrationStatus,
        FinalFieldSource,
        Point,
    )

    values = {
        "session_id": SESSION_ID,
        "report_id": REPORT_ID,
        "report_revision": 2,
        "shot_id": SHOT_ID,
        "shot_sequence": 1,
        "struck_ball": 3,
        "ball_positions": (
            BallPosition(ball_id=0, position=Point(x_mm=100, y_mm=100)),
            BallPosition(ball_id=3, position=Point(x_mm=900, y_mm=500)),
        ),
        "actual_path": ApprovedActualPath(
            status=ApprovedActualPathStatus.AVAILABLE,
            points=(
                Point(x_mm=100, y_mm=100),
                Point(x_mm=400, y_mm=220),
                Point(x_mm=700, y_mm=400),
            ),
            source=FinalFieldSource.CV,
            confidence=0.9,
        ),
        "calibration_id": CALIBRATION_ID,
        "calibration_revision_id": CALIBRATION_REVISION_ID,
        "calibration_revision_number": 2,
        "calibration_status": CalibrationStatus.ACTIVE,
        "table_profile_id": "dilan-table-3",
        "table_length_mm": 2540,
        "table_width_mm": 1270,
        "table_configuration_fingerprint": "b" * 64,
    }
    values.update(overrides)
    return ApprovedVisualizationProjection(**values)


def _selection(**overrides):
    from billiards_ai.shared.models import (
        TypicalShotRuleVersion,
        TypicalShotSelection,
        TypicalShotSelectionReason,
        TypicalShotSelectionResult,
    )

    values = {
        "session_id": SESSION_ID,
        "report_id": REPORT_ID,
        "report_revision": 2,
        "rule_version": TypicalShotRuleVersion.V1,
        "selections": (
            TypicalShotSelection(
                shot_id=SHOT_ID,
                sequence=1,
                selection_rank=1,
                selection_reason=TypicalShotSelectionReason.SEVERE_DEVIATION,
                rule_version=TypicalShotRuleVersion.V1,
            ),
        ),
    }
    values.update(overrides)
    return TypicalShotSelectionResult(**values)


def _ideal(**overrides):
    from billiards_ai.shared.models import (
        IdealPathBall,
        IdealPathEvent,
        IdealPathEventKind,
        IdealPathPocket,
        IdealPathPoint,
        IdealPathProvenance,
        IdealPathResult,
        IdealPathStatus,
    )

    provenance = IdealPathProvenance(
        calibration_id=CALIBRATION_ID,
        calibration_revision_id=CALIBRATION_REVISION_ID,
        calibration_revision_number=2,
        table_profile_id="dilan-table-3",
        table_configuration_fingerprint="b" * 64,
        cue_ball=IdealPathBall(ball_id=0, position=IdealPathPoint(x_mm=100, y_mm=100)),
        object_balls=(IdealPathBall(ball_id=3, position=IdealPathPoint(x_mm=900, y_mm=500)),),
        target_ball_id=3,
    )
    values = {
        "status": IdealPathStatus.SUCCEEDED,
        "reason": None,
        "adapter_version": "pooltool-adapter-1.0",
        "engine_name": "pooltool",
        "engine_version": "0.6.0",
        "source_configuration_version": "chinese-eight-ball-9ft-v1",
        "provenance": provenance,
        "events": (
            IdealPathEvent(
                sequence=1,
                kind=IdealPathEventKind.CUE_STRIKE,
                ball_id=0,
                position=IdealPathPoint(x_mm=100, y_mm=100),
            ),
            IdealPathEvent(
                sequence=2,
                kind=IdealPathEventKind.BALL_CONTACT,
                ball_id=0,
                other_ball_id=3,
                position=IdealPathPoint(x_mm=850, y_mm=480),
            ),
            IdealPathEvent(
                sequence=3,
                kind=IdealPathEventKind.BALL_CONTACT,
                ball_id=3,
                other_ball_id=0,
                position=IdealPathPoint(x_mm=900, y_mm=500),
            ),
            IdealPathEvent(
                sequence=4,
                kind=IdealPathEventKind.CUSHION_CONTACT,
                ball_id=0,
                position=IdealPathPoint(x_mm=1100, y_mm=700),
            ),
            IdealPathEvent(
                sequence=5,
                kind=IdealPathEventKind.POCKETED,
                ball_id=3,
                pocket=IdealPathPocket.TOP_RIGHT,
                position=IdealPathPoint(x_mm=2540, y_mm=1270),
            ),
            IdealPathEvent(
                sequence=6,
                kind=IdealPathEventKind.STOP,
                ball_id=0,
                position=IdealPathPoint(x_mm=1200, y_mm=800),
            ),
        ),
    }
    values.update(overrides)
    return IdealPathResult(**values)


@dataclass
class FakeRepository:
    projection: object
    calls: int = 0

    def get_approved_visualization_projection(self, session_id, report_id, revision, shot_id):
        self.calls += 1
        assert (session_id, report_id, revision, shot_id) == (
            SESSION_ID,
            REPORT_ID,
            2,
            SHOT_ID,
        )
        return self.projection


def _generate(*, projection=None, selection=None, ideal=None, revision=2):
    from billiards_ai.cloud.engine.visualization import generate_visualization_asset

    repository = FakeRepository(projection or _projection())
    result = generate_visualization_asset(
        repository=repository,
        session_id=SESSION_ID,
        report_id=REPORT_ID,
        revision=revision,
        shot_id=SHOT_ID,
        selection=selection or _selection(),
        ideal_result=ideal or _ideal(),
    )
    return repository, result


def test_success_asset_is_canonical_deterministic_accessible_and_keeps_balls_separate():
    from billiards_ai.cloud.engine.visualization import canonical_visualization_asset_bytes
    from billiards_ai.shared.models import (
        VisualizationLineStyle,
        VisualizationPathKind,
        VisualizationStatus,
    )

    repository, first = _generate()
    _, second = _generate()

    assert repository.calls == 1
    assert first.status is VisualizationStatus.SUCCEEDED
    assert first.asset == second.asset
    assert canonical_visualization_asset_bytes(first.asset) == canonical_visualization_asset_bytes(
        second.asset
    )
    assert first.asset.asset_id.endswith(first.asset.content_sha256)
    assert first.asset.table.width_mm / first.asset.table.height_mm == 2
    assert tuple(path.kind for path in first.asset.paths) == (
        VisualizationPathKind.ACTUAL_CUE,
        VisualizationPathKind.IDEAL_CUE,
        VisualizationPathKind.IDEAL_OBJECT,
    )
    assert first.asset.paths[0].ball_id == first.asset.paths[1].ball_id == 0
    assert first.asset.paths[2].ball_id == 3
    assert first.asset.paths[0].style.color != first.asset.paths[1].style.color
    assert first.asset.paths[0].style.line_style is VisualizationLineStyle.SOLID
    assert first.asset.paths[1].style.line_style is VisualizationLineStyle.DASHED
    assert all(path.label for path in first.asset.paths)
    assert tuple(item.label for item in first.asset.legend) == ("实际路径", "理想路径")
    assert all(word in first.asset.text_summary for word in ("实际", "理想", "来源", "置信度"))
    serialized = json.dumps(first.asset.model_dump(mode="json"), ensure_ascii=False).lower()
    assert all(
        forbidden not in serialized
        for forbidden in (
            "url",
            "http",
            "oss://",
            "file_path",
            "media_id",
            "pixel",
            "normalized",
            "raw_observation",
            "animation",
            "autoplay",
        )
    )


def test_event_order_is_preserved_per_ball_never_connected_across_balls():
    result = _generate()[1]
    cue = result.asset.paths[1]
    target = result.asset.paths[2]
    assert tuple((point.x_mm, point.y_mm) for point in cue.points) == (
        (100, 100),
        (850, 480),
        (1100, 700),
        (1200, 800),
    )
    assert tuple((point.x_mm, point.y_mm) for point in target.points) == (
        (900, 500),
        (2540, 1270),
    )


@pytest.mark.parametrize(
    ("case", "expected_status"),
    [
        ("calibration", "CALIBRATION_NOT_ACTIVE"),
        ("actual_missing", "ACTUAL_PATH_UNAVAILABLE"),
        ("actual_unreviewed", "SOURCE_NOT_APPROVED"),
        ("ideal_failed", "IDEAL_PATH_UNSUPPORTED"),
        ("selection_revision", "SELECTION_MISMATCH"),
        ("selection_shot", "SELECTION_MISMATCH"),
        ("ideal_calibration", "PROVENANCE_MISMATCH"),
        ("ideal_table", "PROVENANCE_MISMATCH"),
        ("ideal_cue_start", "PROVENANCE_MISMATCH"),
        ("ideal_target", "PROVENANCE_MISMATCH"),
        ("projection_revision", "REVISION_MISMATCH"),
    ],
)
def test_failure_matrix_is_typed_and_never_leaks_geometry(case, expected_status):
    from billiards_ai.shared.models import (
        ApprovedActualPath,
        ApprovedActualPathReason,
        ApprovedActualPathStatus,
        CalibrationStatus,
        IdealPathBall,
        IdealPathPoint,
        IdealPathReason,
        IdealPathStatus,
    )

    projection = _projection()
    selection = _selection()
    ideal = _ideal()
    if case == "calibration":
        projection = projection.model_copy(update={"calibration_status": CalibrationStatus.STALE})
    elif case in {"actual_missing", "actual_unreviewed"}:
        status = (
            ApprovedActualPathStatus.UNAVAILABLE
            if case == "actual_missing"
            else ApprovedActualPathStatus.NOT_REVIEWED
        )
        reason = (
            ApprovedActualPathReason.NOT_OBSERVED
            if case == "actual_missing"
            else ApprovedActualPathReason.NOT_REVIEWED
        )
        projection = projection.model_copy(
            update={"actual_path": ApprovedActualPath(status=status, reason=reason)}
        )
    elif case == "ideal_failed":
        ideal = ideal.__class__(
            status=IdealPathStatus.UNSUPPORTED_INTENT,
            reason=IdealPathReason.INTENT_NOT_EXPLICIT,
            adapter_version=ideal.adapter_version,
            engine_name=ideal.engine_name,
            engine_version=ideal.engine_version,
            source_configuration_version=ideal.source_configuration_version,
        )
    elif case == "selection_revision":
        selection = _selection(report_revision=1)
    elif case == "selection_shot":
        selection = _selection(selections=())
    elif case.startswith("ideal_"):
        provenance = ideal.provenance
        if case == "ideal_calibration":
            provenance = provenance.model_copy(update={"calibration_revision_number": 3})
        elif case == "ideal_table":
            provenance = provenance.model_copy(update={"table_configuration_fingerprint": "c" * 64})
        elif case == "ideal_cue_start":
            provenance = provenance.model_copy(
                update={
                    "cue_ball": IdealPathBall(
                        ball_id=0,
                        position=IdealPathPoint(x_mm=101, y_mm=100),
                    )
                }
            )
        else:
            projection = projection.model_copy(update={"struck_ball": None})
        if case != "ideal_target":
            ideal = ideal.model_copy(update={"provenance": provenance})
    else:
        projection = projection.model_copy(update={"report_revision": 1})

    result = _generate(projection=projection, selection=selection, ideal=ideal)[1]
    assert result.status.value == expected_status
    assert result.asset is None
    assert set(result.model_dump(mode="json")) == {"schema_version", "status", "reason", "asset"}


def test_boolean_revision_fails_before_repository_access():
    from billiards_ai.cloud.engine.visualization import generate_visualization_asset
    from billiards_ai.shared.models import VisualizationStatus

    repository = FakeRepository(_projection())
    result = generate_visualization_asset(
        repository=repository,
        session_id=SESSION_ID,
        report_id=REPORT_ID,
        revision=True,
        shot_id=SHOT_ID,
        selection=_selection(),
        ideal_result=_ideal(),
    )
    assert result.status is VisualizationStatus.INVALID_INPUT
    assert result.asset is None
    assert repository.calls == 0


def test_independent_binding_accepts_exact_current_revision_and_rejects_stale_lineage():
    from billiards_ai.cloud.engine.visualization import bind_visualization_asset
    from billiards_ai.shared.models import VisualizationBindingStatus

    asset = _generate()[1].asset
    exact = bind_visualization_asset(
        asset,
        session_id=SESSION_ID,
        report_id=REPORT_ID,
        revision=2,
        shot_id=SHOT_ID,
    )
    assert exact.status is VisualizationBindingStatus.SUCCEEDED
    assert exact.binding.source == asset.source

    for field, value in (
        ("session_id", UUID("10000000-0000-4000-8000-000000000099")),
        ("report_id", UUID("30000000-0000-4000-8000-000000000099")),
        ("revision", 3),
        ("shot_id", UUID("50000000-0000-4000-8000-000000000099")),
    ):
        kwargs = {
            "session_id": SESSION_ID,
            "report_id": REPORT_ID,
            "revision": 2,
            "shot_id": SHOT_ID,
        }
        kwargs[field] = value
        rejected = bind_visualization_asset(asset, **kwargs)
        assert rejected.status is not VisualizationBindingStatus.SUCCEEDED
        assert rejected.binding is None


@pytest.mark.parametrize(
    ("status", "reason", "expected_reason"),
    [
        ("UNSUPPORTED_INTENT", "intent_not_explicit", "ideal_intent_not_explicit"),
        ("UNSUPPORTED_INTENT", "intent_not_reviewed", "ideal_intent_not_reviewed"),
        ("UNSUPPORTED_INTENT", "calibration_not_active", "ideal_calibration_not_active"),
        ("UNSUPPORTED_INTENT", "target_ball_missing", "ideal_target_ball_missing"),
        ("ENGINE_UNAVAILABLE", "engine_unavailable", "ideal_engine_unavailable"),
        ("SIMULATION_FAILED", "adapter_failure", "ideal_adapter_failure"),
    ],
)
def test_ideal_failure_reasons_remain_complete_and_distinguishable(status, reason, expected_reason):
    from billiards_ai.shared.models import IdealPathReason, IdealPathResult, IdealPathStatus

    ideal = _ideal()
    failed = IdealPathResult(
        status=IdealPathStatus[status],
        reason=IdealPathReason(reason),
        adapter_version=ideal.adapter_version,
        engine_name=ideal.engine_name,
        engine_version=ideal.engine_version,
        source_configuration_version=ideal.source_configuration_version,
    )
    result = _generate(ideal=failed)[1]

    assert result.asset is None
    assert result.reason.value == expected_reason


def test_insufficient_actual_points_are_not_reported_as_unobserved():
    from billiards_ai.shared.models import (
        ApprovedActualPath,
        ApprovedActualPathReason,
        ApprovedActualPathStatus,
    )

    projection = _projection(
        actual_path=ApprovedActualPath(
            status=ApprovedActualPathStatus.UNAVAILABLE,
            reason=ApprovedActualPathReason.INSUFFICIENT_POINTS,
        )
    )
    result = _generate(projection=projection)[1]

    assert result.asset is None
    assert result.reason.value == "actual_path_insufficient_points"


def test_public_generate_and_bind_return_invalid_input_for_foreign_objects():
    from billiards_ai.cloud.engine.visualization import (
        bind_visualization_asset,
        generate_visualization_asset,
    )
    from billiards_ai.shared.models import VisualizationBindingStatus, VisualizationStatus

    class ForeignObject:
        def __init__(self):
            self.value = "not-a-model"

    try:
        generated = generate_visualization_asset(
            repository=FakeRepository(_projection()),
            session_id=SESSION_ID,
            report_id=REPORT_ID,
            revision=2,
            shot_id=SHOT_ID,
            selection=ForeignObject(),
            ideal_result=_ideal(),
        )
    except AttributeError as exc:  # pragma: no cover - this is the regression being removed
        pytest.fail(f"generate leaked AttributeError: {exc}")
    assert generated.status is VisualizationStatus.INVALID_INPUT

    try:
        bound = bind_visualization_asset(
            ForeignObject(),
            session_id=SESSION_ID,
            report_id=REPORT_ID,
            revision=2,
            shot_id=SHOT_ID,
        )
    except AttributeError as exc:  # pragma: no cover - this is the regression being removed
        pytest.fail(f"bind leaked AttributeError: {exc}")
    assert bound.status is VisualizationBindingStatus.INVALID_INPUT
