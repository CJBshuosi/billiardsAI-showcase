"""Story 5.3 strict, deterministic visualization contracts."""

from __future__ import annotations

import hashlib
import json
from uuid import UUID

import pytest
from pydantic import ValidationError

SESSION_ID = UUID("10000000-0000-4000-8000-000000000001")
REPORT_ID = UUID("30000000-0000-4000-8000-000000000001")
SHOT_ID = UUID("50000000-0000-4000-8000-000000000001")
CALIBRATION_ID = UUID("60000000-0000-4000-8000-000000000001")
CALIBRATION_REVISION_ID = UUID("60000000-0000-4000-8000-000000000002")


def _api():
    from billiards_ai.shared.models import visualization

    names = (
        "VisualizationActualProvenance",
        "VisualizationAsset",
        "VisualizationAssetBinding",
        "VisualizationBall",
        "VisualizationCalibrationProvenance",
        "VisualizationIdealProvenance",
        "VisualizationLegendItem",
        "VisualizationLineStyle",
        "VisualizationPathKind",
        "VisualizationPathSeries",
        "VisualizationPathStyle",
        "VisualizationPoint",
        "VisualizationSelectionProvenance",
        "VisualizationSemanticToken",
        "VisualizationSourceLineage",
        "VisualizationTable",
    )
    return {name: getattr(visualization, name) for name in names}


def _asset():
    api = _api()
    point = api["VisualizationPoint"]
    actual_style = api["VisualizationPathStyle"](
        semantic_token=api["VisualizationSemanticToken"].ACTUAL,
        color="#1565C0",
        line_style=api["VisualizationLineStyle"].SOLID,
        direct_label="实际母球路径",
    )
    ideal_style = api["VisualizationPathStyle"](
        semantic_token=api["VisualizationSemanticToken"].IDEAL,
        color="#C62828",
        line_style=api["VisualizationLineStyle"].DASHED,
        direct_label="理想路径",
    )
    payload = dict(
        source=api["VisualizationSourceLineage"](
            session_id=SESSION_ID,
            report_id=REPORT_ID,
            report_revision=2,
            shot_id=SHOT_ID,
        ),
        selection=api["VisualizationSelectionProvenance"](
            rule_version="1.0",
            selection_reason="severe_deviation",
        ),
        table=api["VisualizationTable"](
            width_mm=2540,
            height_mm=1270,
        ),
        calibration=api["VisualizationCalibrationProvenance"](
            calibration_id=CALIBRATION_ID,
            revision_id=CALIBRATION_REVISION_ID,
            revision_number=2,
            table_configuration_fingerprint="b" * 64,
        ),
        actual=api["VisualizationActualProvenance"](
            source="CV",
            confidence=0.9,
        ),
        ideal=api["VisualizationIdealProvenance"](
            adapter_version="pooltool-adapter-1.0",
            engine_name="pooltool",
            engine_version="0.6.0",
            configuration_version="chinese-eight-ball-9ft-v1",
        ),
        balls=(
            api["VisualizationBall"](ball_id=0, position=point(x_mm=100, y_mm=100)),
            api["VisualizationBall"](ball_id=3, position=point(x_mm=900, y_mm=500)),
        ),
        paths=(
            api["VisualizationPathSeries"](
                kind=api["VisualizationPathKind"].ACTUAL_CUE,
                ball_id=0,
                points=(point(x_mm=100, y_mm=100), point(x_mm=700, y_mm=400)),
                style=actual_style,
                label="实际母球路径",
            ),
            api["VisualizationPathSeries"](
                kind=api["VisualizationPathKind"].IDEAL_CUE,
                ball_id=0,
                points=(point(x_mm=100, y_mm=100), point(x_mm=850, y_mm=480)),
                style=ideal_style,
                label="理想母球路径",
            ),
            api["VisualizationPathSeries"](
                kind=api["VisualizationPathKind"].IDEAL_OBJECT,
                ball_id=3,
                points=(point(x_mm=900, y_mm=500), point(x_mm=2540, y_mm=1270)),
                style=ideal_style,
                label="理想目标球路径",
            ),
        ),
        legend=(
            api["VisualizationLegendItem"](
                semantic_token=api["VisualizationSemanticToken"].ACTUAL,
                color="#1565C0",
                line_style=api["VisualizationLineStyle"].SOLID,
                label="实际路径",
            ),
            api["VisualizationLegendItem"](
                semantic_token=api["VisualizationSemanticToken"].IDEAL,
                color="#C62828",
                line_style=api["VisualizationLineStyle"].DASHED,
                label="理想路径",
            ),
        ),
        text_summary="实际母球路径来源为 CV，置信度 90%；理想路径来源为 pooltool 0.6.0。",
    )
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema_version": "1.0",
                "spec_version": "actual-ideal-overlay/1.0",
                **{
                    key: (
                        [item.model_dump(mode="json") for item in value]
                        if isinstance(value, tuple)
                        else value.model_dump(mode="json")
                        if hasattr(value, "model_dump")
                        else value
                    )
                    for key, value in payload.items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return api["VisualizationAsset"](
        asset_id=f"visualization-{digest}",
        content_sha256=digest,
        **payload,
    )


def test_visualization_asset_is_strict_frozen_and_json_round_trips():
    asset = _asset()
    api = _api()

    assert asset.schema_version == "1.0"
    assert asset.spec_version == "actual-ideal-overlay/1.0"
    assert api["VisualizationAsset"].model_validate_json(asset.model_dump_json()) == asset
    assert json.loads(asset.model_dump_json())["table"]["unit"] == "mm"
    with pytest.raises(ValidationError, match="frozen"):
        asset.text_summary = "changed"
    with pytest.raises(ValidationError, match="unexpected"):
        api["VisualizationAsset"].model_validate(asset.model_dump() | {"unexpected": True})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source", "report_revision"), True),
        (("table", "width_mm"), True),
        (("balls", 0, "ball_id"), True),
        (("balls", 0, "position", "x_mm"), 2541),
        (("balls", 0, "position", "y_mm"), 1271),
    ],
)
def test_python_and_model_copy_reject_bool_extra_and_out_of_bounds(path, value):
    asset = _asset()
    payload = asset.model_dump(mode="python")
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        type(asset).model_validate(payload)
    with pytest.raises(ValidationError):
        asset.model_copy(update=payload)


def test_asset_enforces_canonical_ball_path_layer_and_legend_order():
    asset = _asset()
    with pytest.raises(ValidationError, match="ball|sorted"):
        type(asset).model_validate(asset.model_dump() | {"balls": tuple(reversed(asset.balls))})
    with pytest.raises(ValidationError, match="path|order"):
        type(asset).model_validate(asset.model_dump() | {"paths": tuple(reversed(asset.paths))})
    with pytest.raises(ValidationError, match="legend|order"):
        type(asset).model_validate(asset.model_dump() | {"legend": tuple(reversed(asset.legend))})


def test_asset_rejects_a_digest_or_id_that_does_not_match_its_business_content():
    asset = _asset()
    with pytest.raises(ValidationError, match="digest"):
        type(asset).model_validate(asset.model_dump() | {"content_sha256": "0" * 64})
    with pytest.raises(ValidationError, match="digest"):
        type(asset).model_validate(asset.model_dump() | {"asset_id": "visualization-" + "0" * 64})


def test_asset_has_no_media_external_pixel_animation_or_free_text_fields():
    payload = _asset().model_dump(mode="json")
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    forbidden = (
        "http://",
        "https://",
        "oss://",
        "file_path",
        "media_id",
        "pixel",
        "normalized",
        "raw_observation",
        "animation",
        "autoplay",
        "pooltool_object",
        "provider_text",
    )
    assert all(item not in serialized for item in forbidden)


def test_revision_binding_is_strict_and_unambiguously_exact():
    api = _api()
    asset = _asset()
    binding = api["VisualizationAssetBinding"](
        asset_id=asset.asset_id,
        asset_content_sha256=asset.content_sha256,
        source=asset.source,
        target_session_id=SESSION_ID,
        target_report_id=REPORT_ID,
        target_report_revision=2,
        target_shot_id=SHOT_ID,
    )

    assert (
        api["VisualizationAssetBinding"].model_validate_json(binding.model_dump_json()) == binding
    )
    for field, value in (
        ("target_session_id", UUID("10000000-0000-4000-8000-000000000099")),
        ("target_report_id", UUID("30000000-0000-4000-8000-000000000099")),
        ("target_report_revision", 3),
        ("target_shot_id", UUID("50000000-0000-4000-8000-000000000099")),
    ):
        with pytest.raises(ValidationError, match="source|target|lineage"):
            api["VisualizationAssetBinding"].model_validate(binding.model_dump() | {field: value})


@pytest.mark.parametrize("coordinate", [True, "100"])
def test_raw_trajectory_review_and_approved_paths_reject_coerced_millimetres(coordinate):
    from billiards_ai.shared.models import (
        ApprovedActualPath,
        ApprovedActualPathStatus,
        FinalFieldSource,
        ReviewActualPath,
        TrajectorySample,
    )

    raw_points = ({"x_mm": coordinate, "y_mm": 100}, {"x_mm": 200, "y_mm": 200})
    with pytest.raises(ValidationError):
        TrajectorySample.model_validate({"timestamp_ms": 1, "position": raw_points[0]})
    with pytest.raises(ValidationError):
        ReviewActualPath.model_validate(
            {"points": raw_points, "source": FinalFieldSource.CV, "confidence": 0.9}
        )
    with pytest.raises(ValidationError):
        ApprovedActualPath.model_validate(
            {
                "status": ApprovedActualPathStatus.AVAILABLE,
                "points": raw_points,
                "source": FinalFieldSource.CV,
                "confidence": 0.9,
            }
        )


def test_available_actual_path_requires_the_shared_confidence_threshold():
    from billiards_ai.shared.constants import CONFIDENCE_ABSTAIN_BELOW
    from billiards_ai.shared.models import (
        ApprovedActualPath,
        ApprovedActualPathStatus,
        FinalFieldSource,
        Point,
    )

    with pytest.raises(ValidationError, match="confidence"):
        ApprovedActualPath(
            status=ApprovedActualPathStatus.AVAILABLE,
            points=(Point(x_mm=100, y_mm=100), Point(x_mm=200, y_mm=200)),
            source=FinalFieldSource.CV,
            confidence=CONFIDENCE_ABSTAIN_BELOW - 0.01,
        )


def test_segment_anchor_must_equal_the_before_snapshot_cue_ball():
    from billiards_ai.shared.models import (
        BallPosition,
        Point,
        ShotSegment,
        TableSnapshot,
        TrajectorySample,
    )

    before = TableSnapshot(
        timestamp_ms=0,
        positions=(BallPosition(ball_id=0, position=Point(x_mm=100, y_mm=100)),),
    )
    after = TableSnapshot(
        timestamp_ms=200,
        positions=(BallPosition(ball_id=0, position=Point(x_mm=200, y_mm=200)),),
    )
    with pytest.raises(ValidationError, match="anchor|snapshot|cue"):
        ShotSegment(
            t_start_ms=100,
            t_end_ms=200,
            ball_positions_before=before,
            ball_positions_after=after,
            cue_ball_path=(
                TrajectorySample(timestamp_ms=0, position=Point(x_mm=101, y_mm=100)),
                TrajectorySample(timestamp_ms=200, position=Point(x_mm=200, y_mm=200)),
            ),
        )


def _payload_with_current_asset_digest(asset, **updates):
    payload = asset.model_dump(mode="json") | updates
    content = {
        key: value for key, value in payload.items() if key not in {"asset_id", "content_sha256"}
    }
    digest = hashlib.sha256(
        json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload | {"asset_id": f"visualization-{digest}", "content_sha256": digest}


def test_asset_and_binding_enforce_geometry_legend_and_digest_relationships():
    api = _api()
    asset = _asset()
    wrong_paths = list(asset.model_dump(mode="json")["paths"])
    wrong_paths[0]["points"][0] = {"x_mm": 101, "y_mm": 100}
    with pytest.raises(ValidationError, match="start|ball|path"):
        api["VisualizationAsset"].model_validate(
            _payload_with_current_asset_digest(asset, paths=wrong_paths)
        )

    wrong_legend = list(asset.model_dump(mode="json")["legend"])
    wrong_legend[0]["label"] = "伪造图例"
    with pytest.raises(ValidationError, match="legend|label"):
        api["VisualizationAsset"].model_validate(
            _payload_with_current_asset_digest(asset, legend=wrong_legend)
        )

    with pytest.raises(ValidationError, match="hash|digest|asset"):
        api["VisualizationAssetBinding"](
            asset_id=asset.asset_id,
            asset_content_sha256="0" * 64,
            source=asset.source,
            target_session_id=SESSION_ID,
            target_report_id=REPORT_ID,
            target_report_revision=2,
            target_shot_id=SHOT_ID,
        )


@pytest.mark.parametrize(
    "unsafe_metadata",
    ("https://evil.example/x", "../../secret", "pooltool\nleak", "file:///tmp/x"),
)
def test_ideal_and_visualization_metadata_reject_external_or_controlled_text_leakage(
    unsafe_metadata,
):
    from billiards_ai.shared.models import IdealPathReason, IdealPathResult, IdealPathStatus

    with pytest.raises(ValidationError):
        IdealPathResult(
            status=IdealPathStatus.ENGINE_UNAVAILABLE,
            reason=IdealPathReason.ENGINE_UNAVAILABLE,
            adapter_version="1.0",
            engine_name=unsafe_metadata,
            engine_version="0.6.0",
            source_configuration_version="chinese-eight-ball-9ft-v1",
        )
