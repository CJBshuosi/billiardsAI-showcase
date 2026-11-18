"""感知层契约的可执行部分（AD-13）。

这些测试存在的理由：契约写在文档里挡不住任何人，写成断言才有约束力。
两种感知实现（VLM / YOLO）都必须让这组测试通过。
"""

import inspect
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from billiards_ai.shared.constants import TABLE_LENGTH_MM, TABLE_WIDTH_MM
from billiards_ai.shared.models import Confidence, Point, Shot, TargetZoneSource, Zone


def make_shot(**overrides) -> Shot:
    base = dict(
        id=uuid4(),
        session_id=uuid4(),
        sequence=1,
        t_start_ms=0,
        t_end_ms=3200,
        ball_positions_before={0: Point(x_mm=600, y_mm=400)},
        ball_positions_after={0: Point(x_mm=1800, y_mm=900)},
        cue_ball_end=Point(x_mm=1800, y_mm=900),
        confidence=Confidence.HIGH,
        created_at=datetime.now(UTC),
    )
    return Shot(**{**base, **overrides})


class TestCoordinateSystem:
    """AD-13a — 位置量一律台面毫米整数，不得出现像素或归一化坐标。"""

    def test_point_rejects_non_integer(self):
        with pytest.raises(ValueError):
            Point(x_mm="600.5", y_mm=400)

    def test_normalized_coords_are_not_silently_accepted(self):
        """归一化坐标（0-1）会落在台面左下角，is_on_table 无法拦截。
        这就是为什么契约靠的是「填充者遵守单位约定」，不是靠类型系统——
        感知实现必须在写入前完成标定转换（AD-18）。"""
        suspicious = Point(x_mm=0, y_mm=1)
        assert suspicious.is_on_table()

    def test_off_table_is_detectable(self):
        assert not Point(x_mm=TABLE_LENGTH_MM + 50, y_mm=400).is_on_table()
        assert not Point(x_mm=600, y_mm=TABLE_WIDTH_MM + 50).is_on_table()

    def test_distance_is_physical_mm(self):
        a = Point(x_mm=0, y_mm=0)
        b = Point(x_mm=300, y_mm=400)
        assert a.distance_to(b) == 500


class TestAbstention:
    """AD-5 — 弃权是硬约束，不得给出走位结论。"""

    def test_abstained_shot_yields_no_positional_verdict(self):
        shot = make_shot(
            confidence=Confidence.ABSTAIN,
            target_zone=Zone(center=Point(x_mm=1800, y_mm=900), radius_mm=200),
            target_zone_source=TargetZoneSource.POST_HOC_INFERRED,
            deviation_mm=0,
        )
        assert shot.is_abstained
        assert not shot.has_positional_verdict

    def test_missing_target_zone_yields_no_verdict(self):
        shot = make_shot(confidence=Confidence.HIGH, deviation_mm=None)
        assert not shot.has_positional_verdict

    def test_confident_shot_with_zone_yields_verdict(self):
        shot = make_shot(
            confidence=Confidence.HIGH,
            target_zone=Zone(center=Point(x_mm=1700, y_mm=850), radius_mm=150),
            target_zone_source=TargetZoneSource.POST_HOC_INFERRED,
            deviation_mm=112,
        )
        assert shot.has_positional_verdict


class TestTargetZoneSource:
    """AD-13d — 意图来源必须显式，禁止跨来源合并统计。"""

    def test_defaults_to_none_not_a_guess(self):
        assert make_shot().target_zone_source is TargetZoneSource.NONE

    def test_sources_are_distinct_values(self):
        assert TargetZoneSource.SCENE_PRESET != TargetZoneSource.POST_HOC_INFERRED

    def test_zone_deviation_is_zero_inside(self):
        zone = Zone(center=Point(x_mm=1000, y_mm=600), radius_mm=200)
        assert zone.deviation_of(Point(x_mm=1050, y_mm=620)) == 0
        assert zone.contains(Point(x_mm=1050, y_mm=620))

    def test_zone_deviation_measures_from_boundary(self):
        zone = Zone(center=Point(x_mm=1000, y_mm=600), radius_mm=200)
        assert zone.deviation_of(Point(x_mm=1500, y_mm=600)) == 300


class TestUpperLayerBoundary:
    """AD-13b — trajectory 与 raw_observation 上层禁读。

    静态无法强制，但可以让边界显式且可审查：任何 import Shot 的
    上层模块出现这两个字段名，review 时应被拦下。
    """

    RESTRICTED = {"trajectory", "raw_observation"}

    def test_restricted_fields_exist_and_are_optional(self):
        shot = make_shot()
        assert shot.trajectory is None
        assert shot.raw_observation == {}

    def test_restricted_fields_are_documented_as_forbidden(self):
        for name in self.RESTRICTED:
            desc = Shot.model_fields[name].description or ""
            assert "禁读" in desc, f"{name} 缺少禁读标注，上层会误用"

    def test_scalar_derivatives_are_the_sanctioned_surface(self):
        """上层能拿到的东西必须足以做评分，否则会被逼去读 trajectory。"""
        shot = make_shot(
            struck_ball=3,
            potted_balls=[3],
            cushion_count=2,
            travel_distance_mm=1400,
        )
        assert shot.struck_ball == 3
        assert shot.potted_balls == [3]
        assert shot.cushion_count == 2
        assert shot.travel_distance_mm == 1400
        assert shot.duration_ms == 3200


class TestSegmentationReproducibility:
    """AD-13c — 切分权威。两种实现共用同一组静止阈值常量。"""

    def test_stillness_constants_are_shared_not_duplicated(self):
        from billiards_ai.shared import constants

        src = inspect.getsource(constants)
        assert "STILL_THRESHOLD_MM" in src
        assert "STILL_DURATION_MS" in src

    def test_sequence_starts_at_one(self):
        with pytest.raises(ValueError):
            make_shot(sequence=0)
