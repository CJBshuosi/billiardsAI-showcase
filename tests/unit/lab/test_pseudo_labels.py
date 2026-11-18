from __future__ import annotations

from billiards_ai.lab.pseudo_labels import (
    FrameGeometry,
    InstanceCandidate,
    MotionContext,
    ReviewStrata,
    fuse_candidates,
    select_review_items,
)

GEOMETRY = FrameGeometry(width_px=1920, height_px=1080, table_bounds_px=(100, 80, 1820, 1000))
NO_MOTION = MotionContext(active_ratio=0.0, speed_px_per_s=0.0)


def _candidate(
    teacher_id: str,
    *,
    center: tuple[float, float] = (960.0, 540.0),
    confidence: float = 0.92,
    occluded: bool = False,
) -> InstanceCandidate:
    return InstanceCandidate(
        asset_id="video-001",
        frame_index=12,
        timestamp_ms=400,
        teacher_id=teacher_id,
        teacher_version="1.0",
        teacher_sha256="a" * 64,
        class_name="ball",
        confidence=confidence,
        center_px=center,
        box_xyxy_px=(center[0] - 10, center[1] - 10, center[0] + 10, center[1] + 10),
        occluded=occluded,
    )


def test_fusion_never_promotes_teacher_output_to_gold() -> None:
    labels = fuse_candidates((_candidate("rfdetr"),), GEOMETRY, NO_MOTION)

    assert len(labels) == 1
    assert labels[0].provenance == "pseudo"
    assert labels[0].review_status == "unreviewed"
    assert labels[0].teacher_ids == ("rfdetr",)


def test_fusion_rejects_out_of_table_and_impossible_ball_size() -> None:
    outside = _candidate("classic-cv", center=(20.0, 20.0))
    huge = _candidate("rfdetr").model_copy(update={"box_xyxy_px": (600.0, 200.0, 1300.0, 900.0)})

    assert fuse_candidates((outside, huge), GEOMETRY, NO_MOTION) == ()


def test_nearby_teachers_fuse_and_preserve_each_source() -> None:
    first = _candidate("classic-cv", center=(960.0, 540.0), confidence=0.80)
    second = _candidate("rfdetr", center=(964.0, 542.0), confidence=0.94)

    labels = fuse_candidates((second, first), GEOMETRY, NO_MOTION)

    assert len(labels) == 1
    assert labels[0].teacher_ids == ("classic-cv", "rfdetr")
    assert labels[0].teacher_agreement is True
    assert labels[0].confidence == 0.87


def test_disagreement_and_occlusion_rank_before_easy_consensus() -> None:
    labels = (
        *fuse_candidates((_candidate("rfdetr", confidence=0.45),), GEOMETRY, NO_MOTION),
        *fuse_candidates(
            (_candidate("classic-cv", center=(700.0, 500.0), occluded=True),),
            GEOMETRY,
            NO_MOTION,
        ),
        *fuse_candidates(
            (
                _candidate("classic-cv", center=(1200.0, 650.0)),
                _candidate("rfdetr", center=(1202.0, 652.0)),
            ),
            GEOMETRY,
            NO_MOTION,
        ),
    )

    selected = select_review_items(labels, limit=2, strata=ReviewStrata())

    assert [item.reason for item in selected] == ["teacher_disagreement", "occlusion"]
    assert all(item.review_status == "unreviewed" for item in selected)


def test_review_selection_is_deterministic_and_does_not_duplicate_labels() -> None:
    labels = fuse_candidates(
        (
            _candidate("classic-cv", center=(110.0, 100.0), confidence=0.62),
            _candidate("rfdetr", center=(110.0, 100.0), confidence=0.62),
        ),
        GEOMETRY,
        MotionContext(active_ratio=0.2, speed_px_per_s=900.0),
    )

    selected = select_review_items(labels, limit=10, strata=ReviewStrata())

    assert len(selected) == len(labels)
    assert len({item.label_id for item in selected}) == len(selected)


def test_review_limit_covers_hard_strata_before_more_of_same_reason() -> None:
    disagreements = tuple(
        label
        for index in range(3)
        for label in fuse_candidates(
            (_candidate("rfdetr", center=(500.0 + index * 80, 400.0)),),
            GEOMETRY,
            NO_MOTION,
        )
    )
    occlusion = fuse_candidates(
        (_candidate("classic-cv", center=(1000.0, 600.0), occluded=True),),
        GEOMETRY,
        NO_MOTION,
    )

    selected = select_review_items((*disagreements, *occlusion), limit=2, strata=ReviewStrata())

    assert [item.reason for item in selected] == ["teacher_disagreement", "occlusion"]
