from __future__ import annotations

import pytest
from pydantic import ValidationError

from billiards_ai.lab.propagation import (
    InstanceAbsence,
    InstancePrompt,
    MaskObservation,
    PilotQaConfig,
    PropagationResult,
    PropagationWindow,
    RelayEvidence,
    agreement_frames,
    initial_window_plan,
    relay_successor,
    replan_after_oom,
)


def _prompt(instance_id: str = "ball-1") -> InstancePrompt:
    return InstancePrompt(
        asset_id="video-1",
        frame_index=0,
        instance_id=instance_id,
        class_name="ball",
        source_kind="gold",
        source_id="gold-0",
        box_xyxy_px=(10.0, 10.0, 20.0, 20.0),
    )


def _observation(frame_index: int) -> MaskObservation:
    return MaskObservation(
        asset_id="video-1",
        frame_index=frame_index,
        instance_id="ball-1",
        class_name="ball",
        prompt_frame_index=0,
        center_px=(15.0, 15.0),
        box_xyxy_px=(10.0, 10.0, 20.0, 20.0),
        area_px=80,
        mask_ref=f"masks/ball-1/{frame_index:06d}.png",
    )


def _result(window: PropagationWindow, frames: tuple[int, ...]) -> PropagationResult:
    return PropagationResult(
        backend_id="fake",
        backend_version="1.0",
        window=window,
        prompts=(_prompt(),),
        observations=tuple(_observation(frame) for frame in frames),
    )


def test_initial_window_plan_uses_default_30_10_for_short_and_long_half_open_shots() -> None:
    short = initial_window_plan(asset_id="video-1", start_frame=4, end_frame=18)
    long = initial_window_plan(asset_id="video-1", start_frame=0, end_frame=100)

    assert [(item.start_frame, item.end_frame) for item in short.windows] == [(4, 18)]
    assert [(item.start_frame, item.end_frame) for item in long.windows] == [
        (0, 30),
        (20, 50),
        (40, 70),
        (60, 90),
        (80, 100),
    ]
    assert long.coverage_proof.missing_frames == ()
    assert long.coverage_proof.covered_frames == tuple(range(100))


def test_plan_hash_is_canonical_and_changes_with_the_plan() -> None:
    first = initial_window_plan(asset_id="video-1", start_frame=0, end_frame=50)
    same = initial_window_plan(asset_id="video-1", start_frame=0, end_frame=50)
    changed = initial_window_plan(asset_id="video-1", start_frame=0, end_frame=51)

    assert first.plan_hash == same.plan_hash
    assert len(first.plan_hash) == 64
    assert first.plan_hash != changed.plan_hash


def test_oom_replan_preserves_committed_attempt_and_replaces_failed_tail_from_failed_start() -> (
    None
):
    original = initial_window_plan(asset_id="video-1", start_frame=0, end_frame=50)
    committed = original.with_attempt_status(ordinal=0, status="committed")
    failed = committed.windows[1]

    replanned = replan_after_oom(plan=committed, failed_window=failed)

    attempt_summary = [
        (item.window.start_frame, item.window.end_frame, item.status) for item in replanned.attempts
    ]
    assert attempt_summary == [
        (0, 30, "committed"),
        (20, 50, "failed"),
        (20, 35, "planned"),
        (30, 45, "planned"),
        (40, 50, "planned"),
    ]
    assert [(item.start_frame, item.end_frame) for item in replanned.windows] == [
        (0, 30),
        (20, 35),
        (30, 45),
        (40, 50),
    ]
    assert replanned.coverage_proof.missing_frames == ()


def test_mixed_transition_agreement_is_actual_intersection_without_relay_frame() -> None:
    plan = initial_window_plan(asset_id="video-1", start_frame=0, end_frame=50)
    replanned = replan_after_oom(
        plan=plan.with_attempt_status(ordinal=0, status="committed"),
        failed_window=plan.windows[1],
    )

    assert agreement_frames(replanned.windows[0], replanned.windows[1]) == tuple(range(21, 30))


def test_successor_relays_only_its_start_frame_and_records_an_absence_when_missing() -> None:
    predecessor = PropagationWindow(asset_id="video-1", ordinal=0, start_frame=0, end_frame=30)
    successor = PropagationWindow(asset_id="video-1", ordinal=1, start_frame=20, end_frame=50)
    prior_result = _result(predecessor, (19, 21))

    evidence = relay_successor(predecessor=prior_result, successor=successor)

    assert evidence[0].relay_frame == 20
    assert evidence[0].relay_mask_ref is None
    assert evidence[0].absence is not None
    assert evidence[0].absence.frame_index == 20
    assert evidence[0].absence.reason == "missing_relay_mask"


def test_successor_relay_uses_the_mask_at_exact_start_when_available() -> None:
    predecessor = PropagationWindow(asset_id="video-1", ordinal=0, start_frame=0, end_frame=30)
    successor = PropagationWindow(asset_id="video-1", ordinal=1, start_frame=20, end_frame=50)

    evidence = relay_successor(predecessor=_result(predecessor, (20, 21)), successor=successor)

    assert evidence[0].relay_mask_ref == "masks/ball-1/000020.png"
    assert evidence[0].absence is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asset_id", "video-2"),
        ("class_name", "cue"),
        ("instance_id", "ball-2"),
        ("frame_index", 21),
        ("prompt_frame_index", 1),
    ],
)
def test_relay_evidence_rejects_an_absence_that_does_not_match_its_prompt(
    field: str,
    value: str | int,
) -> None:
    predecessor = PropagationWindow(asset_id="video-1", ordinal=0, start_frame=0, end_frame=30)
    successor = PropagationWindow(asset_id="video-1", ordinal=1, start_frame=20, end_frame=50)
    absence = InstanceAbsence(
        asset_id="video-1",
        frame_index=20,
        instance_id="ball-1",
        class_name="ball",
        prompt_frame_index=0,
        reason="missing_relay_mask",
    ).model_copy(update={field: value})

    with pytest.raises(ValidationError, match="relay absence"):
        RelayEvidence(
            predecessor_window=predecessor,
            successor_window=successor,
            instance_id="ball-1",
            class_name="ball",
            prompt_frame_index=0,
            relay_frame=20,
            absence=absence,
            agreement_frames=tuple(range(21, 30)),
        )


def test_frozen_qa_config_has_a_stable_hash_and_rejects_threshold_overrides() -> None:
    config = PilotQaConfig()

    assert config.canonical_hash == PilotQaConfig().canonical_hash
    with pytest.raises(ValidationError, match="frozen"):
        PilotQaConfig(minimum_overlap_iou=0.3)
