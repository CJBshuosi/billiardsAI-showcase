from __future__ import annotations

import pytest
from pydantic import ValidationError

from billiards_ai.lab.propagation import (
    InstanceAbsence,
    InstancePrompt,
    MaskObservation,
    PilotQaConfig,
    PilotQaIdentity,
    PropagationFailure,
    PropagationResult,
    PropagationWindow,
)


def _window() -> PropagationWindow:
    return PropagationWindow(
        asset_id="video-1",
        ordinal=0,
        start_frame=30,
        end_frame=60,
    )


def _prompt(*, instance_id: str = "ball-1") -> InstancePrompt:
    return InstancePrompt(
        asset_id="video-1",
        frame_index=30,
        instance_id=instance_id,
        class_name="ball",
        source_kind="gold",
        source_id="gold-frame-30",
        box_xyxy_px=(100.0, 200.0, 120.0, 220.0),
    )


def _observation(*, frame_index: int = 31, asset_id: str = "video-1") -> MaskObservation:
    return MaskObservation(
        asset_id=asset_id,
        frame_index=frame_index,
        instance_id="ball-1",
        class_name="ball",
        prompt_frame_index=30,
        center_px=(111.0, 211.0),
        box_xyxy_px=(101.0, 201.0, 121.0, 221.0),
        area_px=314,
        mask_ref="masks/ball-1/000031.png",
    )


def test_prompt_requires_one_confined_prompt_geometry() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        _prompt().model_copy(update={"box_xyxy_px": None}, deep=True).__class__(
            **_prompt().model_dump(exclude={"box_xyxy_px"})
        )

    with pytest.raises(ValidationError, match="relative and confined"):
        InstancePrompt(
            asset_id="video-1",
            frame_index=30,
            instance_id="ball-1",
            class_name="ball",
            source_kind="gold",
            source_id="gold-frame-30",
            mask_ref="../outside.png",
        )


def test_window_is_a_non_empty_half_open_interval() -> None:
    with pytest.raises(ValidationError, match="non-empty half-open"):
        PropagationWindow(
            asset_id="video-1",
            ordinal=0,
            start_frame=30,
            end_frame=30,
        )


def test_result_rejects_duplicate_prompt_instance_on_the_same_frame() -> None:
    duplicate = _prompt().model_copy(update={"source_id": "second-source"})

    with pytest.raises(ValidationError, match="prompt instance IDs must be unique"):
        PropagationResult(
            backend_id="fake",
            backend_version="1.0",
            window=_window(),
            prompts=(_prompt(), duplicate),
        )


def test_result_confines_observations_and_failures_to_its_window() -> None:
    with pytest.raises(ValidationError, match="observation must fall inside"):
        PropagationResult(
            backend_id="fake",
            backend_version="1.0",
            window=_window(),
            prompts=(_prompt(),),
            observations=(_observation(frame_index=60),),
        )

    with pytest.raises(ValidationError, match="failure must reference the result asset"):
        PropagationResult(
            backend_id="fake",
            backend_version="1.0",
            window=_window(),
            prompts=(_prompt(),),
            failures=(
                PropagationFailure(
                    asset_id="video-2",
                    frame_index=31,
                    code="synthetic_failure",
                    message="expected test failure",
                ),
            ),
        )


def test_mask_observation_is_always_unreviewed_pseudo_evidence() -> None:
    observation = _observation()

    assert observation.provenance == "pseudo"
    assert observation.review_status == "unreviewed"

    with pytest.raises(ValidationError):
        MaskObservation.model_validate(
            {
                **observation.model_dump(),
                "provenance": "gold",
                "review_status": "accepted",
            }
        )


def test_result_keeps_a_valid_absence_alongside_positive_observations() -> None:
    result = PropagationResult(
        backend_id="fake",
        backend_version="1.0",
        window=_window(),
        prompts=(_prompt(), _prompt(instance_id="ball-2")),
        observations=(_observation(),),
        absences=(
            InstanceAbsence(
                asset_id="video-1",
                frame_index=31,
                instance_id="ball-2",
                class_name="ball",
                prompt_frame_index=30,
                reason="missing_relay_mask",
            ),
        ),
    )

    assert result.observations[0].instance_id == "ball-1"
    assert result.absences[0].instance_id == "ball-2"


def test_result_rejects_malformed_absence_that_conflicts_with_mask_evidence() -> None:
    with pytest.raises(ValidationError, match="cannot both be observed and absent"):
        PropagationResult(
            backend_id="fake",
            backend_version="1.0",
            window=_window(),
            prompts=(_prompt(),),
            observations=(_observation(),),
            absences=(
                InstanceAbsence(
                    asset_id="video-1",
                    frame_index=31,
                    instance_id="ball-1",
                    class_name="ball",
                    prompt_frame_index=30,
                    reason="empty_mask",
                ),
            ),
        )


def test_result_binds_the_frozen_qa_config_hash_into_raw_identity() -> None:
    result = PropagationResult(
        backend_id="fake",
        backend_version="1.0",
        window=_window(),
        prompts=(_prompt(),),
    )

    assert result.pilot_qa_config_hash == PilotQaConfig().canonical_hash

    with pytest.raises(ValidationError, match="frozen QA configuration"):
        PropagationResult(
            backend_id="fake",
            backend_version="1.0",
            window=_window(),
            prompts=(_prompt(),),
            pilot_qa_config_hash="0" * 64,
        )


@pytest.mark.parametrize("artifact_kind", ["worker_job", "raw_run", "evaluation_report"])
def test_qa_identity_binds_each_pilot_artifact_to_the_canonical_config(
    artifact_kind: str,
) -> None:
    identity = PilotQaIdentity(artifact_kind=artifact_kind)

    assert identity.pilot_qa_config_hash == PilotQaConfig().canonical_hash


def test_qa_identity_rejects_a_mismatched_config_hash() -> None:
    with pytest.raises(ValidationError, match="frozen QA configuration"):
        PilotQaIdentity(artifact_kind="worker_job", pilot_qa_config_hash="0" * 64)
