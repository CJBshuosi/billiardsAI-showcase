from __future__ import annotations

from billiards_ai.lab.propagation import InstancePrompt, PropagationWindow
from billiards_ai.lab.propagation.fake import FakePropagationBackend


def _prompt(instance_id: str, *, left: float) -> InstancePrompt:
    return InstancePrompt(
        asset_id="video-1",
        frame_index=10,
        instance_id=instance_id,
        class_name="ball",
        source_kind="gold",
        source_id=f"gold-{instance_id}",
        box_xyxy_px=(left, 20.0, left + 10.0, 30.0),
    )


def _window() -> PropagationWindow:
    return PropagationWindow(asset_id="video-1", ordinal=0, start_frame=10, end_frame=13)


def test_fake_backend_propagates_each_prompt_deterministically() -> None:
    backend = FakePropagationBackend(frame_step_px=(2.0, 1.0))

    result = backend.propagate(window=_window(), prompts=(_prompt("ball-a", left=10.0),))

    assert [item.frame_index for item in result.observations] == [10, 11, 12]
    assert [item.center_px for item in result.observations] == [
        (15.0, 25.0),
        (17.0, 26.0),
        (19.0, 27.0),
    ]
    assert [item.mask_ref for item in result.observations] == [
        "masks/ball-a/000010.png",
        "masks/ball-a/000011.png",
        "masks/ball-a/000012.png",
    ]
    assert result.failures == ()


def test_fake_backend_keeps_same_class_instances_independent() -> None:
    backend = FakePropagationBackend()

    result = backend.propagate(
        window=_window(),
        prompts=(_prompt("ball-b", left=40.0), _prompt("ball-a", left=10.0)),
    )

    assert {(item.frame_index, item.instance_id) for item in result.observations} == {
        (10, "ball-a"),
        (10, "ball-b"),
        (11, "ball-a"),
        (11, "ball-b"),
        (12, "ball-a"),
        (12, "ball-b"),
    }


def test_fake_backend_returns_explicit_failure_and_stops_at_that_frame() -> None:
    backend = FakePropagationBackend(fail_at_frame=12)

    result = backend.propagate(window=_window(), prompts=(_prompt("ball-a", left=10.0),))

    assert [item.frame_index for item in result.observations] == [10, 11]
    assert len(result.failures) == 1
    assert result.failures[0].frame_index == 12
    assert result.failures[0].code == "synthetic_failure"
