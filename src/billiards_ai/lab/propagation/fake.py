"""Deterministic no-media backend used to exercise propagation boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from billiards_ai.lab.propagation.contracts import (
    InstancePrompt,
    MaskObservation,
    PropagationFailure,
    PropagationResult,
    PropagationWindow,
)


@dataclass(frozen=True)
class FakePropagationBackend:
    frame_step_px: tuple[float, float] = (0.0, 0.0)
    fail_at_frame: int | None = None
    backend_id: str = "fake-propagation"
    backend_version: str = "1.0"

    def propagate(
        self,
        *,
        window: PropagationWindow,
        prompts: tuple[InstancePrompt, ...],
    ) -> PropagationResult:
        observations: list[MaskObservation] = []
        failures: list[PropagationFailure] = []
        ordered_prompts = tuple(
            sorted(prompts, key=lambda item: (item.frame_index, item.instance_id))
        )

        for frame_index in range(window.start_frame, window.end_frame):
            if self.fail_at_frame == frame_index:
                failures.append(
                    PropagationFailure(
                        asset_id=window.asset_id,
                        frame_index=frame_index,
                        code="synthetic_failure",
                        message="fake backend stopped at the configured frame",
                    )
                )
                break
            for prompt in ordered_prompts:
                if frame_index < prompt.frame_index:
                    continue
                if prompt.box_xyxy_px is None:
                    failures.append(
                        PropagationFailure(
                            asset_id=window.asset_id,
                            frame_index=frame_index,
                            code="unsupported_mask_prompt",
                            message="fake backend requires box prompt geometry",
                        )
                    )
                    continue
                delta = frame_index - prompt.frame_index
                shift_x = self.frame_step_px[0] * delta
                shift_y = self.frame_step_px[1] * delta
                left, top, right, bottom = prompt.box_xyxy_px
                box = (
                    round(left + shift_x, 6),
                    round(top + shift_y, 6),
                    round(right + shift_x, 6),
                    round(bottom + shift_y, 6),
                )
                observations.append(
                    MaskObservation(
                        asset_id=window.asset_id,
                        frame_index=frame_index,
                        instance_id=prompt.instance_id,
                        class_name=prompt.class_name,
                        prompt_frame_index=prompt.frame_index,
                        center_px=(
                            round((box[0] + box[2]) / 2, 6),
                            round((box[1] + box[3]) / 2, 6),
                        ),
                        box_xyxy_px=box,
                        area_px=max(1, round((right - left) * (bottom - top))),
                        mask_ref=f"masks/{prompt.instance_id}/{frame_index:06d}.png",
                    )
                )

        return PropagationResult(
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            window=window,
            prompts=ordered_prompts,
            observations=tuple(observations),
            failures=tuple(failures),
        )
