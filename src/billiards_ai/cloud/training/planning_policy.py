"""Deterministic workload and validity policy for Story 9.6 planning drafts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import StrictInt, model_validator

from billiards_ai.shared.models.events import CoreContractModel


class TrainingPlanningPolicyV1(CoreContractModel):
    """Frozen engineering limits; model providers cannot override these values."""

    schema_version: Literal["1.0"] = "1.0"
    policy_version: Literal["training-planning-policy-v1"] = "training-planning-policy-v1"
    maximum_blocks: Literal[2] = 2
    maximum_open_load_blocks: Literal[1] = 1
    maximum_total_repetitions: Literal[40] = 40
    seconds_per_repetition: Literal[45] = 45
    maximum_estimated_duration_seconds: Literal[1800] = 1800
    valid_for_seconds: Literal[604800] = 604800
    maximum_provider_payload_bytes: Literal[32768] = 32768
    maximum_provider_output_bytes: Literal[16384] = 16384
    maximum_provider_output_nodes: Literal[128] = 128
    maximum_provider_output_depth: Literal[8] = 8

    @model_validator(mode="after")
    def limits_are_internally_consistent(self) -> Self:
        if self.maximum_open_load_blocks > self.maximum_blocks:
            raise ValueError("open-load block cap cannot exceed nominal cap")
        if (
            self.maximum_total_repetitions * self.seconds_per_repetition
            > self.maximum_estimated_duration_seconds
        ):
            raise ValueError("repetition and duration caps are inconsistent")
        return self

    def allowed_blocks_for_open_plans(self, *, open_plan_count: StrictInt) -> int:
        if (
            isinstance(open_plan_count, bool)
            or not isinstance(open_plan_count, int)
            or open_plan_count < 0
        ):
            raise ValueError("open plan count must be a non-negative strict integer")
        if open_plan_count:
            return self.maximum_open_load_blocks
        return self.maximum_blocks


__all__ = ["TrainingPlanningPolicyV1"]
