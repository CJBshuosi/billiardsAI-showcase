"""Binary readiness boundary for future Product V1 realtime work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _GateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RealtimePrerequisites(_GateModel):
    offline_brain_status: str
    teacher_export_compatibility: str
    identity_export_compatibility: str
    cue_stick_aim_line_annotation_status: str
    cue_dataset_status: str
    candidate_device_list_status: str
    target_hardware_benchmark_status: str
    power_thermal_protocol_status: str
    offline_degradation_status: str
    commercial_runtime_licenses_status: str


class RealtimePerformanceTargets(_GateModel):
    table_state_hz_min: float = Field(gt=0)
    aim_line_hz_min: float = Field(gt=0)
    end_to_end_guidance_ms_max: float = Field(gt=0)


class RealtimeGateConfig(_GateModel):
    schema_version: Literal["1.0"]
    gate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    stage: Literal["Product-V1"]
    prerequisites: RealtimePrerequisites
    performance_targets: RealtimePerformanceTargets


@dataclass(frozen=True)
class RealtimeReadinessDecision:
    allowed: bool
    blockers: tuple[str, ...]


_APPROVED_FIELDS = (
    "teacher_export_compatibility",
    "identity_export_compatibility",
    "cue_stick_aim_line_annotation_status",
    "cue_dataset_status",
    "candidate_device_list_status",
    "target_hardware_benchmark_status",
    "power_thermal_protocol_status",
    "offline_degradation_status",
    "commercial_runtime_licenses_status",
)


def evaluate_realtime_readiness(payload: Any) -> RealtimeReadinessDecision:
    try:
        config = RealtimeGateConfig.model_validate(payload)
    except ValidationError:
        return RealtimeReadinessDecision(False, ("configuration_invalid",))

    blockers = []
    prerequisites = config.prerequisites
    if prerequisites.offline_brain_status != "PASS":
        blockers.append("offline_brain_status")
    for field in _APPROVED_FIELDS:
        if getattr(prerequisites, field) != "approved":
            blockers.append(field)

    targets = config.performance_targets
    if targets.table_state_hz_min < 8:
        blockers.append("table_state_hz_min")
    if targets.aim_line_hz_min < 15:
        blockers.append("aim_line_hz_min")
    if targets.end_to_end_guidance_ms_max > 200:
        blockers.append("end_to_end_guidance_ms_max")
    return RealtimeReadinessDecision(not blockers, tuple(blockers))
