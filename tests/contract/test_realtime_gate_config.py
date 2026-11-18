from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from billiards_ai.lab.realtime_gate import evaluate_realtime_readiness

CONFIG = Path("configs/algorithm/realtime-gate.yaml")


def _ready_payload() -> dict:
    return {
        "schema_version": "1.0",
        "gate_id": "product-v1-realtime-readiness-r1",
        "stage": "Product-V1",
        "prerequisites": {
            "offline_brain_status": "PASS",
            "teacher_export_compatibility": "approved",
            "identity_export_compatibility": "approved",
            "cue_stick_aim_line_annotation_status": "approved",
            "cue_dataset_status": "approved",
            "candidate_device_list_status": "approved",
            "target_hardware_benchmark_status": "approved",
            "power_thermal_protocol_status": "approved",
            "offline_degradation_status": "approved",
            "commercial_runtime_licenses_status": "approved",
        },
        "performance_targets": {
            "table_state_hz_min": 8,
            "aim_line_hz_min": 15,
            "end_to_end_guidance_ms_max": 200,
        },
    }


def test_checked_in_gate_blocks_premature_realtime_work() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    decision = evaluate_realtime_readiness(payload)

    assert decision.allowed is False
    assert "offline_brain_status" in decision.blockers
    assert "cue_dataset_status" in decision.blockers
    assert "target_hardware_benchmark_status" in decision.blockers


@pytest.mark.parametrize(
    "field",
    [
        "offline_brain_status",
        "teacher_export_compatibility",
        "identity_export_compatibility",
        "cue_stick_aim_line_annotation_status",
        "cue_dataset_status",
        "candidate_device_list_status",
        "target_hardware_benchmark_status",
        "power_thermal_protocol_status",
        "offline_degradation_status",
        "commercial_runtime_licenses_status",
    ],
)
def test_each_unapproved_prerequisite_blocks_realtime_start(field: str) -> None:
    payload = _ready_payload()
    payload["prerequisites"][field] = "pending"

    decision = evaluate_realtime_readiness(payload)

    assert decision.allowed is False
    assert field in decision.blockers


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("table_state_hz_min", 7.99),
        ("aim_line_hz_min", 14.99),
        ("end_to_end_guidance_ms_max", 200.01),
    ],
)
def test_weaker_performance_target_blocks_realtime_start(field: str, unsafe_value: float) -> None:
    payload = _ready_payload()
    payload["performance_targets"][field] = unsafe_value

    decision = evaluate_realtime_readiness(payload)

    assert decision.allowed is False
    assert field in decision.blockers


def test_gate_allows_planning_only_when_every_frozen_condition_is_ready() -> None:
    payload = deepcopy(_ready_payload())

    decision = evaluate_realtime_readiness(payload)

    assert decision.allowed is True
    assert decision.blockers == ()
