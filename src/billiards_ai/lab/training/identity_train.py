"""Frozen identity gates and checkpoint/ONNX semantic parity checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IdentityGateResult:
    passed: bool
    violations: tuple[str, ...]


def evaluate_identity_gates(
    *,
    group_macro_f1: float,
    cue_f1: float,
    eight_f1: float,
    solid_f1: float,
    stripe_f1: float,
    forced_exact_without_evidence: int,
) -> IdentityGateResult:
    values = (group_macro_f1, cue_f1, eight_f1, solid_f1, stripe_f1)
    if any(value < 0 or value > 1 for value in values) or forced_exact_without_evidence < 0:
        raise ValueError("identity metrics are outside their valid ranges")
    checks = (
        ("group_macro_f1", group_macro_f1 >= 0.90),
        ("cue_f1", cue_f1 >= 0.98),
        ("eight_f1", eight_f1 >= 0.98),
        ("solid_f1", solid_f1 >= 0.90),
        ("stripe_f1", stripe_f1 >= 0.90),
        (
            "forced_exact_without_evidence",
            forced_exact_without_evidence == 0,
        ),
    )
    violations = tuple(name for name, passed in checks if not passed)
    return IdentityGateResult(passed=not violations, violations=violations)


def verify_export_parity(
    *,
    pytorch_logits: dict[str, tuple[float, ...]],
    onnx_logits: dict[str, tuple[float, ...]],
    pytorch_class_maps: dict[str, tuple[str, ...]],
    onnx_class_maps: dict[str, tuple[str, ...]],
    absolute_tolerance: float = 1e-4,
) -> None:
    expected_heads = set(pytorch_class_maps)
    if not expected_heads or not expected_heads.issubset({"group", "exact", "known"}):
        raise ValueError("checkpoint output heads are invalid")
    if (
        set(pytorch_logits) != expected_heads
        or set(onnx_logits) != expected_heads
        or set(pytorch_class_maps) != expected_heads
        or set(onnx_class_maps) != expected_heads
    ):
        raise ValueError("ONNX output heads differ from checkpoint output heads")
    if pytorch_class_maps != onnx_class_maps:
        raise ValueError("ONNX class map differs from checkpoint class map")
    for head in sorted(expected_heads):
        expected_logits = pytorch_logits[head]
        actual_logits = onnx_logits[head]
        class_map = pytorch_class_maps[head]
        if not expected_logits or len(expected_logits) != len(actual_logits):
            raise ValueError("ONNX logits have a different shape")
        if len(expected_logits) != len(class_map):
            raise ValueError("logits do not match the class map")
        maximum_error = max(
            abs(expected - actual)
            for expected, actual in zip(expected_logits, actual_logits, strict=True)
        )
        if maximum_error > absolute_tolerance:
            raise ValueError("ONNX logits differ from checkpoint logits")
