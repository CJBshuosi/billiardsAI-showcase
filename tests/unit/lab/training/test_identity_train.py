import pytest

from billiards_ai.lab.training.identity_train import (
    evaluate_identity_gates,
    verify_export_parity,
)


def test_identity_gates_enforce_group_quality_and_optional_exact_safety() -> None:
    passing = evaluate_identity_gates(
        group_macro_f1=0.91,
        cue_f1=0.99,
        eight_f1=0.985,
        solid_f1=0.92,
        stripe_f1=0.93,
        forced_exact_without_evidence=0,
    )
    failing = evaluate_identity_gates(
        group_macro_f1=0.91,
        cue_f1=0.99,
        eight_f1=0.985,
        solid_f1=0.92,
        stripe_f1=0.93,
        forced_exact_without_evidence=1,
    )

    assert passing.passed is True
    assert failing.passed is False
    assert "forced_exact_without_evidence" in failing.violations


def test_export_parity_requires_same_class_maps_and_logits_for_both_heads() -> None:
    verify_export_parity(
        pytorch_logits={"group": (0.1, 0.9), "exact": (0.8, 0.2)},
        onnx_logits={"group": (0.100001, 0.899999), "exact": (0.8, 0.2)},
        pytorch_class_maps={"group": ("solid", "stripe"), "exact": ("4", "12")},
        onnx_class_maps={"group": ("solid", "stripe"), "exact": ("4", "12")},
        absolute_tolerance=1e-4,
    )

    with pytest.raises(ValueError, match="class map"):
        verify_export_parity(
            pytorch_logits={"group": (0.1, 0.9), "exact": (0.8, 0.2)},
            onnx_logits={"group": (0.1, 0.9), "exact": (0.8, 0.2)},
            pytorch_class_maps={"group": ("solid", "stripe"), "exact": ("4", "12")},
            onnx_class_maps={"group": ("stripe", "solid"), "exact": ("4", "12")},
        )


def test_export_parity_accepts_group_only_when_exact_head_is_deferred() -> None:
    verify_export_parity(
        pytorch_logits={"group": (0.1, 0.9)},
        onnx_logits={"group": (0.100001, 0.899999)},
        pytorch_class_maps={"group": ("solid", "stripe")},
        onnx_class_maps={"group": ("solid", "stripe")},
        absolute_tolerance=1e-4,
    )
