from __future__ import annotations

import json
from pathlib import Path

from billiards_ai.lab.training.identity_group_training_run import (
    run_group_short_training,
)
from tests.unit.lab.training.test_identity_training_run import _reviewed_artifacts


class FakeGroupBackend:
    name = "fake-group"
    version = "1.0"

    def train(self, **kwargs: object) -> None:
        assert kwargs["exact_class_map"] == ()
        Path(kwargs["checkpoint_path"]).write_bytes(b"checkpoint")
        Path(kwargs["metrics_path"]).write_text(
            json.dumps(
                {
                    "trained_heads": ["group"],
                    "exact_head": {"status": "deferred", "trained": False},
                    "group_macro_f1": 0.40,
                    "cue_f1": 0.99,
                    "eight_f1": 0.99,
                    "solid_f1": 0.20,
                    "stripe_f1": 0.20,
                    "forced_exact_without_evidence": 0,
                }
            ),
            encoding="utf-8",
        )

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None:
        assert checkpoint_path.is_file()
        output_path.write_bytes(b"onnx")

    def parity_logits(self, **kwargs: object) -> tuple[object, object]:
        logits = {"group": (0.1, 0.2, 0.3, 0.4, 0.5)}
        return logits, logits


def test_short_training_preserves_failed_gate_diagnostics_and_defers_exact_head(
    tmp_path: Path,
) -> None:
    candidates, verdict = _reviewed_artifacts(tmp_path)

    result = run_group_short_training(
        candidate_artifact=candidates,
        verdict_artifact=verdict,
        config_path=Path("configs/algorithm/training/identity.yaml"),
        output_root=tmp_path / "runs",
        backend=FakeGroupBackend(),
    )

    manifest = json.loads((result.output / "manifest.json").read_text(encoding="utf-8"))
    assert result.status == "PASS"
    assert manifest["gate_status"] == "FAIL"
    assert manifest["release_eligible"] is False
    assert manifest["trained_heads"] == ["group"]
    assert manifest["exact_head"] == {"status": "deferred", "trained": False}
    assert (result.output / "metrics.json").is_file()
    assert (result.output / "group-head.onnx").is_file()


def test_short_training_accepts_four_known_probabilities_with_unknown_rejection(
    tmp_path: Path,
) -> None:
    candidates, verdict = _reviewed_artifacts(tmp_path)
    backend = FakeGroupBackend()
    backend.parity_class_map = ("cue", "solid", "eight", "stripe")  # type: ignore[attr-defined]

    def parity_logits(**kwargs: object) -> tuple[object, object]:
        logits = {"group": (0.1, 0.2, 0.3, 0.4)}
        return logits, logits

    backend.parity_logits = parity_logits  # type: ignore[method-assign]

    result = run_group_short_training(
        candidate_artifact=candidates,
        verdict_artifact=verdict,
        config_path=Path("configs/algorithm/training/identity.yaml"),
        output_root=tmp_path / "hierarchical-runs",
        backend=backend,
    )

    assert result.status == "PASS"


def test_short_training_verifies_independent_knownness_onnx_output(tmp_path: Path) -> None:
    candidates, verdict = _reviewed_artifacts(tmp_path)
    backend = FakeGroupBackend()
    backend.parity_class_maps = {  # type: ignore[attr-defined]
        "group": ("cue", "solid", "eight", "stripe"),
        "known": ("known",),
    }

    def parity_logits(**kwargs: object) -> tuple[object, object]:
        logits = {"group": (0.1, 0.2, 0.3, 0.4), "known": (0.8,)}
        return logits, logits

    backend.parity_logits = parity_logits  # type: ignore[method-assign]

    result = run_group_short_training(
        candidate_artifact=candidates,
        verdict_artifact=verdict,
        config_path=Path("configs/algorithm/training/identity.yaml"),
        output_root=tmp_path / "knownness-runs",
        backend=backend,
    )

    assert result.status == "PASS"


def test_short_training_overlays_scoped_corrections_and_records_identity(
    tmp_path: Path,
) -> None:
    candidates, verdict = _reviewed_artifacts(tmp_path)
    from billiards_ai.lab.training.identity_review import IdentityReviewSession

    correction = IdentityReviewSession.open(
        candidate_artifact=candidates,
        state_path=tmp_path / "correction-state.json",
        crop_ids=(
            json.loads((verdict / "verdict.json").read_text(encoding="utf-8"))["decisions"][0][
                "crop_id"
            ],
        ),
    )
    original = json.loads((verdict / "verdict.json").read_text(encoding="utf-8"))["decisions"][0]
    correction.record(
        crop_id=original["crop_id"],
        group=original["group"],
        exact_identity=original["exact_identity"],
        exact_evidence=original["exact_evidence"],
        visibility=original["visibility"],
        orientation=original["orientation"],
        note="validated correction scope",
    )
    correction_verdict = correction.export_verdict(tmp_path / "corrections")

    result = run_group_short_training(
        candidate_artifact=candidates,
        verdict_artifact=verdict,
        correction_verdict_artifact=correction_verdict,
        config_path=Path("configs/algorithm/training/identity.yaml"),
        output_root=tmp_path / "corrected-runs",
        backend=FakeGroupBackend(),
    )

    manifest = json.loads((result.output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["correction_verdict_artifact_id"] == correction_verdict.name
