from __future__ import annotations

import json
from pathlib import Path

import pytest

from billiards_ai.lab.training.identity_training_run import (
    IdentityTrainingError,
    run_identity_training,
)
from tests.unit.lab.training.test_identity_review import _session


class FakeIdentityTrainingBackend:
    name = "fake-identity"
    version = "1.0"

    def train(
        self,
        *,
        train_records: tuple[object, ...],
        valid_records: tuple[object, ...],
        group_class_map: tuple[str, ...],
        exact_class_map: tuple[str, ...],
        checkpoint_path: Path,
        metrics_path: Path,
    ) -> None:
        assert train_records and valid_records
        assert group_class_map == ("cue", "solid", "eight", "stripe", "unknown")
        assert exact_class_map[0] == "cue" and exact_class_map[-1] == "unknown"
        checkpoint_path.write_bytes(b"identity-checkpoint")
        metrics_path.write_text(
            json.dumps(
                {
                    "group_macro_f1": 0.92,
                    "cue_f1": 0.99,
                    "eight_f1": 0.99,
                    "solid_f1": 0.93,
                    "stripe_f1": 0.94,
                    "forced_exact_without_evidence": 0,
                    "exact_labeled_count": 2,
                    "exact_macro_f1": 0.75,
                }
            ),
            encoding="utf-8",
        )

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None:
        assert checkpoint_path.read_bytes() == b"identity-checkpoint"
        output_path.write_bytes(b"identity-onnx")

    def parity_logits(self, *, checkpoint_path: Path, onnx_path: Path) -> tuple[object, object]:
        assert checkpoint_path.is_file() and onnx_path.is_file()
        logits = {
            "group": tuple(0.1 for _ in range(5)),
            "exact": tuple(0.1 for _ in range(17)),
        }
        return logits, logits


def _reviewed_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    session = _session(tmp_path)
    labels = ("cue", "8")
    for crop_id, label in zip(session.crop_ids, labels, strict=True):
        session.record(
            crop_id=crop_id,
            exact_identity=label,
            visibility="visible",
            orientation="number_visible",
        )
    return session.candidate_artifact, session.export_verdict(tmp_path / "verdicts")


def test_training_run_publishes_verified_checkpoint_onnx_metrics_and_manifest(
    tmp_path: Path,
) -> None:
    candidates, verdict = _reviewed_artifacts(tmp_path)

    result = run_identity_training(
        candidate_artifact=candidates,
        verdict_artifact=verdict,
        config_path=Path("configs/algorithm/training/identity.yaml"),
        output_root=tmp_path / "runs",
        backend=FakeIdentityTrainingBackend(),
    )

    assert result.status == "PASS"
    assert result.output.name == result.run_id
    assert (result.output / "checkpoint.pt").read_bytes() == b"identity-checkpoint"
    assert (result.output / "model.onnx").read_bytes() == b"identity-onnx"
    manifest = json.loads((result.output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gate_status"] == "PASS"
    assert manifest["backend"] == {"name": "fake-identity", "version": "1.0"}
    assert manifest["class_maps"]["group"] == [
        "cue",
        "solid",
        "eight",
        "stripe",
        "unknown",
    ]
    assert manifest["class_maps"]["exact"] == [
        "cue",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "unknown",
    ]


def test_training_run_rejects_failed_frozen_identity_gates(tmp_path: Path) -> None:
    candidates, verdict = _reviewed_artifacts(tmp_path)
    backend = FakeIdentityTrainingBackend()

    original_train = backend.train

    def failing_train(**kwargs: object) -> None:
        original_train(**kwargs)  # type: ignore[arg-type]
        Path(kwargs["metrics_path"]).write_text(
            json.dumps(
                {
                    "group_macro_f1": 0.20,
                    "cue_f1": 0.99,
                    "eight_f1": 0.99,
                    "solid_f1": 0.93,
                    "stripe_f1": 0.94,
                    "forced_exact_without_evidence": 0,
                    "exact_labeled_count": 2,
                    "exact_macro_f1": 0.75,
                }
            ),
            encoding="utf-8",
        )

    backend.train = failing_train  # type: ignore[method-assign]
    with pytest.raises(IdentityTrainingError, match="frozen identity gates"):
        run_identity_training(
            candidate_artifact=candidates,
            verdict_artifact=verdict,
            config_path=Path("configs/algorithm/training/identity.yaml"),
            output_root=tmp_path / "runs",
            backend=backend,
        )
