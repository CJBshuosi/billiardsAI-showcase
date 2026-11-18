from __future__ import annotations

import json
from pathlib import Path

from billiards_ai.lab.training.identity_group_error_review import (
    build_identity_group_error_review,
)
from billiards_ai.lab.training.identity_review import (
    IdentityReviewSession,
    load_reviewed_identity_dataset,
)
from tests.unit.lab.training.test_identity_training_run import _reviewed_artifacts


class FakePredictor:
    def predict_records(
        self, *, checkpoint_path: Path, records: tuple[object, ...]
    ) -> tuple[dict[str, object], ...]:
        assert checkpoint_path.read_bytes() == b"checkpoint"
        return tuple(
            {
                "crop_id": record.crop_id,
                "truth": record.group,
                "predicted": "stripe" if record.group != "stripe" else "cue",
                "probabilities": [0.05, 0.05, 0.05, 0.85],
                "confidence": 0.85,
                "margin": 0.80,
            }
            for record in records
        )


def test_build_error_review_selects_only_frozen_validation_mistakes(
    tmp_path: Path,
) -> None:
    candidates, verdict = _reviewed_artifacts(tmp_path)
    valid = load_reviewed_identity_dataset(
        candidate_artifact=candidates,
        verdict_artifact=verdict,
        split="valid",
    )
    training_run = tmp_path / "training-run"
    training_run.mkdir()
    (training_run / "group-head.npz").write_bytes(b"checkpoint")
    (training_run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": training_run.name,
                "candidate_artifact_id": candidates.name,
                "verdict_artifact_id": verdict.name,
                "backend": {
                    "name": "dinov2-frozen-hierarchical-group-mlp",
                    "version": "2.0",
                },
            }
        ),
        encoding="utf-8",
    )

    review = build_identity_group_error_review(
        candidate_artifact=candidates,
        verdict_artifact=verdict,
        training_run=training_run,
        state_path=tmp_path / "error-review-state.json",
        predictor=FakePredictor(),
    )

    assert review.crop_ids == (valid.records[0].crop_id,)
    assert review.reviewed_count == 0
    assert review.suggestions[review.crop_ids[0]].group == valid.records[0].group
    assert review.contexts[review.crop_ids[0]]["predicted"] == "stripe"


def test_build_error_review_rejects_training_run_for_other_labels(tmp_path: Path) -> None:
    candidates, verdict = _reviewed_artifacts(tmp_path)
    training_run = tmp_path / "training-run"
    training_run.mkdir()
    (training_run / "group-head.npz").write_bytes(b"checkpoint")
    (training_run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": training_run.name,
                "candidate_artifact_id": "wrong-candidate",
                "verdict_artifact_id": verdict.name,
            }
        ),
        encoding="utf-8",
    )

    try:
        build_identity_group_error_review(
            candidate_artifact=candidates,
            verdict_artifact=verdict,
            training_run=training_run,
            state_path=tmp_path / "review-state.json",
            predictor=FakePredictor(),
        )
    except ValueError as error:
        assert "training run identity" in str(error)
    else:
        raise AssertionError("mismatched training run was accepted")


def test_build_error_review_uses_correction_verdict_as_frozen_truth(
    tmp_path: Path,
) -> None:
    candidates, verdict = _reviewed_artifacts(tmp_path)
    valid = load_reviewed_identity_dataset(
        candidate_artifact=candidates,
        verdict_artifact=verdict,
        split="valid",
    )
    target_id = valid.records[0].crop_id
    correction_session = IdentityReviewSession.open(
        candidate_artifact=candidates,
        state_path=tmp_path / "correction-state.json",
        crop_ids=(target_id,),
    )
    correction_session.record(
        crop_id=target_id,
        group="stripe",
        exact_identity=None,
        exact_evidence=None,
        visibility="visible",
        orientation="number_hidden",
    )
    correction = correction_session.export_verdict(tmp_path / "corrections")
    training_run = tmp_path / "training-run"
    training_run.mkdir()
    (training_run / "group-head.npz").write_bytes(b"checkpoint")
    (training_run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": training_run.name,
                "candidate_artifact_id": candidates.name,
                "verdict_artifact_id": verdict.name,
                "correction_verdict_artifact_id": correction.name,
            }
        ),
        encoding="utf-8",
    )

    review = build_identity_group_error_review(
        candidate_artifact=candidates,
        verdict_artifact=verdict,
        correction_verdict_artifact=correction,
        training_run=training_run,
        state_path=tmp_path / "error-review-state.json",
        predictor=FakePredictor(),
    )

    assert review.crop_ids == (target_id,)
    assert review.suggestions[target_id].group == "stripe"
    assert review.contexts[target_id]["truth"] == "stripe"


def test_build_error_review_rejects_missing_training_correction_verdict(
    tmp_path: Path,
) -> None:
    candidates, verdict = _reviewed_artifacts(tmp_path)
    training_run = tmp_path / "training-run"
    training_run.mkdir()
    (training_run / "group-head.npz").write_bytes(b"checkpoint")
    (training_run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": training_run.name,
                "candidate_artifact_id": candidates.name,
                "verdict_artifact_id": verdict.name,
                "correction_verdict_artifact_id": "required-correction",
            }
        ),
        encoding="utf-8",
    )

    try:
        build_identity_group_error_review(
            candidate_artifact=candidates,
            verdict_artifact=verdict,
            training_run=training_run,
            state_path=tmp_path / "review-state.json",
            predictor=FakePredictor(),
        )
    except ValueError as error:
        assert "training run identity" in str(error)
    else:
        raise AssertionError("training correction verdict was silently omitted")
