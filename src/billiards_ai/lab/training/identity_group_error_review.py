"""Build a focused human review session from frozen validation mistakes."""

from __future__ import annotations

import json
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from billiards_ai.lab.training.identity_review import (
    IdentityReviewDecision,
    IdentityReviewSession,
    _candidate_records,
    _verify_verdict,
    load_reviewed_identity_dataset,
)
from billiards_ai.lab.training.identity_review_server import make_identity_review_server


class IdentityGroupPredictor(Protocol):
    def predict_records(
        self, *, checkpoint_path: Path, records: tuple[object, ...]
    ) -> tuple[dict[str, object], ...]: ...


def _load_training_manifest(training_run: Path) -> dict[str, object]:
    try:
        manifest = json.loads((training_run / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("training run identity is invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("training run identity is invalid")
    return manifest


def build_identity_group_error_review(
    *,
    candidate_artifact: Path,
    verdict_artifact: Path,
    training_run: Path,
    state_path: Path,
    predictor: IdentityGroupPredictor,
    correction_verdict_artifact: Path | None = None,
) -> IdentityReviewSession:
    """Open an explicit review containing only frozen-validation mistakes."""

    candidate_artifact = candidate_artifact.resolve()
    verdict_artifact = verdict_artifact.resolve()
    correction_verdict_artifact = (
        correction_verdict_artifact.resolve() if correction_verdict_artifact is not None else None
    )
    training_run = training_run.resolve()
    manifest = _load_training_manifest(training_run)
    if (
        not training_run.is_dir()
        or manifest.get("run_id") != training_run.name
        or manifest.get("candidate_artifact_id") != candidate_artifact.name
        or manifest.get("verdict_artifact_id") != verdict_artifact.name
        or manifest.get("correction_verdict_artifact_id")
        != (correction_verdict_artifact.name if correction_verdict_artifact is not None else None)
    ):
        raise ValueError("training run identity does not match reviewed labels")
    checkpoint = training_run / "group-head.npz"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("training run checkpoint is missing or unsafe")

    valid = load_reviewed_identity_dataset(
        candidate_artifact=candidate_artifact,
        verdict_artifact=verdict_artifact,
        correction_verdict_artifact=correction_verdict_artifact,
        split="valid",
    )
    predictions = predictor.predict_records(
        checkpoint_path=checkpoint,
        records=valid.records,
    )
    expected_ids = {record.crop_id for record in valid.records}
    if len(predictions) != len(valid.records):
        raise ValueError("validation predictions are incomplete")
    prediction_by_id: dict[str, dict[str, object]] = {}
    allowed = {"cue", "solid", "eight", "stripe", "unknown"}
    for prediction in predictions:
        crop_id = prediction.get("crop_id")
        truth = prediction.get("truth")
        predicted = prediction.get("predicted")
        probabilities = prediction.get("probabilities")
        if (
            not isinstance(crop_id, str)
            or crop_id in prediction_by_id
            or crop_id not in expected_ids
            or truth not in allowed
            or predicted not in allowed
            or not isinstance(probabilities, (list, tuple))
            or len(probabilities) != 4
        ):
            raise ValueError("validation predictions are invalid")
        prediction_by_id[crop_id] = prediction
    if set(prediction_by_id) != expected_ids:
        raise ValueError("validation predictions are incomplete")

    candidate_id, _ = _candidate_records(candidate_artifact)
    original = {
        decision.crop_id: decision
        for decision in _verify_verdict(verdict_artifact, candidate_id=candidate_id)
    }
    if correction_verdict_artifact is not None:
        original.update(
            {
                decision.crop_id: decision
                for decision in _verify_verdict(
                    correction_verdict_artifact,
                    candidate_id=candidate_id,
                )
            }
        )
    mistakes = tuple(
        record.crop_id
        for record in valid.records
        if prediction_by_id[record.crop_id]["truth"]
        != prediction_by_id[record.crop_id]["predicted"]
    )
    if not mistakes:
        raise ValueError("training run has no frozen-validation mistakes")
    suggestions: dict[str, IdentityReviewDecision] = {
        crop_id: original[crop_id] for crop_id in mistakes
    }
    contexts = {
        crop_id: {
            "truth": prediction_by_id[crop_id]["truth"],
            "predicted": prediction_by_id[crop_id]["predicted"],
            "probabilities": list(prediction_by_id[crop_id]["probabilities"]),
            "confidence": prediction_by_id[crop_id].get("confidence"),
            "margin": prediction_by_id[crop_id].get("margin"),
        }
        for crop_id in mistakes
    }
    return IdentityReviewSession.open(
        candidate_artifact=candidate_artifact,
        state_path=state_path,
        crop_ids=mistakes,
        suggestions=suggestions,
        contexts=contexts,
    )


def run_identity_group_error_review_cli(
    *,
    candidate_artifact: Path,
    verdict_artifact: Path,
    training_run: Path,
    asset_root: Path,
    output_root: Path,
    port: int,
    open_browser: bool,
    device: str = "cuda",
    batch_size: int = 8,
    correction_verdict_artifact: Path | None = None,
    feature_factory: Callable[..., Any] | None = None,
    backend_factory: Callable[..., Any] | None = None,
    server_factory: Callable[..., Any] = make_identity_review_server,
) -> int:
    """Launch the loopback UI for only the frozen-validation mistakes."""

    if feature_factory is None or backend_factory is None:
        from billiards_ai.lab.training.identity_dinov2 import (
            Dinov2IdentityFeatureBackend,
        )
        from billiards_ai.lab.training.identity_hierarchical_backend import (
            Dinov2HierarchicalGroupTrainingBackend,
        )

        feature_factory = feature_factory or Dinov2IdentityFeatureBackend
        backend_factory = backend_factory or Dinov2HierarchicalGroupTrainingBackend
    features = feature_factory(
        asset_root=asset_root,
        device=device,
        batch_size=batch_size,
    )
    predictor = backend_factory(
        candidate_root=candidate_artifact,
        feature_backend=features,
        augmentation_count=0,
    )
    output_root = output_root.resolve()
    session = build_identity_group_error_review(
        candidate_artifact=candidate_artifact,
        verdict_artifact=verdict_artifact,
        correction_verdict_artifact=correction_verdict_artifact,
        training_run=training_run,
        state_path=output_root / "states" / f"{training_run.resolve().name}.json",
        predictor=predictor,
    )
    server = server_factory(
        session=session,
        export_root=output_root / "verdicts",
        port=port,
    )
    url = f"http://127.0.0.1:{server.server_port}/"
    print(
        json.dumps(
            {
                "status": "READY",
                "url": url,
                "error_count": len(session.crop_ids),
                "reviewed_count": session.reviewed_count,
                "state_path": str(session.state_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
