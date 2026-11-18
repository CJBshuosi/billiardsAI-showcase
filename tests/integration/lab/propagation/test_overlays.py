from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.propagation.evaluation import evaluate_pilot_run
from billiards_ai.lab.propagation.overlays import (
    EvaluationOverlayError,
    render_evaluation_overlays,
)
from tests.integration.lab.propagation.test_pilot_evaluation import _published_partial_run


def _evaluated_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    original = Path.write_bytes
    png = cv2.imencode(".png", np.full((48, 64, 3), 35, dtype=np.uint8))[1].tobytes()

    def write_synthetic_images(path: Path, data: bytes) -> int:
        if "overlays" in path.parts:
            data = png
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", write_synthetic_images)
    raw_run, evaluation_manifest = _published_partial_run(tmp_path)
    result = evaluate_pilot_run(
        raw_run=raw_run,
        evaluation_manifest_path=evaluation_manifest,
        output_root=tmp_path / "evaluation",
    )
    return raw_run, result


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _publish_json(root: Path, filename: str, payload: dict) -> Path:
    body = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    target = root / hashlib.sha256(body).hexdigest()
    target.mkdir(parents=True)
    (target / filename).write_bytes(body)
    (target / filename).chmod(0o444)
    target.chmod(0o555)
    return target


def test_evaluation_overlays_freeze_separate_bound_mp4_and_review_jpegs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_run, evaluation = _evaluated_run(tmp_path, monkeypatch)
    raw_before = _tree_snapshot(raw_run)

    rendered = render_evaluation_overlays(
        raw_run=raw_run,
        evaluation_artifact=evaluation.evaluation_path,
        review_sample=evaluation.review_sample_path,
        output_root=tmp_path / "renders",
        fps=2.0,
    )

    manifest = json.loads((rendered.path / "render-manifest.json").read_text())
    assert rendered.path.name == rendered.content_hash
    assert manifest["raw_run_hash"] == raw_run.name
    assert manifest["evaluation_artifact_hash"] == evaluation.evaluation_path.name
    assert manifest["review_sample_hash"] == evaluation.review_sample_path.name
    assert manifest["qa_identity"] == evaluation.report.qa_identity.model_dump(mode="json")
    assert {item["kind"] for item in manifest["unit_evidence"]} == {
        "match",
        "miss",
    }
    assert {item["stratum"] for item in manifest["unit_evidence"]} >= {
        "clean_match",
        "miss",
    }
    inventory_paths = {item["path"] for item in manifest["inventory"]}
    assert "review.mp4" in inventory_paths
    assert len([path for path in inventory_paths if path.endswith(".jpg")]) == len(
        manifest["unit_evidence"]
    )
    assert all((rendered.path / path).is_file() for path in inventory_paths)
    assert not rendered.path.stat().st_mode & 0o222
    assert _tree_snapshot(raw_run) == raw_before
    assert not any(path.startswith("evaluation") for path in raw_before)


def test_evaluation_overlays_reject_rehashed_or_symlinked_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_run, evaluation = _evaluated_run(tmp_path, monkeypatch)
    evaluation.evaluation_path.chmod(0o755)
    report = evaluation.evaluation_path / "report.json"
    report.chmod(0o644)
    report.write_text(report.read_text() + " ", encoding="utf-8")

    with pytest.raises(EvaluationOverlayError, match="evaluation artifact"):
        render_evaluation_overlays(
            raw_run=raw_run,
            evaluation_artifact=evaluation.evaluation_path,
            review_sample=evaluation.review_sample_path,
            output_root=tmp_path / "renders",
        )


def test_evaluation_overlays_reject_a_symlinked_output_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_run, evaluation = _evaluated_run(tmp_path, monkeypatch)
    external = tmp_path / "external-output"
    external.mkdir()
    (tmp_path / "linked-output").symlink_to(external, target_is_directory=True)

    with pytest.raises(EvaluationOverlayError, match="output root.*unsafe"):
        render_evaluation_overlays(
            raw_run=raw_run,
            evaluation_artifact=evaluation.evaluation_path,
            review_sample=evaluation.review_sample_path,
            output_root=tmp_path / "linked-output" / "nested",
        )
    assert not list(external.iterdir())


def test_evaluation_overlays_require_exact_task7_population_and_render_all_unit_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_run, evaluation = _evaluated_run(tmp_path, monkeypatch)
    report = json.loads((evaluation.evaluation_path / "report.json").read_text())
    sample = json.loads((evaluation.review_sample_path / "review-sample.json").read_text())

    mismatched = json.loads(json.dumps(sample))
    mismatched["population_counts"]["clean_match"] += 1
    mismatched_sample = _publish_json(tmp_path / "mismatch", "review-sample.json", mismatched)
    with pytest.raises(EvaluationOverlayError, match="population"):
        render_evaluation_overlays(
            raw_run=raw_run,
            evaluation_artifact=evaluation.evaluation_path,
            review_sample=mismatched_sample,
            output_root=tmp_path / "rejected-renders",
        )

    missing_stratum_report = json.loads(json.dumps(report))
    missing_stratum_sample = json.loads(json.dumps(sample))
    for payload in (missing_stratum_report["review_population_counts"],):
        payload.pop("false_positive")
    for field in (
        "population_counts",
        "primary_quota_sample_counts",
        "sample_counts",
    ):
        missing_stratum_sample[field].pop("false_positive")
    shortened_evaluation = _publish_json(
        tmp_path / "short-evaluation", "report.json", missing_stratum_report
    )
    shortened_sample = _publish_json(
        tmp_path / "short-sample", "review-sample.json", missing_stratum_sample
    )
    with pytest.raises(EvaluationOverlayError, match="strata"):
        render_evaluation_overlays(
            raw_run=raw_run,
            evaluation_artifact=shortened_evaluation,
            review_sample=shortened_sample,
            output_root=tmp_path / "short-renders",
        )

    clean = next(item for item in sample["units"] if item["stratum"] == "clean_match")
    miss = next(item for item in sample["units"] if item["stratum"] == "miss")
    flagged = json.loads(json.dumps(clean))
    flagged.update(
        unit_id="b" * 64,
        stratum="flagged_or_ambiguous_match",
        qa_flags=["center_jump"],
    )
    flagged["prediction"]["qa_flags"] = ["center_jump"]
    false_positive = json.loads(json.dumps(clean))
    false_positive.update(
        unit_id="c" * 64,
        unit_type="unmatched_prediction",
        stratum="false_positive",
        ground_truth_id=None,
        ground_truth=None,
        normalized_center_error=None,
    )
    custom_counts = {
        "clean_match": 1,
        "flagged_or_ambiguous_match": 1,
        "miss": 1,
        "false_positive": 1,
    }
    sample.update(
        population_counts=custom_counts,
        primary_quota_sample_counts=custom_counts,
        sample_counts=custom_counts,
        units=[clean, flagged, miss, false_positive],
    )
    report["review_population_counts"] = custom_counts
    custom_evaluation = _publish_json(tmp_path / "custom-evaluation", "report.json", report)
    custom_sample = _publish_json(tmp_path / "custom-sample", "review-sample.json", sample)

    rendered = render_evaluation_overlays(
        raw_run=raw_run,
        evaluation_artifact=custom_evaluation,
        review_sample=custom_sample,
        output_root=tmp_path / "all-types-renders",
    )

    evidence = rendered.manifest.unit_evidence
    assert {item.kind for item in evidence} == {"match", "miss", "false_positive"}
    assert {item.stratum for item in evidence} == set(custom_counts)
    assert all(
        {"gate:acceptable_without_redraw", "gate:severe_drift_or_wrong_target"}
        <= set(item.annotations)
        for item in evidence
    )
    assert "qa:center_jump" in next(
        item.annotations for item in evidence if item.stratum == "flagged_or_ambiguous_match"
    )


@pytest.mark.parametrize(
    ("field", "foreign_value"),
    [
        ("dataset_id", "d" * 64),
        ("prompt_manifest_id", "e" * 64),
        ("input_bundle_id", "f" * 64),
        ("qa_identity", None),
    ],
)
def test_evaluation_overlays_reject_rehashed_foreign_provenance_even_with_correct_raw_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    foreign_value: str | None,
) -> None:
    raw_run, evaluation = _evaluated_run(tmp_path, monkeypatch)
    report = json.loads((evaluation.evaluation_path / "report.json").read_text())
    sample = json.loads((evaluation.review_sample_path / "review-sample.json").read_text())
    if field == "qa_identity":
        sample[field] = {
            "artifact_kind": "raw_run",
            "pilot_qa_config_hash": report[field]["pilot_qa_config_hash"],
        }
    else:
        report[field] = foreign_value
        sample[field] = foreign_value
    foreign_evaluation = _publish_json(
        tmp_path / f"foreign-evaluation-{field}", "report.json", report
    )
    foreign_sample = _publish_json(
        tmp_path / f"foreign-sample-{field}", "review-sample.json", sample
    )

    with pytest.raises(EvaluationOverlayError, match="provenance|QA identity"):
        render_evaluation_overlays(
            raw_run=raw_run,
            evaluation_artifact=foreign_evaluation,
            review_sample=foreign_sample,
            output_root=tmp_path / f"foreign-render-{field}",
        )


def test_evaluation_overlay_publication_rejects_a_review_sample_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_run, evaluation = _evaluated_run(tmp_path, monkeypatch)
    sample_file = evaluation.review_sample_path / "review-sample.json"
    original_replace = os.replace

    def mutate_source_during_publish(source: Path, destination: Path) -> None:
        if Path(destination).parent.name == "evaluation-renders":
            evaluation.review_sample_path.chmod(0o755)
            sample_file.chmod(0o644)
            sample_file.write_text(sample_file.read_text() + " ", encoding="utf-8")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", mutate_source_during_publish)

    with pytest.raises(EvaluationOverlayError, match="review sample|rendering"):
        render_evaluation_overlays(
            raw_run=raw_run,
            evaluation_artifact=evaluation.evaluation_path,
            review_sample=evaluation.review_sample_path,
            output_root=tmp_path / "raced-renders",
        )
    assert not [path for path in (tmp_path / "raced-renders" / "evaluation-renders").glob("*")]
