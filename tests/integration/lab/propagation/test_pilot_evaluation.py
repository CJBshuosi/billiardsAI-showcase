from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from billiards_ai.lab.propagation.contracts import (
    InstancePrompt,
    MaskObservation,
    PropagationResult,
    PropagationWindow,
)
from billiards_ai.lab.propagation.evaluation import PilotEvaluationError, evaluate_pilot_run
from billiards_ai.lab.propagation.pilot_artifacts import (
    ExecutedGoldPrompt,
    PilotAttemptFailure,
    PilotRunManifest,
    PilotShotFailure,
    publish_pilot_run,
)
from billiards_ai.lab.propagation.pilot_data import (
    FROZEN_VIDEO_IDS,
    EvaluationCircle,
    EvaluationManifest,
    PromptCircle,
    PromptManifest,
    ShotDefinition,
)
from billiards_ai.lab.propagation.pilot_planning import WindowAttempt, WindowPlan


def _published_partial_run(tmp_path: Path) -> tuple[Path, Path]:
    staging = tmp_path / "staging"
    plans: list[WindowPlan] = []
    results: list[PropagationResult] = []
    circles: list[EvaluationCircle] = []
    prompt_circles: list[PromptCircle] = []
    executed_gold_prompts: list[ExecutedGoldPrompt] = []
    video_ids = (*FROZEN_VIDEO_IDS, *FROZEN_VIDEO_IDS, *FROZEN_VIDEO_IDS[:5])
    for index in range(25):
        asset = video_ids[index]
        start_frame = index * 3
        end_frame = start_frame + 3
        window = PropagationWindow(
            asset_id=asset, ordinal=0, start_frame=start_frame, end_frame=end_frame
        )
        shot_gold_prompts: list[InstancePrompt] = []
        for circle_index in range(29 if index < 17 else 28):
            annotation_id = f"seed-{index}" if circle_index == 0 else f"seed-{index}-{circle_index}"
            instance_id = f"ball-{index}" if circle_index == 0 else f"ball-{index}-{circle_index}"
            prompt_circles.append(
                PromptCircle(
                    annotation_id=annotation_id,
                    video_id=asset,
                    shot_index=index,
                    source_frame_index=start_frame,
                    source_frame_sha256=f"{index + 100:064x}",
                    center_x_px=10 + circle_index,
                    center_y_px=10,
                    radius_px=5,
                )
            )
            shot_gold_prompts.append(
                InstancePrompt(
                    asset_id=asset,
                    frame_index=start_frame,
                    instance_id=instance_id,
                    class_name="ball",
                    source_kind="gold",
                    source_id=annotation_id,
                    box_xyxy_px=(5 + circle_index, 5, 15 + circle_index, 15),
                )
            )
        executed_gold_prompts.extend(
            ExecutedGoldPrompt(
                asset_id=asset,
                shot_index=index,
                shot_start=start_frame,
                shot_end=end_frame,
                prompt=prompt,
            )
            for prompt in shot_gold_prompts
        )
        for role, frame_index in (("middle", start_frame + 1), ("end", start_frame + 2)):
            circles.append(
                EvaluationCircle(
                    annotation_id=f"ground-{index}-{role}",
                    video_id=asset,
                    shot_index=index,
                    source_frame_index=frame_index,
                    source_frame_sha256=f"{index * 2 + frame_index - 1:064x}",
                    center_x_px=10,
                    center_y_px=10,
                    radius_px=5,
                    role=role,
                )
            )
        if index == 24:
            plans.append(
                WindowPlan(
                    asset_id=asset,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    attempts=(
                        WindowAttempt(
                            window=window,
                            status="failed",
                            failure_code="synthetic_failure",
                        ),
                    ),
                )
            )
            continue
        plans.append(
            WindowPlan(
                asset_id=asset,
                start_frame=start_frame,
                end_frame=end_frame,
                attempts=(WindowAttempt(window=window, status="committed"),),
            )
        )
        observations: list[MaskObservation] = []
        for frame_index in (start_frame + 1, start_frame + 2):
            mask_ref = f"masks/{asset}/{frame_index:08d}.png"
            mask = staging / mask_ref
            overlay = staging / "overlays" / asset / f"{frame_index:08d}.png"
            mask.parent.mkdir(parents=True, exist_ok=True)
            overlay.parent.mkdir(parents=True, exist_ok=True)
            mask.write_bytes(f"mask-{index}-{frame_index}".encode())
            overlay.write_bytes(f"overlay-{index}-{frame_index}".encode())
            observations.append(
                MaskObservation(
                    asset_id=asset,
                    frame_index=frame_index,
                    instance_id=f"ball-{index}",
                    class_name="ball",
                    prompt_frame_index=start_frame,
                    center_px=(10, 10),
                    box_xyxy_px=(5, 5, 15, 15),
                    area_px=78,
                    mask_ref=mask_ref,
                )
            )
        results.append(
            PropagationResult(
                backend_id="synthetic",
                backend_version="1.0",
                window=window,
                prompts=tuple(shot_gold_prompts),
                observations=tuple(observations),
            )
        )
    failed_window = plans[-1].attempts[0].window
    circles.append(
        circles[0].model_copy(
            update={
                "annotation_id": "ground-0-middle-extra",
                "center_x_px": 30,
                "center_y_px": 30,
            }
        )
    )
    shot_definitions = tuple(
        ShotDefinition(
            video_id=video_ids[index],
            shot_index=index,
            start_frame=index * 3,
            end_frame=index * 3 + 3,
        )
        for index in range(25)
    )
    prompt_body = {
        "schema_version": "2.0",
        "dataset_id": "c" * 64,
        "shots": [item.model_dump(mode="json") for item in shot_definitions],
        "circles": [item.model_dump(mode="json") for item in prompt_circles],
    }
    prompt_manifest = PromptManifest(
        manifest_id=hashlib.sha256(
            json.dumps(prompt_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **prompt_body,
    )
    manifest = PilotRunManifest(
        input_bundle_id="a" * 64,
        prompt_manifest=prompt_manifest,
        executed_gold_prompts=tuple(executed_gold_prompts),
        outcome="partial_success",
        plans=tuple(plans),
        attempt_failures=(
            PilotAttemptFailure(
                window=failed_window,
                code="synthetic_failure",
                message="synthetic explicit failure",
            ),
        ),
        failed_shots=(
            PilotShotFailure(
                asset_id=failed_window.asset_id,
                start_frame=failed_window.start_frame,
                end_frame=failed_window.end_frame,
                code="synthetic_failure",
                message="synthetic explicit failure",
            ),
        ),
    )
    run = publish_pilot_run(
        staging_root=staging,
        manifest=manifest,
        results=tuple(results),
        output_root=tmp_path / "raw-runs",
    )
    body = {
        "schema_version": "2.0",
        "dataset_id": "c" * 64,
        "prompt_manifest_id": prompt_manifest.manifest_id,
        "shots": [item.model_dump(mode="json") for item in shot_definitions],
        "circles": [circle.model_dump(mode="json") for circle in circles],
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    evaluation = EvaluationManifest(manifest_id=hashlib.sha256(encoded).hexdigest(), **body)
    evaluation_path = tmp_path / "evaluation-manifest.json"
    evaluation_path.write_text(evaluation.model_dump_json(), encoding="utf-8")
    return run, evaluation_path


def test_evaluate_accepts_structurally_complete_partial_success_and_freezes_separate_artifacts(
    tmp_path: Path,
) -> None:
    run, evaluation_manifest = _published_partial_run(tmp_path)

    result = evaluate_pilot_run(
        raw_run=run,
        evaluation_manifest_path=evaluation_manifest,
        output_root=tmp_path / "evaluation-output",
    )

    assert result.report.coverage.completed_shots == 24
    assert result.report.coverage.total_shots == 25
    assert result.report.set_metrics.ground_truth_count == 51
    assert result.report.set_metrics.true_positive_support == 48
    assert result.report.set_metrics.false_negative_count == 3
    assert result.report.set_metrics.recall == 48 / 51
    assert result.report.frozen_gates.automated_gates_passed is True
    assert result.report.exploratory_track_continuity.status == "not_evaluated"
    assert result.evaluation_path.parent.name == "evaluations"
    assert result.review_sample_path.parent.name == "review-samples"
    assert result.evaluation_path != result.review_sample_path
    assert not result.evaluation_path.stat().st_mode & 0o222
    assert not result.review_sample_path.stat().st_mode & 0o222
    review = json.loads((result.review_sample_path / "review-sample.json").read_text())
    assert review["raw_run_hash"] == run.name
    assert review["evaluation_manifest_hash"] == result.report.evaluation_manifest_hash
    assert review["population_counts"]["miss"] == 3
    miss_ids = {
        item["ground_truth"]["annotation_id"]
        for item in review["units"]
        if item["unit_type"] == "unmatched_ground_truth"
    }
    assert {"ground-0-middle-extra", "ground-24-middle", "ground-24-end"} <= miss_ids
    clean = next(item for item in review["units"] if item["unit_type"] == "matched_pair")
    assert clean["ground_truth"]["radius_px"] == 5
    assert (run / clean["prediction"]["mask_ref"]).is_file()


def test_evaluate_recall_collapses_when_ground_truth_is_in_a_different_pixel_space(
    tmp_path: Path,
) -> None:
    """Bug 6 signature: GT and predictions must share a pixel space.

    Predictions are SAM2 mask centroids in source pixels; if the evaluation
    circles are in a different space (the pre-fix bug left them in review
    pixels), every pair exceeds one radius and recall collapses to 0 with the
    prediction/GT counts still ~equal - exactly what the live run reported.
    build_pilot_manifests now maps evaluation circles into source space so this
    cannot happen; this test pins the evaluator's dependence on aligned spaces.
    """

    run, evaluation_manifest = _published_partial_run(tmp_path)

    aligned = evaluate_pilot_run(
        raw_run=run,
        evaluation_manifest_path=evaluation_manifest,
        output_root=tmp_path / "aligned-output",
    )
    assert aligned.report.set_metrics.recall == 48 / 51
    assert aligned.report.set_metrics.true_positive_support == 48

    # Shift every ground-truth circle far from the predictions (a wrong pixel
    # space) without touching the raw run's source-pixel predictions.
    payload = json.loads(evaluation_manifest.read_text())
    for circle in payload["circles"]:
        circle["center_x_px"] = circle["center_x_px"] + 500.0
        circle["center_y_px"] = circle["center_y_px"] + 500.0
    misaligned_manifest = _write_rehashed_evaluation(
        tmp_path, payload, name="evaluation-misaligned-space.json"
    )

    misaligned = evaluate_pilot_run(
        raw_run=run,
        evaluation_manifest_path=misaligned_manifest,
        output_root=tmp_path / "misaligned-output",
    )

    metrics = misaligned.report.set_metrics
    assert metrics.recall == 0.0
    assert metrics.true_positive_support == 0
    # Counts are preserved even though nothing matches - the live-run symptom.
    assert metrics.ground_truth_count == 51
    assert metrics.prediction_count == aligned.report.set_metrics.prediction_count
    assert metrics.median_normalized_center_error is None
    assert misaligned.report.frozen_gates.pilot_status == "automated_gate_failed"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _make_writable(root: Path) -> None:
    root.chmod(0o755)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _forge_raw_run(
    tmp_path: Path,
    run: Path,
    *,
    mutate_manifest=lambda value: value,
    mutate_results=lambda rows: rows,
    mutate_inventory=lambda rows: rows,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    forged = tmp_path / "forged"
    shutil.copytree(run, forged)
    _make_writable(forged)
    manifest = mutate_manifest(json.loads((forged / "manifest.json").read_text()))
    results = [json.loads(line) for line in (forged / "results.jsonl").read_text().splitlines()]
    inventory_payload = json.loads((forged / "inventory.json").read_text())
    results = mutate_results(results)
    inventory_payload["files"] = mutate_inventory(inventory_payload["files"])
    (forged / "results.jsonl").write_bytes(
        b"".join(_canonical_bytes(row) + b"\n" for row in results)
    )
    (forged / "inventory.json").write_bytes(_canonical_bytes(inventory_payload) + b"\n")
    identity = {
        "schema_version": "1.0",
        "manifest": {key: value for key, value in manifest.items() if key != "content_hash"},
        "results": results,
        "inventory": inventory_payload["files"],
    }
    content_hash = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    manifest["content_hash"] = content_hash
    (forged / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
    target = tmp_path / content_hash
    forged.rename(target)
    _make_immutable(target)
    return target


def _write_rehashed_evaluation(tmp_path: Path, payload: dict, *, name: str) -> Path:
    body = {key: value for key, value in payload.items() if key != "manifest_id"}
    payload["manifest_id"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    changed = tmp_path / name
    changed.write_bytes(_canonical_bytes(payload) + b"\n")
    return changed


def test_evaluate_rejects_joint_rehash_that_shifts_a_ground_truth_circle_to_raw_summary(
    tmp_path: Path,
) -> None:
    run, evaluation_manifest = _published_partial_run(tmp_path)
    attacked_payload = json.loads(evaluation_manifest.read_text())
    removable = next(
        circle
        for circle in attacked_payload["circles"]
        if circle["annotation_id"] == "ground-0-middle-extra"
    )
    attacked_payload["circles"] = [
        circle
        for circle in attacked_payload["circles"]
        if circle["annotation_id"] != removable["annotation_id"]
    ]
    assert any(
        circle["source_frame_index"] == removable["source_frame_index"]
        and circle["video_id"] == removable["video_id"]
        for circle in attacked_payload["circles"]
    )
    attacked_evaluation = _write_rehashed_evaluation(
        tmp_path, attacked_payload, name="evaluation-after-joint-forgery.json"
    )
    attacked_run = _forge_raw_run(
        tmp_path / "attacked-raw",
        run,
        mutate_manifest=lambda manifest: {**manifest, "prompt_circle_count": 718},
    )

    with pytest.raises(PilotEvaluationError, match="prompt|snapshot|circle|manifest"):
        evaluate_pilot_run(
            raw_run=attacked_run,
            evaluation_manifest_path=attacked_evaluation,
            output_root=tmp_path / "attacked-output",
        )


def test_evaluate_rejects_joint_rehash_with_unused_valid_prompt_circle(
    tmp_path: Path,
) -> None:
    run, evaluation_manifest = _published_partial_run(tmp_path)
    raw_payload = json.loads((run / "manifest.json").read_text())
    prompt = raw_payload["prompt_manifest"]
    unused = dict(prompt["circles"][0])
    unused["annotation_id"] = "unused-valid-start-circle"
    unused["center_x_px"] = 123.0
    prompt["circles"].append(unused)
    prompt_body = {key: value for key, value in prompt.items() if key != "manifest_id"}
    prompt["manifest_id"] = hashlib.sha256(_canonical_bytes(prompt_body)).hexdigest()

    evaluation_payload = json.loads(evaluation_manifest.read_text())
    evaluation_payload["circles"] = [
        circle
        for circle in evaluation_payload["circles"]
        if circle["annotation_id"] != "ground-0-middle-extra"
    ]
    evaluation_payload["prompt_manifest_id"] = prompt["manifest_id"]
    attacked_evaluation = _write_rehashed_evaluation(
        tmp_path,
        evaluation_payload,
        name="evaluation-with-shifted-unused-prompt-circle.json",
    )
    attacked_run = _forge_raw_run(
        tmp_path / "unused-valid-prompt-circle",
        run,
        mutate_manifest=lambda manifest: {**manifest, "prompt_manifest": prompt},
    )

    with pytest.raises(PilotEvaluationError, match="gold|prompt|circle|exact|result|active plan"):
        evaluate_pilot_run(
            raw_run=attacked_run,
            evaluation_manifest_path=attacked_evaluation,
            output_root=tmp_path / "rejected-output",
        )


@pytest.mark.parametrize("mutation", ["declared_id", "start_frame"])
def test_evaluate_rejects_rehashed_forged_embedded_prompt_snapshot(
    tmp_path: Path, mutation: str
) -> None:
    run, evaluation_manifest = _published_partial_run(tmp_path)
    raw_payload = json.loads((run / "manifest.json").read_text())
    prompt = raw_payload["prompt_manifest"]
    if mutation == "declared_id":
        prompt["manifest_id"] = "0" * 64
    else:
        prompt["circles"][0]["source_frame_index"] += 1
        prompt_body = {key: value for key, value in prompt.items() if key != "manifest_id"}
        prompt["manifest_id"] = hashlib.sha256(_canonical_bytes(prompt_body)).hexdigest()
        evaluation_payload = json.loads(evaluation_manifest.read_text())
        evaluation_payload["prompt_manifest_id"] = prompt["manifest_id"]
        evaluation_manifest = _write_rehashed_evaluation(
            tmp_path,
            evaluation_payload,
            name="evaluation-for-forged-prompt-snapshot.json",
        )
    forged = _forge_raw_run(
        tmp_path / f"forged-prompt-{mutation}",
        run,
        mutate_manifest=lambda manifest: {
            **manifest,
            "prompt_manifest": prompt,
        },
    )

    with pytest.raises(PilotEvaluationError, match="raw-run|prompt|snapshot|manifest"):
        evaluate_pilot_run(
            raw_run=forged,
            evaluation_manifest_path=evaluation_manifest,
            output_root=tmp_path / "rejected-output",
        )


@pytest.mark.parametrize("mutation", ["undeclared", "missing"])
def test_evaluate_rejects_rehashed_results_that_do_not_exactly_match_active_plan(
    tmp_path: Path, mutation: str
) -> None:
    run, evaluation_manifest = _published_partial_run(tmp_path)

    def mutate(rows: list[dict]) -> list[dict]:
        if mutation == "undeclared":
            rows[0]["window"]["ordinal"] = 99
            return rows
        return rows[1:]

    forged = _forge_raw_run(tmp_path / mutation, run, mutate_results=mutate)

    with pytest.raises(PilotEvaluationError, match="active|plan|result"):
        evaluate_pilot_run(
            raw_run=forged,
            evaluation_manifest_path=evaluation_manifest,
            output_root=tmp_path / "rejected-output",
        )


def test_evaluate_rejects_rehashed_inventory_with_a_still_referenced_mask_removed(
    tmp_path: Path,
) -> None:
    run, evaluation_manifest = _published_partial_run(tmp_path)
    removed = "masks/video1/00000001.png"

    def mutate_inventory(rows: list[dict]) -> list[dict]:
        return [row for row in rows if row["path"] != removed]

    forged_root = tmp_path / "dangling"
    forged_root.mkdir()
    forged = _forge_raw_run(forged_root, run, mutate_inventory=mutate_inventory)
    _make_writable(forged)
    (forged / removed).unlink()
    _make_immutable(forged)

    with pytest.raises(PilotEvaluationError, match="reference|inventory|payload"):
        evaluate_pilot_run(
            raw_run=forged,
            evaluation_manifest_path=evaluation_manifest,
            output_root=tmp_path / "rejected-output",
        )


@pytest.mark.parametrize(
    "mutation", ["removed_failed_gt", "role_substitution", "shot_substitution"]
)
def test_evaluate_rejects_rehashed_incomplete_or_substituted_frozen_evaluation_population(
    tmp_path: Path, mutation: str
) -> None:
    run, evaluation_manifest = _published_partial_run(tmp_path)
    payload = json.loads(evaluation_manifest.read_text())
    if mutation == "removed_failed_gt":
        payload["circles"] = [
            circle for circle in payload["circles"] if circle["annotation_id"] != "ground-24-middle"
        ]
    elif mutation == "role_substitution":
        payload["circles"][0]["role"] = "end"
    else:
        payload["shots"][0]["video_id"] = "video999"
    body = {key: value for key, value in payload.items() if key != "manifest_id"}
    payload["manifest_id"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    changed = tmp_path / f"evaluation-{mutation}.json"
    changed.write_bytes(_canonical_bytes(payload) + b"\n")

    with pytest.raises(PilotEvaluationError, match="frozen|evaluation|shot|role|video|circle"):
        evaluate_pilot_run(
            raw_run=run,
            evaluation_manifest_path=changed,
            output_root=tmp_path / "rejected-output",
        )


def test_evaluate_rejects_relocked_structured_evidence_that_no_longer_matches_raw_identity(
    tmp_path: Path,
) -> None:
    run, evaluation_manifest = _published_partial_run(tmp_path)
    observations = run / "observations.jsonl"
    original = observations.read_text(encoding="utf-8")
    observations.chmod(0o644)
    observations.write_text(original.replace('"center_px":[10.0,10.0]', '"center_px":[11.0,10.0]'))
    observations.chmod(0o444)

    with pytest.raises(PilotEvaluationError, match="identity|structured"):
        evaluate_pilot_run(
            raw_run=run,
            evaluation_manifest_path=evaluation_manifest,
            output_root=tmp_path / "rejected-output",
        )
