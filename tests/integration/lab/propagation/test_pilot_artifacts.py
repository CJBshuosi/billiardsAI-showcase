from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from billiards_ai.lab.propagation import (
    InstanceAbsence,
    InstancePrompt,
    MaskObservation,
    PropagationFailure,
    PropagationResult,
    PropagationWindow,
    WindowAttempt,
    WindowPlan,
)
from billiards_ai.lab.propagation.pilot_artifacts import (
    ExecutedGoldPrompt,
    PilotArtifactError,
    PilotAttemptFailure,
    PilotRunManifest,
    PilotShotFailure,
    publish_pilot_run,
    verify_pilot_run_evidence,
)
from billiards_ai.lab.propagation.pilot_data import (
    PromptCircle,
    PromptManifest,
    ShotDefinition,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(mask_ref: str = "masks/ball-a/00000004.png") -> PropagationResult:
    window = PropagationWindow(asset_id="synthetic-video", ordinal=0, start_frame=4, end_frame=6)
    prompt = InstancePrompt(
        asset_id="synthetic-video",
        frame_index=4,
        instance_id="ball-a",
        class_name="ball",
        source_kind="gold",
        source_id="seed-a",
        box_xyxy_px=(1.0, 1.0, 5.0, 5.0),
    )
    observation = MaskObservation(
        asset_id="synthetic-video",
        frame_index=4,
        instance_id="ball-a",
        class_name="ball",
        prompt_frame_index=4,
        center_px=(3.0, 3.0),
        box_xyxy_px=(1.0, 1.0, 5.0, 5.0),
        area_px=16,
        mask_ref=mask_ref,
        qa_flags=("center_jump",),
    )
    absence = InstanceAbsence(
        asset_id="synthetic-video",
        frame_index=5,
        instance_id="ball-a",
        class_name="ball",
        prompt_frame_index=4,
        reason="empty_mask",
        qa_flags=("empty_mask",),
    )
    failure = PropagationFailure(
        asset_id="synthetic-video",
        frame_index=5,
        code="synthetic_warning",
        message="synthetic only",
    )
    return PropagationResult(
        backend_id="synthetic",
        backend_version="1.0",
        window=window,
        prompts=(prompt,),
        observations=(observation,),
        absences=(absence,),
        failures=(failure,),
    )


def _manifest() -> PilotRunManifest:
    window = PropagationWindow(asset_id="synthetic-video", ordinal=0, start_frame=4, end_frame=6)
    return PilotRunManifest(
        input_bundle_id="a" * 64,
        prompt_manifest=_prompt_manifest(),
        executed_gold_prompts=(
            {
                "asset_id": "synthetic-video",
                "shot_index": 0,
                "shot_start": 4,
                "shot_end": 6,
                "prompt": _result().prompts[0],
            },
        ),
        plans=(
            WindowPlan(
                asset_id="synthetic-video",
                start_frame=4,
                end_frame=6,
                attempts=(WindowAttempt(window=window, status="committed"),),
            ),
        ),
        telemetry=({"device": "synthetic", "processed_frames": 2},),
    )


def _prompt_manifest() -> PromptManifest:
    body = {
        "schema_version": "2.0",
        "dataset_id": "d" * 64,
        "shots": [
            ShotDefinition(
                video_id="synthetic-video", shot_index=0, start_frame=4, end_frame=6
            ).model_dump(mode="json")
        ],
        "circles": [
            PromptCircle(
                annotation_id="seed-a",
                video_id="synthetic-video",
                shot_index=0,
                source_frame_index=4,
                source_frame_sha256="f" * 64,
                center_x_px=3,
                center_y_px=3,
                radius_px=2,
            ).model_dump(mode="json")
        ],
    }
    manifest_id = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PromptManifest(manifest_id=manifest_id, **body)


def _snapshot_manifest() -> PilotRunManifest:
    window = PropagationWindow(asset_id="synthetic-video", ordinal=0, start_frame=4, end_frame=6)
    return PilotRunManifest(
        input_bundle_id="a" * 64,
        prompt_manifest=_prompt_manifest(),
        executed_gold_prompts=(
            {
                "asset_id": "synthetic-video",
                "shot_index": 0,
                "shot_start": 4,
                "shot_end": 6,
                "prompt": _result().prompts[0],
            },
        ),
        plans=(
            WindowPlan(
                asset_id="synthetic-video",
                start_frame=4,
                end_frame=6,
                attempts=(WindowAttempt(window=window, status="committed"),),
            ),
        ),
        telemetry=({"device": "synthetic", "processed_frames": 2},),
    )


def test_manifest_rejects_legacy_bare_executed_gold_prompts() -> None:
    payload = _snapshot_manifest().model_dump(mode="json", exclude_computed_fields=True)
    payload["executed_gold_prompts"] = [_result().prompts[0].model_dump(mode="json")]

    with pytest.raises(ValueError, match="executed_gold_prompts|prompt|field|required"):
        PilotRunManifest.model_validate(payload)


def test_manifest_accepts_owner_qualified_executed_gold_prompt() -> None:
    payload = _snapshot_manifest().model_dump(mode="json", exclude_computed_fields=True)
    payload["executed_gold_prompts"] = [
        {
            "asset_id": "synthetic-video",
            "shot_index": 0,
            "shot_start": 4,
            "shot_end": 6,
            "prompt": _result().prompts[0].model_dump(mode="json"),
        }
    ]

    manifest = PilotRunManifest.model_validate(payload)

    assert manifest.executed_gold_prompts[0].shot_index == 0
    assert manifest.executed_gold_prompts[0].prompt == _result().prompts[0]


@pytest.mark.parametrize(
    "prompt_update",
    [
        {"asset_id": "wrong-video"},
        {"frame_index": 5},
        {"class_name": "cue"},
        {"source_kind": "pseudo"},
    ],
)
def test_executed_gold_prompt_rejects_owner_prompt_mismatch(
    prompt_update: dict[str, object],
) -> None:
    prompt = _result().prompts[0].model_copy(update=prompt_update)

    with pytest.raises(ValueError, match="owner|asset|start|gold|ball"):
        ExecutedGoldPrompt(
            asset_id="synthetic-video",
            shot_index=0,
            shot_start=4,
            shot_end=6,
            prompt=prompt,
        )


def test_executed_gold_prompt_rejects_extra_fields() -> None:
    with pytest.raises(ValueError, match="extra|unexpected"):
        ExecutedGoldPrompt(
            asset_id="synthetic-video",
            shot_index=0,
            shot_start=4,
            shot_end=6,
            prompt=_result().prompts[0],
            unexpected="legacy-owner-claim",
        )


def _overlapping_shot_evidence() -> tuple[
    PilotRunManifest, tuple[PropagationResult, PropagationResult]
]:
    shots = (
        ShotDefinition(video_id="overlap-video", shot_index=0, start_frame=4, end_frame=10),
        ShotDefinition(video_id="overlap-video", shot_index=1, start_frame=5, end_frame=11),
    )
    circles = tuple(
        PromptCircle(
            annotation_id=f"overlap-seed-{index}",
            video_id=shot.video_id,
            shot_index=shot.shot_index,
            source_frame_index=shot.start_frame,
            source_frame_sha256=f"{index + 20:064x}",
            center_x_px=3 + index,
            center_y_px=3,
            radius_px=2,
        )
        for index, shot in enumerate(shots)
    )
    prompt_body = {
        "schema_version": "2.0",
        "dataset_id": "d" * 64,
        "shots": [shot.model_dump(mode="json") for shot in shots],
        "circles": [circle.model_dump(mode="json") for circle in circles],
    }
    prompt_manifest = PromptManifest(
        manifest_id=hashlib.sha256(
            json.dumps(prompt_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **prompt_body,
    )
    prompts = tuple(
        InstancePrompt(
            asset_id=shot.video_id,
            frame_index=shot.start_frame,
            instance_id=f"overlap-ball-{index}",
            class_name="ball",
            source_kind="gold",
            source_id=circles[index].annotation_id,
            box_xyxy_px=(1 + index, 1, 5 + index, 5),
        )
        for index, shot in enumerate(shots)
    )
    windows = tuple(
        PropagationWindow(
            asset_id=shot.video_id,
            ordinal=0,
            start_frame=shot.start_frame,
            end_frame=shot.end_frame,
        )
        for shot in shots
    )
    plans = tuple(
        WindowPlan(
            asset_id=shot.video_id,
            start_frame=shot.start_frame,
            end_frame=shot.end_frame,
            attempts=(WindowAttempt(window=windows[index], status="committed"),),
        )
        for index, shot in enumerate(shots)
    )
    manifest = PilotRunManifest(
        input_bundle_id="a" * 64,
        prompt_manifest=prompt_manifest,
        executed_gold_prompts=tuple(
            {
                "asset_id": shot.video_id,
                "shot_index": shot.shot_index,
                "shot_start": shot.start_frame,
                "shot_end": shot.end_frame,
                "prompt": prompts[index],
            }
            for index, shot in enumerate(shots)
        ),
        plans=plans,
    )
    results = tuple(
        PropagationResult(
            backend_id="synthetic",
            backend_version="1.0",
            window=windows[index],
            prompts=(prompts[index],),
        )
        for index in range(2)
    )
    return manifest, results


def _failed_overlapping_shot_evidence() -> PilotRunManifest:
    manifest, _ = _overlapping_shot_evidence()
    plans = tuple(
        plan.model_copy(
            update={
                "attempts": (
                    plan.attempts[0].model_copy(
                        update={"status": "failed", "failure_code": "cuda_oom"}
                    ),
                )
            }
        )
        for plan in manifest.plans
    )
    return PilotRunManifest(
        input_bundle_id=manifest.input_bundle_id,
        prompt_manifest=manifest.prompt_manifest,
        executed_gold_prompts=manifest.executed_gold_prompts,
        outcome="failed",
        plans=plans,
        attempt_failures=tuple(
            PilotAttemptFailure(
                window=plan.attempts[0].window,
                code="cuda_oom",
                message="synthetic OOM",
            )
            for plan in plans
        ),
        failed_shots=tuple(
            PilotShotFailure(
                asset_id=plan.asset_id,
                start_frame=plan.start_frame,
                end_frame=plan.end_frame,
                code="cuda_oom",
                message="synthetic OOM",
            )
            for plan in plans
        ),
    )


def _staging(tmp_path: Path) -> Path:
    staging = tmp_path / "staging"
    mask = staging / "masks" / "ball-a" / "00000004.png"
    overlay = staging / "overlays" / "synthetic-video" / "00000004.png"
    mask.parent.mkdir(parents=True)
    overlay.parent.mkdir(parents=True)
    mask.write_bytes(b"synthetic-mask")
    overlay.write_bytes(b"synthetic-overlay")
    (overlay.parent / "00000005.png").write_bytes(b"synthetic-overlay-absence")
    return staging


def _relay_result(prompt_mask_ref: str) -> PropagationResult:
    prompt = (
        _result().prompts[0].model_copy(update={"box_xyxy_px": None, "mask_ref": prompt_mask_ref})
    )
    return _result().model_copy(update={"prompts": (prompt,)})


def test_publish_pilot_run_inventories_payloads_and_freezes_raw_evidence(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    run = publish_pilot_run(
        staging_root=staging,
        manifest=_manifest(),
        results=(_result(),),
        output_root=tmp_path / "runs",
    )

    inventory = json.loads((run / "inventory.json").read_text(encoding="utf-8"))
    assert inventory["files"] == [
        {
            "bytes": 14,
            "path": "masks/ball-a/00000004.png",
            "sha256": _sha256(run / "masks/ball-a/00000004.png"),
        },
        {
            "bytes": 17,
            "path": "overlays/synthetic-video/00000004.png",
            "sha256": _sha256(run / "overlays/synthetic-video/00000004.png"),
        },
        {
            "bytes": 25,
            "path": "overlays/synthetic-video/00000005.png",
            "sha256": _sha256(run / "overlays/synthetic-video/00000005.png"),
        },
    ]
    assert {item.name for item in run.iterdir()} == {
        "absences.jsonl",
        "coverage.jsonl",
        "failures.jsonl",
        "inventory.json",
        "links.jsonl",
        "manifest.json",
        "masks",
        "observations.jsonl",
        "overlays",
        "plan.json",
        "prompts.jsonl",
        "qa.jsonl",
        "results.jsonl",
        "telemetry.jsonl",
    }
    observation = json.loads((run / "observations.jsonl").read_text(encoding="utf-8"))
    assert observation["provenance"] == "pseudo"
    assert observation["review_status"] == "unreviewed"
    stored = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert stored["qa_identity"]["artifact_kind"] == "raw_run"
    assert stored["content_hash"] == run.name
    assert not run.stat().st_mode & 0o222
    stored_result = json.loads((run / "results.jsonl").read_text(encoding="utf-8"))
    assert stored_result["backend_id"] == "synthetic"
    assert stored_result["observations"][0]["instance_id"] == "ball-a"


def test_publish_embeds_only_the_verified_prompt_snapshot_without_mutable_summaries(
    tmp_path: Path,
) -> None:
    run = publish_pilot_run(
        staging_root=_staging(tmp_path),
        manifest=_snapshot_manifest(),
        results=(_result(),),
        output_root=tmp_path / "runs",
    )

    stored = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert stored["prompt_manifest"] == _prompt_manifest().model_dump(mode="json")
    assert (
        not {
            "dataset_id",
            "prompt_manifest_id",
            "prompt_circle_count",
            "prompt_frame_count",
            "prompt_shots_hash",
        }
        & stored.keys()
    )
    assert "role" not in json.dumps(stored["prompt_manifest"])
    assert "mask_ref" not in json.dumps(stored["prompt_manifest"])


def test_publish_rejects_a_forged_embedded_prompt_snapshot_id(tmp_path: Path) -> None:
    manifest = _snapshot_manifest()
    forged_prompt = manifest.prompt_manifest.model_copy(update={"manifest_id": "0" * 64})
    forged = manifest.model_copy(update={"prompt_manifest": forged_prompt})

    with pytest.raises(PilotArtifactError, match="prompt manifest|identity|snapshot"):
        publish_pilot_run(
            staging_root=_staging(tmp_path),
            manifest=forged,
            results=(_result(),),
            output_root=tmp_path / "runs",
        )


def test_verify_pilot_run_rejects_missing_gold_prompt_evidence() -> None:
    result = _result().model_copy(update={"prompts": ()})

    with pytest.raises(PilotArtifactError, match="gold|prompt|circle|exact"):
        verify_pilot_run_evidence(_snapshot_manifest(), (result,))


def test_verify_pilot_run_rejects_duplicate_gold_prompt_evidence() -> None:
    prompt = _result().prompts[0]
    duplicate = prompt.model_copy(update={"instance_id": "ball-duplicate"})
    result = _result().model_copy(update={"prompts": (prompt, duplicate)})

    with pytest.raises(PilotArtifactError, match="gold|prompt|circle|exact|duplicate"):
        verify_pilot_run_evidence(_snapshot_manifest(), (result,))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asset_id", "wrong-video"),
        ("frame_index", 5),
        ("source_id", "wrong-seed"),
        ("class_name", "cue"),
    ],
)
def test_verify_pilot_run_rejects_wrong_gold_prompt_provenance(field: str, value: object) -> None:
    prompt = _result().prompts[0].model_copy(update={field: value})
    result = _result().model_copy(update={"prompts": (prompt,)})

    with pytest.raises(PilotArtifactError, match="gold|prompt|circle|exact"):
        verify_pilot_run_evidence(_snapshot_manifest(), (result,))


def test_verify_pilot_run_rejects_cross_swapped_gold_between_overlapping_shots() -> None:
    manifest, results = _overlapping_shot_evidence()
    swapped = (
        results[0].model_copy(update={"prompts": results[1].prompts}),
        results[1].model_copy(update={"prompts": results[0].prompts}),
    )

    with pytest.raises(PilotArtifactError, match="owner|shot|gold|prompt|exact"):
        verify_pilot_run_evidence(manifest, swapped)


def test_verify_rejects_cross_swapped_executed_payloads_for_failed_shots() -> None:
    manifest = _failed_overlapping_shot_evidence()
    first, second = manifest.executed_gold_prompts
    forged = manifest.model_copy(
        update={
            "executed_gold_prompts": (
                first.model_copy(update={"prompt": second.prompt}),
                second.model_copy(update={"prompt": first.prompt}),
            )
        }
    )

    with pytest.raises(PilotArtifactError, match="owner|shot|gold|prompt|exact"):
        verify_pilot_run_evidence(forged, ())


def test_verify_rejects_cross_swapped_executed_owners_for_failed_shots() -> None:
    manifest = _failed_overlapping_shot_evidence()
    first, second = manifest.executed_gold_prompts
    first_owner = {
        "asset_id": first.asset_id,
        "shot_index": first.shot_index,
        "shot_start": first.shot_start,
        "shot_end": first.shot_end,
    }
    second_owner = {
        "asset_id": second.asset_id,
        "shot_index": second.shot_index,
        "shot_start": second.shot_start,
        "shot_end": second.shot_end,
    }
    forged = manifest.model_copy(
        update={
            "executed_gold_prompts": (
                first.model_copy(update=second_owner),
                second.model_copy(update=first_owner),
            )
        }
    )

    with pytest.raises(PilotArtifactError, match="owner|shot|gold|prompt|exact"):
        verify_pilot_run_evidence(forged, ())


def test_verify_pilot_run_rejects_ambiguous_active_attempt_ownership() -> None:
    manifest, results = _overlapping_shot_evidence()
    common = PropagationWindow(asset_id="overlap-video", ordinal=0, start_frame=5, end_frame=10)
    left = PropagationWindow(asset_id="overlap-video", ordinal=1, start_frame=4, end_frame=5)
    right = PropagationWindow(asset_id="overlap-video", ordinal=1, start_frame=10, end_frame=11)
    plans = (
        WindowPlan(
            asset_id="overlap-video",
            start_frame=4,
            end_frame=10,
            attempts=(
                WindowAttempt(window=left, status="committed"),
                WindowAttempt(window=common, status="committed"),
            ),
        ),
        WindowPlan(
            asset_id="overlap-video",
            start_frame=5,
            end_frame=11,
            attempts=(
                WindowAttempt(window=common, status="committed"),
                WindowAttempt(window=right, status="committed"),
            ),
        ),
    )
    ambiguous_manifest = PilotRunManifest(
        input_bundle_id=manifest.input_bundle_id,
        prompt_manifest=manifest.prompt_manifest,
        executed_gold_prompts=manifest.executed_gold_prompts,
        plans=plans,
    )
    pseudo = InstancePrompt(
        asset_id="overlap-video",
        frame_index=10,
        instance_id="relay-ball",
        class_name="ball",
        source_kind="pseudo",
        source_id="overlap-relay",
        box_xyxy_px=(1, 1, 5, 5),
    )
    ambiguous_results = (
        results[0].model_copy(update={"window": left}),
        results[1].model_copy(update={"window": common}),
        PropagationResult(
            backend_id="synthetic",
            backend_version="1.0",
            window=right,
            prompts=(pseudo,),
        ),
    )

    with pytest.raises(PilotArtifactError, match="ambiguous|owner|shot|attempt"):
        verify_pilot_run_evidence(ambiguous_manifest, ambiguous_results)


def test_publish_identity_includes_structured_results_and_payload_inventory(tmp_path: Path) -> None:
    first = publish_pilot_run(
        staging_root=_staging(tmp_path / "one"),
        manifest=_manifest(),
        results=(_result(),),
        output_root=tmp_path / "runs",
    )
    altered = _result().model_copy(
        update={"observations": (_result().observations[0].model_copy(update={"area_px": 15}),)}
    )
    second = publish_pilot_run(
        staging_root=_staging(tmp_path / "two"),
        manifest=_manifest(),
        results=(altered,),
        output_root=tmp_path / "runs",
    )
    third_staging = _staging(tmp_path / "three")
    (third_staging / "masks" / "ball-a" / "00000004.png").write_bytes(b"changed-mask")
    third = publish_pilot_run(
        staging_root=third_staging,
        manifest=_manifest(),
        results=(_result(),),
        output_root=tmp_path / "runs",
    )

    assert len({first.name, second.name, third.name}) == 3


@pytest.mark.parametrize(
    ("mask_ref", "mutation", "message"),
    [
        ("masks/ball-a/00000004.png", "missing", "missing payload"),
        ("masks/ball-a/00000004.png", "extra", "payload set"),
        ("masks/ball-a/00000004.png", "symlink", "symlink"),
        ("/tmp/escape.png", "none", "confined"),
        ("masks/../escape.png", "none", "confined"),
    ],
)
def test_publish_rejects_unsafe_or_non_exact_payloads(
    tmp_path: Path, mask_ref: str, mutation: str, message: str
) -> None:
    staging = _staging(tmp_path)
    if mutation == "missing":
        (staging / "masks" / "ball-a" / "00000004.png").unlink()
    elif mutation == "extra":
        (staging / "masks" / "extra.png").write_bytes(b"extra")
    elif mutation == "symlink":
        os.symlink(staging / "masks" / "ball-a" / "00000004.png", staging / "masks" / "alias.png")

    with pytest.raises((PilotArtifactError, ValueError), match=message):
        publish_pilot_run(
            staging_root=staging,
            manifest=_manifest(),
            results=(_result(mask_ref),),
            output_root=tmp_path / "runs",
        )


def test_publish_rejects_duplicate_payload_references(tmp_path: Path) -> None:
    duplicate = _result().model_copy(
        update={
            "observations": (
                _result().observations[0],
                _result().observations[0].model_copy(update={"instance_id": "ball-b"}),
            )
        }
    )
    with pytest.raises(PilotArtifactError, match="duplicate"):
        publish_pilot_run(
            staging_root=_staging(tmp_path),
            manifest=_manifest(),
            results=(duplicate,),
            output_root=tmp_path / "runs",
        )


def test_publish_reuses_verified_immutable_directory_and_cleans_only_own_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = _staging(tmp_path)
    output = tmp_path / "runs"
    first = publish_pilot_run(
        staging_root=staging, manifest=_manifest(), results=(_result(),), output_root=output
    )
    assert (
        publish_pilot_run(
            staging_root=staging, manifest=_manifest(), results=(_result(),), output_root=output
        )
        == first
    )
    sentinel = output / ".pending-not-ours"
    sentinel.mkdir()

    from billiards_ai.lab.propagation import pilot_artifacts

    def fail_publish(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic interrupt")

    monkeypatch.setattr(pilot_artifacts.os, "replace", fail_publish)
    with pytest.raises(OSError, match="synthetic interrupt"):
        publish_pilot_run(
            staging_root=_staging(tmp_path / "interrupted"),
            manifest=_manifest(),
            results=(_result().model_copy(update={"backend_version": "2.0"}),),
            output_root=output,
        )
    assert sentinel.is_dir()
    assert not list(output.glob(".pending-*")) or list(output.glob(".pending-*")) == [sentinel]


def test_publish_rejects_a_payload_changed_after_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = _staging(tmp_path)
    from billiards_ai.lab.propagation import pilot_artifacts

    real_copyfile = pilot_artifacts.shutil.copyfile

    def mutate_source(source: Path, destination: Path) -> Path:
        copied = real_copyfile(source, destination)
        if source.name == "00000004.png":
            source.write_bytes(b"changed-after-inventory")
        return copied

    monkeypatch.setattr(pilot_artifacts.shutil, "copyfile", mutate_source)

    with pytest.raises(PilotArtifactError, match="changed while publishing"):
        publish_pilot_run(
            staging_root=staging,
            manifest=_manifest(),
            results=(_result(),),
            output_root=tmp_path / "runs",
        )
    assert not list((tmp_path / "runs").glob(".pending-*"))


def test_publish_rejects_a_tampered_existing_content_addressed_directory(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    output = tmp_path / "runs"
    run = publish_pilot_run(
        staging_root=staging, manifest=_manifest(), results=(_result(),), output_root=output
    )
    payload = run / "masks" / "ball-a" / "00000004.png"
    payload.chmod(0o644)
    payload.write_bytes(b"tampered")

    with pytest.raises(PilotArtifactError, match="published raw run payload changed"):
        publish_pilot_run(
            staging_root=staging,
            manifest=_manifest(),
            results=(_result(),),
            output_root=output,
        )


def test_publish_never_freezes_promoted_raw_observations(tmp_path: Path) -> None:
    result = _result().model_copy(
        update={
            "observations": (
                _result().observations[0].model_copy(update={"review_status": "approved"}),
            )
        }
    )
    with pytest.raises(PilotArtifactError, match="pseudo/unreviewed"):
        publish_pilot_run(
            staging_root=_staging(tmp_path),
            manifest=_manifest(),
            results=(result,),
            output_root=tmp_path / "runs",
        )


def test_publish_requires_results_for_exactly_the_active_manifest_windows(tmp_path: Path) -> None:
    with pytest.raises(PilotArtifactError, match="active manifest windows"):
        publish_pilot_run(
            staging_root=_staging(tmp_path),
            manifest=_manifest(),
            results=(),
            output_root=tmp_path / "runs",
        )


def test_publish_rejects_symlink_inserted_into_its_pending_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = _staging(tmp_path)
    from billiards_ai.lab.propagation import pilot_artifacts

    real_copyfile = pilot_artifacts.shutil.copyfile

    def insert_symlink(source: Path, destination: Path) -> Path:
        copied = real_copyfile(source, destination)
        if source.name == "00000004.png":
            (destination.parent / "injected-link").symlink_to(tmp_path / "outside")
        return copied

    monkeypatch.setattr(pilot_artifacts.shutil, "copyfile", insert_symlink)

    with pytest.raises(PilotArtifactError, match="symlink"):
        publish_pilot_run(
            staging_root=staging,
            manifest=_manifest(),
            results=(_result(),),
            output_root=tmp_path / "runs",
        )
    assert not list((tmp_path / "runs").glob(".pending-*"))


@pytest.mark.parametrize("status", ["failed", "superseded"])
def test_publish_rejects_results_for_non_active_manifest_attempts(
    tmp_path: Path, status: str
) -> None:
    plan = (
        _manifest()
        .plans[0]
        .model_copy(
            update={
                "attempts": (
                    WindowAttempt(
                        window=_result().window,
                        status=status,
                        failure_code="synthetic" if status == "failed" else None,
                    ),
                )
            }
        )
    )
    manifest = _manifest().model_copy(update={"plans": (plan,)})

    with pytest.raises(PilotArtifactError, match="active manifest windows"):
        publish_pilot_run(
            staging_root=_staging(tmp_path),
            manifest=manifest,
            results=(_result(),),
            output_root=tmp_path / "runs",
        )


def test_publish_rejects_payloads_in_structured_artifact_namespace(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    (staging / "masks" / "ball-a" / "00000004.png").unlink()
    (staging / "manifest.json").write_bytes(b"mask-in-reserved-name")

    with pytest.raises(PilotArtifactError, match="reserved"):
        publish_pilot_run(
            staging_root=staging,
            manifest=_manifest(),
            results=(_result("manifest.json"),),
            output_root=tmp_path / "runs",
        )
    assert not list((tmp_path / "runs").glob("[0-9a-f]" * 64))


def test_publish_inventories_relay_prompt_masks(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    relay = staging / "masks" / "relay" / "00000004.png"
    relay.parent.mkdir()
    relay.write_bytes(b"relay-mask")

    run = publish_pilot_run(
        staging_root=staging,
        manifest=_manifest(),
        results=(_relay_result("masks/relay/00000004.png"),),
        output_root=tmp_path / "runs",
    )

    inventory = json.loads((run / "inventory.json").read_text(encoding="utf-8"))
    assert {item["path"] for item in inventory["files"]} >= {"masks/relay/00000004.png"}


def test_publish_rejects_dangling_or_shared_relay_prompt_masks(tmp_path: Path) -> None:
    with pytest.raises(PilotArtifactError, match="missing payload"):
        publish_pilot_run(
            staging_root=_staging(tmp_path / "dangling"),
            manifest=_manifest(),
            results=(_relay_result("masks/relay/missing.png"),),
            output_root=tmp_path / "runs",
        )
    with pytest.raises(PilotArtifactError, match="duplicate"):
        publish_pilot_run(
            staging_root=_staging(tmp_path / "shared"),
            manifest=_manifest(),
            results=(_relay_result("masks/ball-a/00000004.png"),),
            output_root=tmp_path / "runs",
        )


def test_publish_revalidates_pending_tree_immediately_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = _staging(tmp_path)
    from billiards_ai.lab.propagation import pilot_artifacts

    make_immutable = pilot_artifacts._make_immutable

    def mutate_after_immutable(temporary: Path) -> None:
        make_immutable(temporary)
        payload = temporary / "masks" / "ball-a" / "00000004.png"
        payload.chmod(0o644)
        payload.write_bytes(b"mutated-after-immutable")

    monkeypatch.setattr(pilot_artifacts, "_make_immutable", mutate_after_immutable)

    with pytest.raises(PilotArtifactError, match="pending publication"):
        publish_pilot_run(
            staging_root=staging,
            manifest=_manifest(),
            results=(_result(),),
            output_root=tmp_path / "runs",
        )
    assert not list((tmp_path / "runs").glob(".pending-*"))
    assert not list((tmp_path / "runs").glob("[0-9a-f]" * 64))


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode checks are not portable on Windows")
def test_publish_rejects_a_writable_existing_content_addressed_directory(tmp_path: Path) -> None:
    staging = _staging(tmp_path)
    output = tmp_path / "runs"
    run = publish_pilot_run(
        staging_root=staging, manifest=_manifest(), results=(_result(),), output_root=output
    )
    payload = run / "masks" / "ball-a" / "00000004.png"
    payload.chmod(0o644)

    with pytest.raises(PilotArtifactError, match="writable"):
        publish_pilot_run(
            staging_root=staging,
            manifest=_manifest(),
            results=(_result(),),
            output_root=output,
        )
