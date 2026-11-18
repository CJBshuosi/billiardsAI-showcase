from __future__ import annotations

import json
from pathlib import Path

import pytest

from billiards_ai.lab.training import identity_review
from billiards_ai.lab.training.identity_crop_export import export_identity_crop_candidates
from billiards_ai.lab.training.identity_review import (
    IdentityReviewError,
    IdentityReviewSession,
    _verify_verdict,
    load_reviewed_identity_dataset,
)
from tests.unit.lab.training.test_identity_crop_export import _bootstrap_dataset


def _session(tmp_path: Path) -> IdentityReviewSession:
    candidates = export_identity_crop_candidates(
        bootstrap_dataset=_bootstrap_dataset(tmp_path),
        output_root=tmp_path / "candidate-artifacts",
        crop_size_px=64,
        context_scale=2.0,
    )
    return IdentityReviewSession.open(
        candidate_artifact=candidates.path,
        state_path=tmp_path / "review-state.json",
    )


def test_review_state_persists_intrinsic_and_group_only_labels(tmp_path: Path) -> None:
    session = _session(tmp_path)
    crop_ids = session.crop_ids

    session.record(
        crop_id=crop_ids[0],
        exact_identity="8",
        visibility="visible",
        orientation="number_visible",
        note="clear eight",
    )
    session.record(
        crop_id=crop_ids[1],
        exact_identity="unknown",
        visibility="partial",
        orientation="ambiguous",
    )
    reopened = IdentityReviewSession.open(
        candidate_artifact=session.candidate_artifact,
        state_path=session.state_path,
    )

    assert reopened.reviewed_count == 2
    assert reopened.decisions[crop_ids[0]].exact_identity == "8"
    assert reopened.decisions[crop_ids[1]].group == "unknown"
    assert reopened.decisions[crop_ids[1]].exact_identity is None


def test_schema_v1_state_migrates_without_losing_decisions(tmp_path: Path) -> None:
    session = _session(tmp_path)
    crop_ids = session.crop_ids
    state = {
        "schema_version": "1.0",
        "candidate_artifact_id": session.candidate_id,
        "revision": 2,
        "decisions": [
            {
                "crop_id": crop_ids[0],
                "exact_identity": "4",
                "visibility": "visible",
                "orientation": "number_visible",
                "note": "legacy solid",
            },
            {
                "crop_id": crop_ids[1],
                "exact_identity": "unknown",
                "visibility": "partial",
                "orientation": "ambiguous",
                "note": None,
            },
        ],
    }
    session.state_path.write_text(json.dumps(state), encoding="utf-8")

    reopened = IdentityReviewSession.open(
        candidate_artifact=session.candidate_artifact,
        state_path=session.state_path,
    )

    assert reopened.decisions[crop_ids[0]].group == "solid"
    assert reopened.decisions[crop_ids[0]].exact_identity == "4"
    assert reopened.decisions[crop_ids[0]].exact_evidence == "known_set_mapping"
    assert reopened.decisions[crop_ids[1]].group == "unknown"
    assert reopened.decisions[crop_ids[1]].exact_identity is None

    reopened.record(
        crop_id=crop_ids[1],
        group="stripe",
        exact_identity=None,
        exact_evidence=None,
        visibility="partial",
        orientation="number_hidden",
    )
    persisted = json.loads(session.state_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "2.0"


def test_verdict_export_requires_every_crop_and_builds_group_isolated_dataset(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    with pytest.raises(IdentityReviewError, match="all crops"):
        session.export_verdict(tmp_path / "verdicts")

    for index, crop_id in enumerate(session.crop_ids):
        session.record(
            crop_id=crop_id,
            exact_identity="cue" if index == 0 else "unknown",
            visibility="visible" if index == 0 else "occluded",
            orientation="number_visible" if index == 0 else "number_hidden",
        )
    verdict = session.export_verdict(tmp_path / "verdicts")

    train = load_reviewed_identity_dataset(
        candidate_artifact=session.candidate_artifact,
        verdict_artifact=verdict,
        split="train",
    )
    valid = load_reviewed_identity_dataset(
        candidate_artifact=session.candidate_artifact,
        verdict_artifact=verdict,
        split="valid",
    )
    assert len(train.records) == len(valid.records) == 1
    assert train.records[0].source_asset_id == "video-train"
    assert valid.records[0].source_asset_id == "video-valid"
    assert {train.records[0].group, valid.records[0].group} == {"cue", "unknown"}
    assert {train.records[0].exact_identity, valid.records[0].exact_identity} == {
        "cue",
        None,
    }


def test_review_rejects_a_crop_id_outside_the_frozen_candidate_artifact(tmp_path: Path) -> None:
    session = _session(tmp_path)

    with pytest.raises(IdentityReviewError, match="unknown crop"):
        session.record(
            crop_id="f" * 64,
            exact_identity="1",
            visibility="visible",
            orientation="number_visible",
        )


def test_schema_v1_verdict_cannot_enter_group_first_training(tmp_path: Path) -> None:
    verdict = tmp_path / ("a" * 64)
    verdict.mkdir()
    (verdict / "verdict.json").write_text(
        json.dumps(
            {
                "artifact_id": verdict.name,
                "schema_version": "1.0",
                "candidate_artifact_id": "candidate-v1",
                "review_count": 0,
                "decisions": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(IdentityReviewError, match="identity verdict is invalid"):
        _verify_verdict(verdict, candidate_id="candidate-v1")


def test_error_subset_requires_explicit_review_and_overlays_corrections(
    tmp_path: Path,
) -> None:
    original = _session(tmp_path)
    for index, crop_id in enumerate(original.crop_ids):
        original.record(
            crop_id=crop_id,
            group="cue" if index == 0 else "eight",
            exact_identity="cue" if index == 0 else "8",
            exact_evidence="group_intrinsic",
            visibility="visible",
            orientation="number_visible",
        )
    verdict = original.export_verdict(tmp_path / "verdicts")
    target_id = original.crop_ids[0]
    suggestion = original.decisions[target_id]
    subset = IdentityReviewSession.open(
        candidate_artifact=original.candidate_artifact,
        state_path=tmp_path / "error-review-state.json",
        crop_ids=(target_id,),
        suggestions={target_id: suggestion},
        contexts={
            target_id: {
                "truth": "cue",
                "predicted": "stripe",
                "confidence": 0.82,
                "margin": 0.61,
            }
        },
    )

    assert subset.crop_ids == (target_id,)
    assert subset.reviewed_count == 0
    assert subset.suggestions[target_id] == suggestion
    assert subset.contexts[target_id]["predicted"] == "stripe"
    with pytest.raises(IdentityReviewError, match="all crops"):
        subset.export_verdict(tmp_path / "corrections")

    subset.record(
        crop_id=target_id,
        group="stripe",
        exact_identity=None,
        exact_evidence=None,
        visibility="visible",
        orientation="number_hidden",
    )
    correction = subset.export_verdict(tmp_path / "corrections")
    records = []
    for split in ("train", "valid"):
        records.extend(
            load_reviewed_identity_dataset(
                candidate_artifact=original.candidate_artifact,
                verdict_artifact=verdict,
                correction_verdict_artifact=correction,
                split=split,
            ).records
        )

    corrected = next(record for record in records if record.crop_id == target_id)
    assert corrected.group == "stripe"
    assert corrected.exact_identity is None


def test_merge_corrections_preserves_old_labels_and_overlays_new_labels(
    tmp_path: Path,
) -> None:
    original = _session(tmp_path)
    first_id, second_id = original.crop_ids
    for crop_id in original.crop_ids:
        original.record(
            crop_id=crop_id,
            group="solid",
            exact_identity=None,
            exact_evidence=None,
            visibility="visible",
            orientation="number_hidden",
        )
    original_verdict = original.export_verdict(tmp_path / "verdicts")

    old = IdentityReviewSession.open(
        candidate_artifact=original.candidate_artifact,
        state_path=tmp_path / "old-state.json",
        crop_ids=(first_id, second_id),
    )
    old.record(
        crop_id=first_id,
        group="cue",
        exact_identity="cue",
        exact_evidence="group_intrinsic",
        visibility="visible",
        orientation="number_visible",
    )
    old.record(
        crop_id=second_id,
        group="eight",
        exact_identity="8",
        exact_evidence="group_intrinsic",
        visibility="visible",
        orientation="number_visible",
    )
    old_verdict = old.export_verdict(tmp_path / "old-corrections")

    new = IdentityReviewSession.open(
        candidate_artifact=original.candidate_artifact,
        state_path=tmp_path / "new-state.json",
        crop_ids=(first_id,),
    )
    new.record(
        crop_id=first_id,
        group="stripe",
        exact_identity=None,
        exact_evidence=None,
        visibility="partial",
        orientation="number_hidden",
    )
    new_verdict = new.export_verdict(tmp_path / "new-corrections")

    merged = identity_review.merge_identity_correction_verdicts(
        candidate_artifact=original.candidate_artifact,
        base_correction_verdict_artifact=old_verdict,
        overlay_correction_verdict_artifact=new_verdict,
        output_root=tmp_path / "merged-corrections",
    )

    records = []
    for split in ("train", "valid"):
        records.extend(
            load_reviewed_identity_dataset(
                candidate_artifact=original.candidate_artifact,
                verdict_artifact=original_verdict,
                correction_verdict_artifact=merged,
                split=split,
            ).records
        )
    by_id = {record.crop_id: record for record in records}
    assert by_id[first_id].group == "stripe"
    assert by_id[second_id].group == "eight"
    payload = json.loads((merged / "verdict.json").read_text(encoding="utf-8"))
    assert payload["source_correction_artifact_ids"] == [
        old_verdict.name,
        new_verdict.name,
    ]
    assert payload["merge_policy"] == "overlay_wins"
