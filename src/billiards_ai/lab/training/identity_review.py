"""Local exact ball-identity review with immutable verdict export."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from billiards_ai.lab.training.identity_crop_export import _verify_published
from billiards_ai.lab.training.identity_dataset import (
    IdentityCrop,
    IdentityDataset,
    build_identity_dataset,
)
from billiards_ai.lab.training.identity_labels import (
    BallGroup,
    ExactBallIdentity,
    ExactIdentityEvidence,
    migrate_v1_exact_identity,
    validate_identity_label,
)


class IdentityReviewError(RuntimeError):
    """Identity review state is incomplete, corrupt, or mismatched."""


class IdentityReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    crop_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    group: BallGroup
    exact_identity: ExactBallIdentity | None = None
    exact_evidence: ExactIdentityEvidence | None = None
    visibility: Literal["visible", "partial", "occluded", "unknown"]
    orientation: Literal["number_visible", "number_hidden", "ambiguous"]
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def identity_is_consistent(self) -> Self:
        validate_identity_label(
            group=self.group,
            exact_identity=self.exact_identity,
            exact_evidence=self.exact_evidence,
        )
        return self


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IdentityReviewError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise IdentityReviewError(f"{label} is invalid")
    return value


def _candidate_records(candidate_artifact: Path) -> tuple[str, dict[str, dict[str, object]]]:
    try:
        verified = _verify_published(candidate_artifact)
    except RuntimeError as exc:
        raise IdentityReviewError(str(exc)) from exc
    manifest = _load_object(candidate_artifact / "manifest.json", label="candidate manifest")
    crops = manifest.get("crops")
    if not isinstance(crops, list):
        raise IdentityReviewError("candidate manifest is invalid")
    records: dict[str, dict[str, object]] = {}
    for crop in crops:
        if not isinstance(crop, dict) or not isinstance(crop.get("crop_id"), str):
            raise IdentityReviewError("candidate crop record is invalid")
        crop_id = crop["crop_id"]
        if crop_id in records:
            raise IdentityReviewError("candidate crop IDs are not unique")
        records[crop_id] = crop
    return verified.artifact_id, records


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _make_immutable(root: Path) -> None:
    if os.name == "nt":
        return
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _publish_identity_verdict(
    *,
    candidate_id: str,
    decisions: dict[str, IdentityReviewDecision],
    review_revision: int,
    output_root: Path,
    kind: str = "ball_identity_human_verdict",
    extra_identity: dict[str, object] | None = None,
) -> Path:
    labels: dict[str, int] = {}
    groups: dict[str, int] = {}
    for decision in decisions.values():
        groups[decision.group] = groups.get(decision.group, 0) + 1
        if decision.exact_identity is not None:
            labels[decision.exact_identity] = labels.get(decision.exact_identity, 0) + 1
    identity = {
        "schema_version": "2.0",
        "kind": kind,
        "candidate_artifact_id": candidate_id,
        "review_revision": review_revision,
        "review_count": len(decisions),
        "group_counts": dict(sorted(groups.items())),
        "label_counts": dict(sorted(labels.items())),
        "exact_label_count": sum(labels.values()),
        "group_only_count": len(decisions) - sum(labels.values()),
        "decisions": [decisions[crop_id].model_dump(mode="json") for crop_id in sorted(decisions)],
        **(extra_identity or {}),
    }
    artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / artifact_id
    if target.exists():
        _verify_verdict(target, candidate_id=candidate_id)
        return target
    temporary = output_root / f".pending-{uuid4().hex}"
    try:
        temporary.mkdir()
        (temporary / "verdict.json").write_bytes(
            _canonical_bytes({"artifact_id": artifact_id, **identity})
        )
        _make_immutable(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    _verify_verdict(target, candidate_id=candidate_id)
    return target


class IdentityReviewSession:
    def __init__(
        self,
        *,
        candidate_artifact: Path,
        candidate_id: str,
        crop_records: dict[str, dict[str, object]],
        state_path: Path,
        decisions: dict[str, IdentityReviewDecision],
        revision: int,
        suggestions: dict[str, IdentityReviewDecision] | None = None,
        contexts: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.candidate_artifact = candidate_artifact
        self.candidate_id = candidate_id
        self._crop_records = crop_records
        self.state_path = state_path
        self.decisions = decisions
        self.revision = revision
        self.suggestions = suggestions or {}
        self.contexts = contexts or {}

    @classmethod
    def open(
        cls,
        *,
        candidate_artifact: Path,
        state_path: Path,
        crop_ids: tuple[str, ...] | None = None,
        suggestions: dict[str, IdentityReviewDecision] | None = None,
        contexts: dict[str, dict[str, object]] | None = None,
    ) -> IdentityReviewSession:
        candidate_artifact = candidate_artifact.resolve()
        candidate_id, records = _candidate_records(candidate_artifact)
        if crop_ids is not None:
            if not crop_ids or len(set(crop_ids)) != len(crop_ids):
                raise IdentityReviewError("identity review crop scope is invalid")
            if any(crop_id not in records for crop_id in crop_ids):
                raise IdentityReviewError("identity review crop scope is invalid")
            records = {crop_id: records[crop_id] for crop_id in crop_ids}
        suggestions = suggestions or {}
        contexts = contexts or {}
        if any(crop_id not in records for crop_id in suggestions) or any(
            crop_id not in records for crop_id in contexts
        ):
            raise IdentityReviewError("identity review context is outside crop scope")
        decisions: dict[str, IdentityReviewDecision] = {}
        revision = 0
        if state_path.exists():
            state = _load_object(state_path, label="identity review state")
            schema_version = state.get("schema_version")
            if (
                schema_version not in {"1.0", "2.0"}
                or state.get("candidate_artifact_id") != candidate_id
                or not isinstance(state.get("revision"), int)
                or not isinstance(state.get("decisions"), list)
                or (crop_ids is not None and state.get("scope_crop_ids") != sorted(records))
            ):
                raise IdentityReviewError("identity review state does not match candidates")
            revision = int(state["revision"])
            try:
                if schema_version == "1.0":
                    migrated = []
                    for item in state["decisions"]:
                        if not isinstance(item, dict):
                            raise ValueError("legacy decision is invalid")
                        group, exact, evidence = migrate_v1_exact_identity(
                            str(item.get("exact_identity"))
                        )
                        migrated.append(
                            {
                                **item,
                                "group": group,
                                "exact_identity": exact,
                                "exact_evidence": evidence,
                            }
                        )
                    parsed = tuple(IdentityReviewDecision.model_validate(item) for item in migrated)
                else:
                    parsed = tuple(
                        IdentityReviewDecision.model_validate(item) for item in state["decisions"]
                    )
            except (TypeError, ValueError) as exc:
                raise IdentityReviewError("identity review decisions are invalid") from exc
            if any(item.crop_id not in records for item in parsed):
                raise IdentityReviewError("identity review state contains an unknown crop")
            decisions = {item.crop_id: item for item in parsed}
            if len(decisions) != len(parsed):
                raise IdentityReviewError("identity review decisions contain duplicate crops")
        return cls(
            candidate_artifact=candidate_artifact,
            candidate_id=candidate_id,
            crop_records=records,
            state_path=state_path,
            decisions=decisions,
            revision=revision,
            suggestions=suggestions,
            contexts=contexts,
        )

    @property
    def crop_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._crop_records))

    @property
    def reviewed_count(self) -> int:
        return len(self.decisions)

    def _persist(self) -> None:
        payload = {
            "schema_version": "2.0",
            "candidate_artifact_id": self.candidate_id,
            "scope_crop_ids": list(self.crop_ids),
            "revision": self.revision,
            "decisions": [
                self.decisions[crop_id].model_dump(mode="json")
                for crop_id in sorted(self.decisions)
            ],
        }
        _write_atomic(self.state_path, _canonical_bytes(payload))

    def record(
        self,
        *,
        crop_id: str,
        group: str | None = None,
        exact_identity: str | None = None,
        exact_evidence: str | None = None,
        visibility: str,
        orientation: str,
        note: str | None = None,
    ) -> None:
        if crop_id not in self._crop_records:
            raise IdentityReviewError("identity review references an unknown crop")
        if group is None:
            if exact_identity is None:
                raise IdentityReviewError("identity review decision is invalid")
            try:
                group, exact_identity, exact_evidence = migrate_v1_exact_identity(exact_identity)
            except ValueError as exc:
                raise IdentityReviewError("identity review decision is invalid") from exc
        try:
            decision = IdentityReviewDecision(
                crop_id=crop_id,
                group=group,
                exact_identity=exact_identity,
                exact_evidence=exact_evidence,
                visibility=visibility,
                orientation=orientation,
                note=note.strip() if note and note.strip() else None,
            )
        except ValueError as exc:
            raise IdentityReviewError("identity review decision is invalid") from exc
        self.decisions[crop_id] = decision
        self.revision += 1
        self._persist()

    def export_verdict(self, output_root: Path) -> Path:
        if set(self.decisions) != set(self._crop_records):
            raise IdentityReviewError("all crops must be reviewed before verdict export")
        return _publish_identity_verdict(
            candidate_id=self.candidate_id,
            decisions=self.decisions,
            review_revision=self.revision,
            output_root=output_root,
        )


def _verify_verdict(path: Path, *, candidate_id: str) -> tuple[IdentityReviewDecision, ...]:
    payload = _load_object(path / "verdict.json", label="identity verdict")
    artifact_id = payload.get("artifact_id")
    decisions = payload.get("decisions")
    if (
        path.is_symlink()
        or not path.is_dir()
        or not isinstance(artifact_id, str)
        or path.name != artifact_id
        or payload.get("schema_version") != "2.0"
        or payload.get("candidate_artifact_id") != candidate_id
        or not isinstance(decisions, list)
    ):
        raise IdentityReviewError("identity verdict is invalid")
    identity = {key: value for key, value in payload.items() if key != "artifact_id"}
    if hashlib.sha256(_canonical_bytes(identity)).hexdigest() != artifact_id:
        raise IdentityReviewError("identity verdict hash changed")
    try:
        parsed = tuple(IdentityReviewDecision.model_validate(item) for item in decisions)
    except (TypeError, ValueError) as exc:
        raise IdentityReviewError("identity verdict decisions are invalid") from exc
    if len(parsed) != payload.get("review_count") or len({item.crop_id for item in parsed}) != len(
        parsed
    ):
        raise IdentityReviewError("identity verdict count is invalid")
    return parsed


def merge_identity_correction_verdicts(
    *,
    candidate_artifact: Path,
    base_correction_verdict_artifact: Path,
    overlay_correction_verdict_artifact: Path,
    output_root: Path,
) -> Path:
    """Publish cumulative corrections, with the overlay winning by crop ID."""

    candidate_artifact = candidate_artifact.resolve()
    base_correction_verdict_artifact = base_correction_verdict_artifact.resolve()
    overlay_correction_verdict_artifact = overlay_correction_verdict_artifact.resolve()
    candidate_id, records = _candidate_records(candidate_artifact)
    base = _verify_verdict(base_correction_verdict_artifact, candidate_id=candidate_id)
    overlay = _verify_verdict(overlay_correction_verdict_artifact, candidate_id=candidate_id)
    merged = {decision.crop_id: decision for decision in base}
    merged.update({decision.crop_id: decision for decision in overlay})
    if any(crop_id not in records for crop_id in merged):
        raise IdentityReviewError("identity correction contains an unknown crop")
    return _publish_identity_verdict(
        candidate_id=candidate_id,
        decisions=merged,
        review_revision=len(base) + len(overlay),
        output_root=output_root,
        kind="ball_identity_cumulative_correction_verdict",
        extra_identity={
            "source_correction_artifact_ids": [
                base_correction_verdict_artifact.name,
                overlay_correction_verdict_artifact.name,
            ],
            "merge_policy": "overlay_wins",
        },
    )


def load_reviewed_identity_dataset(
    *,
    candidate_artifact: Path,
    verdict_artifact: Path,
    correction_verdict_artifact: Path | None = None,
    split: str,
) -> IdentityDataset:
    candidate_id, records = _candidate_records(candidate_artifact)
    base_decisions = _verify_verdict(verdict_artifact, candidate_id=candidate_id)
    if {item.crop_id for item in base_decisions} != set(records):
        raise IdentityReviewError("identity verdict does not cover all candidate crops")
    decisions_by_id = {item.crop_id: item for item in base_decisions}
    if correction_verdict_artifact is not None:
        corrections = _verify_verdict(correction_verdict_artifact, candidate_id=candidate_id)
        if any(item.crop_id not in records for item in corrections):
            raise IdentityReviewError("identity correction contains an unknown crop")
        decisions_by_id.update({item.crop_id: item for item in corrections})
    decisions = tuple(decisions_by_id[crop_id] for crop_id in sorted(decisions_by_id))
    crops = []
    for decision in decisions:
        record = records[decision.crop_id]
        try:
            crops.append(
                IdentityCrop(
                    crop_id=decision.crop_id,
                    relative_path=record["relative_path"],
                    sha256=record["sha256"],
                    source_asset_id=record["source_asset_id"],
                    source_frame_id=(f"{record['source_asset_id']}:{record['source_frame_index']}"),
                    source_frame_sha256=record["source_image_sha256"],
                    calibration_revision=None,
                    calibration_sha256=None,
                    detector_bbox_xywh=tuple(record["detector_bbox_xywh"]),
                    detector_mask_polygon_px=tuple(
                        tuple(point) for point in record["detector_mask_polygon_px"]
                    ),
                    group_split=f"{record['source_asset_id']}:{record['source_split']}",
                    group=decision.group,
                    exact_identity=decision.exact_identity,
                    exact_evidence=decision.exact_evidence,
                    visibility=decision.visibility,
                    orientation=decision.orientation,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise IdentityReviewError("reviewed identity crop record is invalid") from exc
    return build_identity_dataset(crops=tuple(crops), source_root=candidate_artifact, split=split)
