from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from billiards_ai.lab.vision.package import canonical_package_id, load_vision_package
from billiards_ai.lab.vision.promotion import VisionPromotionVerdict
from billiards_ai.lab.vision.registry import (
    VisionRegistryError,
    activate_candidate,
    load_active_model,
    rollback_active,
)


def _canonical_bytes(value: object) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode()


def _write_package(root: Path, name: str, checkpoint: bytes) -> tuple[str, str]:
    package_root = root / "packages" / name
    package_root.mkdir(parents=True)
    (package_root / "checkpoint.pth").write_bytes(checkpoint)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "task": "ball_instance_segmentation",
        "family": "rfdetr",
        "model_version": "1.9.0",
        "model_class": "RFDETRSegSmall",
        "artifacts": [
            {
                "role": "checkpoint",
                "relative_path": "checkpoint.pth",
                "byte_count": len(checkpoint),
                "sha256": hashlib.sha256(checkpoint).hexdigest(),
            }
        ],
        "preprocess": {
            "color_space": "RGB",
            "input_width_px": 384,
            "input_height_px": 384,
        },
        "postprocess": {"confidence_threshold": 0.25, "mask_threshold": 0.5},
        "class_map": {"ball": 0},
        "runtime_requirements": ["rfdetr==1.9.0"],
        "training_run_sha256": "1" * 64,
        "code_sha256": "2" * 64,
        "base_weight_sha256": "3" * 64,
        "asset_status_ref": "internal-rnd-only",
    }
    payload["package_id"] = canonical_package_id(payload)
    manifest = package_root / "manifest.json"
    manifest.write_bytes(_canonical_bytes(payload))
    verified = load_vision_package(package_root)
    return verified.identity.package_id, verified.identity.manifest_sha256


def _write_registry(root: Path) -> tuple[Path, str, str]:
    incumbent, incumbent_manifest = _write_package(root, "incumbent", b"incumbent")
    candidate, candidate_manifest = _write_package(root, "candidate", b"candidate")
    registry = root / "registry.json"
    registry.write_bytes(
        _canonical_bytes(
            {
                "schema_version": "1.0",
                "entries": [
                    {
                        "package_id": incumbent,
                        "manifest_sha256": incumbent_manifest,
                        "relative_path": "packages/incumbent",
                    },
                    {
                        "package_id": candidate,
                        "manifest_sha256": candidate_manifest,
                        "relative_path": "packages/candidate",
                    },
                ],
            }
        )
    )
    return registry, incumbent, candidate


def _write_initial_active(root: Path, package_id: str, manifest_sha256: str) -> Path:
    active = root / "active.json"
    payload = {
        "schema_version": "1.0",
        "activation_id": "1" * 64,
        "package_id": package_id,
        "manifest_sha256": manifest_sha256,
        "verdict_sha256": "0" * 64,
        "activated_at": "2026-08-23T11:00:00Z",
        "previous_activation_id": None,
        "previous_package_id": None,
        "action": "promote",
    }
    active.write_bytes(_canonical_bytes(payload))
    return active


def _write_verdict(root: Path, incumbent: str, candidate: str, status: str) -> Path:
    verdict = VisionPromotionVerdict(
        policy_sha256="d" * 64,
        incumbent_package_id=incumbent,
        candidate_package_id=candidate,
        passed_absolute_gates=status == "passed",
        beats_incumbent=status == "passed",
        passed_downstream_regressions=status == "passed",
        promotion_status=status,
        comparisons=(),
        reasons=() if status == "passed" else ("gate failed",),
    )
    path = root / f"verdict-{status}.json"
    path.write_bytes(_canonical_bytes(verdict.model_dump(mode="json")))
    return path


def _setup(root: Path) -> tuple[Path, Path, str, str]:
    registry, incumbent, candidate = _write_registry(root)
    incumbent_manifest = load_vision_package(root / "packages" / "incumbent").identity
    active = _write_initial_active(root, incumbent, incumbent_manifest.manifest_sha256)
    return registry, active, incumbent, candidate


def test_passed_verdict_atomically_activates_registered_candidate(tmp_path: Path) -> None:
    registry, active_path, incumbent, candidate = _setup(tmp_path)

    active = activate_candidate(
        registry_path=registry,
        active_path=active_path,
        history_root=tmp_path / "history",
        verdict_path=_write_verdict(tmp_path, incumbent, candidate, "passed"),
        activated_at="2026-08-23T12:00:00Z",
    )

    assert active.package_id == candidate
    assert active.previous_package_id == incumbent
    assert load_active_model(registry, active_path).package_id == candidate
    assert len(tuple((tmp_path / "history").glob("*.json"))) == 1


@pytest.mark.parametrize("status", ["failed", "incomplete"])
def test_nonpassing_verdict_cannot_change_active(tmp_path: Path, status: str) -> None:
    registry, active_path, incumbent, candidate = _setup(tmp_path)
    before = active_path.read_bytes()

    with pytest.raises(VisionRegistryError, match="passed promotion verdict"):
        activate_candidate(
            registry_path=registry,
            active_path=active_path,
            history_root=tmp_path / "history",
            verdict_path=_write_verdict(tmp_path, incumbent, candidate, status),
            activated_at="2026-08-23T12:00:00Z",
        )

    assert active_path.read_bytes() == before


def test_rollback_restores_only_recorded_previous_package(tmp_path: Path) -> None:
    registry, active_path, incumbent, candidate = _setup(tmp_path)
    promoted = activate_candidate(
        registry_path=registry,
        active_path=active_path,
        history_root=tmp_path / "history",
        verdict_path=_write_verdict(tmp_path, incumbent, candidate, "passed"),
        activated_at="2026-08-23T12:00:00Z",
    )

    rolled_back = rollback_active(
        registry_path=registry,
        active_path=active_path,
        history_root=tmp_path / "history",
        expected_activation_id=promoted.activation_id,
        activated_at="2026-08-23T12:05:00Z",
    )

    assert rolled_back.package_id == incumbent
    assert rolled_back.previous_package_id == candidate
    assert rolled_back.action == "rollback"


def test_stale_activation_id_cannot_rollback(tmp_path: Path) -> None:
    registry, active_path, incumbent, candidate = _setup(tmp_path)
    activate_candidate(
        registry_path=registry,
        active_path=active_path,
        history_root=tmp_path / "history",
        verdict_path=_write_verdict(tmp_path, incumbent, candidate, "passed"),
        activated_at="2026-08-23T12:00:00Z",
    )
    before = active_path.read_bytes()

    with pytest.raises(VisionRegistryError, match="stale"):
        rollback_active(
            registry_path=registry,
            active_path=active_path,
            history_root=tmp_path / "history",
            expected_activation_id="f" * 64,
            activated_at="2026-08-23T12:05:00Z",
        )

    assert active_path.read_bytes() == before
