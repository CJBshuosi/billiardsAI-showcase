"""Verified metadata policy for explicit development-only model acceptance."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from billiards_ai.lab.mainline1_runtime import inspect_development_group_assets


@dataclass(frozen=True, slots=True)
class DevelopmentAcceptancePolicy:
    identity_training_package_id: str
    identity_training_package_sha256: str
    gate_violations: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "mode": "development_model_acceptance",
            "identity_gate_status": "FAIL",
            "identity_training_package_id": self.identity_training_package_id,
            "identity_training_package_sha256": self.identity_training_package_sha256,
            "gate_violations": list(self.gate_violations),
            "formal_model_qualified": False,
            "formal_field_acceptance": "NOT_ASSERTED",
        }

    def verify_identity(self, training_run: Path) -> None:
        if _inspect_policy(training_run) != self:
            raise ValueError("development model acceptance identity changed")


def _inspect_policy(training_run: Path) -> DevelopmentAcceptancePolicy:
    assets = inspect_development_group_assets(training_run)
    raw = assets.manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != assets.manifest_sha256:
        raise ValueError("development model acceptance manifest changed during inspection")
    manifest = json.loads(raw)
    violations = manifest.get("gate_violations")
    if (
        manifest.get("gate_status") != "FAIL"
        or manifest.get("run_id") != assets.root.name
        or not isinstance(violations, list)
        or any(not isinstance(value, str) or not value.strip() for value in violations)
    ):
        raise ValueError(
            "development model acceptance requires an actual FAIL manifest with gate_violations"
        )
    return DevelopmentAcceptancePolicy(assets.root.name, assets.manifest_sha256, tuple(violations))


def resolve_acceptance_policy(args) -> DevelopmentAcceptancePolicy | None:
    if not getattr(args, "development_model_acceptance", False):
        return None
    if (
        getattr(args, "runtime_environment", None) != "development"
        or not getattr(args, "rule_profile", None)
        or getattr(args, "observation_only", False)
        or getattr(args, "identity_gate_status", None) != "FAIL"
    ):
        raise ValueError(
            "development model acceptance requires explicit development, rule profile, "
            "declared FAIL and no observation-only"
        )
    return _inspect_policy(args.identity_training_run)
