import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from tests.support.development_acceptance import write_identity_assets


def policy_args(tmp_path, **overrides):
    values = dict(
        development_model_acceptance=True,
        runtime_environment="development",
        rule_profile="cbsa-chinese-pool-2017-observable-v1",
        observation_only=False,
        identity_gate_status="FAIL",
        identity_training_run=tmp_path / "synthetic-identity",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_does_not_inspect_assets_or_change_legacy_args(tmp_path):
    from billiards_ai.lab.field_assistance.development_acceptance import resolve_acceptance_policy

    assert resolve_acceptance_policy(SimpleNamespace()) is None


def test_verified_failure_snapshot_is_frozen_copy_safe_and_rechecks_inventory(tmp_path):
    from billiards_ai.lab.field_assistance.development_acceptance import resolve_acceptance_policy

    args = policy_args(tmp_path)
    expected = write_identity_assets(args.identity_training_run)
    policy = resolve_acceptance_policy(args)
    assert policy.as_payload() == expected
    policy.as_payload()["gate_violations"].append("invented")
    assert policy.as_payload() == expected
    with pytest.raises(FrozenInstanceError):
        policy.identity_training_package_id = "another"
    policy.verify_identity(args.identity_training_run)
    (args.identity_training_run / "group-head.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError):
        policy.verify_identity(args.identity_training_run)


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime_environment": None},
        {"runtime_environment": "production"},
        {"rule_profile": None},
        {"observation_only": True},
        {"identity_gate_status": "PASS"},
        {"identity_gate_status": "UNDECLARED"},
    ],
)
def test_opt_in_rejects_conflicts_before_inspection(tmp_path, overrides):
    from billiards_ai.lab.field_assistance.development_acceptance import resolve_acceptance_policy

    with pytest.raises(ValueError, match="development model acceptance"):
        resolve_acceptance_policy(policy_args(tmp_path, **overrides))


@pytest.mark.parametrize(
    "damage", ["PASS", "UNDECLARED", "missing", "inventory", "violations", "run_id"]
)
def test_opt_in_requires_actual_verified_fail_manifest(tmp_path, damage):
    from billiards_ai.lab.field_assistance.development_acceptance import resolve_acceptance_policy

    args = policy_args(tmp_path)
    write_identity_assets(args.identity_training_run)
    path = args.identity_training_run / "manifest.json"
    manifest = json.loads(path.read_bytes())
    if damage in ("PASS", "UNDECLARED"):
        manifest["gate_status"] = damage
    elif damage == "missing":
        del manifest["gate_status"]
    elif damage == "inventory":
        manifest["inventory"]["metrics.json"]["sha256"] = "0" * 64
    elif damage == "violations":
        manifest["gate_violations"] = "not a list"
    else:
        manifest["run_id"] = "other"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        resolve_acceptance_policy(args)


@pytest.mark.parametrize("violations", [None, {}, [1], [""], ["valid", None]])
def test_malformed_gate_violations_are_not_invented(tmp_path, violations):
    from billiards_ai.lab.field_assistance.development_acceptance import resolve_acceptance_policy

    args = policy_args(tmp_path)
    write_identity_assets(args.identity_training_run)
    path = args.identity_training_run / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["gate_violations"] = violations
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        resolve_acceptance_policy(args)


def test_changed_manifest_cannot_replace_pinned_acceptance(tmp_path):
    from billiards_ai.lab.field_assistance.development_acceptance import resolve_acceptance_policy

    args = policy_args(tmp_path)
    write_identity_assets(args.identity_training_run)
    policy = resolve_acceptance_policy(args)
    write_identity_assets(args.identity_training_run, gate_violations=["different actual failure"])
    with pytest.raises(ValueError, match="changed"):
        policy.verify_identity(args.identity_training_run)


def test_manifest_bytes_must_match_the_inspected_sha(tmp_path, monkeypatch):
    from billiards_ai.lab.field_assistance import development_acceptance as policy_module

    args = policy_args(tmp_path)
    write_identity_assets(args.identity_training_run)
    inspect_assets = policy_module.inspect_development_group_assets

    def changed_after_inspection(root):
        verified = inspect_assets(root)
        write_identity_assets(root, gate_violations=["replacement failure"])
        return verified

    monkeypatch.setattr(policy_module, "inspect_development_group_assets", changed_after_inspection)
    with pytest.raises(ValueError, match="changed during inspection"):
        policy_module.resolve_acceptance_policy(args)
