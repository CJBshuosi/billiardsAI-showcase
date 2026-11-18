from pathlib import Path

import yaml

ASSET_MANIFEST = Path("configs/algorithm/assets.yaml")
EXPERIMENT_POLICY = Path("configs/algorithm/experiment-policy.yaml")
IDENTITY_TRAINING_CONFIG = Path("configs/algorithm/training/identity.yaml")
REQUIRED_ASSET_FIELDS = {
    "id",
    "kind",
    "source",
    "version",
    "sha256",
    "license",
    "expected_bytes",
    "commercial_status",
    "approval_status",
}


def test_every_algorithm_asset_has_commercial_provenance() -> None:
    data = yaml.safe_load(ASSET_MANIFEST.read_text(encoding="utf-8"))

    assert data["schema_version"] == "1.0"
    assert data["assets"]
    for asset in data["assets"]:
        assert REQUIRED_ASSET_FIELDS <= asset.keys()
        assert asset["commercial_status"] in {
            "allowed",
            "legal_review_required",
            "excluded",
        }
        assert asset["approval_status"] in {"pending", "approved", "rejected"}
        if asset["approval_status"] == "approved":
            assert len(asset["sha256"]) == 64
            assert asset["expected_bytes"] > 0


def test_experiment_policy_keeps_external_actions_pending() -> None:
    policy = yaml.safe_load(EXPERIMENT_POLICY.read_text(encoding="utf-8"))

    assert policy["schema_version"] == "1.0"
    assert policy["locked_test_single_open"] is True
    assert policy["cloud_budget_status"] == "pending"
    assert policy["automatic_shutdown_minutes"] == 30
    assert policy["real_video_purposes"] == [
        "local_view",
        "annotation",
        "benchmark",
        "local_training",
        "cloud_training",
    ]


def test_identity_training_config_references_registered_backbone_assets() -> None:
    manifest = yaml.safe_load(ASSET_MANIFEST.read_text(encoding="utf-8"))
    config = yaml.safe_load(IDENTITY_TRAINING_CONFIG.read_text(encoding="utf-8"))
    assets = {asset["id"]: asset for asset in manifest["assets"]}

    referenced_ids = [config["backbone_asset_id"], *config["backbone_support_asset_ids"]]

    assert config["status"] == "group_head_development_short_training"
    assert None not in referenced_ids
    assert all(asset_id in assets for asset_id in referenced_ids)
    assert all(assets[asset_id]["approval_status"] == "pending" for asset_id in referenced_ids)
    assert assets[config["backbone_asset_id"]]["kind"] == "model_weight"
