import json
from pathlib import Path

from billiards_ai.lab.guidance_artifacts import publish_guidance_run


def test_publish_guidance_run_has_exact_inventory(tmp_path: Path) -> None:
    artifact = publish_guidance_run(
        tmp_path,
        inputs={"table_state_sha256": "a" * 64, "physics": "off"},
        outputs={
            "recommendation.json": {"status": "geometry_only"},
            "guidance-scene.json": {"coordinate_space": "table_mm"},
            "mobile-scene.json": {"routes": []},
        },
    )
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert {item["path"] for item in manifest["inventory"]} == {
        "guidance-scene.json",
        "mobile-scene.json",
        "recommendation.json",
    }
    assert artifact.name == manifest["artifact_id"]


def test_same_inputs_and_outputs_are_content_addressed_identically(tmp_path: Path) -> None:
    arguments = {
        "inputs": {"table_state_sha256": "a" * 64},
        "outputs": {"recommendation.json": {"status": "abstained"}},
    }
    first = publish_guidance_run(tmp_path, **arguments)
    second = publish_guidance_run(tmp_path, **arguments)
    assert first == second


def test_unsafe_output_name_is_rejected(tmp_path: Path) -> None:
    try:
        publish_guidance_run(
            tmp_path,
            inputs={},
            outputs={"../outside.json": {"bad": True}},
        )
    except ValueError as exc:
        assert "safe filename" in str(exc)
    else:
        raise AssertionError("unsafe artifact output was accepted")
