import json
from pathlib import Path

import pytest

from billiards_ai.lab.cli import _parser
from billiards_ai.lab.runner import RunInputs, create_artifact_run


class SyntheticRenderer:
    name = "synthetic-renderer"
    version = "1.0"

    def render(self, *, output_path: Path, overlay_frames: tuple[dict, ...]) -> None:
        output_path.write_bytes(json.dumps(overlay_frames, sort_keys=True).encode())


class FailingRenderer(SyntheticRenderer):
    def render(self, *, output_path: Path, overlay_frames: tuple[dict, ...]) -> None:
        output_path.write_bytes(b"partial")
        raise RuntimeError("synthetic render failure")


def _inputs() -> RunInputs:
    return RunInputs(
        dataset_revision="dataset-r1",
        candidate_id="candidate-a",
        manifest={"seed": 7, "code_commit": "abc123"},
        predictions=(
            {
                "shot_id": "shot-1",
                "events": [
                    {
                        "event_id": "event-1",
                        "type": "first_contact",
                        "timestamp_ms": 120,
                        "supported": True,
                        "confidence": 0.9,
                        "evidence_frame_ids": [3],
                    }
                ],
            },
        ),
        metrics={"first_hit": {"accuracy": 1.0, "denominator": 1}},
        failure_records=(),
        evidence_frames={3: {"timestamp_ms": 120, "labels": ["cue", "8"]}},
    )


def test_run_creates_content_addressed_immutable_artifacts(tmp_path: Path) -> None:
    run_dir = create_artifact_run(tmp_path / "runs", _inputs(), SyntheticRenderer())

    assert run_dir.parent.name == "candidate-a"
    assert run_dir.parent.parent.name == "dataset-r1"
    assert len(run_dir.name) == 64
    assert {
        "manifest.json",
        "predictions.jsonl",
        "metrics.json",
        "report.html",
        "overlay.mp4",
        "failures",
    } <= {path.name for path in run_dir.iterdir()}
    assert not (run_dir.stat().st_mode & 0o222)


def test_failed_run_does_not_overwrite_prior_run(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    original = create_artifact_run(root, _inputs(), SyntheticRenderer())
    original_overlay = (original / "overlay.mp4").read_bytes()

    assert create_artifact_run(root, _inputs(), FailingRenderer()) == original
    assert (original / "overlay.mp4").read_bytes() == original_overlay

    changed = _inputs().model_copy(update={"candidate_id": "candidate-b"})
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        create_artifact_run(root, changed, FailingRenderer())
    assert not list((root / "dataset-r1" / "candidate-b").glob("*"))


def test_lab_cli_exposes_the_planned_workbench_commands() -> None:
    help_text = _parser().format_help()

    for command in ("ingest", "run", "benchmark", "render", "failures"):
        assert command in help_text


def test_run_identifiers_cannot_escape_the_runs_root() -> None:
    payload = _inputs().model_dump()
    payload["candidate_id"] = "../../outside"
    with pytest.raises(ValueError):
        RunInputs.model_validate(payload)
