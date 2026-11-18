from __future__ import annotations

import json
from pathlib import Path

import pytest

from billiards_ai.lab.propagation import InstancePrompt, PropagationWindow
from billiards_ai.lab.propagation.artifacts import write_propagation_run
from billiards_ai.lab.propagation.fake import FakePropagationBackend


def _result():
    window = PropagationWindow(
        asset_id="video-1",
        ordinal=0,
        start_frame=10,
        end_frame=13,
    )
    prompt = InstancePrompt(
        asset_id="video-1",
        frame_index=10,
        instance_id="ball-1",
        class_name="ball",
        source_kind="gold",
        source_id="gold-10",
        box_xyxy_px=(10.0, 20.0, 20.0, 30.0),
    )
    return FakePropagationBackend(fail_at_frame=12).propagate(
        window=window,
        prompts=(prompt,),
    )


def test_writer_creates_deterministic_immutable_artifacts(tmp_path: Path) -> None:
    manifest = {
        "code_commit": "abc123",
        "checkpoint_sha256": "none-fake-backend",
        "input_descriptor_sha256": "a" * 64,
    }

    first = write_propagation_run(root=tmp_path / "runs", result=_result(), manifest=manifest)
    second = write_propagation_run(root=tmp_path / "runs", result=_result(), manifest=manifest)

    assert first == second
    assert len(first.name) == 64
    assert not (first.stat().st_mode & 0o222)
    assert {path.name for path in first.iterdir()} == {
        "failures.jsonl",
        "manifest.json",
        "observations.jsonl",
        "prompts.jsonl",
    }
    stored_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert stored_manifest["backend"] == {"id": "fake-propagation", "version": "1.0"}
    assert stored_manifest["content_hash"] == first.name
    assert stored_manifest["input_descriptor_sha256"] == "a" * 64


def test_writer_orders_jsonl_and_preserves_explicit_failure(tmp_path: Path) -> None:
    run = write_propagation_run(root=tmp_path / "runs", result=_result(), manifest={"seed": 7})

    observations = [
        json.loads(line)
        for line in (run / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failures = [
        json.loads(line)
        for line in (run / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [(item["frame_index"], item["instance_id"]) for item in observations] == [
        (10, "ball-1"),
        (11, "ball-1"),
    ]
    assert failures == [
        {
            "asset_id": "video-1",
            "code": "synthetic_failure",
            "frame_index": 12,
            "message": "fake backend stopped at the configured frame",
        }
    ]


def test_writer_removes_temporary_directory_when_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from billiards_ai.lab.propagation import artifacts

    def fail_publish(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic publish failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_publish)
    root = tmp_path / "runs"

    with pytest.raises(OSError, match="synthetic publish failure"):
        write_propagation_run(root=root, result=_result(), manifest={"seed": 7})

    assert not list(root.glob(".tmp-*"))
    assert not [path for path in root.iterdir() if path.is_dir()]
