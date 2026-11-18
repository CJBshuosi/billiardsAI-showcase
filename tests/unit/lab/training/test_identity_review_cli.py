from __future__ import annotations

import json
from pathlib import Path

from billiards_ai.lab.training.identity_crop_export import export_identity_crop_candidates
from billiards_ai.lab.training.identity_review_cli import (
    run_existing_identity_review_cli,
    run_identity_review_cli,
)
from tests.unit.lab.training.test_identity_crop_export import _bootstrap_dataset


class _FakeServer:
    server_port = 8767

    def __init__(self) -> None:
        self.served = False
        self.closed = False

    def serve_forever(self) -> None:
        self.served = True

    def server_close(self) -> None:
        self.closed = True


def test_cli_prepares_candidates_and_starts_resumable_loopback_review(
    tmp_path: Path, capsys
) -> None:
    dataset = _bootstrap_dataset(tmp_path)
    fake = _FakeServer()

    exit_code = run_identity_review_cli(
        bootstrap_dataset=dataset,
        output_root=tmp_path / "identity-review",
        port=8767,
        open_browser=False,
        server_factory=lambda **_: fake,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert fake.served is True and fake.closed is True
    assert payload["status"] == "READY"
    assert payload["crop_count"] == 2
    assert Path(payload["state_path"]).parent.name == "states"


def test_cli_can_review_an_existing_representative_candidate_artifact(
    tmp_path: Path, capsys
) -> None:
    candidates = export_identity_crop_candidates(
        bootstrap_dataset=_bootstrap_dataset(tmp_path),
        output_root=tmp_path / "candidates",
        crop_size_px=64,
    )
    fake = _FakeServer()

    exit_code = run_existing_identity_review_cli(
        candidate_artifact=candidates.path,
        output_root=tmp_path / "identity-review",
        port=8767,
        open_browser=False,
        server_factory=lambda **_: fake,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["candidate_id"] == candidates.artifact_id
    assert payload["crop_count"] == 2
