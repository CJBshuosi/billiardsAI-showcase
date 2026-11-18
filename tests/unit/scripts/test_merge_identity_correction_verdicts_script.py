from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from billiards_ai.lab.training.identity_review import IdentityReviewSession, _verify_verdict
from tests.unit.lab.training.test_identity_review import _session


def test_merge_correction_cli_publishes_overlay_wins_artifact(tmp_path: Path) -> None:
    original = _session(tmp_path)
    crop_id = original.crop_ids[0]
    old = IdentityReviewSession.open(
        candidate_artifact=original.candidate_artifact,
        state_path=tmp_path / "old.json",
        crop_ids=(crop_id,),
    )
    old.record(
        crop_id=crop_id,
        group="cue",
        exact_identity="cue",
        exact_evidence="group_intrinsic",
        visibility="visible",
        orientation="number_visible",
    )
    old_verdict = old.export_verdict(tmp_path / "old-verdicts")
    new = IdentityReviewSession.open(
        candidate_artifact=original.candidate_artifact,
        state_path=tmp_path / "new.json",
        crop_ids=(crop_id,),
    )
    new.record(
        crop_id=crop_id,
        group="stripe",
        exact_identity=None,
        exact_evidence=None,
        visibility="partial",
        orientation="number_hidden",
    )
    new_verdict = new.export_verdict(tmp_path / "new-verdicts")
    output_root = tmp_path / "merged"
    script = (
        Path(__file__).parents[3]
        / "scripts"
        / "algorithm"
        / "merge_identity_correction_verdicts.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--candidate-artifact",
            str(original.candidate_artifact),
            "--base-correction-verdict-artifact",
            str(old_verdict),
            "--overlay-correction-verdict-artifact",
            str(new_verdict),
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    merged = Path(result["output"])
    decisions = _verify_verdict(merged, candidate_id=original.candidate_id)
    assert result["status"] == "PASS"
    assert decisions[0].group == "stripe"
