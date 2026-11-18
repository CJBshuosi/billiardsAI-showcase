from pathlib import Path


def test_windows_group_error_review_launcher_is_loopback_and_cuda_only() -> None:
    script = (
        Path(__file__).parents[3]
        / "scripts"
        / "algorithm"
        / "windows"
        / "review-identity-group-errors.ps1"
    ).read_text(encoding="utf-8")

    assert '"--candidate-artifact", $CandidateArtifact' in script
    assert '"--verdict-artifact", $VerdictArtifact' in script
    assert '"--correction-verdict-artifact", $CorrectionVerdictArtifact' in script
    assert '"--training-run", $TrainingRun' in script
    assert '"--device", "cuda"' in script
    assert "--open-browser" in script
    assert '"--port", $Port' in script
