from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.algorithm.replay_lightweight_realtime_chain import main


def test_cli_passes_exact_user_session_and_paths_to_replay(tmp_path: Path, capsys) -> None:
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            artifact_id="result-1",
            path=tmp_path / "result-1",
            shot_count=2,
        )

    result = main(
        [
            "--source-artifact",
            str(tmp_path / "source"),
            "--output-root",
            str(tmp_path / "output"),
            "--user-id",
            "user-1",
            "--session-id",
            "session-43",
        ],
        runner=runner,
    )

    assert result == 0
    assert calls == [
        {
            "source_artifact": tmp_path / "source",
            "output_root": tmp_path / "output",
            "user_id": "user-1",
            "session_id": "session-43",
        }
    ]
    assert '"status":"PASS"' in capsys.readouterr().out


def test_cli_reports_validation_failure_without_traceback(tmp_path: Path, capsys) -> None:
    def runner(**kwargs):
        raise ValueError("source is invalid")

    result = main(
        [
            "--source-artifact",
            str(tmp_path / "source"),
            "--output-root",
            str(tmp_path / "output"),
            "--user-id",
            "user-1",
            "--session-id",
            "session-43",
        ],
        runner=runner,
    )

    assert result == 2
    assert "source is invalid" in capsys.readouterr().err
