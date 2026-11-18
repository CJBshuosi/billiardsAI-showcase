from __future__ import annotations

from pathlib import Path

SCRIPT = Path("scripts/algorithm/windows/fetch-dinov2-identity-assets.ps1")


def test_windows_fetcher_freezes_and_verifies_all_approved_dinov2_payloads() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "ed25f3a31f01632728cabb09d1542f84ab7b0056" in source
    assert "88249960" in source
    assert "547" in source
    assert "436" in source
    assert "88250943" in source
    assert "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1" in source
    assert "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d" in source
    assert "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828" in source
    assert "Get-FileHash -Algorithm SHA256" in source
    assert "Move-Item -LiteralPath $Partial -Destination $Target" in source
    assert "AutoModel" not in source
    assert "torch" not in source.lower()
