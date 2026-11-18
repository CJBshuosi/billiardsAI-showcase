"""Contract checks for security-critical operator guidance."""

from pathlib import Path


def test_runbook_forbids_uncoordinated_shared_registry_rollback():
    project_root = Path(__file__).resolve().parents[2]
    runbook = (project_root / "docs/engineering/privacy/real-video-gate.md").read_text(
        encoding="utf-8"
    )

    assert "全局停止所有共享该 registry/key 的 store writer" in runbook
    assert "逐 namespace 恢复" in runbook
