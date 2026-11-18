"""Regression of the approved 2026-08-26 decision, not current delivery scope."""

from pathlib import Path

REENTRY = Path("_bmad-output/planning-artifacts/miniprogram-client-reentry-2026-08-26.md")
DEVELOPMENT = Path("DEVELOPMENT.md")
ARCHITECTURE_SPINE = Path(
    "_bmad-output/planning-artifacts/architecture/"
    "architecture-billiardsAI-2026-07-20/ARCHITECTURE-SPINE.md"
)
DESIGN = "docs/superpowers/specs/2026-08-26-wechat-miniprogram-long-term-user-client-design.md"


def test_miniprogram_reentry_preserves_approved_2026_08_26_history() -> None:
    content = REENTRY.read_text(encoding="utf-8")

    assert "**状态：** 已批准" in content
    assert "Mini Program: identity, session joining, facts reports, clips" in content
    assert "does not carry realtime video, route overlay, CV, or field control" in content
    assert "App A2 history remains valid and implementation is paused" in content
    assert DESIGN in content


def test_historical_development_entry_points_reference_reentry() -> None:
    decision = "miniprogram-client-reentry-2026-08-26.md"

    assert decision in DEVELOPMENT.read_text(encoding="utf-8").split("\n## 历史", 1)[1]
    assert decision in ARCHITECTURE_SPINE.read_text(encoding="utf-8")


def test_historical_architecture_preserves_2026_08_client_boundaries() -> None:
    content = (
        ARCHITECTURE_SPINE.read_text(encoding="utf-8")
        .split("\n## 历史客户端边界", 1)[1]
        .split("\n## Design Paradigm", 1)[0]
    )

    assert "小程序：身份、扫码入场、事实报告、击球片段、收藏与受控分享" in content
    assert "投影/全屏页：现场路线显示" in content
    assert "平板：红蓝角色与比赛状态控制" in content
    assert "雷神现场程序：感知、路线、事实、片段与 outbox 权威" in content
    assert "小程序不承载实时视频、路线 Overlay、CV 或现场控制" in content
