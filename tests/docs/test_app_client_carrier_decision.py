"""Preserve dated A2 contracts; current workflow authority is checked separately."""

from pathlib import Path

import yaml

DECISION = Path("_bmad-output/planning-artifacts/app-client-carrier-change-2026-08-10.md")

ARCHITECTURE_SPINE = Path(
    "_bmad-output/planning-artifacts/architecture/"
    "architecture-billiardsAI-2026-07-20/ARCHITECTURE-SPINE.md"
)

PRODUCT_ADDENDA = (
    Path("_bmad-output/planning-artifacts/prds/prd-billiardsAI-2026-07-19/addendum.md"),
    Path("_bmad-output/planning-artifacts/prds/prd-billiardsAI-mobile-2026-07-21/addendum.md"),
)

UX_AUTHORITIES = (
    Path("_bmad-output/planning-artifacts/ux-designs/ux-billiardsAI-2026-07-27/EXPERIENCE.md"),
    Path("_bmad-output/planning-artifacts/ux-designs/ux-billiardsAI-2026-07-27/DESIGN.md"),
)

EPICS = Path("_bmad-output/planning-artifacts/epics.md")
CORE_INTELLIGENCE_SPRINT_STATUS = Path(
    "_bmad-output/implementation-artifacts/core-intelligence-sprint-status.yaml"
)

DELIVERY_AND_TEST_PLAN = Path(
    "_bmad-output/planning-artifacts/architecture/"
    "core-intelligence-billiardsAI-2026-08-03/06-DELIVERY-AND-TEST-PLAN.md"
)

DEVELOPMENT_GUIDE = Path("DEVELOPMENT.md")
DETAILED_CARD_EXPLANATIONS = Path(
    "_bmad-output/planning-artifacts/architecture/"
    "core-intelligence-billiardsAI-2026-08-03/"
    "07-DETAILED-CARD-EXPLANATIONS.md"
)
FOUR_MAINLINES_DOT = Path(
    "_bmad-output/planning-artifacts/architecture/"
    "core-intelligence-billiardsAI-2026-08-03/"
    "FOUR-MAINLINES-DETAILED.dot"
)
FOUR_MAINLINES_SVG = FOUR_MAINLINES_DOT.with_suffix(".svg")
PROJECT_WORKFLOW = Path("PROJECT-WORKFLOW.md")
OPEN_QUESTIONS = Path("_bmad-output/planning-artifacts/OPEN-QUESTIONS.md")
ASYNC_SERVICE_FLOW = Path("_bmad-output/planning-artifacts/validation/ASYNC-SERVICE-FLOW.md")
ASYNC_BUILD_SPEC = Path(
    "_bmad-output/planning-artifacts/architecture/"
    "architecture-billiardsAI-2026-07-20/ASYNC-BUILD-SPEC.md"
)
BRAND_PROFILE = Path("docs/product/brand-profile.md")
SHAPING_DRAFT = Path("docs/product/shaping/billiards-ai-shaping-draft.md")

HISTORICAL_A2_ARCHITECTURE_SOURCES = (
    ARCHITECTURE_SPINE,
    Path(
        "_bmad-output/planning-artifacts/architecture/"
        "core-intelligence-billiardsAI-2026-08-03/01-FOUR-MAINLINES.md"
    ),
    Path(
        "_bmad-output/planning-artifacts/architecture/"
        "core-intelligence-billiardsAI-2026-08-03/04-REALTIME-COACH.md"
    ),
    Path(
        "_bmad-output/planning-artifacts/architecture/"
        "core-intelligence-billiardsAI-2026-08-03/05-CROSS-MAINLINE-INTERFACES.md"
    ),
    DELIVERY_AND_TEST_PLAN,
)

STALE_ACTIVE_CARRIER_RULES = (
    "participant APP as MiniProgram",
    "### WebSocket 给小程序",
    "所有视觉呈现在用户手机的小程序内完成",
    "边缘与小程序之间禁止直接通信",
    "手机直连仍按 Architecture Spine 禁止",
    "视觉输出的载体是小程序",
)

SUPERSEDED_AUTHORITY_LINKS = (
    "architecture/architecture-billiardsAI-2026-07-20/ARCHITECTURE-SPINE.md",
    "architecture/architecture-billiardsAI-2026-07-20/ASYNC-BUILD-SPEC.md",
    "prds/prd-billiardsAI-mobile-2026-07-21/prd.md",
    "ux-designs/ux-billiardsAI-2026-07-27/DESIGN.md",
    "ux-designs/ux-billiardsAI-2026-07-27/EXPERIENCE.md",
    "ux-designs/ux-billiardsAI-2026-07-27/validation-report.md",
    "epics.md",
    "implementation-readiness-report-2026-07-27-post-correction.md",
    "implementation-readiness-report-2026-07-27-rerun.md",
    "sprint-change-proposal-2026-07-27-readiness-corrections.md",
    "architecture/core-intelligence-billiardsAI-2026-08-03/01-FOUR-MAINLINES.md",
    "architecture/core-intelligence-billiardsAI-2026-08-03/04-REALTIME-COACH.md",
    "architecture/core-intelligence-billiardsAI-2026-08-03/05-CROSS-MAINLINE-INTERFACES.md",
    "architecture/core-intelligence-billiardsAI-2026-08-03/06-DELIVERY-AND-TEST-PLAN.md",
    "architecture/core-intelligence-billiardsAI-2026-08-03/CHANGE-PROPOSAL.md",
)

PRIVACY_GUARDRAILS = (
    "`REAL_VIDEO_GATE` 与逐主体同意缺一不可",
    "每位可识别主体必须对本次 `SERVICE_ANALYSIS` 用途给出有效同意",
    "真实视频、逐杆片段、配对码、流地址、凭据和可识别用户数据不得提交 Git",
    "开发配对码、调试开关和 synthetic `GRANTED` fixture 不能代替该授权",
    "禁止采集、录制、上传、回放、分析或在 App/Staff Web 中展示该真实素材",
)

PHASE_BOUNDARIES = (
    (
        "| A2.0 | 本决定及 Architecture/UX/Epic/Sprint 的载体纠偏 | "
        "不创建 App 工程、不改 `src/`、不改四主线契约、不改已完成 Story |"
    ),
    (
        "| A2.1 | Flutter 工程、质量门、端口、decoder、`PresentationCoordinator` 与原型 fixtures | "
        "不接真实视频、不接生产身份/会话 API、不接 CV/成长/Agent/生产 transport |"
    ),
    (
        "| A2.2 | 局域网真实视频模拟器、WebRTC、原生媒体适配、clock anchor、连接/重连 | "
        "不实现 Overlay 业务；真实人员素材仍受门禁约束 |"
    ),
    (
        "| A2.3 | 手动标定、Overlay、Replay、横竖屏、语音、免触控流程与安全降级 | "
        "原型轨迹仅限 App 内部 V0 类型，不得冒充主线 1/生产规划输出 |"
    ),
    (
        "| A2.4 | 本地分段录像、`shot_id` 索引、Playback API、回放与缓存失效 | "
        "不接生产对象存储、生产媒体签名 URL 或云端 outbox |"
    ),
)

ROLLBACK_GUARDRAILS = (
    "回退到最近已验证的 A2 阶段边界",
    "立即清除旧 Overlay 和实时 cue",
    "不进入 A2.3/A2.4 的交付承诺",
    "不以 App 缓存、店员 Web 或小程序绕过门禁",
    "任何 A2 失败都不会自动恢复小程序建设",
)


def test_client_carrier_decision_supersedes_mini_program_authorities() -> None:
    """The dated approved App decision retains its original correction contract."""
    content = DECISION.read_text(encoding="utf-8")

    assert "**状态：** 已批准" in content
    assert "**日期：** 2026-08-10" in content
    assert "User App: live coaching, replay, growth, plan, report, privacy" in content
    assert "Staff Web: table/session/media/operations" in content
    assert (
        "Mini Program: not in current build scope; future re-entry requires a new product decision"
        in content
    )

    assert "docs/superpowers/specs/2026-08-10-app-a2-real-video-coach-design.md" in content
    assert "architecture-billiardsAI-2026-07-20/ARCHITECTURE-SPINE.md" in content
    assert "prds/prd-billiardsAI-mobile-2026-07-21/prd.md" in content
    assert "ux-designs/ux-billiardsAI-2026-07-27/DESIGN.md" in content
    assert "ux-designs/ux-billiardsAI-2026-07-27/EXPERIENCE.md" in content
    assert "implementation-readiness-report-2026-07-27-post-correction.md" in content
    assert "epics.md" in content

    assert "四条核心智能主线的正式契约不变" in content
    assert "REAL_VIDEO_GATE" in content
    assert "SERVICE_ANALYSIS" in content
    assert "A2.0" in content
    assert "A2.1" in content
    assert "A2.2" in content
    assert "A2.3" in content
    assert "A2.4" in content
    assert "回退条件" in content
    assert "重新作出产品决策" in content


def test_client_carrier_decision_preserves_every_superseded_authority_link() -> None:
    content = DECISION.read_text(encoding="utf-8")

    for authority_link in SUPERSEDED_AUTHORITY_LINKS:
        assert authority_link in content


def test_client_carrier_decision_keeps_privacy_phase_and_rollback_guardrails() -> None:
    content = DECISION.read_text(encoding="utf-8")

    for guardrail in PRIVACY_GUARDRAILS:
        assert guardrail in content

    for phase_boundary in PHASE_BOUNDARIES:
        assert phase_boundary in content

    for guardrail in ROLLBACK_GUARDRAILS:
        assert guardrail in content


def test_historical_a2_architecture_preserves_video_and_authoritative_event_paths() -> None:
    """Historical A2 sources keep video transport separate from domain authority."""
    contents = {
        authority: authority.read_text(encoding="utf-8")
        for authority in HISTORICAL_A2_ARCHITECTURE_SOURCES
    }
    combined = "\n".join(contents.values())

    for content in contents.values():
        assert "app-client-carrier-change-2026-08-10.md" in content

    assert "Edge/Simulator -> local WebRTC -> App" in combined
    assert "RealtimeShotObservation → PlanExecutionPolicy → RealtimeCoachingEvent" in combined
    assert "生产权威事件仍走已批准的 realtime interface" in combined
    assert "App 不推断 `Shot`、`Growth` 或 `Plan`" in combined
    assert "`apps/mobile/`" in combined
    assert "`tools/app_video_simulator/`" in combined


def test_four_mainlines_source_tree_keeps_product_code_under_repository_root() -> None:
    """The architecture tree must not draw src beside the repository root."""
    content = HISTORICAL_A2_ARCHITECTURE_SOURCES[1].read_text(encoding="utf-8")

    assert "\n  apps/mobile/" in content
    assert "\n  tools/app_video_simulator/" in content
    assert "\n  src/billiards_ai/" in content
    assert "\nsrc/billiards_ai/" not in content


def test_historical_a2_architecture_rejects_mini_program_only_realtime_rules() -> None:
    """No active rule may make Mini Program the realtime carrier or ban App video."""
    combined = "\n".join(
        authority.read_text(encoding="utf-8") for authority in HISTORICAL_A2_ARCHITECTURE_SOURCES
    )

    for stale_rule in STALE_ACTIVE_CARRIER_RULES:
        assert stale_rule not in combined


def test_historical_a2_architecture_rejects_legacy_top_down_table_ui_language() -> None:
    """Realtime feedback and independent video must not collapse into a table UI."""
    content = ARCHITECTURE_SPINE.read_text(encoding="utf-8")

    assert "边缘产生权威反馈，经 approved realtime interface 交付 App" in content
    assert "真实视频独立走 local WebRTC" in content
    assert "推送手机俯视图" not in content
    assert "手机俯视图渲染载荷" not in content


def test_historical_a2_delivery_plan_preserves_app_reconnect() -> None:
    """The dated A2 delivery plan retains its App carrier and R8 label."""
    content = DELIVERY_AND_TEST_PLAN.read_text(encoding="utf-8")

    assert "app-client-carrier-change-2026-08-10.md" in content
    assert "| R8 | App 实时呈现与重连 |" in content
    assert "| R8 | 小程序实时呈现与重连 |" not in content


def test_developer_entry_points_and_detailed_cards_use_app_carrier() -> None:
    """Realtime App history remains valid while the mini program has a separate role."""
    development = DEVELOPMENT_GUIDE.read_text(encoding="utf-8").split("\n## 历史", 1)[1]
    cards = DETAILED_CARD_EXPLANATIONS.read_text(encoding="utf-8")
    dot = FOUR_MAINLINES_DOT.read_text(encoding="utf-8")
    svg = FOUR_MAINLINES_SVG.read_text(encoding="utf-8")

    assert "App A2.0 客户端载体变更决定" in development
    assert "miniprogram-client-reentry-2026-08-26.md" in development
    assert "mobile/       iOS/Android Flutter User App" in development
    assert "User App 与 Staff Web 最小闭环" in development

    assert "Story 10.8；A2 App分阶段实施" in cards
    assert "真实顶置视频＋训练指导Overlay" in cards
    assert "R8：接入User App" in cards
    assert "小程序实时教练页面接入" not in cards
    assert "断网时小程序反馈方案" not in cards
    assert "R8：接入小程序" not in cards

    expected = "RealtimeCoachingEvent → approved realtime interface → User App/语音"
    assert expected in dot
    assert "User App/语音" in svg
    assert "WebSocket → 小程序/语音" not in dot
    assert "WebSocket → 小程序/语音" not in svg


def test_current_workflow_references_restart_and_task_process() -> None:
    workflow = PROJECT_WORKFLOW.read_text(encoding="utf-8").split("\n## 历史", 1)[0]
    assert "[PROJECT-RESTART.md](PROJECT-RESTART.md)" in workflow
    assert "[AGENTS.md](AGENTS.md)" in workflow
    assert "apps/mobile/" in workflow
    assert "apps/staff_web/" in workflow


def test_historical_a2_async_guides_preserve_client_and_privacy_boundaries() -> None:
    questions = OPEN_QUESTIONS.read_text(encoding="utf-8")
    flow = ASYNC_SERVICE_FLOW.read_text(encoding="utf-8")
    build = ASYNC_BUILD_SPEC.read_text(encoding="utf-8")
    brand = BRAND_PROFILE.read_text(encoding="utf-8")
    shaping = SHAPING_DRAFT.read_text(encoding="utf-8")

    assert "微信小程序主体（个人/企业）" not in questions
    assert "User App生产身份提供方与商店签名主体" in questions
    assert "客人扫码进小程序" not in flow
    assert "店员端是小程序的一个页面" not in flow
    assert "需要小程序（最小版）" not in flow
    assert "报告推送小程序" not in build
    assert "小程序↔云端 WS+REST" not in build
    assert "User App/Staff Web↔云端 HTTPS/WSS" in build
    assert "小程序/H5：偏正式" not in brand
    assert "历史构形草案" in shaping
    assert "小程序退出当前建设范围" in shaping


def test_real_video_overlay_requires_port_side_presentation_calibration() -> None:
    """Live-video overlays fail closed when presentation mapping is unproven."""
    content = ARCHITECTURE_SPINE.read_text(encoding="utf-8")

    assert "presentation calibration 位于 App 端口之后" in content
    assert "视频、权威事件与 presentation calibration" in content
    assert "清除旧 Overlay 和实时 cue" in content
    assert "不进入共享或四主线生产契约" in content
    assert "渲染层不需要自己的坐标系，也不需要标定" not in content


def test_historical_a2_er_has_no_provider_specific_identity_authority() -> None:
    """The domain ER keeps only provider-neutral internal user identity authority."""
    content = ARCHITECTURE_SPINE.read_text(encoding="utf-8")

    assert "User.id 是领域关系使用的唯一内部用户身份" in content
    assert "string openid" not in content
    assert "string unionid" not in content


def test_historical_a2_prd_addenda_preserve_app_carrier_decision() -> None:
    """Each current PRD layer overrides only its stale client-carrier rules."""
    for addendum in PRODUCT_ADDENDA:
        content = addendum.read_text(encoding="utf-8")

        assert "## A5. Active Addendum — App A2.0 客户端载体纠偏" in content
        assert "app-client-carrier-change-2026-08-10.md" in content
        assert "User App: live coaching, replay, growth, plan, report, privacy" in content
        assert "Staff Web: table/session/media/operations" in content
        assert (
            "Mini Program: not in current build scope; future re-entry requires a "
            "new product decision" in content
        )
        assert "旧 PRD 正文保留为历史基线" in content
        assert "不改写隐私、异步服务行为或四条核心智能主线契约" in content


def test_historical_a2_ux_preserves_carrier_and_accessibility() -> None:
    """New App UX ignores platform-specific widgets without losing semantics."""
    accessibility_semantics = (
        "系统字号放大",
        "状态不得只靠颜色表达",
        "焦点与读屏顺序",
        "reduced motion",
    )

    for authority in UX_AUTHORITIES:
        content = authority.read_text(encoding="utf-8")

        assert "App A2.0 Active Addendum" in content
        assert "app-client-carrier-change-2026-08-10.md" in content
        assert "User App: live coaching, replay, growth, plan, report, privacy" in content
        assert "Staff Web: table/session/media/operations" in content
        assert "WeChat/WeUI 平台与组件规则对新的客户端工作已被取代" in content
        for semantic in accessibility_semantics:
            assert semantic in content


def test_corrected_authorities_record_the_app_decision_audit_date() -> None:
    """Dated Epic, Sprint, and UX metadata retain the A2.0 correction."""
    epics_content = EPICS.read_text(encoding="utf-8")
    assert "lastUpdated: 2026-08-10" in epics_content

    sprint_content = CORE_INTELLIGENCE_SPRINT_STATUS.read_text(encoding="utf-8")
    sprint_status = yaml.safe_load(sprint_content)
    assert sprint_status["last_updated"].date().isoformat() >= "2026-08-10"
    assert any(line.startswith("# last_updated: 2026-08-") for line in sprint_content.splitlines())

    for authority in UX_AUTHORITIES:
        content = authority.read_text(encoding="utf-8")
        metadata = yaml.safe_load(content.split("---", maxsplit=2)[1])
        assert metadata["updated"].isoformat() == "2026-08-10"
        assert "../../app-client-carrier-change-2026-08-10.md" in metadata["sources"]

    design_content = UX_AUTHORITIES[1].read_text(encoding="utf-8")
    design_metadata = yaml.safe_load(design_content.split("---", maxsplit=2)[1])
    assert "App 优先" in design_metadata["description"]
    assert "继承微信小程序与 WeUI 平台规范" not in design_metadata["description"]


def test_story_10_8_and_sprint_track_app_reconnect_fail_closed() -> None:
    """Story and sprint authority name the App behavior and retain backlog state."""
    content = EPICS.read_text(encoding="utf-8")
    story_10_8 = content.split("### Story 10.8:", maxsplit=1)[1].split(
        "### Story 10.9:", maxsplit=1
    )[0]

    assert "app-client-carrier-change-2026-08-10.md" in content
    assert "### Story 10.8: App实时呈现与重连" in content
    assert "### Story 10.8: 小程序实时呈现与重连" not in content
    assert "**Given** WebSocket断开重连" in story_10_8
    assert "runtime snapshot与后续sequence" in story_10_8
    assert "去重、补齐或明确重置" in story_10_8
    assert "freshness" in story_10_8

    sprint_status = yaml.safe_load(CORE_INTELLIGENCE_SPRINT_STATUS.read_text(encoding="utf-8"))
    development_status = sprint_status["development_status"]
    assert development_status["10-8-App实时呈现与重连"] == "backlog"
    assert "10-8-小程序实时呈现与重连" not in development_status


def test_epic_addendum_distinguishes_retained_history_from_corrected_story() -> None:
    """The addendum must not claim the corrected Story 10.8 is old history."""
    content = EPICS.read_text(encoding="utf-8")
    active_addendum = content.split("## Active Addendum — App A2.0 客户端载体纠偏", maxsplit=1)[
        1
    ].split("## Requirements Inventory", maxsplit=1)[0]

    assert "Story 10.8 的原小程序载体措辞已在当前 Story 中纠偏" in active_addendum
    assert "Story 10.8 绑定微信/小程序的内容保留为历史基线" not in active_addendum


def test_story_10_8_privacy_failure_forbids_real_media_processing() -> None:
    """Privacy failure prohibits media handling without absorbing consistency faults."""
    story_10_8 = (
        EPICS.read_text(encoding="utf-8")
        .split("### Story 10.8:", maxsplit=1)[1]
        .split("### Story 10.9:", maxsplit=1)[0]
    )
    privacy_given = (
        "**Given** `REAL_VIDEO_GATE` 未通过，或任一可识别主体缺少本次 `SERVICE_ANALYSIS` 同意"
    )
    consistency_given = (
        "**Given** `REAL_VIDEO_GATE` 与逐主体同意均有效，但视频、事件、标定、"
        "`stream_generation` 或 `session_timebase_id` 无法证明一致"
    )

    assert privacy_given in story_10_8
    assert consistency_given in story_10_8
    privacy_ac = story_10_8.split(privacy_given, maxsplit=1)[1].split(
        consistency_given, maxsplit=1
    )[0]
    assert "fail closed" in privacy_ac
    assert "禁止采集、录制、上传、回放、分析或在 App/Staff Web 中展示该真实素材" in privacy_ac
    assert "`stream_generation`" not in privacy_ac
    assert "`session_timebase_id`" not in privacy_ac


def test_story_10_8_consistency_failure_clears_only_live_guidance() -> None:
    """Authorized but inconsistent media clears guidance without banning media handling."""
    story_10_8 = (
        EPICS.read_text(encoding="utf-8")
        .split("### Story 10.8:", maxsplit=1)[1]
        .split("### Story 10.9:", maxsplit=1)[0]
    )
    consistency_given = (
        "**Given** `REAL_VIDEO_GATE` 与逐主体同意均有效，但视频、事件、标定、"
        "`stream_generation` 或 `session_timebase_id` 无法证明一致"
    )

    assert consistency_given in story_10_8
    consistency_ac = story_10_8.split(consistency_given, maxsplit=1)[1]
    assert "仅停止并清除旧 Overlay 和实时 cue" in consistency_ac
    assert "只显示安全文字状态或停止训练" in consistency_ac
    assert "不得以猜测继续指导" in consistency_ac
    for prohibited_operation in (
        "禁止采集",
        "禁止录制",
        "禁止上传",
        "禁止回放",
        "禁止分析",
        "禁止展示",
    ):
        assert prohibited_operation not in consistency_ac
