from __future__ import annotations

from billiards_ai.lab.field_assistance.pages import (
    render_screen_page,
    render_tablet_page,
)


def test_screen_polls_live_state_and_limits_writes_to_postgame_review() -> None:
    page = render_screen_page()

    assert 'id="route-frame"' in page
    assert "'/media/frame/'" in page
    assert "'/api/screen'" in page
    assert "state.frame.frame_id" in page
    assert "route-choice" not in page
    assert "ball-coordinate" not in page
    assert "<input" not in page
    assert "'/api/control'" not in page
    assert "X-Control-Token" not in page
    assert "data-command=" not in page
    assert 'id="postgame-panel" hidden' in page
    assert 'id="postgame-finish" type="button" disabled' in page
    assert 'id="postgame-retry" type="button"' in page
    assert "'/api/postgame-review/save'" in page
    assert "'/api/postgame-review/complete'" in page
    assert "X-Kiosk-Nonce" in page
    assert 'id="table-code"' in page
    assert 'id="join-prompt"' in page
    assert 'id="red-slot"' in page
    assert 'id="blue-slot"' in page
    assert "state.display.join_prompt" in page
    assert "state.display.red_slot" in page
    assert "state.display.blue_slot" in page
    assert "/api/table-code.png" in page
    assert "!state.match.red_user_id||!state.match.blue_user_id" in page
    assert "control_token" not in page


def test_screen_keeps_match_identity_and_groups_visible_after_setup() -> None:
    page = render_screen_page()

    assert 'id="match-strip"' in page
    assert 'id="red-player"' in page
    assert 'id="red-group"' in page
    assert 'id="blue-player"' in page
    assert 'id="blue-group"' in page
    assert 'id="last-action"' in page
    assert "state.match.red_group" in page
    assert "state.match.blue_group" in page
    assert "state.display.last_action" in page
    assert "state.display.phase==='WAITING_FOR_PLAYERS'" not in page
    assert "state.join.error" in page
    assert 'alt="球桌小程序码"' in page


def test_tablet_exposes_only_replay_and_audited_prestart_emergency_controls() -> None:
    page = render_tablet_page()

    assert "'/api/tablet'" in page
    assert "'/api/control'" in page
    assert "X-Control-Token" in page
    assert 'data-command="set_participant"' not in page
    assert 'data-command="set_breaker"' not in page
    assert 'data-command="assign_groups"' not in page
    assert 'data-command="continue_turn"' not in page
    assert 'data-command="switch_turn"' not in page
    assert 'data-command="start"' in page
    assert 'data-command="pause"' in page
    assert 'data-command="reset"' in page
    assert 'data-command="end_session"' not in page
    assert 'data-command="create_join_qr"' not in page
    assert 'data-command="refresh_join_requests"' not in page
    assert "confirm_participant" not in page
    assert 'id="emergency-reason"' in page
    assert 'data-command="emergency_set_participant"' in page
    assert 'data-command="emergency_set_breaker"' in page
    assert 'data-command="emergency_assign_groups"' in page


def test_tablet_labels_controls_as_administrator_emergency_only() -> None:
    page = render_tablet_page()

    assert "管理员应急修正" in page
    assert "红方临时参与者 ID" in page
    assert "蓝方临时参与者 ID" in page
    assert "扫码加入本场" not in page
    assert "结束本场并生成事实报告" not in page
    assert "逐杆回放" not in page
    assert "收藏" not in page
    assert "下载" not in page
    assert "分享" not in page


def test_tablet_does_not_render_pending_phone_scans() -> None:
    page = render_tablet_page()

    assert "request.display_name" not in page
    assert "request.user_id} · 等待分配" not in page


def test_tablet_does_not_offer_route_or_ball_position_editing() -> None:
    page = render_tablet_page()

    assert "Top-3" not in page
    assert "选择路线" not in page
    assert "修正球位" not in page
    assert "ball-coordinate" not in page


def test_tablet_reports_ended_session_and_locks_field_controls() -> None:
    page = render_tablet_page()

    assert "本场已结束，事实报告已生成" in page
    assert "state.session.status==='ended'" in page
    assert "button.disabled=ended" in page
