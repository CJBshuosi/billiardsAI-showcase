"""Unit tests for REAL_VIDEO_GATE state transitions and approval identity."""

import pwd
from datetime import UTC, datetime, timedelta
from inspect import signature
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.privacy import (
    FileApproverIdentityProvider,
    GateApprovalError,
    JsonGateRepository,
    RealVideoGateService,
)
from billiards_ai.cloud.privacy import service as gate_service_module
from billiards_ai.shared.models import GateRevision, RealVideoGateStatus


@pytest.mark.parametrize(
    "missing_item",
    [
        "table_view_only",
        "audio_disabled",
        "notice_posted",
        "operator_contact_available",
        "separate_service_consent",
        "separate_training_consent",
        "purpose_authorization_defined",
        "access_path_defined",
        "retention_path_defined",
        "deletion_path_defined",
        "marked_session_clips_only",
        "full_stream_stays_local",
    ],
)
def test_missing_checklist_item_cannot_be_approved(
    tmp_path,
    missing_item,
    gate_service_factory,
    gate_scope_factory,
    complete_checklist_factory,
    gate_evidence,
    now,
):
    gate_service = gate_service_factory(tmp_path)
    blocked = gate_service.create_gate(
        scope=gate_scope_factory(checklist=complete_checklist_factory(**{missing_item: False})),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
    )
    with pytest.raises(GateApprovalError):
        gate_service.approve(blocked.gate_id, "4" * 64)
    assert (
        gate_service.repository.get_current(blocked.gate_id).status is RealVideoGateStatus.BLOCKED
    )


def test_missing_evidence_cannot_be_approved(
    tmp_path, gate_service_factory, gate_scope_factory, now
):
    gate_service = gate_service_factory(tmp_path)
    blocked = gate_service.create_gate(
        scope=gate_scope_factory(), inspector_id="inspector-1", inspected_at=now, evidence=()
    )
    with pytest.raises(GateApprovalError):
        gate_service.approve(blocked.gate_id, "4" * 64)


def test_identity_comes_from_protected_credential_not_cli_string(
    tmp_path, gate_service_factory, gate_scope_factory, gate_evidence, now
):
    gate_service = gate_service_factory(tmp_path, approver_id="Mallory")
    blocked = gate_service.create_gate(
        scope=gate_scope_factory(),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
    )
    with pytest.raises(GateApprovalError):
        gate_service.approve(blocked.gate_id, "4" * 64)


def test_credential_must_be_regular_0600_file(tmp_path, credential_file_factory):
    credential = credential_file_factory(tmp_path)
    credential.chmod(0o644)
    provider = FileApproverIdentityProvider()
    with pytest.raises(GateApprovalError):
        provider.current_identity()

    credential.chmod(0o400)
    with pytest.raises(GateApprovalError):
        provider.current_identity()

    credential.chmod(0o600)
    target = tmp_path / "credential-target.json"
    target.write_text(credential.read_text())
    target.chmod(0o600)
    credential.unlink()
    credential.symlink_to(target)
    with pytest.raises(GateApprovalError):
        FileApproverIdentityProvider().current_identity()


def test_credential_path_cannot_be_chosen_by_caller():
    assert list(signature(FileApproverIdentityProvider).parameters) == []


def test_home_environment_cannot_redirect_approval_credential(
    tmp_path, monkeypatch, credential_file_factory
):
    attacker_home = tmp_path / "attacker-home"
    attacker_home.mkdir(mode=0o700)
    credential_file_factory(attacker_home)
    trusted_home = tmp_path / "trusted-account-home"
    trusted_home.mkdir(mode=0o700)
    monkeypatch.setattr(
        pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(trusted_home)),
    )
    monkeypatch.setenv("HOME", str(attacker_home))

    with pytest.raises(GateApprovalError):
        FileApproverIdentityProvider().current_identity()


def test_credential_rejects_unsafe_ancestor_directory(tmp_path, credential_file_factory):
    credential = credential_file_factory(tmp_path)
    credential.parent.parent.chmod(0o777)
    with pytest.raises(GateApprovalError):
        FileApproverIdentityProvider().current_identity()


def test_credential_id_outside_whitelist_is_rejected(
    tmp_path,
    credential_file_factory,
    gate_scope_factory,
    gate_evidence,
    now,
):
    credential_file_factory(tmp_path, credential_id="self-issued-by-a-developer")
    service = RealVideoGateService(JsonGateRepository(tmp_path / "gates.json"))
    blocked = service.create_gate(
        scope=gate_scope_factory(),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
    )
    with pytest.raises(GateApprovalError):
        service.approve(blocked.gate_id, "4" * 64)


def test_approval_cannot_predate_inspection(
    tmp_path, monkeypatch, gate_service_factory, gate_scope_factory, gate_evidence, now
):
    gate_service = gate_service_factory(tmp_path)
    blocked = gate_service.create_gate(
        scope=gate_scope_factory(),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
    )
    monkeypatch.setattr(gate_service_module, "_utc_now", lambda: now - timedelta(seconds=1))
    with pytest.raises(ValidationError):
        gate_service.approve(blocked.gate_id, "4" * 64)


def test_gate_inspection_cannot_be_unreasonably_far_in_the_future(
    tmp_path, gate_service_factory, gate_scope_factory, gate_evidence
):
    gate_service = gate_service_factory(tmp_path)
    with pytest.raises(ValidationError):
        gate_service.create_gate(
            scope=gate_scope_factory(),
            inspector_id="inspector-1",
            inspected_at=datetime(2999, 1, 1, tzinfo=UTC),
            evidence=gate_evidence,
        )


def test_gate_inspection_allows_bounded_clock_skew_without_weakening_approval_order(
    tmp_path,
    monkeypatch,
    gate_service_factory,
    gate_scope_factory,
    gate_evidence,
    now,
):
    from pydantic import ValidationError

    from billiards_ai.cloud.privacy import service as service_module
    from billiards_ai.shared.constants import PRIVACY_MAX_FUTURE_CLOCK_SKEW
    from billiards_ai.shared.models import privacy as privacy_models

    monkeypatch.setattr(service_module, "_utc_now", lambda: now)
    monkeypatch.setattr(privacy_models, "_utc_now", lambda: now)
    service = gate_service_factory(tmp_path)
    inspected_at = now + timedelta(seconds=30)
    blocked = service.create_gate(
        scope=gate_scope_factory(),
        inspector_id="device-clock-ahead",
        inspected_at=inspected_at,
        evidence=gate_evidence,
    )
    assert blocked.created_at == now
    assert blocked.inspected_at == inspected_at

    with pytest.raises(ValidationError):
        service.approve(blocked.gate_id, "4" * 64)

    monkeypatch.setattr(service_module, "_utc_now", lambda: inspected_at)
    monkeypatch.setattr(privacy_models, "_utc_now", lambda: inspected_at)
    approved = service.approve(blocked.gate_id, "4" * 64)
    assert approved.approval is not None
    assert approved.approval.approved_at == inspected_at

    with pytest.raises(ValidationError):
        service.create_gate(
            scope=gate_scope_factory(table_id="table-future"),
            inspector_id="device-clock-too-far-ahead",
            inspected_at=inspected_at + PRIVACY_MAX_FUTURE_CLOCK_SKEW + timedelta(seconds=1),
            evidence=gate_evidence,
        )


def test_revision_created_at_cannot_be_unreasonably_far_in_the_future(
    gate_scope_factory, gate_evidence, now
):
    with pytest.raises(ValidationError):
        GateRevision(
            gate_id=uuid4(),
            revision_id=uuid4(),
            revision_number=1,
            scope=gate_scope_factory(),
            inspector_id="inspector-1",
            inspected_at=now,
            evidence=gate_evidence,
            created_at=datetime(2999, 1, 1, tzinfo=UTC),
        )


def test_create_and_revoke_use_internal_revision_clock(
    tmp_path,
    monkeypatch,
    gate_service_factory,
    gate_scope_factory,
    gate_evidence,
    now,
):
    trusted_created_at = now + timedelta(seconds=1)
    monkeypatch.setattr(gate_service_module, "_utc_now", lambda: trusted_created_at)
    service = gate_service_factory(tmp_path)
    monkeypatch.setattr(gate_service_module, "_utc_now", lambda: trusted_created_at)
    blocked = service.create_gate(
        scope=gate_scope_factory(),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
    )
    assert blocked.created_at == trusted_created_at
    approved = service.approve(blocked.gate_id, "4" * 64)
    revised_at = trusted_created_at + timedelta(seconds=1)
    monkeypatch.setattr(gate_service_module, "_utc_now", lambda: revised_at)
    revised = service.revise(
        approved.gate_id,
        scope=gate_scope_factory(camera_config_id="top-view-v2"),
        inspector_id="inspector-2",
        inspected_at=now,
        evidence=gate_evidence,
        reason="camera moved",
    )
    assert revised.created_at == revised_at
    reapproved = service.approve(revised.gate_id, "5" * 64)
    revoked_at = revised_at + timedelta(seconds=1)
    monkeypatch.setattr(gate_service_module, "_utc_now", lambda: revoked_at)
    revoked = service.revoke(reapproved.gate_id, reason="notice removed")
    assert revoked.created_at == revoked_at


def test_approval_is_append_only_idempotent_and_complete(tmp_path, approved_gate_factory, now):
    gate_service, blocked, approved = approved_gate_factory(tmp_path)
    repeated = gate_service.approve(blocked.gate_id, "4" * 64)
    assert repeated.revision_id == approved.revision_id
    assert approved.status is RealVideoGateStatus.APPROVED
    assert approved.supersedes_revision_id == blocked.revision_id
    assert approved.approval is not None
    assert approved.approval.approver_id == "Boyu"
    assert approved.approval.gate_fingerprint == approved.scope.fingerprint
    assert len(gate_service.repository.get_history(blocked.gate_id)) == 2


def test_changed_scope_creates_new_blocked_revision_and_stales_old_approval(
    tmp_path, approved_gate_factory, gate_scope_factory, gate_evidence, now
):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    changed = gate_service.revise(
        approved.gate_id,
        scope=gate_scope_factory(camera_config_id="top-view-v2"),
        inspector_id="inspector-2",
        inspected_at=now,
        evidence=gate_evidence,
        reason="camera moved",
    )
    assert changed.status is RealVideoGateStatus.BLOCKED
    assert changed.supersedes_revision_id == approved.revision_id
    assert gate_service.repository.get_current(approved.gate_id).revision_id == changed.revision_id


def test_revocation_creates_new_blocked_revision(tmp_path, approved_gate_factory, now):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    revoked = gate_service.revoke(approved.gate_id, reason="notice removed")
    assert revoked.status is RealVideoGateStatus.BLOCKED
    assert revoked.invalidation_reason == "notice removed"


def test_revoked_gate_requires_fresh_inspection_before_reapproval(
    tmp_path, approved_gate_factory, now
):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    revoked = gate_service.revoke(approved.gate_id, reason="现场标识被撤除")
    with pytest.raises(GateApprovalError):
        gate_service.approve(revoked.gate_id, "4" * 64)
    assert gate_service.repository.get_current(revoked.gate_id).revision_id == revoked.revision_id


@pytest.mark.parametrize("reason", ["", " \t\n"])
def test_revoke_rejects_blank_invalidation_reason(tmp_path, approved_gate_factory, now, reason):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    with pytest.raises(ValidationError):
        gate_service.revoke(approved.gate_id, reason=reason)
