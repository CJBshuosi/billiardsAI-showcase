"""Filesystem integration tests for the versioned gate repository."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from billiards_ai.cloud.privacy import (
    GateConflictError,
    GateRepositoryError,
    JsonGateRepository,
    RealVideoGateService,
)
from billiards_ai.cloud.privacy.integrity import GateHeadRegistry
from billiards_ai.shared.models import GateApproval, GateRevision, RealVideoGateStatus


def test_round_trip_permissions_and_unknown_schema_fail_closed(tmp_path, approved_gate_factory):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    store = gate_service.repository.path
    assert store.stat().st_mode & 0o777 == 0o600
    loaded = gate_service.repository.get_current(approved.gate_id)
    assert loaded.model_dump(mode="json") == approved.model_dump(mode="json")

    payload = json.loads(store.read_text())
    payload["schema_version"] = "999"
    store.write_text(json.dumps(payload))
    store.chmod(0o600)
    with pytest.raises(GateRepositoryError):
        gate_service.repository.get_current(approved.gate_id)


def test_repository_canonicalizes_relative_store_path_at_construction(
    tmp_path,
    monkeypatch,
    credential_file_factory,
    gate_scope_factory,
    gate_evidence,
    now,
):
    origin = tmp_path / "origin"
    elsewhere = tmp_path / "elsewhere"
    origin.mkdir(mode=0o700)
    elsewhere.mkdir(mode=0o700)
    credential_file_factory(tmp_path)
    monkeypatch.chdir(origin)
    repository = JsonGateRepository(Path("private/gates.json"))
    expected_path = (origin / "private/gates.json").resolve()
    assert repository.path == expected_path

    service = RealVideoGateService(repository)
    blocked = service.create_gate(
        scope=gate_scope_factory(),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
    )
    monkeypatch.chdir(elsewhere)

    assert repository.path == expected_path
    assert repository.get_current(blocked.gate_id).revision_id == blocked.revision_id


@pytest.mark.parametrize("invalid_digest", [None, 42, "not-a-digest", "A" * 64])
def test_migration_api_rejects_invalid_expected_digest_with_repository_error(
    tmp_path, invalid_digest
):
    store = tmp_path / "legacy.json"
    store.write_text('{"schema_version":"1.0","gates":{}}')
    store.chmod(0o600)

    with pytest.raises(GateRepositoryError, match="expected store SHA-256"):
        JsonGateRepository(store).migrate_legacy_store(
            expected_store_sha256=invalid_digest,
            dry_run=True,
        )


def test_credential_key_ids_are_normalized_before_approval_and_verification(
    tmp_path,
    credential_file_factory,
    gate_scope_factory,
    gate_evidence,
    now,
):
    credential_file_factory(
        tmp_path,
        active_key_id=" key-a ",
        keys={" key-a ": "a" * 64},
    )
    service = RealVideoGateService(JsonGateRepository(tmp_path / "gates.json"))
    blocked = service.create_gate(
        scope=gate_scope_factory(),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
    )
    approved = service.approve(blocked.gate_id, "4" * 64)

    assert approved.approval is not None
    assert approved.approval.key_id == "key-a"
    assert service.repository.get_current(blocked.gate_id).revision_id == approved.revision_id


@pytest.mark.parametrize(
    ("active_key_id", "keys"),
    [(" ", {" ": "a" * 64}), ("\t", {" ": "a" * 64, "\t": "b" * 64})],
)
def test_credential_rejects_whitespace_only_key_id(
    tmp_path, credential_file_factory, active_key_id, keys
):
    from billiards_ai.cloud.privacy import GateApprovalError
    from billiards_ai.cloud.privacy.identity import FileApproverIdentityProvider

    credential_file_factory(tmp_path, active_key_id=active_key_id, keys=keys)

    with pytest.raises(GateApprovalError):
        FileApproverIdentityProvider().current_identity()


@pytest.mark.parametrize(
    "keys",
    [
        {"key-a": "a" * 64, " key-a ": "b" * 64},
        {" key-a": "a" * 64, "key-a ": "b" * 64},
    ],
)
def test_credential_rejects_key_ids_that_collide_after_strip(
    tmp_path, credential_file_factory, keys
):
    from billiards_ai.cloud.privacy import GateApprovalError
    from billiards_ai.cloud.privacy.identity import FileApproverIdentityProvider

    credential_file_factory(tmp_path, active_key_id="key-a", keys=keys)

    with pytest.raises(GateApprovalError):
        FileApproverIdentityProvider().current_identity()


@pytest.mark.parametrize(
    ("active_key_id", "key_id"),
    [(" key-a ", "key-a"), ("key-a", " key-a ")],
)
def test_credential_allows_one_unambiguous_key_id_after_strip(
    tmp_path, credential_file_factory, active_key_id, key_id
):
    from billiards_ai.cloud.privacy.identity import FileApproverIdentityProvider

    credential_file_factory(
        tmp_path,
        active_key_id=active_key_id,
        keys={key_id: "a" * 64},
    )

    identity = FileApproverIdentityProvider().current_identity()
    assert identity.active_key_id == "key-a"
    assert set(identity.approval_hmac_keys) == {"key-a"}


def test_corrupt_insecure_or_symlink_store_fails_closed(tmp_path):
    store = tmp_path / "gates.json"
    store.write_text("not-json")
    store.chmod(0o600)
    with pytest.raises(GateRepositoryError):
        JsonGateRepository(store).get_history(uuid4())

    store.write_text('{"schema_version":"1.0","gates":{}}')
    store.chmod(0o644)
    with pytest.raises(GateRepositoryError):
        JsonGateRepository(store).get_history(uuid4())

    target = tmp_path / "target.json"
    target.write_text('{"schema_version":"1.0","gates":{}}')
    target.chmod(0o600)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(GateRepositoryError):
        JsonGateRepository(link).get_history(uuid4())


def test_non_utf8_store_fails_with_repository_error(tmp_path):
    store = tmp_path / "gates.json"
    store.write_bytes(b"\xff")
    store.chmod(0o600)

    with pytest.raises(GateRepositoryError):
        JsonGateRepository(store).get_history(uuid4())


def test_extreme_revision_time_is_wrapped_and_public_guard_denies(
    tmp_path, approved_gate_factory, now
):
    from billiards_ai.cloud.privacy import authorize_real_video_use
    from billiards_ai.shared.models import AssetContext, MaterialUseDecision, MaterialUsePurpose

    service, _, approved = approved_gate_factory(tmp_path)
    context = AssetContext(
        asset_id="asset-1",
        asset_sha256="8" * 64,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=now,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=False,
    )
    payload = json.loads(service.repository.path.read_text())
    payload["gates"][str(approved.gate_id)][-1]["created_at"] = datetime.max.replace(
        tzinfo=UTC
    ).isoformat()
    payload["gates"][str(approved.gate_id)][-1]["approval"]["approved_at"] = datetime.max.replace(
        tzinfo=UTC
    ).isoformat()
    service.repository.path.write_text(json.dumps(payload))
    service.repository.path.chmod(0o600)

    with pytest.raises(GateRepositoryError):
        service.repository.get_current(approved.gate_id)
    decision = authorize_real_video_use(
        service.repository,
        approved.revision_id,
        context,
        MaterialUsePurpose.SERVICE_ANALYSIS,
    )
    assert decision.decision is MaterialUseDecision.DENY
    assert decision.reason_code == "GATE_RECORD_UNAVAILABLE"


def test_compare_and_swap_rejects_stale_writer(tmp_path, approved_gate_factory):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    current = gate_service.repository.get_current(approved.gate_id)
    changed = current.model_copy(
        update={"revision_id": uuid4(), "revision_number": current.revision_number + 1}
    )
    with pytest.raises(GateConflictError):
        gate_service.repository._append_revision(changed, expected_current_revision_id=uuid4())


def test_forged_approved_revision_is_rejected_without_credential(
    tmp_path, gate_scope_factory, gate_evidence, now
):
    repository = JsonGateRepository(tmp_path / "gates.json")
    blocked = GateRevision(
        gate_id=uuid4(),
        revision_id=uuid4(),
        revision_number=1,
        scope=gate_scope_factory(),
        inspector_id="Mallory",
        inspected_at=now,
        evidence=gate_evidence,
        created_at=now,
    )
    repository._append_revision(blocked, expected_current_revision_id=None)
    forged = GateRevision(
        gate_id=blocked.gate_id,
        revision_id=uuid4(),
        revision_number=2,
        supersedes_revision_id=blocked.revision_id,
        status=RealVideoGateStatus.APPROVED,
        scope=blocked.scope,
        inspector_id=blocked.inspector_id,
        inspected_at=blocked.inspected_at,
        evidence=blocked.evidence,
        approval=GateApproval(
            approver_id="Boyu",
            credential_id="i-made-this-up",
            key_id="phase0-approval-key-v1",
            approved_at=now,
            approval_evidence_sha256="f" * 64,
            gate_fingerprint=blocked.scope.fingerprint,
            approval_mac="0" * 64,
        ),
        created_at=now,
    )
    with pytest.raises(GateRepositoryError):
        repository._append_revision(forged, expected_current_revision_id=blocked.revision_id)
    assert repository.get_current(blocked.gate_id).status is RealVideoGateStatus.BLOCKED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approver_id", "Mallory"),
        ("credential_id", "offline-forgery"),
        ("approval_evidence_sha256", "e" * 64),
    ],
)
def test_offline_tampered_approval_fails_closed(tmp_path, approved_gate_factory, field, value):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    store = gate_service.repository.path
    payload = json.loads(store.read_text())
    payload["gates"][str(approved.gate_id)][-1]["approval"][field] = value
    store.write_text(json.dumps(payload))
    store.chmod(0o600)
    with pytest.raises(GateRepositoryError):
        gate_service.repository.get_current(approved.gate_id)


def test_approved_revision_must_preserve_inspected_scope(
    tmp_path, gate_scope_factory, gate_evidence, now
):
    repository = JsonGateRepository(tmp_path / "gates.json")
    blocked = GateRevision(
        gate_id=uuid4(),
        revision_id=uuid4(),
        revision_number=1,
        scope=gate_scope_factory(
            table_id="table-3",
            checklist=gate_scope_factory().checklist.model_copy(update={"audio_disabled": False}),
        ),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
        created_at=now,
    )
    repository._append_revision(blocked, expected_current_revision_id=None)
    replacement_scope = gate_scope_factory(table_id="table-99")
    forged = GateRevision(
        gate_id=blocked.gate_id,
        revision_id=uuid4(),
        revision_number=2,
        supersedes_revision_id=blocked.revision_id,
        status=RealVideoGateStatus.APPROVED,
        scope=replacement_scope,
        inspector_id=blocked.inspector_id,
        inspected_at=blocked.inspected_at,
        evidence=blocked.evidence,
        approval=GateApproval(
            approver_id="Boyu",
            credential_id="phase0-local-approval",
            key_id="phase0-approval-key-v1",
            approved_at=now,
            approval_evidence_sha256="f" * 64,
            gate_fingerprint=replacement_scope.fingerprint,
            approval_mac="0" * 64,
        ),
        created_at=now,
    )
    with pytest.raises(GateRepositoryError):
        repository._append_revision(forged, expected_current_revision_id=blocked.revision_id)


def test_offline_tampered_inspected_scope_fails_closed(tmp_path, approved_gate_factory):
    gate_service, blocked, approved = approved_gate_factory(tmp_path)
    store = gate_service.repository.path
    payload = json.loads(store.read_text())
    payload["gates"][str(approved.gate_id)][0]["scope"]["table_id"] = "table-99"
    store.write_text(json.dumps(payload))
    store.chmod(0o600)
    with pytest.raises(GateRepositoryError):
        gate_service.repository.get_current(blocked.gate_id)


def test_truncated_revocation_history_fails_closed(tmp_path, approved_gate_factory, now):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    gate_service.revoke(approved.gate_id, reason="notice removed")
    store = gate_service.repository.path
    payload = json.loads(store.read_text())
    payload["gates"][str(approved.gate_id)].pop()
    store.write_text(json.dumps(payload))
    store.chmod(0o600)

    with pytest.raises(GateRepositoryError):
        gate_service.repository.resolve_revision(approved.revision_id)


def test_removed_gate_history_fails_closed(tmp_path, approved_gate_factory):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    store = gate_service.repository.path
    payload = json.loads(store.read_text())
    del payload["gates"][str(approved.gate_id)]
    store.write_text(json.dumps(payload))
    store.chmod(0o600)

    with pytest.raises(GateRepositoryError):
        gate_service.repository.get_history(approved.gate_id)


def test_deleted_anchored_store_file_fails_closed(tmp_path, approved_gate_factory):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    gate_service.repository.path.unlink()

    with pytest.raises(GateRepositoryError):
        gate_service.repository.get_history(approved.gate_id)


def test_integrity_boundary_os_error_is_wrapped_for_public_guard(
    tmp_path, approved_gate_factory, now, monkeypatch
):
    from billiards_ai.cloud.privacy import authorize_real_video_use
    from billiards_ai.cloud.privacy import integrity as integrity_module
    from billiards_ai.shared.models import AssetContext, MaterialUseDecision, MaterialUsePurpose

    gate_service, _, approved = approved_gate_factory(tmp_path)
    context = AssetContext(
        asset_id="asset-1",
        asset_sha256="8" * 64,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=now,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=False,
    )

    def fail_directory_check(*_args, **_kwargs):
        raise OSError("injected stat failure")

    monkeypatch.setattr(integrity_module, "_require_private_directory", fail_directory_check)
    result = authorize_real_video_use(
        gate_service.repository,
        approved.revision_id,
        context,
        MaterialUsePurpose.SERVICE_ANALYSIS,
    )
    assert result.decision is MaterialUseDecision.DENY
    assert result.reason_code == "GATE_RECORD_UNAVAILABLE"


def test_invalid_registry_lock_closes_open_descriptor(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from billiards_ai.cloud.privacy import integrity as integrity_module

    registry = GateHeadRegistry()
    original_open = os.open
    original_fstat = os.fstat
    opened_fds = []

    def recording_open(path, flags, mode=0o777):
        fd = original_open(path, flags, mode)
        opened_fds.append(fd)
        return fd

    def invalid_lock_stat(fd):
        info = original_fstat(fd)
        return SimpleNamespace(st_mode=0o040600, st_uid=info.st_uid)

    monkeypatch.setattr(integrity_module.os, "open", recording_open)
    monkeypatch.setattr(integrity_module.os, "fstat", invalid_lock_stat)
    with pytest.raises(GateRepositoryError):
        with registry._lock():
            pass

    assert len(opened_fds) == 1
    with pytest.raises(OSError):
        original_fstat(opened_fds[0])


def test_offline_tampered_inspection_metadata_on_both_revisions_fails_closed(
    tmp_path, approved_gate_factory, now
):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    store = gate_service.repository.path
    payload = json.loads(store.read_text())
    history = payload["gates"][str(approved.gate_id)]
    forged_inspection_time = now - timedelta(days=1)
    for revision in history:
        revision["inspector_id"] = "Mallory"
        revision["inspected_at"] = forged_inspection_time.isoformat()
        revision["evidence"][0]["evidence_ref"] = "controlled://forged/site-photo.jpg"
        revision["evidence"][0]["sha256"] = "f" * 64
        revision["evidence"][0]["captured_at"] = forged_inspection_time.isoformat()
    store.write_text(json.dumps(payload))
    store.chmod(0o600)

    with pytest.raises(GateRepositoryError):
        gate_service.repository.get_current(approved.gate_id)


@pytest.mark.parametrize("corruption", ["duplicate_revision", "wrong_gate_id"])
def test_corrupt_revision_identity_fails_closed(tmp_path, approved_gate_factory, corruption):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    store = gate_service.repository.path
    payload = json.loads(store.read_text())
    history = payload["gates"][str(approved.gate_id)]
    if corruption == "duplicate_revision":
        history[1]["revision_id"] = history[0]["revision_id"]
    else:
        history[1]["gate_id"] = str(uuid4())
    store.write_text(json.dumps(payload))
    store.chmod(0o600)
    with pytest.raises(GateRepositoryError):
        gate_service.repository.resolve_revision(approved.revision_id)


def test_insecure_store_directory_is_rejected(tmp_path):
    store_directory = tmp_path / "shared"
    store_directory.mkdir(mode=0o700)
    store_directory.chmod(0o777)
    with pytest.raises(GateRepositoryError):
        JsonGateRepository(store_directory / "gates.json").get_history(uuid4())


def test_insecure_store_ancestor_directory_is_rejected(tmp_path):
    unsafe_ancestor = tmp_path / "shared"
    unsafe_ancestor.mkdir(mode=0o700)
    store_directory = unsafe_ancestor / "private"
    store_directory.mkdir(mode=0o700)
    unsafe_ancestor.chmod(0o777)
    with pytest.raises(GateRepositoryError):
        JsonGateRepository(store_directory / "gates.json").get_history(uuid4())


def test_concurrent_identical_approvals_are_idempotent(
    tmp_path, gate_service_factory, gate_scope_factory, gate_evidence, now
):
    gate_service = gate_service_factory(tmp_path)
    blocked = gate_service.create_gate(
        scope=gate_scope_factory(),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: gate_service.approve(blocked.gate_id, "4" * 64),
                range(8),
            )
        )
    assert len({item.revision_id for item in results}) == 1
    assert len(gate_service.repository.get_history(blocked.gate_id)) == 2


def test_store_temp_files_do_not_remain_after_success(tmp_path, approved_gate_factory):
    gate_service, _, _ = approved_gate_factory(tmp_path)
    leftovers = [path for path in tmp_path.iterdir() if path.name.startswith(".gates.json.")]
    assert leftovers == []
    assert os.stat(gate_service.repository.path).st_mode & 0o777 == 0o600


def test_two_stores_have_isolated_registry_namespaces(
    tmp_path, credential_file_factory, gate_scope_factory, gate_evidence, now
):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir(mode=0o700)
    second_dir.mkdir(mode=0o700)
    credential_file_factory(tmp_path)
    first = RealVideoGateService(JsonGateRepository(first_dir / "gates.json"))
    second = RealVideoGateService(JsonGateRepository(second_dir / "gates.json"))

    first_gate = first.create_gate(
        scope=gate_scope_factory(table_id="table-1"),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
    )
    second_gate = second.create_gate(
        scope=gate_scope_factory(table_id="table-2"),
        inspector_id="inspector-2",
        inspected_at=now,
        evidence=gate_evidence,
    )

    assert first.repository.get_current(first_gate.gate_id).revision_id == first_gate.revision_id
    assert second.repository.get_current(second_gate.gate_id).revision_id == second_gate.revision_id
    registry_text = (tmp_path / ".config/billiards-ai/gate-heads.json").read_text()
    assert str(first.repository.path) not in registry_text
    assert str(second.repository.path) not in registry_text
    assert len(json.loads(registry_text)["stores"]) == 2


@pytest.mark.parametrize("failure_stage", ["pending", "store", "finalize"])
def test_append_transaction_recovers_after_each_failure_stage(
    tmp_path,
    approved_gate_factory,
    gate_scope_factory,
    gate_evidence,
    now,
    monkeypatch,
    failure_stage,
):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    repository = gate_service.repository
    original_registry_write = GateHeadRegistry._write_registry_unlocked
    original_store_write = repository._write_unlocked
    saw_pending = False
    injected = False

    def registry_write(registry, payload):
        nonlocal saw_pending, injected
        stores = payload.get("stores", {})
        has_pending = any(item.get("pending") is not None for item in stores.values())
        if has_pending:
            saw_pending = True
        should_fail = (failure_stage == "pending" and has_pending) or (
            failure_stage == "finalize" and saw_pending and not has_pending
        )
        if should_fail and not injected:
            injected = True
            if failure_stage == "pending":
                original_registry_write(registry, payload)
            raise OSError(f"injected {failure_stage} registry failure")
        return original_registry_write(registry, payload)

    def store_write(payload):
        nonlocal injected
        if failure_stage == "store" and not injected:
            injected = True
            raise OSError("injected store replace failure")
        return original_store_write(payload)

    monkeypatch.setattr(GateHeadRegistry, "_write_registry_unlocked", registry_write)
    monkeypatch.setattr(repository, "_write_unlocked", store_write)
    with pytest.raises(GateRepositoryError):
        gate_service.revise(
            approved.gate_id,
            scope=gate_scope_factory(table_id="table-updated"),
            inspector_id="inspector-2",
            inspected_at=now,
            evidence=gate_evidence,
            reason="camera moved",
        )
    assert injected

    monkeypatch.setattr(GateHeadRegistry, "_write_registry_unlocked", original_registry_write)
    monkeypatch.setattr(repository, "_write_unlocked", original_store_write)
    recovered = repository.get_current(approved.gate_id)
    if failure_stage == "finalize":
        assert recovered.status is RealVideoGateStatus.BLOCKED
    else:
        assert recovered.revision_id == approved.revision_id
    registry_payload = json.loads((tmp_path / ".config/billiards-ai/gate-heads.json").read_text())
    assert all(item["pending"] is None for item in registry_payload["stores"].values())


def test_approval_keyring_rotation_preserves_history_and_uses_active_key(
    tmp_path,
    credential_file_factory,
    gate_scope_factory,
    gate_evidence,
    now,
    monkeypatch,
):
    from billiards_ai.cloud.privacy import service as gate_service_module

    monkeypatch.setattr(gate_service_module, "_utc_now", lambda: now)
    credential_file_factory(
        tmp_path,
        active_key_id="key-a",
        keys={"key-a": "a" * 64},
    )
    repository = JsonGateRepository(tmp_path / "gates.json")
    service = RealVideoGateService(repository)
    first_blocked = service.create_gate(
        scope=gate_scope_factory(),
        inspector_id="inspector-1",
        inspected_at=now,
        evidence=gate_evidence,
    )
    first_approved = service.approve(first_blocked.gate_id, "4" * 64)
    assert first_approved.approval is not None
    assert first_approved.approval.key_id == "key-a"

    credential_file_factory(
        tmp_path,
        active_key_id="key-b",
        keys={"key-a": "a" * 64, "key-b": "b" * 64},
    )
    assert repository.get_current(first_approved.gate_id).revision_id == first_approved.revision_id
    second_blocked = service.create_gate(
        scope=gate_scope_factory(table_id="table-4"),
        inspector_id="inspector-2",
        inspected_at=now,
        evidence=gate_evidence,
    )
    second_approved = service.approve(second_blocked.gate_id, "5" * 64)
    assert second_approved.approval is not None
    assert second_approved.approval.key_id == "key-b"

    credential_file_factory(
        tmp_path,
        active_key_id="key-b",
        keys={"key-b": "b" * 64},
    )
    with pytest.raises(GateRepositoryError):
        repository.get_current(first_approved.gate_id)
