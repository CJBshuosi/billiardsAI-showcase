"""Regression coverage for trusted schema 1 registry upgrades."""

import copy
import hashlib
import hmac
import json
import os
import stat
from datetime import datetime

import pytest

from billiards_ai.cloud.privacy import GateRepositoryError, JsonGateRepository
from billiards_ai.cloud.privacy import integrity as integrity_module
from billiards_ai.cloud.privacy.cli import main as gate_cli_main
from billiards_ai.shared.models import GateRevision


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _legacy_chain_digest(raw_history: list[dict]) -> str:
    digest = hashlib.sha256(b"billiards-ai-gate-chain-v1").hexdigest()
    for revision in raw_history:
        digest = hashlib.sha256(bytes.fromhex(digest) + _canonical_json(revision)).hexdigest()
    return digest


def _0e615b1_normalized_chain_digest(raw_history: list[dict]) -> str:
    """Reproduce the normalized-prefix digest emitted by parent 0e615b1."""
    digest = hashlib.sha256(b"billiards-ai-gate-chain-v1").hexdigest()
    for raw_revision in raw_history:
        normalized = copy.deepcopy(raw_revision)
        approval = normalized.get("approval")
        if isinstance(approval, dict):
            if "key_id" not in approval:
                approval["key_id"] = "legacy-v1"
                approval["mac_version"] = "1.0"
            else:
                approval.setdefault("mac_version", "2.0")
        revision = GateRevision.model_validate(normalized)
        digest = hashlib.sha256(
            bytes.fromhex(digest) + _canonical_json(revision.model_dump(mode="json"))
        ).hexdigest()
    return digest


def _legacy_approval_payload(revision: dict, previous_chain_digest: str) -> dict:
    approval = revision["approval"]
    scope_fingerprint = hashlib.sha256(_canonical_json(revision["scope"])).hexdigest()
    inspected_at = datetime.fromisoformat(revision["inspected_at"].replace("Z", "+00:00"))
    approved_at = datetime.fromisoformat(approval["approved_at"].replace("Z", "+00:00"))
    return {
        "schema_version": "1.0",
        "gate_id": revision["gate_id"],
        "revision_id": revision["revision_id"],
        "revision_number": revision["revision_number"],
        "supersedes_revision_id": revision["supersedes_revision_id"],
        "status": revision["status"],
        "scope_fingerprint": scope_fingerprint,
        "evidence": revision["evidence"],
        "inspector_id": revision["inspector_id"],
        "inspected_at": inspected_at.isoformat(),
        "approver_id": approval["approver_id"],
        "credential_id": approval["credential_id"],
        "key_id": approval["key_id"],
        "approved_at": approved_at.isoformat(),
        "approval_evidence_sha256": approval["approval_evidence_sha256"],
        "previous_chain_digest": previous_chain_digest,
    }


def _write_schema1_registry(tmp_path, gate_id, raw_history, current_revision_id):
    integrity_directory = tmp_path / ".config" / "billiards-ai"
    key_payload = json.loads((integrity_directory / "gate-head.key").read_text())
    unsigned_head = {
        "revision_id": str(current_revision_id),
        "revision_number": len(raw_history),
        "chain_digest": _legacy_chain_digest(raw_history),
        "key_id": "phase0-gate-head-v1",
    }
    head_payload = {"gate_id": str(gate_id), **unsigned_head}
    head_mac = hmac.new(
        bytes.fromhex(key_payload["hmac_key"]),
        _canonical_json(head_payload),
        hashlib.sha256,
    ).hexdigest()
    registry_path = integrity_directory / "gate-heads.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "gates": {str(gate_id): {**unsigned_head, "head_mac": head_mac}},
            },
            sort_keys=True,
        )
    )
    registry_path.chmod(0o600)
    return registry_path


def _write_schema2_registry(tmp_path, store, payload):
    registry = integrity_module.GateHeadRegistry()
    key = integrity_module._ChainKey.model_validate(
        json.loads((tmp_path / ".config/billiards-ai/gate-head.key").read_text())
    )
    store_id = integrity_module.store_identity(store)
    histories = {
        gate_key: tuple(GateRevision.model_validate(item) for item in raw_history)
        for gate_key, raw_history in payload["gates"].items()
    }
    values = {
        "key_id": key.key_id,
        "store_digest": integrity_module.store_payload_digest(payload),
        "heads": registry._heads(key, store_id, histories),
        "pending": None,
    }
    registry_path = registry.registry_path
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "stores": {store_id: registry._signed_namespace(key, store_id, values)},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    registry_path.chmod(0o600)
    return registry_path


def _install_schema1_registry(tmp_path, approved_gate_factory):
    service, _, approved = approved_gate_factory(tmp_path)
    store = service.repository.path
    store_payload = json.loads(store.read_text())
    raw_history = store_payload["gates"][str(approved.gate_id)]
    raw_history[-1]["approval"].pop("mac_version")
    raw_history[-1]["approval"]["approval_mac"] = hmac.new(
        bytes.fromhex("a" * 64),
        _canonical_json(
            _legacy_approval_payload(
                raw_history[-1],
                _legacy_chain_digest(raw_history[:-1]),
            )
        ),
        hashlib.sha256,
    ).hexdigest()
    store.write_text(json.dumps(store_payload, sort_keys=True))
    store.chmod(0o600)

    registry_path = _write_schema1_registry(
        tmp_path, approved.gate_id, raw_history, approved.revision_id
    )
    return service, approved, store, registry_path


def _install_multiround_schema1_registry(tmp_path, approved_gate_factory, gate_evidence, now):
    service, _, first_approved = approved_gate_factory(tmp_path)
    first_revised = service.revise(
        first_approved.gate_id,
        scope=first_approved.scope,
        inspector_id="inspector-2",
        inspected_at=now,
        evidence=gate_evidence,
        reason="camera rechecked",
    )
    second_approved = service.approve(first_revised.gate_id, "5" * 64)
    service.revoke(second_approved.gate_id, reason="notice removed")
    second_revised = service.revise(
        second_approved.gate_id,
        scope=second_approved.scope,
        inspector_id="inspector-3",
        inspected_at=now,
        evidence=gate_evidence,
        reason="notice restored",
    )
    final_approved = service.approve(second_revised.gate_id, "6" * 64)

    store = service.repository.path
    store_payload = json.loads(store.read_text())
    raw_history = store_payload["gates"][str(final_approved.gate_id)]
    approval_key = bytes.fromhex("a" * 64)
    for index, revision in enumerate(raw_history):
        approval = revision.get("approval")
        if approval is None:
            continue
        approval.pop("mac_version")
        approval["approval_mac"] = hmac.new(
            approval_key,
            _canonical_json(
                _legacy_approval_payload(
                    revision,
                    _legacy_chain_digest(raw_history[:index]),
                )
            ),
            hashlib.sha256,
        ).hexdigest()
    store.write_text(json.dumps(store_payload, sort_keys=True))
    store.chmod(0o600)
    registry_path = _write_schema1_registry(
        tmp_path,
        final_approved.gate_id,
        raw_history,
        final_approved.revision_id,
    )
    return service, final_approved, store, registry_path


def _install_0e615b1_v2_history(tmp_path, approved_gate_factory, gate_evidence, now):
    """Build schema1-migrated history followed by real 0e615b1 v2 approvals."""
    service, first_approved, store, _ = _install_schema1_registry(tmp_path, approved_gate_factory)
    service.repository.migrate_legacy_store(
        expected_store_sha256=hashlib.sha256(store.read_bytes()).hexdigest(),
        dry_run=False,
    )
    current = first_approved
    for round_number in (2, 3):
        revised = service.revise(
            current.gate_id,
            scope=current.scope,
            inspector_id=f"inspector-{round_number}",
            inspected_at=now,
            evidence=gate_evidence,
            reason=f"historical round {round_number}",
        )
        current = service.approve(revised.gate_id, str(round_number + 3) * 64)

    payload = json.loads(store.read_text())
    raw_history = payload["gates"][str(current.gate_id)]
    approval_key = bytes.fromhex("a" * 64)
    approval_indexes = [
        index for index, revision in enumerate(raw_history) if revision.get("approval")
    ]
    for approval_number, index in enumerate(approval_indexes):
        revision = raw_history[index]
        approval = revision["approval"]
        if approval_number == 0:
            approval.pop("mac_version", None)
            previous_digest = _legacy_chain_digest(raw_history[:index])
        else:
            approval["mac_version"] = "2.0"
            previous_digest = _0e615b1_normalized_chain_digest(raw_history[:index])
        approval["approval_mac"] = hmac.new(
            approval_key,
            _canonical_json(_legacy_approval_payload(revision, previous_digest)),
            hashlib.sha256,
        ).hexdigest()
    store.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    store.chmod(0o600)
    _write_schema2_registry(tmp_path, store, payload)
    return service, current, store


def _trust_file_snapshot(integrity_directory):
    snapshot = {}
    for name in ("gate-head.key", "gate-heads.json"):
        path = integrity_directory / name
        if path.exists():
            snapshot[name] = (path.read_bytes(), path.stat().st_mtime_ns)
        else:
            snapshot[name] = None
    return snapshot


def test_0e615b1_v2_normalized_prefix_history_reads_and_appends_v3(
    tmp_path, approved_gate_factory, gate_evidence, now
):
    service, historical_current, _ = _install_0e615b1_v2_history(
        tmp_path, approved_gate_factory, gate_evidence, now
    )

    assert service.repository.get_current(historical_current.gate_id).revision_id == (
        historical_current.revision_id
    )
    revised = service.revise(
        historical_current.gate_id,
        scope=historical_current.scope,
        inspector_id="inspector-4",
        inspected_at=now,
        evidence=gate_evidence,
        reason="append with raw-prefix v3",
    )
    appended = service.approve(revised.gate_id, "7" * 64)
    assert appended.approval is not None
    assert appended.approval.mac_version == "3.0"
    assert service.repository.get_current(appended.gate_id).revision_id == appended.revision_id


def test_explicit_approval_mac_version_cannot_replay_across_prefix_algorithms(
    tmp_path, approved_gate_factory, gate_evidence, now
):
    service, historical_current, store = _install_0e615b1_v2_history(
        tmp_path, approved_gate_factory, gate_evidence, now
    )
    payload = json.loads(store.read_text())
    history = payload["gates"][str(historical_current.gate_id)]
    history[-1]["approval"]["mac_version"] = "3.0"
    store.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    store.chmod(0o600)

    with pytest.raises(GateRepositoryError, match="approval integrity"):
        service.repository.get_current(historical_current.gate_id)


def test_raw_prefix_v3_mac_cannot_be_replayed_as_v2(tmp_path, approved_gate_factory):
    service, _, approved = approved_gate_factory(tmp_path)
    store = service.repository.path
    payload = json.loads(store.read_text())
    payload["gates"][str(approved.gate_id)][-1]["approval"]["mac_version"] = "2.0"
    store.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    store.chmod(0o600)

    with pytest.raises(GateRepositoryError, match="approval integrity"):
        service.repository.get_current(approved.gate_id)


@pytest.mark.parametrize("key_contents", [None, "corrupt-key"])
@pytest.mark.parametrize("dry_run", [True, False])
def test_missing_registry_with_orphan_key_fails_closed_without_trust_writes(
    tmp_path, approved_gate_factory, key_contents, dry_run
):
    service, _, _ = approved_gate_factory(tmp_path)
    integrity_directory = tmp_path / ".config" / "billiards-ai"
    registry_path = integrity_directory / "gate-heads.json"
    key_path = integrity_directory / "gate-head.key"
    registry_path.unlink()
    if key_contents is not None:
        key_path.write_text(key_contents)
        key_path.chmod(0o600)
    before = _trust_file_snapshot(integrity_directory)

    with pytest.raises(GateRepositoryError, match="registry is missing"):
        service.repository.migrate_legacy_store(
            expected_store_sha256=hashlib.sha256(service.repository.path.read_bytes()).hexdigest(),
            dry_run=dry_run,
        )
    assert _trust_file_snapshot(integrity_directory) == before


@pytest.mark.parametrize("dry_run", [True, False])
def test_deleted_registry_cannot_reanchor_rolled_back_store(
    tmp_path, approved_gate_factory, gate_evidence, now, dry_run
):
    service, _, approved = approved_gate_factory(tmp_path)
    store = service.repository.path
    old_store_bytes = store.read_bytes()
    revised = service.revise(
        approved.gate_id,
        scope=approved.scope,
        inspector_id="inspector-2",
        inspected_at=now,
        evidence=gate_evidence,
        reason="advance trusted head",
    )
    assert revised.revision_number == 3
    integrity_directory = tmp_path / ".config" / "billiards-ai"
    (integrity_directory / "gate-heads.json").unlink()
    store.write_bytes(old_store_bytes)
    store.chmod(0o600)
    before = _trust_file_snapshot(integrity_directory)

    with pytest.raises(GateRepositoryError, match="registry is missing"):
        service.repository.migrate_legacy_store(
            expected_store_sha256=hashlib.sha256(old_store_bytes).hexdigest(),
            dry_run=dry_run,
        )
    assert _trust_file_snapshot(integrity_directory) == before


def test_schema1_multiround_history_migrates_without_rewriting_and_can_append(
    tmp_path, approved_gate_factory, gate_evidence, now
):
    service, final_approved, store, _ = _install_multiround_schema1_registry(
        tmp_path, approved_gate_factory, gate_evidence, now
    )
    original_store = store.read_bytes()
    expected_sha256 = hashlib.sha256(original_store).hexdigest()

    service.repository.migrate_legacy_store(
        expected_store_sha256=expected_sha256,
        dry_run=True,
    )
    assert store.read_bytes() == original_store
    service.repository.migrate_legacy_store(
        expected_store_sha256=expected_sha256,
        dry_run=False,
    )
    assert store.read_bytes() == original_store
    assert service.repository.get_current(final_approved.gate_id).revision_id == (
        final_approved.revision_id
    )

    revised = service.revise(
        final_approved.gate_id,
        scope=final_approved.scope,
        inspector_id="inspector-4",
        inspected_at=now,
        evidence=gate_evidence,
        reason="post migration append",
    )
    appended = service.approve(revised.gate_id, "7" * 64)
    assert service.repository.get_current(appended.gate_id).revision_id == appended.revision_id


@pytest.mark.parametrize("tamper", ["approval", "prefix", "mixed_mac_version"])
def test_schema1_multiround_migration_rejects_tampered_raw_history(
    tmp_path, approved_gate_factory, gate_evidence, now, tamper
):
    service, _, store, registry_path = _install_multiround_schema1_registry(
        tmp_path, approved_gate_factory, gate_evidence, now
    )
    payload = json.loads(store.read_text())
    history = next(iter(payload["gates"].values()))
    if tamper == "approval":
        history[1]["approval"]["approval_evidence_sha256"] = "e" * 64
    elif tamper == "prefix":
        history[0]["invalidation_reason"] = "tampered prefix"
    else:
        history[3]["approval"]["mac_version"] = "2.0"
    store.write_text(json.dumps(payload, sort_keys=True))
    store.chmod(0o600)
    original_registry = registry_path.read_bytes()

    with pytest.raises(GateRepositoryError):
        service.repository.migrate_legacy_store(
            expected_store_sha256=hashlib.sha256(store.read_bytes()).hexdigest(),
            dry_run=True,
        )
    assert registry_path.read_bytes() == original_registry


def test_missing_key_with_existing_namespace_fails_closed_without_dry_run_writes(
    tmp_path, approved_gate_factory
):
    service, _, _ = approved_gate_factory(tmp_path)
    integrity_directory = tmp_path / ".config" / "billiards-ai"
    key_path = integrity_directory / "gate-head.key"
    key_path.unlink()
    before = _trust_file_snapshot(integrity_directory)
    expected_sha256 = hashlib.sha256(service.repository.path.read_bytes()).hexdigest()

    with pytest.raises(GateRepositoryError):
        service.repository.migrate_legacy_store(
            expected_store_sha256=expected_sha256,
            dry_run=True,
        )
    assert _trust_file_snapshot(integrity_directory) == before


@pytest.mark.parametrize("dry_run", [True, False])
def test_missing_shared_key_cannot_migrate_second_store(tmp_path, approved_gate_factory, dry_run):
    service, _, _ = approved_gate_factory(tmp_path)
    second_directory = tmp_path / "second"
    second_directory.mkdir(mode=0o700)
    second_store = second_directory / "gates.json"
    second_store.write_bytes(service.repository.path.read_bytes())
    second_store.chmod(0o600)
    integrity_directory = tmp_path / ".config" / "billiards-ai"
    (integrity_directory / "gate-head.key").unlink()
    before = _trust_file_snapshot(integrity_directory)

    with pytest.raises(GateRepositoryError):
        JsonGateRepository(second_store).migrate_legacy_store(
            expected_store_sha256=hashlib.sha256(second_store.read_bytes()).hexdigest(),
            dry_run=dry_run,
        )
    assert _trust_file_snapshot(integrity_directory) == before


@pytest.mark.parametrize("registry_contents", ['{"schema_version":"2.0","stores":{}}', "{"])
def test_existing_empty_or_corrupt_registry_without_key_cannot_initialize_trust(
    tmp_path, approved_gate_factory, registry_contents
):
    service, _, _ = approved_gate_factory(tmp_path)
    integrity_directory = tmp_path / ".config" / "billiards-ai"
    key_path = integrity_directory / "gate-head.key"
    registry_path = integrity_directory / "gate-heads.json"
    key_path.unlink()
    registry_path.write_text(registry_contents)
    registry_path.chmod(0o600)
    before = _trust_file_snapshot(integrity_directory)

    for dry_run in (True, False):
        with pytest.raises(GateRepositoryError):
            service.repository.migrate_legacy_store(
                expected_store_sha256=hashlib.sha256(
                    service.repository.path.read_bytes()
                ).hexdigest(),
                dry_run=dry_run,
            )
        assert _trust_file_snapshot(integrity_directory) == before


def test_fresh_unanchored_store_dry_run_is_zero_write_then_execute_initializes(
    tmp_path, approved_gate_factory
):
    service, _, approved = approved_gate_factory(tmp_path)
    integrity_directory = tmp_path / ".config" / "billiards-ai"
    (integrity_directory / "gate-head.key").unlink()
    (integrity_directory / "gate-heads.json").unlink()
    before = _trust_file_snapshot(integrity_directory)
    expected_sha256 = hashlib.sha256(service.repository.path.read_bytes()).hexdigest()

    service.repository.migrate_legacy_store(
        expected_store_sha256=expected_sha256,
        dry_run=True,
    )
    assert _trust_file_snapshot(integrity_directory) == before

    service.repository.migrate_legacy_store(
        expected_store_sha256=expected_sha256,
        dry_run=False,
    )
    assert (integrity_directory / "gate-head.key").exists()
    assert (integrity_directory / "gate-heads.json").exists()
    assert service.repository.get_current(approved.gate_id).revision_id == approved.revision_id


def test_schema1_registry_upgrade_is_explicit_verified_and_multi_store_safe(
    tmp_path,
    capsys,
    approved_gate_factory,
    gate_scope_factory,
    gate_evidence,
    now,
):
    service, approved, store, registry_path = _install_schema1_registry(
        tmp_path, approved_gate_factory
    )
    original_store = store.read_bytes()
    original_registry = registry_path.read_bytes()
    expected_sha256 = hashlib.sha256(original_store).hexdigest()

    with pytest.raises(GateRepositoryError):
        service.repository.get_current(approved.gate_id)

    assert (
        gate_cli_main(
            [
                "migrate",
                "--store",
                str(store),
                "--expected-store-sha256",
                expected_sha256,
                "--dry-run",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "迁移验证通过，未写入"
    assert registry_path.read_bytes() == original_registry

    assert (
        gate_cli_main(
            [
                "migrate",
                "--store",
                str(store),
                "--expected-store-sha256",
                expected_sha256,
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "迁移锚定完成"
    assert store.read_bytes() == original_store
    assert (
        JsonGateRepository(store).get_current(approved.gate_id).revision_id == approved.revision_id
    )

    revised = service.revise(
        approved.gate_id,
        scope=approved.scope,
        inspector_id="inspector-2",
        inspected_at=now,
        evidence=gate_evidence,
        reason="post schema1 upgrade",
    )
    assert service.repository.get_current(approved.gate_id).revision_id == revised.revision_id

    second_directory = tmp_path / "second"
    second_directory.mkdir(mode=0o700)
    second_repository = JsonGateRepository(second_directory / "gates.json")
    second_service = type(service)(second_repository)
    second = second_service.create_gate(
        scope=gate_scope_factory(table_id="table-2"),
        inspector_id="inspector-2",
        inspected_at=now,
        evidence=gate_evidence,
    )
    assert second_repository.get_current(second.gate_id).revision_id == second.revision_id
    assert len(json.loads(registry_path.read_text())["stores"]) == 2


def test_schema1_registry_dry_run_rejects_tampered_hmac_without_writing(
    tmp_path, capsys, approved_gate_factory
):
    _, _, store, registry_path = _install_schema1_registry(tmp_path, approved_gate_factory)
    registry = json.loads(registry_path.read_text())
    next(iter(registry["gates"].values()))["head_mac"] = "0" * 64
    registry_path.write_text(json.dumps(registry, sort_keys=True))
    registry_path.chmod(0o600)
    original_registry = registry_path.read_bytes()

    assert (
        gate_cli_main(
            [
                "migrate",
                "--store",
                str(store),
                "--expected-store-sha256",
                hashlib.sha256(store.read_bytes()).hexdigest(),
                "--dry-run",
            ]
        )
        == 5
    )
    assert capsys.readouterr().err.strip() == "不可处理"
    assert registry_path.read_bytes() == original_registry


def test_schema1_registry_upgrade_retry_is_idempotent_after_replace_before_dir_fsync(
    tmp_path, approved_gate_factory, monkeypatch
):
    _, approved, store, registry_path = _install_schema1_registry(tmp_path, approved_gate_factory)
    repository = JsonGateRepository(store)
    expected_sha256 = hashlib.sha256(store.read_bytes()).hexdigest()
    original_fsync = os.fsync
    failed = False

    def fail_first_directory_fsync(fd: int) -> None:
        nonlocal failed
        if not failed and stat.S_ISDIR(os.fstat(fd).st_mode):
            failed = True
            raise OSError("injected directory fsync failure after registry replace")
        original_fsync(fd)

    monkeypatch.setattr(integrity_module.os, "fsync", fail_first_directory_fsync)
    with pytest.raises(GateRepositoryError):
        repository.migrate_legacy_store(
            expected_store_sha256=expected_sha256,
            dry_run=False,
        )
    assert json.loads(registry_path.read_text())["schema_version"] == "2.0"

    monkeypatch.setattr(integrity_module.os, "fsync", original_fsync)
    repository.migrate_legacy_store(
        expected_store_sha256=expected_sha256,
        dry_run=False,
    )
    assert repository.get_current(approved.gate_id).revision_id == approved.revision_id
