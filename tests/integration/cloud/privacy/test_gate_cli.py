"""CLI integration tests for the public REAL_VIDEO_GATE surface."""

import hashlib
import hmac
import json
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest

from billiards_ai.cloud.privacy import (
    GateRepositoryError,
    JsonGateRepository,
    RealVideoGateService,
)
from billiards_ai.cloud.privacy import service as gate_service_module
from billiards_ai.cloud.privacy.cli import main as gate_cli_main


def test_cli_has_no_approver_argument_and_requires_store(capsys):
    with pytest.raises(SystemExit) as exc:
        gate_cli_main(["approve", "--approver", "Boyu"])
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "--store" in stderr or "unrecognized arguments" in stderr

    with pytest.raises(SystemExit) as exc:
        gate_cli_main(
            [
                "approve",
                "--store",
                "gates.json",
                "--gate-id",
                "00000000-0000-4000-8000-000000000000",
                "--credential",
                "approver.json",
                "--approval-evidence-sha256",
                "4" * 64,
            ]
        )
    assert exc.value.code == 2


def test_cli_approval_uses_internal_utc_clock(
    tmp_path,
    monkeypatch,
    capsys,
    credential_file_factory,
    gate_scope_factory,
    gate_evidence,
    now,
):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz is not None else now.replace(tzinfo=None)

    monkeypatch.setattr(gate_service_module, "datetime", FrozenDateTime)
    credential_file_factory(tmp_path)
    store = tmp_path / "gates.json"
    application = tmp_path / "application.json"
    application.write_text(
        json.dumps(
            {
                "scope": gate_scope_factory().model_dump(mode="json"),
                "inspector_id": "inspector-1",
                "inspected_at": now.isoformat(),
                "evidence": [item.model_dump(mode="json") for item in gate_evidence],
            }
        )
    )
    assert gate_cli_main(["create", "--store", str(store), "--application", str(application)]) == 0
    gate_id = UUID(capsys.readouterr().out.strip())

    assert (
        gate_cli_main(
            [
                "approve",
                "--store",
                str(store),
                "--gate-id",
                str(gate_id),
                "--approval-evidence-sha256",
                "4" * 64,
            ]
        )
        == 0
    )
    approved = JsonGateRepository(store).get_current(gate_id)
    assert approved.approval is not None
    assert approved.approval is not None
    assert approved.approval.approved_at == now


def test_cli_rejects_backfilled_approval_time(
    tmp_path,
    capsys,
    credential_file_factory,
    gate_scope_factory,
    gate_evidence,
    now,
):
    credential_file_factory(tmp_path)
    store = tmp_path / "gates.json"
    application = tmp_path / "application.json"
    backfilled_at = now - timedelta(days=1)
    historical_evidence = [
        item.model_copy(update={"captured_at": backfilled_at}).model_dump(mode="json")
        for item in gate_evidence
    ]
    application.write_text(
        json.dumps(
            {
                "scope": gate_scope_factory().model_dump(mode="json"),
                "inspector_id": "inspector-1",
                "inspected_at": backfilled_at.isoformat(),
                "evidence": historical_evidence,
            }
        )
    )
    assert gate_cli_main(["create", "--store", str(store), "--application", str(application)]) == 0
    gate_id = UUID(capsys.readouterr().out.strip())

    with pytest.raises(SystemExit) as exc:
        gate_cli_main(
            [
                "approve",
                "--store",
                str(store),
                "--gate-id",
                str(gate_id),
                "--approval-evidence-sha256",
                "4" * 64,
                "--approved-at",
                backfilled_at.isoformat(),
            ]
        )
    assert exc.value.code == 2


def test_show_outputs_public_summary_not_evidence_path(tmp_path, capsys, approved_gate_factory):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    exit_code = gate_cli_main(
        [
            "show",
            "--store",
            str(gate_service.repository.path),
            "--gate-id",
            str(approved.gate_id),
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert output.strip() == "门禁已批准"
    assert "controlled://" not in output


def test_only_evaluate_can_emit_continue_token(
    tmp_path,
    capsys,
    credential_file_factory,
    gate_scope_factory,
    gate_evidence,
    now,
):
    credential = credential_file_factory(tmp_path)
    store = tmp_path / "gates.json"
    application = tmp_path / "application.json"
    scope = gate_scope_factory()
    application.write_text(
        json.dumps(
            {
                "scope": scope.model_dump(mode="json"),
                "inspector_id": "inspector-1",
                "inspected_at": now.isoformat(),
                "evidence": [item.model_dump(mode="json") for item in gate_evidence],
            }
        )
    )
    assert gate_cli_main(["create", "--store", str(store), "--application", str(application)]) == 0
    create_output = capsys.readouterr().out.strip()
    assert "可继续" not in create_output
    gate_id = UUID(create_output)

    assert gate_cli_main(["show", "--store", str(store), "--gate-id", str(gate_id)]) == 0
    assert "可继续" not in capsys.readouterr().out

    assert credential.exists()
    assert (
        gate_cli_main(
            [
                "approve",
                "--store",
                str(store),
                "--gate-id",
                str(gate_id),
                "--approval-evidence-sha256",
                "4" * 64,
            ]
        )
        == 0
    )
    assert "可继续" not in capsys.readouterr().out

    approved = JsonGateRepository(store).get_current(gate_id)
    assert gate_cli_main(["show", "--store", str(store), "--gate-id", str(gate_id)]) == 0
    assert "可继续" not in capsys.readouterr().out
    context = tmp_path / "asset-context.json"
    context.write_text(
        json.dumps(
            {
                "asset_id": "asset-1",
                "asset_sha256": "8" * 64,
                "gate_scope_fingerprint": approved.scope.fingerprint,
                "captured_at": approved.approval.approved_at.isoformat(),
                "captured_under_revision_id": str(approved.revision_id),
                "bystanders_present": False,
            }
        )
    )
    assert (
        gate_cli_main(
            [
                "evaluate",
                "--store",
                str(store),
                "--gate-revision-id",
                str(approved.revision_id),
                "--asset-context",
                str(context),
                "--purpose",
                "SERVICE_ANALYSIS",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "可继续"


def test_cli_explicitly_migrates_verified_v1_store_without_rewriting_history(
    tmp_path,
    capsys,
    credential_file_factory,
    gate_scope_factory,
    gate_evidence,
    now,
):
    credential = credential_file_factory(tmp_path)
    credential.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "approver_id": "Boyu",
                "credential_id": "phase0-local-approval",
                "approval_hmac_key": "a" * 64,
            }
        )
    )
    credential.chmod(0o600)
    gate_id = uuid4()
    blocked_id = uuid4()
    approved_id = uuid4()
    scope = gate_scope_factory()
    approval_evidence = "4" * 64
    legacy_payload = "|".join(
        (
            str(gate_id),
            str(approved_id),
            scope.fingerprint,
            now.isoformat(),
            approval_evidence,
        )
    ).encode()
    legacy_mac = hmac.new(bytes.fromhex("a" * 64), legacy_payload, hashlib.sha256).hexdigest()
    blocked = {
        "schema_version": "1.0",
        "gate_id": str(gate_id),
        "revision_id": str(blocked_id),
        "revision_number": 1,
        "supersedes_revision_id": None,
        "status": "BLOCKED",
        "scope": scope.model_dump(mode="json"),
        "inspector_id": "inspector-1",
        "inspected_at": now.isoformat(),
        "evidence": [item.model_dump(mode="json") for item in gate_evidence],
        "approval": None,
        "invalidation_reason": None,
        "created_at": now.isoformat(),
    }
    approved = {
        **blocked,
        "revision_id": str(approved_id),
        "revision_number": 2,
        "supersedes_revision_id": str(blocked_id),
        "status": "APPROVED",
        "approval": {
            "approver_id": "Boyu",
            "credential_id": "phase0-local-approval",
            "approved_at": now.isoformat(),
            "approval_evidence_sha256": approval_evidence,
            "gate_fingerprint": scope.fingerprint,
            "approval_mac": legacy_mac,
        },
    }
    store = tmp_path / "legacy-gates.json"
    store.write_text(
        json.dumps(
            {"schema_version": "1.0", "gates": {str(gate_id): [blocked, approved]}},
            sort_keys=True,
        )
    )
    store.chmod(0o600)
    original_bytes = store.read_bytes()
    expected_sha256 = hashlib.sha256(original_bytes).hexdigest()

    with pytest.raises(GateRepositoryError):
        JsonGateRepository(store).get_current(gate_id)
    assert (
        gate_cli_main(
            [
                "migrate",
                "--store",
                str(store),
                "--expected-store-sha256",
                "0" * 64,
            ]
        )
        == 5
    )
    assert capsys.readouterr().err.strip() == "不可处理"
    assert store.read_bytes() == original_bytes
    assert not (tmp_path / ".config/billiards-ai/gate-heads.json").exists()
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
    assert store.read_bytes() == original_bytes
    assert not (tmp_path / ".config/billiards-ai/gate-heads.json").exists()

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
    assert store.read_bytes() == original_bytes
    assert JsonGateRepository(store).get_current(gate_id).revision_id == approved_id

    repository = JsonGateRepository(store)
    service = RealVideoGateService(repository)
    revised = service.revise(
        gate_id,
        scope=scope,
        inspector_id="inspector-2",
        inspected_at=now,
        evidence=gate_evidence,
        reason="migration verification",
    )
    newly_approved = service.approve(revised.gate_id, "5" * 64)
    assert newly_approved.approval is not None
    assert newly_approved.approval.mac_version == "3.0"
    assert newly_approved.approval.key_id == "legacy-v1"
