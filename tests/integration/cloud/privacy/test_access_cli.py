import json
from uuid import uuid4

import pytest

from billiards_ai.cloud.privacy.access_cli import _parser, main


def _write_reviewer_credential(tmp_path) -> None:
    directory = tmp_path / ".config" / "billiards-ai"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    path = directory / "reviewer.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "principal_id": "reviewer-1",
                "roles": ["ACCESS_ADMIN", "REVIEWER"],
                "credential_id": "phase0-reviewer",
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _base_args(tmp_path) -> list[str]:
    return [
        "--access-store",
        str(tmp_path / "access" / "store.json"),
        "--retention-store",
        str(tmp_path / "retention" / "store.json"),
        "--gate-store",
        str(tmp_path / "gates.json"),
    ]


def test_cli_has_no_self_asserted_identity_or_media_export_surface() -> None:
    help_text = _parser().format_help()
    for forbidden in (
        "--reviewer",
        "--actor",
        "--occurred-at",
        "--credential",
        "--locator",
        "--output",
        "download",
        "export",
    ):
        assert forbidden not in help_text.lower()


def test_grant_check_revoke_and_audit_use_trusted_identity(tmp_path, capsys) -> None:
    _write_reviewer_credential(tmp_path)
    session_id = uuid4()
    base = _base_args(tmp_path)

    assert main([*base, "grant", "--session-id", str(session_id), "--reason", "assigned"]) == 0
    granted = json.loads(capsys.readouterr().out)
    assert granted["session_id"] == str(session_id)
    assert set(granted) == {"grant_id", "session_id", "status"}

    assert main([*base, "check", "--session-id", str(session_id)]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked == {
        "grant_id": granted["grant_id"],
        "session_id": str(session_id),
        "status": "ACTIVE",
    }

    assert main([*base, "audit", "--session-id", str(session_id)]) == 0
    assert json.loads(capsys.readouterr().out) == []

    assert main([*base, "revoke", "--grant-id", granted["grant_id"], "--reason", "done"]) == 0
    revoked = json.loads(capsys.readouterr().out)
    assert revoked["status"] == "REVOKED"

    serialized = json.dumps([granted, checked, revoked])
    assert "locator" not in serialized.lower()
    assert str(tmp_path) not in serialized


def test_provider_approval_cli_outputs_only_non_sensitive_status(tmp_path, capsys) -> None:
    _write_reviewer_credential(tmp_path)
    base = _base_args(tmp_path)
    assert (
        main(
            [
                *base,
                "provider-approve",
                "--provider-id",
                "provider-cn-1",
                "--endpoint-region",
                "cn-shanghai",
                "--purpose",
                "SERVICE_ANALYSIS",
                "--cross-border-assessment",
                "DOMESTIC_ONLY",
                "--max-cost-cny",
                "2.50",
                "--evidence-sha256",
                "a" * 64,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"approval_id", "provider_id", "status"}
    assert "evidence" not in json.dumps(payload).lower()


def test_cli_wraps_repository_failure_without_disclosing_path(tmp_path, capsys) -> None:
    _write_reviewer_credential(tmp_path)
    access_store = tmp_path / "access" / "store.json"
    access_store.parent.mkdir(mode=0o700)
    access_store.write_text("{}", encoding="utf-8")
    access_store.chmod(0o644)

    assert main([*_base_args(tmp_path), "audit"]) == 2
    error = capsys.readouterr().err
    assert str(tmp_path) not in error
    assert "operation could not be completed" in error


def test_unknown_export_command_is_rejected(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main([*_base_args(tmp_path), "export"])
