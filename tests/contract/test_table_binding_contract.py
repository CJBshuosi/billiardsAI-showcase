"""Story 6.1 strict table-binding contracts and public-boundary regressions."""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models import (
    JoinTableRequest,
    TableBinding,
    TableBindingErrorCode,
    TableBindingNextAction,
    TableBindingStatus,
    TableBindingView,
    TableCapacityStatus,
    TableQrRegistration,
    TableQrStatus,
    TableRegistration,
    TableRegistrationStatus,
)

TOKEN = "A" * 43
DIGEST = "1" * 64


def _registration(**updates: object) -> TableRegistration:
    values: dict[str, object] = {
        "table_registration_id": uuid4(),
        "venue_id": "venue-synthetic",
        "table_id": "table-03",
        "device_id": "device-synthetic-03",
        "venue_display_label": "Dilan 球厅",
        "table_display_label": "3 号台",
        "current_qr_version": 1,
        "status": TableRegistrationStatus.ACTIVE,
    }
    values.update(updates)
    return TableRegistration(**values)


def _qr(table_registration_id: UUID, **updates: object) -> TableQrRegistration:
    values: dict[str, object] = {
        "qr_registration_id": uuid4(),
        "token_digest": DIGEST,
        "table_registration_id": table_registration_id,
        "qr_version": 1,
        "status": TableQrStatus.CURRENT,
    }
    values.update(updates)
    return TableQrRegistration(**values)


def _binding(table: TableRegistration, qr: TableQrRegistration, **updates: object) -> TableBinding:
    values: dict[str, object] = {
        "binding_id": uuid4(),
        "user_id": uuid4(),
        "table_registration_id": table.table_registration_id,
        "qr_registration_id": qr.qr_registration_id,
        "qr_version": qr.qr_version,
        "status": TableBindingStatus.PENDING_CONSENT,
        "created_at": datetime(2026, 7, 29, 12, 0, tzinfo=UTC),
        "activated_at": None,
    }
    values.update(updates)
    return TableBinding(**values)


@pytest.mark.parametrize(
    "payload",
    [
        {"qr_token": TOKEN},
        {"schema_version": "2.0", "qr_token": TOKEN},
        {"schema_version": "1.0", "qr_token": ""},
        {"schema_version": "1.0", "qr_token": " " + TOKEN},
        {"schema_version": "1.0", "qr_token": "A" * 42},
        {"schema_version": "1.0", "qr_token": "A" * 44},
        {"schema_version": "1.0", "qr_token": "A" * 42 + "="},
        {"schema_version": "1.0", "qr_token": "A" * 42 + "+"},
        {"schema_version": "1.0", "qr_token": "A" * 42 + "\x00"},
        {"schema_version": "1.0", "qr_token": 7},
        {"schema_version": "1.0", "qr_token": True},
        {"schema_version": "1.0", "qr_token": TOKEN, "user_id": str(uuid4())},
        {"schema_version": "1.0", "qr_token": TOKEN, "openid": "synthetic"},
        {"schema_version": "1.0", "qr_token": TOKEN, "session_id": str(uuid4())},
        {"schema_version": "1.0", "qr_token": TOKEN, "device_id": "device"},
        {"schema_version": "1.0", "qr_token": TOKEN, "consent": True},
    ],
)
def test_join_request_rejects_noncanonical_or_authority_bearing_payloads(
    payload: dict[str, object],
) -> None:
    """Removing strict request parsing would accept caller-controlled authority."""
    with pytest.raises(ValidationError):
        JoinTableRequest.model_validate(payload)


def test_join_request_accepts_only_the_canonical_locator_contract() -> None:
    request = JoinTableRequest(schema_version="1.0", qr_token=TOKEN)
    assert request.model_dump(mode="json") == {"schema_version": "1.0", "qr_token": TOKEN}
    assert JoinTableRequest.model_validate_json(request.model_dump_json()) == request


def test_review_qr_token_rejects_noncanonical_base64url_pad_bits() -> None:
    """A 43-character alphabet match is insufficient when trailing pad bits are non-zero."""
    with pytest.raises(ValidationError):
        JoinTableRequest(schema_version="1.0", qr_token="A" * 42 + "B")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("venue_id", "   "),
        ("table_id", "table\n03"),
        ("device_id", "device\x7f03"),
        ("venue_display_label", "venue\x00label"),
        ("venue_display_label", "venue\u202elabel"),
        ("table_display_label", "\t"),
        ("current_qr_version", True),
        ("current_qr_version", "1"),
    ],
)
def test_table_registration_rejects_ambiguous_identifiers_and_numbers(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        _registration(**{field: value})


def test_qr_and_binding_require_uuid4_strict_digest_and_aware_utc() -> None:
    table = _registration()
    with pytest.raises(ValidationError):
        _qr(table.table_registration_id, token_digest="A" * 64)
    with pytest.raises(ValidationError):
        _qr(table.table_registration_id, qr_version=True)
    with pytest.raises(ValidationError):
        TableQrRegistration(
            qr_registration_id=UUID("00000000-0000-1000-8000-000000000001"),
            token_digest=DIGEST,
            table_registration_id=table.table_registration_id,
            qr_version=1,
            status=TableQrStatus.CURRENT,
        )
    qr = _qr(table.table_registration_id)
    with pytest.raises(ValidationError):
        _binding(table, qr, created_at=datetime(2026, 7, 29, 12, 0))
    with pytest.raises(ValidationError):
        _binding(table, qr, binding_id=UUID("00000000-0000-1000-8000-000000000001"))


@pytest.mark.parametrize("ambiguous_time", [0, 1.5, True, False])
def test_review_binding_datetimes_reject_numeric_and_boolean_coercion(
    ambiguous_time: object,
) -> None:
    table = _registration()
    qr = _qr(table.table_registration_id)
    with pytest.raises(ValidationError):
        _binding(table, qr, created_at=ambiguous_time)


def test_review_valid_iso_datetime_json_still_round_trips() -> None:
    table = _registration()
    qr = _qr(table.table_registration_id)
    binding = _binding(table, qr)
    assert TableBinding.model_validate_json(binding.model_dump_json()) == binding


def test_review_sprint_last_updated_does_not_move_backward() -> None:
    sprint = Path(__file__).parents[2] / "_bmad-output/implementation-artifacts/sprint-status.yaml"
    line = next(
        item for item in sprint.read_text(encoding="utf-8").splitlines() if "last_updated:" in item
    )
    actual = datetime.fromisoformat(line.split("last_updated:", 1)[1].strip())
    assert actual >= datetime.fromisoformat("2026-07-29T17:10:00+02:00")


def test_binding_lifecycle_is_strict_frozen_and_round_trips() -> None:
    table = _registration()
    qr = _qr(table.table_registration_id)
    pending = _binding(table, qr)
    assert pending.activated_at is None
    with pytest.raises(ValidationError):
        _binding(
            table,
            qr,
            status=TableBindingStatus.PENDING_CONSENT,
            activated_at=datetime(2026, 7, 29, 12, 1, tzinfo=UTC),
        )
    with pytest.raises(ValidationError):
        _binding(table, qr, status=TableBindingStatus.ACTIVE, activated_at=None)
    with pytest.raises(ValidationError):
        _binding(
            table,
            qr,
            status=TableBindingStatus.ACTIVE,
            activated_at=datetime(2026, 7, 29, 11, 59, tzinfo=UTC),
        )
    with pytest.raises(ValidationError):
        pending.status = TableBindingStatus.ACTIVE
    assert TableBinding.model_validate_json(pending.model_dump_json()) == pending


def test_owner_view_is_a_minimal_non_sensitive_projection() -> None:
    table = _registration()
    qr = _qr(table.table_registration_id)
    binding = _binding(table, qr)
    view = TableBindingView(
        binding_id=binding.binding_id,
        venue_display_label=table.venue_display_label,
        table_display_label=table.table_display_label,
        qr_version=1,
        binding_status=TableBindingStatus.PENDING_CONSENT,
        capacity_status=TableCapacityStatus.AVAILABLE,
        created_at=binding.created_at,
        next_action=TableBindingNextAction.REVIEW_CONSENT,
    )
    payload = json.loads(view.model_dump_json())
    assert set(payload) == {
        "schema_version",
        "binding_id",
        "venue_display_label",
        "table_display_label",
        "qr_version",
        "binding_status",
        "capacity_status",
        "created_at",
        "next_action",
    }
    serialized = view.model_dump_json().lower()
    for forbidden in (
        "user_id",
        "openid",
        "device_id",
        "token",
        "digest",
        "session_id",
        "upload",
        "path",
    ):
        assert forbidden not in serialized


def test_closed_error_enum_contains_only_story_6_1_codes() -> None:
    assert {item.value for item in TableBindingErrorCode} == {
        "IDENTITY_REQUIRED",
        "QR_INVALID",
        "QR_VERSION_UNSUPPORTED",
        "QR_BINDING_INACTIVE",
        "TABLE_BINDING_MISMATCH",
        "TABLE_UNAVAILABLE",
        "TABLE_CAPACITY_REACHED",
        "USER_ALREADY_BOUND",
        "RESOURCE_NOT_FOUND",
        "TABLE_BINDING_REPOSITORY_UNAVAILABLE",
    }


def test_story_6_1_public_surface_has_no_activation_or_session_creation() -> None:
    from billiards_ai.cloud.table_binding import TableBindingService

    assert not hasattr(TableBindingService, "activate")
    assert not hasattr(TableBindingService, "create_session")


def test_story_6_1_modules_do_not_import_out_of_scope_runtime_boundaries() -> None:
    package = Path(__file__).parents[2] / "src" / "billiards_ai" / "cloud" / "table_binding"
    assert package.is_dir()
    forbidden_roots = {"fastapi", "sqlalchemy", "jwt", "requests", "httpx", "wechat"}
    forbidden_imports: list[str] = []
    forbidden_session_symbols: list[str] = []
    for source_path in package.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                forbidden_imports.extend(
                    name.name for name in node.names if name.name.split(".")[0] in forbidden_roots
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in forbidden_roots or ".privacy" in node.module:
                    forbidden_imports.append(node.module)
            elif (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "session" in node.name
            ):
                forbidden_session_symbols.append(node.name)
    assert forbidden_imports == []
    assert set(forbidden_session_symbols) <= {
        "list_staff_sessions",
        "read_staff_session",
    }
