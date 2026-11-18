"""Integration and concurrency coverage for the Story 6.1 JSON authority."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import multiprocessing
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from billiards_ai.cloud.table_binding import (
    JsonTableBindingRepository,
    TableBindingDomainError,
    TableBindingRepositoryError,
)
from billiards_ai.shared.models import (
    TableBinding,
    TableBindingErrorCode,
    TableBindingStatus,
    TableCapacityStatus,
    TableQrRegistration,
    TableRegistration,
    TableRegistrationStatus,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _table(number: int = 3, **updates: object) -> TableRegistration:
    values: dict[str, object] = {
        "table_registration_id": uuid4(),
        "venue_id": "venue-synthetic",
        "table_id": f"table-{number:02d}",
        "device_id": f"device-synthetic-{number:02d}",
        "venue_display_label": "Dilan 球厅",
        "table_display_label": f"{number} 号台",
        "current_qr_version": 1,
        "status": TableRegistrationStatus.ACTIVE,
    }
    values.update(updates)
    return TableRegistration(**values)


def _candidate(
    user_id: UUID,
    table: TableRegistration,
    qr: TableQrRegistration,
    *,
    binding_id: UUID | None = None,
    status: TableBindingStatus = TableBindingStatus.PENDING_CONSENT,
    created_at: datetime = NOW,
) -> TableBinding:
    return TableBinding(
        binding_id=binding_id or uuid4(),
        user_id=user_id,
        table_registration_id=table.table_registration_id,
        qr_registration_id=qr.qr_registration_id,
        qr_version=qr.qr_version,
        status=status,
        created_at=created_at,
        activated_at=created_at if status is TableBindingStatus.ACTIVE else None,
    )


def _payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_binding(path: Path, binding: TableBinding) -> None:
    payload = _payload(path)
    bindings = payload["bindings"]
    assert isinstance(bindings, dict)
    bindings[str(binding.binding_id)] = binding.model_dump(mode="json")
    payload["revision"] = int(payload["revision"]) + 1
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _join_worker(
    path_text: str,
    token: str,
    user_text: str,
    table_payload: dict[str, object],
    qr_payload: dict[str, object],
    start: multiprocessing.synchronize.Event,
    queue: multiprocessing.queues.Queue,
) -> None:
    try:
        table = TableRegistration.model_validate(table_payload)
        qr = TableQrRegistration.model_validate(qr_payload)
        user_id = UUID(user_text)
        start.wait(timeout=10)
        result = JsonTableBindingRepository(path_text).resolve_or_create_pending(
            qr_token=token,
            user_id=user_id,
            candidate_factory=lambda: _candidate(user_id, table, qr),
        )
        queue.put(("ok", str(result.binding.binding_id)))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        queue.put(("error", type(exc).__name__))


def _rotate_worker(
    path_text: str,
    table_id_text: str,
    start: multiprocessing.synchronize.Event,
    queue: multiprocessing.queues.Queue,
) -> None:
    try:
        start.wait(timeout=10)
        result = JsonTableBindingRepository(path_text).rotate_qr(UUID(table_id_text))
        queue.put(("ok", result.qr_token))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        queue.put(("error", type(exc).__name__))


def _provision_worker(
    path_text: str,
    table_payload: dict[str, object],
    start: multiprocessing.synchronize.Event,
    queue: multiprocessing.queues.Queue,
) -> None:
    try:
        start.wait(timeout=10)
        result = JsonTableBindingRepository(path_text).provision_table(
            TableRegistration.model_validate(table_payload)
        )
        queue.put(("ok", result.qr_token))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        queue.put(("error", type(exc).__name__))


def _run_processes(target: object, args_list: list[tuple[object, ...]]) -> list[tuple[str, str]]:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    queue = context.Queue()
    processes = [context.Process(target=target, args=(*args, start, queue)) for args in args_list]
    for process in processes:
        process.start()
    start.set()
    results = [queue.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    return results


def _raise_oserror(*args: object) -> None:
    raise OSError


def test_provisioning_uses_32_byte_csprng_shape_and_persists_only_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import billiards_ai.cloud.table_binding.repository as module

    store = tmp_path / "private" / "table-bindings.json"
    raw = bytes(range(32))
    monkeypatch.setattr(module.secrets, "token_bytes", lambda size: raw)
    repository = JsonTableBindingRepository(store)
    result = repository.provision_table(_table())

    assert result.qr_token == base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    assert len(result.qr_token) == 43
    assert (
        result.qr_registration.token_digest
        == hashlib.sha256(result.qr_token.encode("ascii")).hexdigest()
    )
    stored = store.read_text(encoding="utf-8")
    assert result.qr_token not in stored
    assert result.qr_registration.token_digest in stored
    assert store.stat().st_mode & 0o777 == 0o600
    assert store.parent.stat().st_mode & 0o777 == 0o700
    assert _payload(store)["revision"] == 1


@pytest.mark.parametrize("raw", [b"short", b"\x00" * 32, b"\x00" * 31 + b"\x01"])
def test_low_entropy_or_wrong_length_token_generation_has_zero_change(
    tmp_path: Path, raw: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    import billiards_ai.cloud.table_binding.repository as module

    store = tmp_path / "private" / "table-bindings.json"
    monkeypatch.setattr(module.secrets, "token_bytes", lambda size: raw)
    repository = JsonTableBindingRepository(store)
    with pytest.raises(TableBindingRepositoryError):
        repository.provision_table(_table())
    assert not store.exists()


def test_duplicate_digest_during_provisioning_preserves_existing_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import billiards_ai.cloud.table_binding.repository as module

    store = tmp_path / "private" / "table-bindings.json"
    raw = bytes(range(32))
    monkeypatch.setattr(module.secrets, "token_bytes", lambda size: raw)
    repository = JsonTableBindingRepository(store)
    repository.provision_table(_table(3))
    before = store.read_bytes()
    with pytest.raises(TableBindingRepositoryError):
        repository.provision_table(_table(4))
    assert store.read_bytes() == before


def test_rotation_is_one_atomic_revision_and_old_locator_stops_working(tmp_path: Path) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    before = _payload(store)

    rotated = repository.rotate_qr(provisioned.table_registration.table_registration_id)

    after = _payload(store)
    assert after["revision"] == int(before["revision"]) + 1
    assert rotated.qr_token not in store.read_text(encoding="utf-8")
    statuses = {item["status"] for item in after["qr_registrations"].values()}
    assert statuses == {"CURRENT", "ROTATED"}
    with pytest.raises(TableBindingDomainError) as raised:
        repository.resolve_or_create_pending(
            qr_token=provisioned.qr_token,
            user_id=uuid4(),
            candidate_factory=lambda: _candidate(
                uuid4(), provisioned.table_registration, provisioned.qr_registration
            ),
        )
    assert raised.value.code is TableBindingErrorCode.QR_BINDING_INACTIVE


def test_pending_bindings_never_consume_capacity_and_each_success_mutates_once(
    tmp_path: Path,
) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    for offset in range(5):
        user_id = uuid4()
        result = repository.resolve_or_create_pending(
            qr_token=provisioned.qr_token,
            user_id=user_id,
            candidate_factory=lambda user_id=user_id, offset=offset: _candidate(
                user_id,
                provisioned.table_registration,
                provisioned.qr_registration,
                created_at=NOW + timedelta(seconds=offset),
            ),
        )
        assert result.binding.status is TableBindingStatus.PENDING_CONSENT
        assert result.capacity_status is TableCapacityStatus.AVAILABLE
    assert _payload(store)["revision"] == 6


def test_zero_one_two_active_capacity_snapshots_and_full_failure_is_unchanged(
    tmp_path: Path,
) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    table = provisioned.table_registration
    qr = provisioned.qr_registration

    first_user = uuid4()
    _seed_binding(store, _candidate(first_user, table, qr, status=TableBindingStatus.ACTIVE))
    pending_user = uuid4()
    one_active = repository.resolve_or_create_pending(
        qr_token=provisioned.qr_token,
        user_id=pending_user,
        candidate_factory=lambda: _candidate(pending_user, table, qr),
    )
    assert one_active.capacity_status is TableCapacityStatus.ONE_ACTIVE

    second_user = uuid4()
    _seed_binding(store, _candidate(second_user, table, qr, status=TableBindingStatus.ACTIVE))
    before = store.read_bytes()
    with pytest.raises(TableBindingDomainError) as raised:
        repository.resolve_or_create_pending(
            qr_token=provisioned.qr_token,
            user_id=uuid4(),
            candidate_factory=lambda: _candidate(uuid4(), table, qr),
        )
    assert raised.value.code is TableBindingErrorCode.TABLE_CAPACITY_REACHED
    assert store.read_bytes() == before


def test_same_user_rescan_is_byte_stable_even_with_new_candidate_values(tmp_path: Path) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    user_id = uuid4()
    first = repository.resolve_or_create_pending(
        qr_token=provisioned.qr_token,
        user_id=user_id,
        candidate_factory=lambda: _candidate(
            user_id, provisioned.table_registration, provisioned.qr_registration
        ),
    )
    before = store.read_bytes()
    second = repository.resolve_or_create_pending(
        qr_token=provisioned.qr_token,
        user_id=user_id,
        candidate_factory=lambda: _candidate(
            user_id,
            provisioned.table_registration,
            provisioned.qr_registration,
            created_at=NOW + timedelta(days=1),
        ),
    )
    assert second.binding == first.binding
    assert store.read_bytes() == before


def test_same_user_active_rescan_is_byte_stable(tmp_path: Path) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    owner_id = uuid4()
    active = _candidate(
        owner_id,
        provisioned.table_registration,
        provisioned.qr_registration,
        status=TableBindingStatus.ACTIVE,
    )
    _seed_binding(store, active)
    before = store.read_bytes()
    resolved = repository.resolve_or_create_pending(
        qr_token=provisioned.qr_token,
        user_id=owner_id,
        candidate_factory=lambda: _candidate(
            owner_id,
            provisioned.table_registration,
            provisioned.qr_registration,
        ),
    )
    assert resolved.binding == active
    assert resolved.capacity_status is TableCapacityStatus.ONE_ACTIVE
    assert store.read_bytes() == before


def test_same_user_rescan_after_qr_rotation_keeps_original_binding(tmp_path: Path) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    owner_id = uuid4()
    original = repository.resolve_or_create_pending(
        qr_token=provisioned.qr_token,
        user_id=owner_id,
        candidate_factory=lambda: _candidate(
            owner_id,
            provisioned.table_registration,
            provisioned.qr_registration,
        ),
    )
    rotated = repository.rotate_qr(provisioned.table_registration.table_registration_id)
    before = store.read_bytes()
    resolved = repository.resolve_or_create_pending(
        qr_token=rotated.qr_token,
        user_id=owner_id,
        candidate_factory=lambda: _candidate(
            owner_id,
            rotated.table_registration,
            rotated.qr_registration,
        ),
    )
    assert resolved.binding == original.binding
    assert store.read_bytes() == before


def test_same_user_cannot_move_to_another_table_and_failure_is_byte_stable(
    tmp_path: Path,
) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    first = repository.provision_table(_table(3))
    second = repository.provision_table(_table(4))
    user_id = uuid4()
    repository.resolve_or_create_pending(
        qr_token=first.qr_token,
        user_id=user_id,
        candidate_factory=lambda: _candidate(
            user_id,
            first.table_registration,
            first.qr_registration,
        ),
    )
    before = store.read_bytes()
    with pytest.raises(TableBindingDomainError) as raised:
        repository.resolve_or_create_pending(
            qr_token=second.qr_token,
            user_id=user_id,
            candidate_factory=lambda: _candidate(
                user_id,
                second.table_registration,
                second.qr_registration,
            ),
        )
    assert raised.value.code is TableBindingErrorCode.USER_ALREADY_BOUND
    assert store.read_bytes() == before


def test_qr_and_table_failure_matrix_is_closed_and_has_zero_binding_writes(tmp_path: Path) -> None:
    cases = [
        ("unknown", TableBindingErrorCode.QR_INVALID),
        ("revoked", TableBindingErrorCode.QR_BINDING_INACTIVE),
        ("mismatch", TableBindingErrorCode.TABLE_BINDING_MISMATCH),
        ("inactive", TableBindingErrorCode.TABLE_UNAVAILABLE),
    ]
    for case, expected in cases:
        store = tmp_path / case / "private" / "table-bindings.json"
        repository = JsonTableBindingRepository(store)
        provisioned = repository.provision_table(_table())
        token = provisioned.qr_token
        payload = _payload(store)
        if case == "unknown":
            token = "Z" * 43
        elif case == "revoked":
            payload["qr_registrations"][str(provisioned.qr_registration.qr_registration_id)][
                "status"
            ] = "REVOKED"
        elif case == "mismatch":
            payload["qr_registrations"][str(provisioned.qr_registration.qr_registration_id)][
                "qr_version"
            ] = 2
        else:
            payload["table_registrations"][
                str(provisioned.table_registration.table_registration_id)
            ]["status"] = "INACTIVE"
        store.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        store.chmod(0o600)
        before = store.read_bytes()
        with pytest.raises(TableBindingDomainError) as raised:
            repository.resolve_or_create_pending(
                qr_token=token,
                user_id=uuid4(),
                candidate_factory=lambda provisioned=provisioned: _candidate(
                    uuid4(), provisioned.table_registration, provisioned.qr_registration
                ),
            )
        assert raised.value.code is expected
        assert store.read_bytes() == before


def test_qr_reference_to_missing_table_returns_binding_mismatch_without_mutation(
    tmp_path: Path,
) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    payload = _payload(store)
    payload["qr_registrations"][str(provisioned.qr_registration.qr_registration_id)][
        "table_registration_id"
    ] = str(uuid4())
    store.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    store.chmod(0o600)
    before = store.read_bytes()
    with pytest.raises(TableBindingDomainError) as raised:
        repository.resolve_or_create_pending(
            qr_token=provisioned.qr_token,
            user_id=uuid4(),
            candidate_factory=lambda: _candidate(
                uuid4(),
                provisioned.table_registration,
                provisioned.qr_registration,
            ),
        )
    assert raised.value.code is TableBindingErrorCode.TABLE_BINDING_MISMATCH
    assert store.read_bytes() == before


def test_owner_lookup_hides_foreign_and_absent_records(tmp_path: Path) -> None:
    repository = JsonTableBindingRepository(tmp_path / "private" / "table-bindings.json")
    provisioned = repository.provision_table(_table())
    owner_id = uuid4()
    created = repository.resolve_or_create_pending(
        qr_token=provisioned.qr_token,
        user_id=owner_id,
        candidate_factory=lambda: _candidate(
            owner_id, provisioned.table_registration, provisioned.qr_registration
        ),
    )
    assert repository.find_owned(binding_id=created.binding.binding_id, user_id=owner_id) == created
    assert repository.find_owned(binding_id=created.binding.binding_id, user_id=uuid4()) is None
    assert repository.find_owned(binding_id=uuid4(), user_id=owner_id) is None


def test_binding_to_qr_relationship_corruption_fails_closed(tmp_path: Path) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    owner_id = uuid4()
    created = repository.resolve_or_create_pending(
        qr_token=provisioned.qr_token,
        user_id=owner_id,
        candidate_factory=lambda: _candidate(
            owner_id,
            provisioned.table_registration,
            provisioned.qr_registration,
        ),
    )
    payload = _payload(store)
    qr = payload["qr_registrations"][str(provisioned.qr_registration.qr_registration_id)]
    qr["qr_version"] = 2
    store.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    store.chmod(0o600)
    with pytest.raises(TableBindingRepositoryError):
        repository.find_owned(binding_id=created.binding.binding_id, user_id=owner_id)


def test_threaded_pending_and_same_user_rescan_have_no_lost_updates(tmp_path: Path) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    users = [uuid4() for _ in range(12)]

    def join(user_id: UUID) -> UUID:
        return repository.resolve_or_create_pending(
            qr_token=provisioned.qr_token,
            user_id=user_id,
            candidate_factory=lambda: _candidate(
                user_id, provisioned.table_registration, provisioned.qr_registration
            ),
        ).binding.binding_id

    with ThreadPoolExecutor(max_workers=12) as executor:
        distinct_ids = list(executor.map(join, users))
    assert len(set(distinct_ids)) == 12

    repeated_user = uuid4()
    with ThreadPoolExecutor(max_workers=12) as executor:
        repeated_ids = list(executor.map(join, [repeated_user] * 12))
    assert len(set(repeated_ids)) == 1
    payload = _payload(store)
    assert len(payload["bindings"]) == 13
    assert payload["revision"] == 14


def test_threaded_first_store_provisioning_has_no_lost_updates(tmp_path: Path) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    tables = [_table(number) for number in range(1, 9)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda table: JsonTableBindingRepository(store).provision_table(table),
                tables,
            )
        )
    assert len({item.qr_token for item in results}) == 8
    payload = _payload(store)
    assert len(payload["table_registrations"]) == 8
    assert len(payload["qr_registrations"]) == 8
    assert payload["revision"] == 8


def test_threaded_rotation_serializes_every_new_current(tmp_path: Path) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                repository.rotate_qr,
                [provisioned.table_registration.table_registration_id] * 8,
            )
        )
    assert len({item.qr_token for item in results}) == 8
    payload = _payload(store)
    statuses = [item["status"] for item in payload["qr_registrations"].values()]
    assert payload["revision"] == 9
    assert statuses.count("CURRENT") == 1
    assert statuses.count("ROTATED") == 8


def test_multiprocessing_pending_rescan_and_rotation_have_no_lost_updates(
    tmp_path: Path,
) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    shared_user = uuid4()
    users = [shared_user, shared_user, uuid4(), uuid4(), uuid4()]
    args = [
        (
            str(store),
            provisioned.qr_token,
            str(user_id),
            provisioned.table_registration.model_dump(mode="json"),
            provisioned.qr_registration.model_dump(mode="json"),
        )
        for user_id in users
    ]
    results = _run_processes(_join_worker, args)
    assert {status for status, _ in results} == {"ok"}
    binding_ids = [value for _, value in results]
    assert len(set(binding_ids)) == 4
    assert sorted(binding_ids.count(value) for value in set(binding_ids)) == [1, 1, 1, 2]
    payload = _payload(store)
    assert len(payload["bindings"]) == 4
    assert payload["revision"] == 5

    rotation_results = _run_processes(
        _rotate_worker,
        [(str(store), str(provisioned.table_registration.table_registration_id))] * 4,
    )
    assert {status for status, _ in rotation_results} == {"ok"}
    assert len({token for _, token in rotation_results}) == 4
    payload = _payload(store)
    assert payload["revision"] == 9
    statuses = [item["status"] for item in payload["qr_registrations"].values()]
    assert statuses.count("CURRENT") == 1
    assert statuses.count("ROTATED") == 4


def test_multiprocessing_first_store_provisioning_is_serialized(tmp_path: Path) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    tables = [_table(number) for number in range(1, 6)]
    results = _run_processes(
        _provision_worker,
        [(str(store), table.model_dump(mode="json")) for table in tables],
    )
    assert {status for status, _ in results} == {"ok"}
    assert len({token for _, token in results}) == 5
    payload = _payload(store)
    assert payload["revision"] == 5
    assert len(payload["table_registrations"]) == 5


def test_store_size_depth_and_record_limits_fail_closed(tmp_path: Path) -> None:
    deeply_nested: object = []
    for _ in range(34):
        deeply_nested = [deeply_nested]
    stores = {
        "size": b"{" + b" " * 4_194_304 + b"}",
        "depth": json.dumps(
            {
                "schema_version": "1.0",
                "revision": 0,
                "table_registrations": {},
                "qr_registrations": {},
                "bindings": deeply_nested,
            }
        ).encode(),
        "records": json.dumps(
            {
                "schema_version": "1.0",
                "revision": 0,
                "table_registrations": {str(index): {} for index in range(1001)},
                "qr_registrations": {},
                "bindings": {},
            }
        ).encode(),
    }
    for name, raw in stores.items():
        store = tmp_path / name / "private" / "table-bindings.json"
        store.parent.mkdir(mode=0o700, parents=True)
        store.write_bytes(raw)
        store.chmod(0o600)
        with pytest.raises(TableBindingRepositoryError):
            JsonTableBindingRepository(store).find_owned(binding_id=uuid4(), user_id=uuid4())


@pytest.mark.parametrize("failure", ["corrupt", "symlink", "permissions"])
def test_untrusted_store_conditions_fail_closed(tmp_path: Path, failure: str) -> None:
    store = tmp_path / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    if failure == "corrupt":
        store.write_text("{not-json", encoding="utf-8")
        store.chmod(0o600)
    elif failure == "symlink":
        target = tmp_path / "target.json"
        target.write_bytes(store.read_bytes())
        target.chmod(0o600)
        store.unlink()
        store.symlink_to(target)
    else:
        store.chmod(0o644)
    with pytest.raises(TableBindingRepositoryError):
        repository.resolve_or_create_pending(
            qr_token=provisioned.qr_token,
            user_id=uuid4(),
            candidate_factory=lambda: _candidate(
                uuid4(), provisioned.table_registration, provisioned.qr_registration
            ),
        )


def test_lock_replace_and_fsync_failures_preserve_old_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import billiards_ai.cloud.table_binding.repository as module

    for failure in ("lock", "write", "replace", "fsync"):
        store = tmp_path / failure / "private" / "table-bindings.json"
        repository = JsonTableBindingRepository(store)
        provisioned = repository.provision_table(_table())
        before = store.read_bytes()
        with monkeypatch.context() as scoped:
            if failure == "lock":
                scoped.setattr(module.fcntl, "flock", _raise_oserror)
            elif failure == "write":
                scoped.setattr(module.tempfile, "NamedTemporaryFile", _raise_oserror)
            elif failure == "replace":
                scoped.setattr(module.os, "replace", _raise_oserror)
            else:
                scoped.setattr(module.os, "fsync", _raise_oserror)
            with pytest.raises(TableBindingRepositoryError):
                repository.rotate_qr(provisioned.table_registration.table_registration_id)
        assert store.read_bytes() == before
        assert (
            JsonTableBindingRepository(store).find_owned(binding_id=uuid4(), user_id=uuid4())
            is None
        )


def test_review_repository_constructor_exposes_no_generator_bypass() -> None:
    parameters = inspect.signature(JsonTableBindingRepository).parameters
    assert "token_bytes_factory" not in parameters
    assert "uuid_factory" not in parameters


def test_review_symlink_ancestor_is_rejected(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    repository = JsonTableBindingRepository(alias / "private" / "table-bindings.json")
    with pytest.raises(TableBindingRepositoryError):
        repository.find_owned(binding_id=uuid4(), user_id=uuid4())


@pytest.mark.parametrize("unsafe", ["mode", "nlink"])
def test_review_existing_lock_metadata_is_rejected_without_repair(
    tmp_path: Path, unsafe: str
) -> None:
    store = tmp_path / unsafe / "private" / "table-bindings.json"
    store.parent.mkdir(mode=0o700, parents=True)
    lock = store.with_name(f".{store.name}-lock")
    lock.write_bytes(b"")
    lock.chmod(0o644 if unsafe == "mode" else 0o600)
    if unsafe == "nlink":
        os.link(lock, store.parent / "second-lock-link")
    before_mode = stat.S_IMODE(lock.stat().st_mode)
    with pytest.raises(TableBindingRepositoryError):
        JsonTableBindingRepository(store).find_owned(binding_id=uuid4(), user_id=uuid4())
    assert stat.S_IMODE(lock.stat().st_mode) == before_mode


def test_review_directory_fsync_failure_durably_restores_previous_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import billiards_ai.cloud.table_binding.repository as module

    store = tmp_path / "directory-fsync" / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    before = store.read_bytes()
    real_fsync = module.os.fsync
    calls = 0

    def fail_commit_directory_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", fail_commit_directory_fsync)
    with pytest.raises(TableBindingRepositoryError):
        repository.rotate_qr(provisioned.table_registration.table_registration_id)
    assert calls == 4
    assert store.read_bytes() == before


def test_review_rollback_fsync_failure_reports_commit_state_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import billiards_ai.cloud.table_binding.repository as module

    store = tmp_path / "rollback-fsync" / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    real_fsync = module.os.fsync
    calls = 0

    def fail_commit_and_rollback_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise OSError("synthetic durability failure")
        real_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", fail_commit_and_rollback_fsync)
    with pytest.raises(TableBindingRepositoryError) as raised:
        repository.rotate_qr(provisioned.table_registration.table_registration_id)
    assert type(raised.value).__name__ == "TableBindingCommitStateUnknownError"
    assert calls == 3


def test_review_generation_exception_and_uuid_collision_are_zero_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import billiards_ai.cloud.table_binding.repository as module

    exception_store = tmp_path / "generation" / "private" / "table-bindings.json"

    def broken_token_bytes(size: int) -> bytes:
        raise RuntimeError("synthetic CSPRNG failure")

    monkeypatch.setattr(module.secrets, "token_bytes", broken_token_bytes)
    with pytest.raises(TableBindingRepositoryError):
        JsonTableBindingRepository(exception_store).provision_table(_table())
    assert not exception_store.exists()

    collision_store = tmp_path / "collision" / "private" / "table-bindings.json"
    fixed_qr_id = uuid4()
    raw_values = iter((bytes(range(32)), bytes(range(1, 33))))
    monkeypatch.setattr(module.secrets, "token_bytes", lambda size: next(raw_values))
    monkeypatch.setattr(module, "uuid4", lambda: fixed_qr_id)
    collision_repository = JsonTableBindingRepository(collision_store)
    collision_repository.provision_table(_table(31))
    before = collision_store.read_bytes()
    with pytest.raises(TableBindingRepositoryError):
        collision_repository.provision_table(_table(32))
    assert collision_store.read_bytes() == before


@pytest.mark.parametrize(
    "corruption", ["active_overflow", "no_current", "two_current", "wrong_version"]
)
def test_review_root_authority_invariants_fail_closed(tmp_path: Path, corruption: str) -> None:
    store = tmp_path / corruption / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(_table())
    payload = _payload(store)
    table = provisioned.table_registration
    qr = provisioned.qr_registration
    if corruption == "active_overflow":
        for _ in range(3):
            binding = _candidate(uuid4(), table, qr, status=TableBindingStatus.ACTIVE)
            payload["bindings"][str(binding.binding_id)] = binding.model_dump(mode="json")
    elif corruption == "no_current":
        payload["qr_registrations"][str(qr.qr_registration_id)]["status"] = "ROTATED"
    elif corruption == "two_current":
        second = TableQrRegistration(
            qr_registration_id=uuid4(),
            token_digest="2" * 64,
            table_registration_id=table.table_registration_id,
            qr_version=2,
            status="CURRENT",
        )
        payload["qr_registrations"][str(second.qr_registration_id)] = second.model_dump(mode="json")
    else:
        payload["qr_registrations"][str(qr.qr_registration_id)]["qr_version"] = 2
    store.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    store.chmod(0o600)
    with pytest.raises(TableBindingRepositoryError):
        repository.find_owned(binding_id=uuid4(), user_id=uuid4())


@pytest.mark.parametrize("level", ["root", "nested"])
def test_review_duplicate_json_keys_at_any_level_fail_closed(tmp_path: Path, level: str) -> None:
    store = tmp_path / level / "private" / "table-bindings.json"
    repository = JsonTableBindingRepository(store)
    repository.provision_table(_table())
    raw = store.read_text(encoding="utf-8")
    if level == "root":
        raw = raw.replace('"revision":1', '"revision":0,"revision":1', 1)
    else:
        raw = raw.replace(
            '"venue_id":"venue-synthetic"',
            '"venue_id":"hidden","venue_id":"venue-synthetic"',
            1,
        )
    store.write_text(raw, encoding="utf-8")
    store.chmod(0o600)
    with pytest.raises(TableBindingRepositoryError):
        repository.find_owned(binding_id=uuid4(), user_id=uuid4())
