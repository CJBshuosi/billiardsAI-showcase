from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.field_assistance.cloud_gateway import FieldSessionLease

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/algorithm/retire_existing_field_session.py"


def _load():
    assert SCRIPT.is_file(), "existing-session retirement helper is missing"
    spec = importlib.util.spec_from_file_location("retire_existing_session", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Gateway:
    def __init__(self, *, failure: OSError | None = None) -> None:
        self.failure = failure
        self.closed: list[str] = []

    def close_field_session(self, session_id: str) -> FieldSessionLease:
        if self.failure is not None:
            raise self.failure
        self.closed.append(session_id)
        return FieldSessionLease(session_id, "raytheon-01", "ended")


def _open_cloud_session(path: Path) -> SQLiteFieldRecordStore:
    store = SQLiteFieldRecordStore(path, session_id_factory=lambda: "session-old")
    store.open_or_resume_session(artifact_id="development-replay")
    store.record_cloud_session_open("session-old")
    return store


def test_retirement_closes_the_cloud_lease_then_ends_the_existing_local_session(
    tmp_path: Path,
) -> None:
    module = _load()
    path = tmp_path / "field-state.sqlite3"
    store = _open_cloud_session(path)

    result = module.retire_existing_session(
        state_db=path,
        gateway=_Gateway(),
        expected_table_id="raytheon-01",
    )

    assert result == {
        "schema_version": "1.0",
        "status": "RETIRED",
        "session_id": "session-old",
        "table_id": "raytheon-01",
    }
    assert store.load_session("session-old").status.value == "ended"
    assert store.list_pending_cloud_close_session_ids() == ()
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT status FROM field_cloud_close_intents WHERE session_id='session-old'"
        ).fetchone() == ("complete",)


def test_retirement_leaves_local_session_open_when_cloud_close_fails(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "field-state.sqlite3"
    store = _open_cloud_session(path)

    with pytest.raises(OSError, match="SYNC_CONFLICT"):
        module.retire_existing_session(
            state_db=path,
            gateway=_Gateway(failure=OSError("SYNC_CONFLICT")),
            expected_table_id="raytheon-01",
        )

    assert store.load_session("session-old").status.value == "open"
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT status FROM field_cloud_close_intents WHERE session_id='session-old'"
        ).fetchone() == ("open",)


def test_retirement_is_a_noop_when_the_installed_database_has_no_open_session(
    tmp_path: Path,
) -> None:
    module = _load()
    path = tmp_path / "field-state.sqlite3"
    SQLiteFieldRecordStore(path)

    result = module.retire_existing_session(
        state_db=path,
        gateway=_Gateway(),
        expected_table_id="raytheon-01",
    )

    assert result == {
        "schema_version": "1.0",
        "status": "NOT_NEEDED",
        "session_id": None,
        "table_id": "raytheon-01",
    }


def test_retirement_ignores_a_local_session_that_never_acquired_a_cloud_lease(
    tmp_path: Path,
) -> None:
    module = _load()
    path = tmp_path / "field-state.sqlite3"
    store = SQLiteFieldRecordStore(path, session_id_factory=lambda: "session-local")
    store.open_or_resume_session(artifact_id="failed-cloud-open")
    gateway = _Gateway()

    result = module.retire_existing_session(
        state_db=path,
        gateway=gateway,
        expected_table_id="raytheon-01",
    )

    assert result["status"] == "NOT_NEEDED"
    assert gateway.closed == []
    assert store.load_session("session-local").status.value == "open"


def test_retirement_completes_an_already_ended_local_session_pending_cloud_close(
    tmp_path: Path,
) -> None:
    module = _load()
    path = tmp_path / "field-state.sqlite3"
    store = _open_cloud_session(path)
    store.end_session("session-old")
    gateway = _Gateway()

    result = module.retire_existing_session(
        state_db=path,
        gateway=gateway,
        expected_table_id="raytheon-01",
    )

    assert result["status"] == "RETIRED"
    assert gateway.closed == ["session-old"]
    assert store.list_pending_cloud_close_session_ids() == ()


def test_retirement_rejects_ambiguous_open_session_authority(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "field-state.sqlite3"
    ids = iter(("session-a", "session-b"))
    store = SQLiteFieldRecordStore(path, session_id_factory=lambda: next(ids))
    first = store.open_or_resume_session(artifact_id="artifact-a")
    second = store.open_or_resume_session(artifact_id="artifact-b")
    store.record_cloud_session_open(first.session_id)
    store.record_cloud_session_open(second.session_id)

    with pytest.raises(ValueError, match="exactly one"):
        module.retire_existing_session(
            state_db=path,
            gateway=_Gateway(),
            expected_table_id="raytheon-01",
        )
