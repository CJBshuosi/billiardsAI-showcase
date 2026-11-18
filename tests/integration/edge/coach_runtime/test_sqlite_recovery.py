from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from billiards_ai.edge.coach_runtime.persistence import (
    PersistenceErrorCode,
    RuntimePersistenceError,
    SqliteRuntimeStore,
)


def test_partial_catalog_is_not_mistaken_for_a_migrated_database(tmp_path: Path) -> None:
    path = tmp_path / "partial.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE runtime_catalog(schema_version INTEGER NOT NULL)")
    with pytest.raises(RuntimePersistenceError) as caught:
        SqliteRuntimeStore(path, device_root_key=b"x" * 32)
    assert caught.value.code is PersistenceErrorCode.CORRUPT_DATABASE


def test_sqlite_header_corruption_is_a_typed_fail_closed_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(b"not a sqlite database")
    with pytest.raises(RuntimePersistenceError) as caught:
        SqliteRuntimeStore(path, device_root_key=b"x" * 32)
    assert caught.value.code is PersistenceErrorCode.CORRUPT_DATABASE
