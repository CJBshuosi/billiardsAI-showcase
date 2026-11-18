from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

from billiards_ai.cloud.db import create_database_engine, create_session_factory

PROJECT_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.fail(
            "PostgreSQL integration tests require the dedicated TEST_DATABASE_URL; "
            "DATABASE_URL is intentionally ignored"
        )
    parsed = make_url(url)
    if parsed.drivername != "postgresql+asyncpg":
        pytest.fail("TEST_DATABASE_URL must use the postgresql+asyncpg driver")
    database_name = parsed.database or ""
    if "test" not in database_name.lower():
        pytest.fail("TEST_DATABASE_URL must name a dedicated database containing 'test'")
    return url


@pytest.fixture
def migrated_quality_gate(test_database_url: str):
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", test_database_url)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    yield
    command.downgrade(config, "base")


@pytest.fixture
async def database_session_factory(migrated_quality_gate, test_database_url: str):
    engine = create_database_engine(test_database_url)
    factory = create_session_factory(engine)
    yield factory
    await engine.dispose()
