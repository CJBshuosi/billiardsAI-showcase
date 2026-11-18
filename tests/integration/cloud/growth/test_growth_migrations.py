from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[4]
GROWTH_TABLES = {
    "growth_evidence_batches",
    "growth_evidence_inbox",
    "growth_evidence_sources",
    "growth_metric_authorities",
    "growth_snapshots",
    "growth_snapshot_idempotency",
    "growth_snapshot_heads",
    "growth_metrics",
    "growth_plan_outcomes",
    "growth_outcome_inbox",
    "core_intelligence_outbox",
}


def _config(url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    return config


def _catalog(sync_connection):
    catalog = inspect(sync_connection)
    foreign_keys = {table: catalog.get_foreign_keys(table) for table in sorted(GROWTH_TABLES)}
    return {
        "tables": set(catalog.get_table_names()),
        "foreign_keys": foreign_keys,
        "head_uniques": catalog.get_unique_constraints("growth_snapshots"),
        "metric_indexes": catalog.get_indexes("growth_metrics"),
        "snapshot_fks": catalog.get_foreign_keys("growth_snapshots"),
        "receipt_fks": catalog.get_foreign_keys("growth_snapshot_idempotency"),
        "outbox_columns": {
            item["name"] for item in catalog.get_columns("core_intelligence_outbox")
        },
        "outbox_indexes": catalog.get_indexes("core_intelligence_outbox"),
    }


@pytest.mark.asyncio
async def test_growth_migration_empty_round_trip_and_catalog(test_database_url: str) -> None:
    config = _config(test_database_url)
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == ["20260816_03"]
    await asyncio.to_thread(command.downgrade, config, "base")
    await asyncio.to_thread(command.upgrade, config, "head")
    engine = create_async_engine(test_database_url)
    try:
        async with engine.connect() as connection:
            revision = (
                await connection.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one()
            catalog = await connection.run_sync(_catalog)
        assert revision == "20260816_03"
        assert GROWTH_TABLES <= catalog["tables"]
        assert {item["name"] for item in catalog["head_uniques"]} >= {
            "uq_growth_snapshot_version",
            "uq_growth_snapshot_content",
            "uq_growth_snapshot_head_target",
        }
        assert {item["name"] for item in catalog["metric_indexes"]} >= {
            "ix_growth_metrics_snapshot",
            "ix_growth_metrics_query",
        }
        assert {item["name"] for item in catalog["snapshot_fks"]} >= {
            "fk_growth_snapshot_adjacent_predecessor"
        }
        assert {item["name"] for item in catalog["receipt_fks"]} >= {
            "fk_growth_snapshot_receipt_exact_target"
        }
        assert catalog["outbox_columns"] >= {
            "owner_user_id",
            "aggregate_namespace",
            "available_at",
            "published_at",
            "attempt_count",
            "claimed_by",
            "claimed_at",
            "claim_until",
            "claim_token",
            "last_error",
        }
        assert {item["name"] for item in catalog["outbox_indexes"]} >= {
            "ix_core_intelligence_outbox_pending"
        }
        referenced = {
            key["referred_table"] for values in catalog["foreign_keys"].values() for key in values
        }
        assert not any("media" in table or "recording" in table for table in referenced)
    finally:
        await engine.dispose()
    await asyncio.to_thread(command.downgrade, config, "20260810_02")
    await asyncio.to_thread(command.upgrade, config, "head")
