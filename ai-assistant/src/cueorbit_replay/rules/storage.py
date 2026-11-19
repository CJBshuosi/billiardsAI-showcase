"""Dedicated pgvector index storage; no access to original/M1 databases."""

import hashlib
import json

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    ForeignKey,
    MetaData,
    String,
    Table,
    and_,
    insert,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine

from ..contracts import ImportResult
from ..repository import ImportConflict
from ..settings import guarded_url
from .contracts import RuleChunk, RulesContext, RuleSourceManifestV1
from .embeddings import validate_vector

metadata = MetaData()
indexes = Table(
    "rule_indexes",
    metadata,
    Column("index_id", String, primary_key=True),
    Column("ruleset_id", String, nullable=False),
    Column("version", String, nullable=False),
    Column("embedding_revision", String, nullable=False),
    Column("corpus_hash", String, nullable=False),
    Column("manifest", JSONB, nullable=False),
)
chunks_table = Table(
    "rule_chunks",
    metadata,
    Column("index_id", ForeignKey("rule_indexes.index_id"), primary_key=True),
    Column("chunk_id", String, primary_key=True),
    Column("rule_number", String, nullable=False),
    Column("heading", String, nullable=False),
    Column("text", String, nullable=False),
    Column("embedding", Vector(384), nullable=False),
    Column("payload", JSONB, nullable=False),
)


def index_id(
    ruleset_id: str,
    version: str,
    revision: str,
    source_id: str,
    extraction_version: str,
) -> str:
    return hashlib.sha256(
        f"{ruleset_id}|{version}|{revision}|{source_id}|{extraction_version}".encode()
    ).hexdigest()


class RuleStore:
    @staticmethod
    def _manifest_scope(manifest):
        return and_(
            indexes.c.ruleset_id == manifest.ruleset_id,
            indexes.c.version == manifest.effective_date,
            indexes.c.embedding_revision == manifest.embedding_revision,
            indexes.c.manifest["source_id"].astext == manifest.source_id,
            indexes.c.manifest["extraction_version"].astext
            == manifest.extraction_version,
        )

    def __init__(self, dsn: str, *, test_only=False):
        url = guarded_url(dsn)
        if url.database not in {
            "cueorbit_replay_m1_m2",
            "cueorbit_replay_m1_m2_test",
        } or (test_only and url.database != "cueorbit_replay_m1_m2_test"):
            raise ValueError("M2_DATABASE_REQUIRED")
        self.engine = create_async_engine(url, echo=False)

    async def initialize(self):
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(metadata.create_all)

    async def close(self):
        await self.engine.dispose()

    async def import_corpus(
        self, manifest: RuleSourceManifestV1, chunks, embeddings
    ) -> ImportResult:
        manifest = RuleSourceManifestV1.model_validate_json(manifest.model_dump_json())
        chunks = tuple(
            RuleChunk.model_validate_json(c.model_dump_json()) for c in chunks
        )
        if (
            not chunks
            or len(chunks) != len(embeddings)
            or len({c.chunk_id for c in chunks}) != len(chunks)
        ):
            raise ValueError("INVALID_CORPUS_BATCH")
        vectors = [validate_vector(v) for v in embeddings]
        for chunk in chunks:
            if (
                chunk.source_id,
                chunk.ruleset_id,
                chunk.version,
                chunk.embedding_revision,
                chunk.embedding_model,
                chunk.data_kind,
            ) != (
                manifest.source_id,
                manifest.ruleset_id,
                manifest.effective_date,
                manifest.embedding_revision,
                manifest.embedding_model,
                manifest.data_kind,
            ):
                raise ValueError("MIXED_CORPUS_IDENTITY")
        pairs = sorted(
            zip(chunks, vectors, strict=True), key=lambda pair: pair[0].chunk_id
        )
        payload = {
            "manifest": manifest.model_dump(mode="json", exclude={"retrieved_at"}),
            "chunks": [(c.model_dump(mode="json"), v) for c, v in pairs],
        }
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        identity = index_id(
            manifest.ruleset_id,
            manifest.effective_date,
            manifest.embedding_revision,
            manifest.source_id,
            manifest.extraction_version,
        )
        async with self.engine.begin() as conn:
            await conn.execute(text("SELECT pg_advisory_xact_lock(641171)"))
            existing = (
                (
                    await conn.execute(
                        select(indexes).where(self._manifest_scope(manifest))
                    )
                )
                .mappings()
                .first()
            )
            if existing:
                if existing["corpus_hash"] != digest:
                    raise ImportConflict()
                return ImportResult(status="unchanged")
            await conn.execute(
                insert(indexes).values(
                    index_id=identity,
                    ruleset_id=manifest.ruleset_id,
                    version=manifest.effective_date,
                    embedding_revision=manifest.embedding_revision,
                    corpus_hash=digest,
                    manifest=manifest.model_dump(mode="json"),
                )
            )
            await conn.execute(
                insert(chunks_table),
                [
                    dict(
                        index_id=identity,
                        chunk_id=c.chunk_id,
                        rule_number=c.rule_number,
                        heading=c.heading,
                        text=c.text,
                        embedding=v,
                        payload=c.model_dump(mode="json"),
                    )
                    for c, v in pairs
                ],
            )
        return ImportResult(status="imported")

    async def context_for(self, manifest) -> RulesContext | None:
        async with self.engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        select(indexes).where(self._manifest_scope(manifest))
                    )
                )
                .mappings()
                .first()
            )
        return (
            RulesContext(
                ruleset_id=row["ruleset_id"],
                version=row["version"],
                corpus_hash=row["corpus_hash"],
                embedding_revision=row["embedding_revision"],
            )
            if row
            else None
        )

    async def list_contexts(self) -> tuple[RulesContext, ...]:
        async with self.engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        select(indexes).order_by(
                            indexes.c.ruleset_id,
                            indexes.c.version,
                            indexes.c.embedding_revision,
                        )
                    )
                )
                .mappings()
                .all()
            )
        return tuple(
            RulesContext(
                ruleset_id=r["ruleset_id"],
                version=r["version"],
                corpus_hash=r["corpus_hash"],
                embedding_revision=r["embedding_revision"],
            )
            for r in rows
        )
