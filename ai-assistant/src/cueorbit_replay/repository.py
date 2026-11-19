"""Dedicated PostgreSQL storage. Imports atomic; queries parameterized."""

import hashlib
import json
from typing import Protocol

from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from .contracts import ImportResult, ReplayBundleV1, ReplaySession, ReplayShot
from .settings import guarded_url

metadata = MetaData()
bundles = Table(
    "replay_bundles",
    metadata,
    Column("bundle_id", String, primary_key=True),
    Column("session_id", String, unique=True, nullable=False),
    Column("content_hash", String, nullable=False),
    Column("revision", Integer, nullable=False),
    Column("session", JSONB, nullable=False),
    Column("media", JSONB, nullable=False),
)
shots = Table(
    "replay_shots",
    metadata,
    Column("shot_id", String, primary_key=True),
    Column("bundle_id", ForeignKey("replay_bundles.bundle_id"), nullable=False),
    Column("session_id", String, nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("outcome", String, nullable=False),
    Column("evidence_id", String, unique=True, nullable=False),
    Column("payload", JSONB, nullable=False),
)


class RepositoryPort(Protocol):
    async def list_sessions(self) -> list[ReplaySession]: ...
    async def get_summary(self, session_id: str) -> dict | None: ...
    async def list_shots(
        self, session_id: str, *, outcome=None, offset=0, limit=20
    ) -> list[ReplayShot]: ...
    async def get_shot(
        self, session_id: str, *, shot_id=None, sequence=None
    ) -> ReplayShot | None: ...


class ImportConflict(ValueError):
    def __init__(self):
        super().__init__("IMPORT_CONFLICT")


class SessionRepository:
    def __init__(self, dsn: str):
        self.engine = create_async_engine(guarded_url(dsn), echo=False)

    async def initialize(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def close(self):
        await self.engine.dispose()

    async def import_bundle(self, bundle: ReplayBundleV1) -> ImportResult:
        bundle = ReplayBundleV1.model_validate_json(bundle.model_dump_json())
        payload = bundle.model_dump(mode="json")
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        try:
            async with self.engine.begin() as conn:
                # Serialize small local imports, including colliding global evidence IDs.
                await conn.execute(text("SELECT pg_advisory_xact_lock(618241115)"))
                existing = (
                    (
                        await conn.execute(
                            select(bundles).where(
                                (bundles.c.bundle_id == bundle.bundle_id)
                                | (bundles.c.session_id == bundle.session.session_id)
                            )
                        )
                    )
                    .mappings()
                    .first()
                )
                if existing:
                    if existing["content_hash"] == digest:
                        return ImportResult(status="unchanged")
                    raise ImportConflict()
                await conn.execute(
                    bundles.insert().values(
                        bundle_id=bundle.bundle_id,
                        session_id=bundle.session.session_id,
                        content_hash=digest,
                        revision=bundle.revision,
                        session=payload["session"],
                        media=payload["media"],
                    )
                )
                if bundle.shots:
                    await conn.execute(
                        shots.insert(),
                        [
                            dict(
                                shot_id=s.shot_id,
                                bundle_id=bundle.bundle_id,
                                session_id=s.session_id,
                                sequence=s.sequence,
                                outcome=s.outcome,
                                evidence_id=s.evidence_id,
                                payload=s.model_dump(mode="json"),
                            )
                            for s in bundle.shots
                        ],
                    )
        except IntegrityError:
            raise ImportConflict() from None
        return ImportResult(status="imported")

    async def list_sessions(self):
        async with self.engine.connect() as conn:
            rows = await conn.execute(
                select(bundles.c.session).order_by(bundles.c.session_id)
            )
            return [ReplaySession.model_validate(r[0]) for r in rows]

    async def get_summary(self, session_id):
        async with self.engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        select(bundles).where(bundles.c.session_id == session_id)
                    )
                )
                .mappings()
                .first()
            )
            if not row:
                return None
            counts = dict(
                (
                    await conn.execute(
                        select(shots.c.outcome, func.count())
                        .where(shots.c.session_id == session_id)
                        .group_by(shots.c.outcome)
                    )
                ).all()
            )
            refs = tuple(
                (
                    await conn.execute(
                        select(shots.c.evidence_id)
                        .where(shots.c.session_id == session_id)
                        .order_by(shots.c.sequence)
                    )
                ).scalars()
            )
            return dict(
                session_id=session_id,
                title=row["session"]["title"],
                revision=row["revision"],
                data_kind="synthetic",
                recorded_shot_count=sum(counts.values()),
                pocket_observed_count=counts.get("pocket_observed", 0),
                no_pocket_observed_count=counts.get("no_pocket_observed", 0),
                unknown_count=counts.get("unknown", 0),
                limitations=row["session"]["limitations"],
                evidence_refs=list(refs),
            )

    async def list_shots(self, session_id, *, outcome=None, offset=0, limit=20):
        query = select(shots.c.payload).where(shots.c.session_id == session_id)
        if outcome is not None:
            query = query.where(shots.c.outcome == outcome)
        async with self.engine.connect() as conn:
            rows = await conn.execute(
                query.order_by(shots.c.sequence).offset(offset).limit(limit)
            )
            return [ReplayShot.model_validate(r[0]) for r in rows]

    async def get_shot(self, session_id, *, shot_id=None, sequence=None):
        if (shot_id is None) == (sequence is None):
            raise ValueError("exactly one locator")
        query = select(shots.c.payload).where(shots.c.session_id == session_id)
        query = (
            query.where(shots.c.shot_id == shot_id)
            if shot_id is not None
            else query.where(shots.c.sequence == sequence)
        )
        async with self.engine.connect() as conn:
            row = (await conn.execute(query)).first()
            return ReplayShot.model_validate(row[0]) if row else None
