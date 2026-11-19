"""Two transparent baselines: English FTS and exact pgvector cosine."""

import re

from pydantic import TypeAdapter
from sqlalchemy import and_, func, literal_column, or_, select

from .contracts import RuleChunk, RuleHit, RuleNumber, RulesContext
from .embeddings import validate_vector
from .storage import RuleStore, chunks_table, indexes


def check_top_k(top_k):
    if type(top_k) is not int or not 1 <= top_k <= 5:
        raise ValueError("TOP_K_OUT_OF_RANGE")


class RuleRepository(RuleStore):
    @staticmethod
    def _scope(context: RulesContext):
        return and_(
            indexes.c.ruleset_id == context.ruleset_id,
            indexes.c.version == context.version,
            indexes.c.corpus_hash == context.corpus_hash,
            indexes.c.embedding_revision == context.embedding_revision,
        )

    async def search_keyword(
        self, query: str, context: RulesContext, top_k: int
    ) -> tuple[RuleHit, ...]:
        check_top_k(top_k)
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("INVALID_QUERY")
        locators = re.findall(
            r"\b(?:II|A)\.[1-9][0-9]*(?:\.[1-9][0-9]*)*\b", query, re.IGNORECASE
        )
        if locators:
            exact = {}
            for locator in dict.fromkeys(n.upper() for n in locators):
                for chunk in await self.get_rule(locator, context):
                    exact[chunk.chunk_id] = chunk
            return tuple(
                RuleHit(chunk=c, score=1.0, method="keyword")
                for c in list(exact.values())[:top_k]
            )
        language = literal_column("'english'::regconfig")
        document = func.to_tsvector(
            language, chunks_table.c.heading + " " + chunks_table.c.text
        )
        terms = func.websearch_to_tsquery(language, query)
        rank = func.ts_rank_cd(document, terms)
        statement = (
            select(chunks_table.c.payload, rank.label("score"))
            .select_from(chunks_table.join(indexes))
            .where(self._scope(context), document.op("@@")(terms))
            .order_by(rank.desc(), chunks_table.c.chunk_id)
            .limit(top_k)
        )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return tuple(
            RuleHit(
                chunk=RuleChunk.model_validate(row["payload"]),
                score=float(row["score"]),
                method="keyword",
            )
            for row in rows
        )

    async def search_vector(
        self, vector: list[float], context: RulesContext, top_k: int
    ) -> tuple[RuleHit, ...]:
        check_top_k(top_k)
        vector = validate_vector(vector)
        distance = chunks_table.c.embedding.cosine_distance(vector)
        statement = (
            select(chunks_table.c.payload, distance.label("distance"))
            .select_from(chunks_table.join(indexes))
            .where(self._scope(context))
            .order_by(distance, chunks_table.c.chunk_id)
            .limit(top_k)
        )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return tuple(
            RuleHit(
                chunk=RuleChunk.model_validate(row["payload"]),
                score=1 - float(row["distance"]),
                method="vector",
            )
            for row in rows
        )

    async def get_rule(
        self, rule_number: str, context: RulesContext
    ) -> tuple[RuleChunk, ...]:
        if not isinstance(rule_number, str):
            raise ValueError("INVALID_RULE_NUMBER")
        number = TypeAdapter(RuleNumber).validate_python(rule_number.upper())
        statement = (
            select(chunks_table.c.payload)
            .select_from(chunks_table.join(indexes))
            .where(
                self._scope(context),
                or_(
                    chunks_table.c.rule_number == number,
                    chunks_table.c.rule_number.like(number + ".%"),
                ),
            )
            .order_by(chunks_table.c.rule_number, chunks_table.c.chunk_id)
        )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(statement)).scalars().all()
        return tuple(RuleChunk.model_validate(row) for row in rows)
