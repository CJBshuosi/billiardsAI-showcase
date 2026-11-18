"""Durable bounded SQL discovery; the existing pinned executor owns physical work."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from billiards_ai.edge.route_assistance.owner_device_delivery import _state, _state_key, _time

if TYPE_CHECKING:
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore


@dataclass(frozen=True, slots=True)
class LocalCleanupCandidate:
    kind: Literal["owner", "discard", "recording"]
    session_id: str


class LocalMediaCleanupScheduler:
    def __init__(
        self,
        store: SQLiteFieldRecordStore,
        managed_scope: ManagedSessionScope | None = None,
        exclude_managed: bool = False,
    ):
        if type(exclude_managed) is not bool:
            raise ValueError("invalid cleanup exclusion")
        self.store = store
        self.scope = None if managed_scope is None else store._managed_scope_values(managed_scope)
        self.exclude_managed = exclude_managed
        self.key = _state_key("local-cleanup", self.scope, exclude_managed)

    def _candidate_in(self, db, kind, now):
        if kind == "owner" and self.scope is None:
            # Owner authority is never adopted from a current/default scope.
            return None
        table = {
            "owner": "field_owner_media_deletions",
            "discard": "field_local_media_discards",
            "recording": "field_recording_cleanup",
        }[kind]
        statuses = "'pending','retry'" if kind == "owner" else "'pending'"
        query = (
            f" FROM {table} d WHERE ((d.status IN ({statuses}) AND "
            "d.next_attempt_at_utc<=?) OR (d.status='leased' AND d.lease_until_utc<=?)) "
        )
        parameters = [now, now]
        if self.scope is not None:
            query += (
                "AND EXISTS (SELECT 1 FROM field_managed_open_requests m "
                "WHERE m.session_id=d.session_id AND m.environment_id=? AND m.device_id=? "
                "AND m.table_id=? AND m.phase='materialized') "
            )
            parameters.extend(self.scope)
        if kind != "owner" and self.exclude_managed:
            query += (
                "AND d.session_id NOT LIKE 'managed-%' AND NOT EXISTS "
                "(SELECT 1 FROM field_managed_open_requests m WHERE m.session_id=d.session_id) "
            )
        key = self.key + ":" + kind
        state = _state(db, key, "owner")
        cursor, upper = state["cursor"], state["upper_key"]
        # At most one old finite sweep and one fresh wrap, never a Python scan.
        for _ in range(2):
            if upper is None:
                upper = db.execute("SELECT MAX(d.session_id)" + query, parameters).fetchone()[0]
                cursor = ""
            row = (
                None
                if upper is None
                else db.execute(
                    "SELECT d.session_id"
                    + query
                    + "AND d.session_id>? AND d.session_id<=? ORDER BY d.session_id LIMIT 1",
                    (*parameters, cursor, upper),
                ).fetchone()
            )
            if row is not None:
                # This transaction commits before the caller sees/resolves a candidate.
                db.execute(
                    "UPDATE field_local_media_schedule_state SET cursor=?,upper_key=? "
                    "WHERE state_key=?",
                    (row[0], upper, key),
                )
                return LocalCleanupCandidate(kind, row[0])
            db.execute(
                "UPDATE field_local_media_schedule_state SET cursor='',upper_key=NULL "
                "WHERE state_key=?",
                (key,),
            )
            if upper is None:
                break
            cursor, upper = "", None
        return None

    def next_candidate(self, now: datetime) -> LocalCleanupCandidate | None:
        due = _time(now)
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            kind = _state(db, self.key, "owner")["turn"]
            order = ("owner", "discard", "recording")
            start = order.index(kind)
            candidate = None
            for offset in range(len(order)):
                candidate = self._candidate_in(db, order[(start + offset) % len(order)], due)
                if candidate is not None:
                    break
            if candidate is not None:
                db.execute(
                    "UPDATE field_local_media_schedule_state SET turn=? WHERE state_key=?",
                    (order[(order.index(candidate.kind) + 1) % len(order)], self.key),
                )
            return candidate
