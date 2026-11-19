from cueorbit_replay.fixtures import load_builtin_bundles


class FixtureRepository:
    """Test-only isolated fixture double; live/default app always PostgreSQL."""

    def __init__(self):
        self.bundles = {b.session.session_id: b for b in load_builtin_bundles()}

    async def list_sessions(self):
        return [b.session for b in self.bundles.values()]

    async def get_summary(self, session_id):
        b = self.bundles.get(session_id)
        if not b:
            return None
        return dict(
            session_id=session_id,
            title=b.session.title,
            revision=b.revision,
            data_kind="synthetic",
            recorded_shot_count=len(b.shots),
            pocket_observed_count=sum(s.outcome == "pocket_observed" for s in b.shots),
            no_pocket_observed_count=sum(
                s.outcome == "no_pocket_observed" for s in b.shots
            ),
            unknown_count=sum(s.outcome == "unknown" for s in b.shots),
            limitations=list(b.session.limitations),
            evidence_refs=[s.evidence_id for s in b.shots],
        )

    async def list_shots(self, session_id, *, outcome=None, offset=0, limit=20):
        rows = sorted(
            (
                s
                for s in self.bundles[session_id].shots
                if outcome is None or s.outcome == outcome
            ),
            key=lambda s: s.sequence,
        )
        return rows[offset : offset + limit]

    async def get_shot(self, session_id, *, shot_id=None, sequence=None):
        return next(
            (
                s
                for s in self.bundles[session_id].shots
                if s.shot_id == shot_id or s.sequence == sequence
            ),
            None,
        )


class ScriptedModel:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.messages = []
        self.timeouts = []

    async def next_turn(self, messages, tool_schemas, *, timeout_s, required_tool=None):
        self.messages.append(messages.copy())
        self.timeouts.append(timeout_s)
        result = next(self.turns)
        if isinstance(result, Exception):
            raise result
        return result
