"""Bind USB shot boundaries to field match, observable rules, and route authority.

Unprofiled legacy sessions still require explicit confirmation.  The pinned Chinese
eight-ball profile may rotate or end only from durable, complete observable facts;
missing facts abstain without mutating match state.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from typing import Literal

from billiards_ai.edge.route_assistance.control import MatchControlSnapshot
from billiards_ai.edge.route_assistance.eight_ball_rules import (
    RuleOutcomeKind,
    decide_eight_ball,
)
from billiards_ai.edge.route_assistance.models import GuidanceDisplay
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import (
    FieldSessionStatus,
    FieldShotRecord,
    LiveMatchMode,
    LiveShotFact,
    SessionLimitCode,
    ShotMediaRange,
)
from billiards_ai.edge.route_assistance.rule_facts import RealtimeRuleFacts
from billiards_ai.edge.route_assistance.turn_confirmation import TurnConfirmationRef
from billiards_ai.lab.field_assistance.guidance import (
    FieldGuidanceResult,
    build_match_field_guidance,
)
from billiards_ai.lab.field_assistance.observations import build_snapshot_table_state
from billiards_ai.lab.mainline1_product_realtime_runtime import LiveSettledShot
from billiards_ai.lab.shot_local.models import ObservationFrame

LiveRuleStatus = Literal[
    "READY",
    "EVALUATING",
    "AWAITING_PLACEMENT",
    "RULES_BLOCKED",
    "RULES_PENDING",
    "SESSION_ENDED",
]


def _legacy_observed_outcome(facts: tuple[dict[str, object], ...]) -> str:
    if not facts:
        return "unknown"
    return (
        "observed-"
        + sha256(json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )


def _authoritative_turn_action(
    shot: LiveSettledShot,
    facts: tuple[dict[str, object], ...],
    *,
    session_id: str,
    match_revision: int,
) -> Literal["continue_turn", "switch_turn"] | None:
    """Accept one exact, coverage-backed legacy turn decision and nothing else."""

    decisions = tuple(fact for fact in facts if fact.get("event_type") == "turn_decision")
    if len(decisions) != 1:
        return None
    fact = decisions[0]
    expected_keys = {
        "event_type",
        "action",
        "session_id",
        "shot_id",
        "source_epoch_id",
        "match_revision",
    }
    action = fact.get("action")
    coverage = shot.fact_coverage.get("turn_decision")
    if (
        set(fact) != expected_keys
        or action not in {"continue_turn", "switch_turn"}
        or fact.get("session_id") != session_id
        or fact.get("shot_id") != shot.shot_id
        or fact.get("source_epoch_id") != shot.source_epoch_id
        or fact.get("match_revision") != match_revision
        or coverage is None
        or coverage.status != "observed"
    ):
        return None
    return action


@dataclass(frozen=True, slots=True)
class LiveMatchProjection:
    session_id: str
    shot_id: str
    match: MatchControlSnapshot
    rule_status: LiveRuleStatus
    confirmation_ref: TurnConfirmationRef | None
    guidance: GuidanceDisplay | None
    route_abstained_reason: str | None
    rule_pending_reason: str | None = None


class LiveMatchCoordinator:
    def __init__(
        self,
        *,
        store: SQLiteFieldRecordStore,
        session_id: str,
        source_epoch_id: str,
        calibration_id: str | None,
        mode: LiveMatchMode | None = None,
    ) -> None:
        if not source_epoch_id.strip():
            raise ValueError("live match requires a source epoch")
        self._store = store
        self._session_id = session_id
        self._source_epoch_id = source_epoch_id
        self._calibration_id = calibration_id
        self._match = store.load_session(session_id).match
        frozen_mode = None if self._match.mode is None else self._match.mode.value
        if mode is not None and frozen_mode is not None and mode != frozen_mode:
            raise ValueError("live match mode conflicts with frozen settings")
        self._fallback_mode: LiveMatchMode = mode or "assistance"
        self._settled: LiveSettledShot | None = None
        self._settled_reference: TurnConfirmationRef | None = None
        self._settled_evidence_digest: str | None = None
        self._guidance_reference: TurnConfirmationRef | None = None
        self._guidance_result: FieldGuidanceResult | None = None
        self._route_reason: str | None = None
        self._pending_reason = "AUTHORITATIVE_RULE_FACTS_INCOMPLETE"
        self._stable_observation_frame_id: int | None = None

    @property
    def guidance_result(self) -> FieldGuidanceResult | None:
        # Refresh authority before exposing a cached route.
        _ = self.snapshot
        return self._guidance_result

    @property
    def snapshot(self) -> LiveMatchProjection:
        session = self._store.load_session(self._session_id)
        if session.match != self._match:
            self._clear_guidance()
            self._match = session.match
        shots = self._store.list_shots(self._session_id)
        shot_id = shots[-1].shot_id if shots else ""
        receipt = self._store.get_turn_confirmation(self._session_id, shot_id) if shots else None
        reference = None
        if session.status is FieldSessionStatus.ENDED:
            status: LiveRuleStatus = "SESSION_ENDED"
            self._clear_guidance()
        elif session.match.rule_profile_id is not None:
            rule_receipt = (
                None
                if not shots
                else self._store.get_rule_decision(self._session_id, shots[-1].shot_id)
            )
            if session.match.rack_phase.value == "ended":
                # The deterministic terminal result is committed, but the
                # session remains open until SHOT_ENQUEUED proves the postgame
                # job and media reference are durable.
                status = "EVALUATING"
                self._clear_guidance()
                self._pending_reason = "TERMINAL_JOB_NOT_DURABLE"
            elif not shots:
                status = "READY"
            elif shots[-1].settled_ms is None or rule_receipt is None:
                status = "EVALUATING"
                self._clear_guidance()
            elif rule_receipt.status != "applied":
                status = "RULES_PENDING"
                if rule_receipt.status != "abstained":
                    self._clear_guidance()
                self._pending_reason = f"RULE_DECISION_{rule_receipt.status.upper()}"
            elif session.match.cue_ball_placement.value != "none":
                status = "AWAITING_PLACEMENT"
                self._clear_guidance()
            else:
                status = "READY"
        elif (
            receipt is not None
            and receipt.status == "resolved"
            and receipt.reference.match_revision + 1 == session.match.revision
        ):
            status = "READY"
            if self._guidance_result is not None and (
                self._guidance_reference != receipt.reference
                or not self._observations_match(shots[-1], receipt.reference)
            ):
                self._clear_guidance()
                self._route_reason = "settled_observation_lineage_mismatch"
        else:
            status = "RULES_PENDING"
            self._clear_guidance()
            if receipt is not None and receipt.status == "pending":
                reference = receipt.reference
        return LiveMatchProjection(
            self._session_id,
            shot_id,
            session.match,
            status,
            reference,
            None if self._guidance_result is None else self._guidance_result.display,
            self._route_reason,
            self._pending_reason if status in {"RULES_PENDING", "RULES_BLOCKED"} else None,
        )

    def _clear_guidance(self) -> None:
        self._guidance_result = None
        self._guidance_reference = None
        self._route_reason = None

    def observe_stable(
        self,
        observation: ObservationFrame,
        *,
        source_epoch_id: str,
    ) -> LiveMatchProjection:
        """Plan from the current trusted stable scene without claiming rule authority."""
        if source_epoch_id != self._source_epoch_id:
            raise ValueError("stale stable observation source epoch")
        if observation.frame_id < 0 or not observation.observations:
            raise ValueError("stable observation is empty")
        session = self._store.load_session(self._session_id)
        if session.status is FieldSessionStatus.ENDED:
            self._clear_guidance()
            return self.snapshot
        shots = self._store.list_shots(self._session_id)
        if shots and shots[-1].settled_ms is None:
            self._clear_guidance()
            return self.snapshot
        self._match = session.match
        if self._current_mode() == "observation":
            self._clear_guidance()
            self._stable_observation_frame_id = observation.frame_id
            return self.snapshot
        if self._match.cue_ball_placement.value != "none":
            self._clear_guidance()
            self._route_reason = "cue_ball_placement_required"
            return self.snapshot
        if (
            self._stable_observation_frame_id == observation.frame_id
            and self._guidance_result is not None
        ):
            return self.snapshot
        self._clear_guidance()
        self._stable_observation_frame_id = observation.frame_id
        try:
            state = build_snapshot_table_state(
                observation,
                calibration_id=self._calibration_id or "",
                source_ref=(f"usb:{source_epoch_id}/stable#frame={observation.frame_id}"),
                scene_generation=self._match.revision,
            )
            self._guidance_result = build_match_field_guidance(
                state=state,
                match=self._match,
                scene_generation=self._match.revision,
            )
            self._route_reason = self._guidance_result.abstained_reason
        except ValueError as exc:
            self._route_reason = f"snapshot_unplannable:{exc}"
        return self.snapshot

    def reject_stable_observation(self, reason: str) -> LiveMatchProjection:
        """Revoke only route input while leaving capture and shot authority alive."""
        self._clear_guidance()
        self._route_reason = f"stable_observation_rejected:{reason}"
        return self.snapshot

    def _current_mode(self) -> LiveMatchMode:
        return self._fallback_mode if self._match.mode is None else self._match.mode.value

    def _observations_match(
        self,
        persisted: FieldShotRecord,
        reference: TurnConfirmationRef,
    ) -> bool:
        shot = self._settled
        return bool(
            shot is not None
            and self._settled_reference == reference
            and reference.session_id == persisted.session_id == self._session_id
            and reference.shot_id == persisted.shot_id == shot.shot_id
            and persisted.media.source_kind == "usb_live"
            and persisted.media.source_epoch_id == shot.source_epoch_id == self._source_epoch_id
            and persisted.live_evidence_digest == self._settled_evidence_digest
            and self._settled_evidence_digest is not None
            and persisted.live_job_id == shot.job_id
        )

    def shot_started(self, *, shot_id: str, started_ms: int) -> LiveMatchProjection:
        self._clear_guidance()
        self._stable_observation_frame_id = None
        session = self._store.load_session(self._session_id)
        self._match = session.match
        shots = self._store.list_shots(self._session_id)
        existing = next((shot for shot in shots if shot.shot_id == shot_id), None)
        if existing is not None:
            if (
                existing.started_ms != started_ms
                or existing.media.source_epoch_id != self._source_epoch_id
            ):
                raise ValueError("conflicting live shot start")
            return self.snapshot
        source_event = sha256(
            f"{self._session_id}|{self._source_epoch_id}|{shot_id}|{started_ms}".encode()
        ).hexdigest()
        shot = FieldShotRecord(
            self._session_id,
            shot_id,
            len(shots) + 1,
            session.match.active_role,
            started_ms,
            None,
            None,
            None,
            None,
            ShotMediaRange(
                session.artifact_id,
                f"usb:{self._source_epoch_id}",
                "usb_live",
                started_ms,
                started_ms + 1,
                self._source_epoch_id,
            ),
            (SessionLimitCode.NO_GLOBAL_FIXED_BALL_ID, SessionLimitCode.NO_AUTOMATIC_FOUL_VERDICT),
        )
        self._store.record_shot_start(shot, source_start_event_id=source_event)
        self._settled = None
        self._pending_reason = "AUTHORITATIVE_RULE_FACTS_INCOMPLETE"
        return self.snapshot

    def settle(
        self,
        shot: LiveSettledShot,
        *,
        facts: tuple[dict[str, object], ...],
    ) -> LiveMatchProjection:
        if shot.source_epoch_id != self._source_epoch_id:
            raise ValueError("stale shot source epoch")
        shots = self._store.list_shots(self._session_id)
        original = next((row for row in shots if row.shot_id == shot.shot_id), None)
        if original is None:
            raise ValueError("settled shot has no persisted start")
        if shots[-1].shot_id != shot.shot_id:
            raise ValueError("stale settled shot")
        if self._store.load_session(self._session_id).status is FieldSessionStatus.ENDED:
            raise ValueError("stale ended session")
        settled_ms = shot.settled_at_ns // 1_000_000
        evidence_digest = sha256(
            json.dumps(
                {"shot": asdict(shot), "facts": facts},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        settled = replace(
            original,
            settled_ms=settled_ms,
            media=replace(original.media, end_ms=settled_ms),
            live_evidence_digest=evidence_digest,
            live_job_id=shot.job_id,
        )
        live_fact = self._live_shot_fact(
            settled,
            shot,
            observed_outcome=_legacy_observed_outcome(facts),
        )
        self._store.record_shot_settle(
            settled,
            source_epoch_id=shot.source_epoch_id,
            live_fact=live_fact,
        )
        if self._settled is not None and self._settled.shot_id == shot.shot_id:
            if self._settled != shot:
                raise ValueError("conflicting settled observation")
            return self.snapshot
        self._clear_guidance()
        self._settled = shot
        receipt = self._store.get_turn_confirmation(self._session_id, shot.shot_id)
        self._settled_reference = None if receipt is None else receipt.reference
        self._settled_evidence_digest = evidence_digest
        automatic_action = _authoritative_turn_action(
            shot,
            facts,
            session_id=self._session_id,
            match_revision=self._match.revision,
        )
        if automatic_action is not None and receipt is not None:
            command_id = f"edge-auto-{evidence_digest[:48]}"
            self._match = self._store.apply_confirmed_turn(
                command_id,
                automatic_action,
                receipt.reference,
            )
            self._build_guidance(receipt.reference)
            self._pending_reason = ""
            return self.snapshot
        observed = bool(facts) and all(
            isinstance(fact.get("event_type"), str)
            and (coverage := shot.fact_coverage.get(fact["event_type"])) is not None
            and coverage.status == "observed"
            for fact in facts
        )
        self._pending_reason = (
            "COMPLETE_TURN_RULE_EVALUATOR_UNSUPPORTED"
            if observed and shot.publishable and shot.shot_abstention_reason is None
            else "AUTHORITATIVE_RULE_FACTS_INCOMPLETE"
        )
        return self.snapshot

    def settle_profiled(
        self,
        shot: LiveSettledShot,
        facts: RealtimeRuleFacts,
        *,
        elapsed_ms: float,
        fact_preimage_json: str | None = None,
    ) -> LiveMatchProjection:
        """Commit one bounded observable-rule result and refresh realtime guidance."""

        session = self._store.load_session(self._session_id)
        if session.match.rule_profile_id is None:
            raise ValueError("profiled settlement requires a pinned rule profile")
        if (
            facts.session_id != self._session_id
            or facts.shot_id != shot.shot_id
            or facts.source_epoch_id != shot.source_epoch_id
            or facts.profile_id != session.match.rule_profile_id
        ):
            raise ValueError("profiled settlement lineage mismatch")
        if shot.source_epoch_id != self._source_epoch_id:
            raise ValueError("profiled settlement source lineage mismatch")
        shots = self._store.list_shots(self._session_id)
        original = next((row for row in shots if row.shot_id == shot.shot_id), None)
        if original is None or shots[-1].shot_id != shot.shot_id:
            raise ValueError("profiled settlement has no latest persisted start")
        settled_ms = shot.settled_at_ns // 1_000_000
        settled = replace(
            original,
            settled_ms=settled_ms,
            media=replace(original.media, end_ms=settled_ms),
            live_evidence_digest=facts.settled_evidence_digest,
            live_job_id=shot.job_id,
        )
        existing_receipt = self._store.get_rule_decision(self._session_id, shot.shot_id)
        decision = None if existing_receipt is not None else decide_eight_ball(session.match, facts)
        receipt = self._store.settle_and_apply_automatic_rule_decision(
            settled,
            facts,
            decision,
            source_epoch_id=shot.source_epoch_id,
            elapsed_ms=elapsed_ms,
            fact_preimage_json=fact_preimage_json,
            live_fact=self._live_shot_fact(
                settled,
                shot,
                observed_outcome=f"rules-{facts.digest}",
            ),
        )
        self._settled = shot
        self._settled_reference = None
        self._settled_evidence_digest = facts.settled_evidence_digest
        self._match = self._store.load_session(self._session_id).match
        self._clear_guidance()
        if (
            receipt.status == "applied"
            and json.loads(receipt.decision_json).get("kind") == RuleOutcomeKind.APPLY.value
            and json.loads(receipt.decision_json).get("action") != "end"
            and self._match.cue_ball_placement.value == "none"
        ):
            self._build_profiled_guidance()
        elif receipt.status != "applied":
            self._pending_reason = f"RULE_DECISION_{receipt.status.upper()}"
        return self.snapshot

    def _live_shot_fact(
        self,
        settled: FieldShotRecord,
        shot: LiveSettledShot,
        *,
        observed_outcome: str,
    ) -> LiveShotFact | None:
        existing = self._store.get_live_shot_fact(settled.session_id, settled.shot_id)
        if existing is not None:
            return existing
        ownership = self._store.get_shot_ownership(settled.session_id, settled.shot_id)
        if ownership is None or ownership.evidence_status != "bound_at_start":
            return None
        role = ownership.actor_role
        actor_user_id = ownership.actor_user_id
        return LiveShotFact.create(
            session_id=settled.session_id,
            shot_id=settled.shot_id,
            sequence=settled.sequence,
            start_frame_id=shot.start_frame_id,
            settle_frame_id=shot.settled_frame_id,
            mode=self._current_mode(),
            actor_role=role,
            actor_user_id=actor_user_id,
            observed_outcome=observed_outcome,
            rule_revision=self._match.revision,
            recording_reference=(
                f"session-recording:{shot.source_epoch_id}:"
                f"{shot.start_frame_id}-{shot.settled_frame_id}"
            ),
        )

    def finalize_profiled_terminal(self, *, shot_id: str, job_id: str) -> LiveMatchProjection:
        self._store.finalize_automatic_terminal_decision(
            self._session_id,
            shot_id,
            job_id,
        )
        self._match = self._store.load_session(self._session_id).match
        self._clear_guidance()
        return self.snapshot

    def confirm_turn(
        self,
        command_id: str,
        action: Literal["continue_turn", "switch_turn"],
        reference: TurnConfirmationRef,
    ) -> LiveMatchProjection:
        current = self.snapshot
        if reference.session_id != self._session_id or reference.shot_id != current.shot_id:
            raise ValueError("stale session or shot confirmation")
        if current.rule_status == "SESSION_ENDED" or current.match.revision not in {
            reference.match_revision,
            reference.match_revision + 1,
        }:
            raise ValueError("stale match revision confirmation")
        self._match = self._store.apply_confirmed_turn(command_id, action, reference)
        self._build_guidance(reference)
        return self.snapshot

    def _build_guidance(self, reference: TurnConfirmationRef) -> None:
        self._clear_guidance()
        if self._current_mode() == "observation":
            return
        shot = self._settled
        if shot is None:
            self._route_reason = "settled_observation_unavailable"
            return
        persisted = self._store.list_shots(self._session_id)
        if not persisted or not self._observations_match(persisted[-1], reference):
            self._route_reason = "settled_observation_lineage_mismatch"
            return
        if not shot.publishable or shot.shot_abstention_reason is not None:
            self._route_reason = shot.shot_abstention_reason or "settled_observation_unpublishable"
            return
        if not self._calibration_id:
            self._route_reason = "calibration_unavailable"
            return
        terminal = next(
            (frame for frame in shot.observations if frame.frame_id == shot.settled_frame_id), None
        )
        if terminal is None:
            self._route_reason = "settled_observation_unavailable"
            return
        try:
            state = build_snapshot_table_state(
                terminal,
                calibration_id=self._calibration_id,
                source_ref=f"usb:{shot.source_epoch_id}/{shot.shot_id}#frame={shot.settled_frame_id}",
                scene_generation=self._match.revision,
            )
            self._guidance_result = build_match_field_guidance(
                state=state,
                match=self._match,
                scene_generation=self._match.revision,
            )
            self._guidance_reference = reference
            self._route_reason = self._guidance_result.abstained_reason
        except ValueError as exc:
            self._route_reason = f"snapshot_unplannable:{exc}"

    def _build_profiled_guidance(self) -> None:
        self._clear_guidance()
        if self._current_mode() == "observation":
            return
        shot = self._settled
        if shot is None:
            self._route_reason = "settled_observation_unavailable"
            return
        persisted = self._store.list_shots(self._session_id)
        if (
            not persisted
            or persisted[-1].shot_id != shot.shot_id
            or persisted[-1].media.source_kind != "usb_live"
            or persisted[-1].media.source_epoch_id != shot.source_epoch_id
            or persisted[-1].live_evidence_digest != self._settled_evidence_digest
            or persisted[-1].live_job_id != shot.job_id
        ):
            self._route_reason = "settled_observation_lineage_mismatch"
            return
        if not shot.publishable or shot.shot_abstention_reason is not None:
            self._route_reason = shot.shot_abstention_reason or "settled_observation_unpublishable"
            return
        if not self._calibration_id:
            self._route_reason = "calibration_unavailable"
            return
        terminal = next(
            (frame for frame in shot.observations if frame.frame_id == shot.settled_frame_id), None
        )
        if terminal is None:
            self._route_reason = "settled_observation_unavailable"
            return
        try:
            state = build_snapshot_table_state(
                terminal,
                calibration_id=self._calibration_id,
                source_ref=(
                    f"usb:{shot.source_epoch_id}/{shot.shot_id}#frame={shot.settled_frame_id}"
                ),
                scene_generation=self._match.revision,
            )
            self._guidance_result = build_match_field_guidance(
                state=state,
                match=self._match,
                scene_generation=self._match.revision,
            )
            self._route_reason = self._guidance_result.abstained_reason
        except ValueError as exc:
            self._route_reason = f"snapshot_unplannable:{exc}"

    def refresh_confirmed_turn(self, reference: TurnConfirmationRef) -> LiveMatchProjection:
        """Replan after the existing authenticated cloud ledger applied this receipt."""
        current = self.snapshot
        receipt = self._store.get_turn_confirmation(self._session_id, reference.shot_id)
        if (
            current.rule_status != "READY"
            or receipt is None
            or receipt.reference != reference
            or receipt.status != "resolved"
            or current.shot_id != reference.shot_id
        ):
            raise ValueError("stale turn confirmation")
        self._build_guidance(reference)
        return self.snapshot

    def end_and_rollover(self) -> tuple[str, str]:
        ended = self._store.end_session(self._session_id)
        next_session = self._store.open_or_resume_session(
            artifact_id=ended.artifact_id,
            rule_profile_id=ended.match.rule_profile_id,
        )
        self._session_id = next_session.session_id
        self._match = next_session.match
        self._settled = None
        self._clear_guidance()
        return ended.session_id, next_session.session_id
