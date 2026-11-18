"""Project one retained realtime tracked window into observable rule facts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from billiards_ai.edge.route_assistance.control import CueBallPlacement, WristbandRole
from billiards_ai.lab.events.contact import observe_first_contact
from billiards_ai.lab.events.cushion import observe_cushion_contacts
from billiards_ai.lab.events.models import BallState, EventStatus, EvidenceEvent, TrackedFrame
from billiards_ai.lab.events.pot import observe_pots
from billiards_ai.lab.shot_local.events import ShotLocalEventConfig
from billiards_ai.lab.shot_local.models import BallGroup

if TYPE_CHECKING:
    from billiards_ai.lab.mainline1_realtime_refinement import (
        FactCoverageVerdict,
        RealtimeEvidenceCoverage,
    )


class FactTruth(StrEnum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RealtimePotFact:
    track_id: str
    group: BallGroup
    pocket_id: str
    last_seen_frame_id: int
    first_missing_frame_id: int
    confirmed_through_frame_id: int
    evidence_frame_ids: tuple[int, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class RealtimeRuleFacts:
    session_id: str
    shot_id: str
    source_epoch_id: str
    match_revision_at_start: int
    actor_role: WristbandRole
    start_frame_id: int
    settled_frame_id: int
    settled_evidence_digest: str
    first_contact_group: BallGroup | None
    pots: tuple[RealtimePotFact, ...]
    explicit_no_pot: FactTruth
    cue_ball_scratch: FactTruth
    post_contact_rail_or_pot: FactTruth
    distinct_break_object_rail_count: int | None
    before_counts: Mapping[BallGroup, int]
    after_counts: Mapping[BallGroup, int]
    object_ball_off_table: FactTruth
    evidence_frame_ids: tuple[int, ...]
    abstention_reasons: tuple[str, ...]
    digest: str
    profile_id: str = ""
    cue_ball_placement_observed: FactTruth = FactTruth.TRUE


@dataclass(frozen=True, slots=True)
class RealtimeRuleFactsBuild:
    facts: RealtimeRuleFacts
    normalization_preimage_json: str


class RuleFactDeadlineExceeded(TimeoutError):
    """The realtime fact scan reached its caller-owned monotonic deadline."""


class _SettledShot(Protocol):
    shot_id: str
    start_frame_id: int
    settled_frame_id: int
    settled_at_ns: int
    publishable: bool
    shot_abstention_reason: str | None
    fact_coverage: Mapping[str, FactCoverageVerdict]
    source_epoch_id: str
    job_id: str
    rule_frames: tuple[TrackedFrame, ...]
    rule_coverage: RealtimeEvidenceCoverage | None
    event_config: ShotLocalEventConfig | None
    event_config_digest: str | None


def _canonical_preimage_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(_canonical_preimage_json(payload).encode()).hexdigest()


def validate_realtime_rule_fact_preimage(
    facts_payload: Mapping[str, object],
    normalization_preimage_json: str,
    *,
    expected_digest: str,
) -> dict[str, object]:
    """Verify the retained v1 hash input, not the quality of its observations."""

    def reject(reason: str) -> None:
        raise ValueError(f"rule fact preimage {reason}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                reject("has duplicate keys")
            result[key] = value
        return result

    def nonfinite(_value):
        reject("contains nonfinite values")

    def finite(value):
        if type(value) is float and not math.isfinite(value):
            reject("contains nonfinite values")
        if type(value) is dict:
            for item in value.values():
                finite(item)
        elif type(value) is list:
            for item in value:
                finite(item)

    try:
        if type(normalization_preimage_json) is not str or not isinstance(facts_payload, Mapping):
            reject("input type is invalid")
        if (
            type(expected_digest) is not str
            or re.fullmatch("[0-9a-f]{64}", expected_digest) is None
        ):
            reject("digest is invalid")
        fact_keys = {item.name for item in fields(RealtimeRuleFacts)}
        if set(facts_payload) != fact_keys or facts_payload["digest"] != expected_digest:
            reject("fact fields or digest conflict")
        normalized = json.loads(
            normalization_preimage_json, object_pairs_hook=unique, parse_constant=nonfinite
        )
        if type(normalized) is not dict or set(normalized) != (fact_keys - {"digest"}) | {
            "event_config_digest",
            "required_cue_ball_placement",
        }:
            reject("normalization fields conflict")
        finite(normalized)
        if _canonical_preimage_json(normalized) != normalization_preimage_json:
            reject("is not canonical v1 JSON")
        if hashlib.sha256(normalization_preimage_json.encode()).hexdigest() != expected_digest:
            reject("hash conflicts")
        placement = normalized["required_cue_ball_placement"]
        if type(placement) is not str or placement not in {item.value for item in CueBallPlacement}:
            reject("required placement is invalid")
        config_digest = normalized["event_config_digest"]
        if config_digest is not None and (
            type(config_digest) is not str or re.fullmatch("[0-9a-f]{64}", config_digest) is None
        ):
            reject("event configuration digest is invalid")
        projected = {
            key: value
            for key, value in normalized.items()
            if key not in {"event_config_digest", "required_cue_ball_placement"}
        }
        expected = {key: value for key, value in facts_payload.items() if key != "digest"}
        # Canonical JSON comparison preserves bool/int and number distinctions;
        # Python mapping equality alone would accept True == 1.
        if _canonical_preimage_json(projected) != _canonical_preimage_json(expected):
            reject("fact projection conflicts")
        return normalized
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("rule fact preimage "):
            raise
        raise ValueError("rule fact preimage is invalid") from exc


def canonical_event_config_digest(config: ShotLocalEventConfig) -> str:
    return _canonical_digest(asdict(config))


def _deadline(
    monotonic_ns: Callable[[], int], deadline_ns: int, *, stage: str | None = None
) -> None:
    if monotonic_ns() >= deadline_ns:
        suffix = "" if stage is None else f" during {stage}"
        raise RuleFactDeadlineExceeded(f"realtime rule fact deadline exceeded{suffix}")


class _DeadlineFrames(Sequence[TrackedFrame]):
    def __init__(
        self,
        frames: Sequence[TrackedFrame],
        *,
        monotonic_ns: Callable[[], int],
        deadline_ns: int,
        stage: str,
    ) -> None:
        self._frames = frames
        self._monotonic_ns = monotonic_ns
        self._deadline_ns = deadline_ns
        self._stage = stage

    def __len__(self) -> int:
        _deadline(self._monotonic_ns, self._deadline_ns, stage=self._stage)
        return len(self._frames)

    def __getitem__(self, index: int | slice) -> TrackedFrame | _DeadlineFrames:
        _deadline(self._monotonic_ns, self._deadline_ns, stage=self._stage)
        if isinstance(index, slice):
            return _DeadlineFrames(
                self._frames[index],
                monotonic_ns=self._monotonic_ns,
                deadline_ns=self._deadline_ns,
                stage=self._stage,
            )
        return self._frames[index]


def _group(identity: str) -> BallGroup:
    prefix = identity.split(":", 1)[0]
    try:
        return BallGroup(prefix)
    except ValueError:
        return BallGroup.UNKNOWN


def _counts(balls: Sequence[BallState]) -> dict[BallGroup, int]:
    counts = Counter(_group(ball.identity) for ball in balls)
    return {
        group: counts[group]
        for group in (
            BallGroup.CUE,
            BallGroup.EIGHT,
            BallGroup.SOLID,
            BallGroup.STRIPE,
        )
    }


def _frame_by_id(frames: Sequence[TrackedFrame]) -> dict[int, TrackedFrame]:
    return {frame.frame_id: frame for frame in frames}


def _unique_ball(frame: TrackedFrame, identity: str) -> BallState | None:
    candidates = tuple(ball for ball in frame.balls if ball.identity == identity)
    return candidates[0] if len(candidates) == 1 else None


def _coverage_observed(
    shot: _SettledShot,
    name: str,
    reasons: set[str],
    *,
    record_reason: bool = True,
) -> bool:
    verdict = shot.fact_coverage.get(name)
    if verdict is not None and verdict.status == "observed":
        return True
    reason = None if verdict is None else verdict.reason
    if record_reason:
        reasons.add(f"{name}_{reason or 'coverage_missing'}")
    return False


def _pot_fact(
    event: EvidenceEvent,
    *,
    frames: Sequence[TrackedFrame],
    frame_index: Mapping[int, TrackedFrame],
    before_by_track: Mapping[str, BallState],
    after_by_track: Mapping[str, BallState],
    uncertain_by_track: Mapping[str, tuple[tuple[int, str], ...]],
    pocket_radius_mm: float,
    trusted_after_frame_id: int,
) -> RealtimePotFact | None:
    if event.status is not EventStatus.OBSERVED or len(event.evidence_frame_ids) != 3:
        return None
    last_seen, first_missing, persistence_confirmed = event.evidence_frame_ids
    if trusted_after_frame_id < persistence_confirmed:
        return None
    last_frame = frame_index.get(last_seen)
    if last_frame is None or event.actor_identity is None:
        return None
    candidate = _unique_ball(last_frame, event.actor_identity)
    if candidate is None or not candidate.track_id:
        return None
    before = before_by_track.get(candidate.track_id)
    if before is None or _group(before.identity) is not _group(candidate.identity):
        return None
    if candidate.track_id in after_by_track:
        return None
    if any(
        ball.track_id == candidate.track_id
        for frame in frames
        if frame.frame_id >= first_missing
        for ball in frame.balls
    ):
        return None
    states = uncertain_by_track.get(candidate.track_id, ())
    if (first_missing, "lost") not in states or any(
        frame_id < first_missing or state != "lost" for frame_id, state in states
    ):
        return None
    if not event.reason or not event.reason.startswith("pocket:"):
        return None
    pocket_id = event.reason.removeprefix("pocket:")
    group = _group(candidate.identity)
    if group is BallGroup.UNKNOWN:
        return None

    # A replacement track of the same semantic ball beside the same pocket is
    # indistinguishable from an ID switch, not positive pot evidence.
    if any(
        other.track_id not in before_by_track
        and _group(other.identity) is group
        and (other.x_mm - candidate.x_mm) ** 2 + (other.y_mm - candidate.y_mm) ** 2
        <= pocket_radius_mm**2
        for other in after_by_track.values()
    ):
        return None
    return RealtimePotFact(
        track_id=candidate.track_id,
        group=group,
        pocket_id=pocket_id,
        last_seen_frame_id=last_seen,
        first_missing_frame_id=first_missing,
        confirmed_through_frame_id=trusted_after_frame_id,
        evidence_frame_ids=tuple(
            dict.fromkeys((*event.evidence_frame_ids, trusted_after_frame_id))
        ),
        confidence=event.confidence,
    )


def _settled_digest(shot: _SettledShot, frames: Sequence[TrackedFrame]) -> str:
    coverage = shot.rule_coverage
    return _canonical_digest(
        {
            "shot_id": shot.shot_id,
            "source_epoch_id": shot.source_epoch_id,
            "start_frame_id": shot.start_frame_id,
            "settled_frame_id": shot.settled_frame_id,
            "settled_at_ns": shot.settled_at_ns,
            "job_id": shot.job_id,
            "publishable": shot.publishable,
            "shot_abstention_reason": shot.shot_abstention_reason,
            "fact_coverage": {
                name: {"status": verdict.status, "reason": verdict.reason}
                for name, verdict in sorted(shot.fact_coverage.items())
            },
            "rule_frames": [asdict(frame) for frame in frames],
            "coverage": None if coverage is None else asdict(coverage),
            "event_config_digest": shot.event_config_digest,
        }
    )


def build_abstained_realtime_rule_facts_with_preimage(
    shot: _SettledShot,
    *,
    session_id: str,
    actor_role: WristbandRole,
    match_revision: int,
    profile_id: str,
    reason: str,
    required_cue_ball_placement: CueBallPlacement = CueBallPlacement.NONE,
) -> RealtimeRuleFactsBuild:
    """Build a complete-lineage, unknown-observable receipt after a hard fact gap."""

    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("rule fact abstention reason is required")
    frames = tuple(shot.rule_frames)
    before_counts = _counts(()) if not frames else _counts(frames[0].balls)
    after_counts = _counts(()) if not frames else _counts(frames[-1].balls)
    evidence_frame_ids = tuple(sorted({shot.start_frame_id, shot.settled_frame_id}))
    settled_evidence_digest = _settled_digest(shot, frames)
    abstention_reasons = (normalized_reason,)
    normalized = {
        "session_id": session_id,
        "shot_id": shot.shot_id,
        "source_epoch_id": shot.source_epoch_id,
        "match_revision_at_start": match_revision,
        "actor_role": actor_role.value,
        "profile_id": profile_id,
        "required_cue_ball_placement": required_cue_ball_placement.value,
        "cue_ball_placement_observed": FactTruth.UNKNOWN.value,
        "start_frame_id": shot.start_frame_id,
        "settled_frame_id": shot.settled_frame_id,
        "settled_evidence_digest": settled_evidence_digest,
        "first_contact_group": None,
        "pots": [],
        "explicit_no_pot": FactTruth.UNKNOWN.value,
        "cue_ball_scratch": FactTruth.UNKNOWN.value,
        "post_contact_rail_or_pot": FactTruth.UNKNOWN.value,
        "distinct_break_object_rail_count": None,
        "before_counts": {group.value: count for group, count in before_counts.items()},
        "after_counts": {group.value: count for group, count in after_counts.items()},
        "object_ball_off_table": FactTruth.UNKNOWN.value,
        "evidence_frame_ids": list(evidence_frame_ids),
        "abstention_reasons": list(abstention_reasons),
        "event_config_digest": shot.event_config_digest,
    }
    normalization_preimage_json = _canonical_preimage_json(normalized)
    facts = RealtimeRuleFacts(
        session_id=session_id,
        shot_id=shot.shot_id,
        source_epoch_id=shot.source_epoch_id,
        match_revision_at_start=match_revision,
        actor_role=actor_role,
        start_frame_id=shot.start_frame_id,
        settled_frame_id=shot.settled_frame_id,
        settled_evidence_digest=settled_evidence_digest,
        first_contact_group=None,
        pots=(),
        explicit_no_pot=FactTruth.UNKNOWN,
        cue_ball_scratch=FactTruth.UNKNOWN,
        post_contact_rail_or_pot=FactTruth.UNKNOWN,
        distinct_break_object_rail_count=None,
        before_counts=before_counts,
        after_counts=after_counts,
        object_ball_off_table=FactTruth.UNKNOWN,
        evidence_frame_ids=evidence_frame_ids,
        abstention_reasons=abstention_reasons,
        digest=hashlib.sha256(normalization_preimage_json.encode()).hexdigest(),
        profile_id=profile_id,
        cue_ball_placement_observed=FactTruth.UNKNOWN,
    )
    return RealtimeRuleFactsBuild(facts, normalization_preimage_json)


def build_realtime_rule_facts_with_preimage(
    shot: _SettledShot,
    *,
    session_id: str,
    actor_role: WristbandRole,
    match_revision: int,
    profile_id: str,
    required_cue_ball_placement: CueBallPlacement = CueBallPlacement.NONE,
    deadline_ns: int,
    monotonic_ns: Callable[[], int],
) -> RealtimeRuleFactsBuild:
    """Use only the immutable settled snapshot; no media or detector access."""

    _deadline(monotonic_ns, deadline_ns)
    frames = tuple(shot.rule_frames)
    reasons: set[str] = set()
    for _frame in frames:
        _deadline(monotonic_ns, deadline_ns)

    coverage = shot.rule_coverage
    config = shot.event_config
    trusted_before_frame_id: int | None = None
    trusted_after_frame_id: int | None = None
    expected_ids = () if not frames else tuple(range(frames[0].frame_id, frames[-1].frame_id + 1))
    if not frames:
        reasons.add("rule_frames_missing")
    elif tuple(frame.frame_id for frame in frames) != expected_ids:
        reasons.add("rule_frame_gap")
    if coverage is None:
        reasons.add("rule_coverage_missing")
    else:
        if coverage.source_frame_ids != expected_ids:
            reasons.add("source_frame_gap")
        if not coverage.anchor_frame_ids:
            reasons.add("anchor_coverage_missing")
        else:
            trusted_before_frame_id = max(
                (
                    frame_id
                    for frame_id in coverage.anchor_frame_ids
                    if frame_id < shot.start_frame_id
                ),
                default=None,
            )
            trusted_after_frame_id = max(
                (
                    frame_id
                    for frame_id in coverage.anchor_frame_ids
                    if frame_id > shot.settled_frame_id
                ),
                default=None,
            )
            if trusted_before_frame_id is None:
                reasons.add("trusted_before_anchor_missing")
            if trusted_after_frame_id is None:
                reasons.add("trusted_after_anchor_missing")
            if any(frame_id not in expected_ids for frame_id in coverage.anchor_frame_ids):
                reasons.add("anchor_source_lineage_mismatch")
        if not coverage.segment_sha256:
            reasons.add("segment_hashes_missing")
        if tuple(sorted(set(coverage.uncertain_frame_ids))) != tuple(
            sorted({row[0] for row in coverage.uncertain_track_states})
        ):
            reasons.add("uncertain_lineage_mismatch")
    if config is None:
        reasons.add("event_config_missing")
    elif shot.event_config_digest != canonical_event_config_digest(config):
        reasons.add("event_config_digest_mismatch")
    if not shot.publishable or shot.shot_abstention_reason is not None:
        reasons.add(shot.shot_abstention_reason or "shot_not_publishable")
    for name in ("table_state_before", "table_state_after"):
        _coverage_observed(shot, name, reasons)
    for frame in frames:
        track_ids = tuple(ball.track_id for ball in frame.balls)
        if not all(track_ids) or len(track_ids) != len(set(track_ids)):
            reasons.add(f"rule_track_identity_invalid:{frame.frame_id}")
        for ball in frame.balls:
            group = _group(ball.identity)
            canonical = (
                ball.identity in {"cue", "eight"}
                if group in {BallGroup.CUE, BallGroup.EIGHT}
                else ball.identity == f"{group.value}:{ball.track_id}"
            )
            if not canonical:
                reasons.add(f"rule_identity_invalid:{ball.track_id}")
            if group is BallGroup.UNKNOWN:
                reasons.add(f"unknown_ball_identity:{ball.track_id}")

    settled_evidence_digest = _settled_digest(shot, frames)
    before = None
    after = None
    if frames:
        frames_by_id = _frame_by_id(frames)
        before = frames_by_id.get(trusted_before_frame_id, frames[0])
        after = frames_by_id.get(trusted_after_frame_id, frames[-1])
    before_counts = {} if before is None else _counts(before.balls)
    after_counts = {} if after is None else _counts(after.balls)

    cue_ball_placement_observed = FactTruth.TRUE
    if required_cue_ball_placement is not CueBallPlacement.NONE:
        cue_ball_placement_observed = FactTruth.UNKNOWN
        if before is not None and config is not None:
            before_cues = tuple(
                ball for ball in before.balls if _group(ball.identity) is BallGroup.CUE
            )
            if len(before_cues) == 1:
                cue = before_cues[0]
                inside_table = (
                    0 <= cue.x_mm <= config.table_length_mm
                    and 0 <= cue.y_mm <= config.table_width_mm
                )
                behind_line = (
                    required_cue_ball_placement is CueBallPlacement.ANYWHERE
                    or cue.x_mm <= config.table_length_mm / 4
                )
                if inside_table and behind_line:
                    cue_ball_placement_observed = FactTruth.TRUE
                elif inside_table:
                    cue_ball_placement_observed = FactTruth.FALSE
        if cue_ball_placement_observed is not FactTruth.TRUE:
            reasons.add("cue_ball_placement_unverified")

    contract_valid = not reasons and coverage is not None and config is not None

    first_contact_group: BallGroup | None = None
    pots: tuple[RealtimePotFact, ...] = ()
    explicit_no_pot = FactTruth.UNKNOWN
    cue_ball_scratch = FactTruth.UNKNOWN
    post_contact_rail_or_pot = FactTruth.UNKNOWN
    distinct_break_object_rail_count: int | None = None
    object_ball_off_table = FactTruth.UNKNOWN
    evidence_ids: set[int] = set()

    if contract_valid and before is not None and after is not None and len(frames) >= 2:
        assert coverage is not None
        assert config is not None
        assert trusted_after_frame_id is not None
        frame_index = _frame_by_id(frames)
        before_by_track = {ball.track_id: ball for ball in before.balls if ball.track_id}
        after_by_track = {ball.track_id: ball for ball in after.balls if ball.track_id}
        uncertain_by_track_lists: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for frame_id, track_id, state in coverage.uncertain_track_states:
            uncertain_by_track_lists[track_id].append((frame_id, state))
        uncertain_by_track = {
            track_id: tuple(rows) for track_id, rows in uncertain_by_track_lists.items()
        }

        contact_event = observe_first_contact(
            _DeadlineFrames(
                frames,
                monotonic_ns=monotonic_ns,
                deadline_ns=deadline_ns,
                stage="first_contact",
            ),
            contact_distance_mm=config.first_contact_distance_mm,
            ambiguity_mm=config.first_contact_ambiguity_mm,
            cue_motion_min_mm=config.cue_motion_min_mm,
        )
        _deadline(monotonic_ns, deadline_ns)
        contact_observed = _coverage_observed(shot, "first_contact", reasons, record_reason=False)
        if contact_observed and contact_event.status is EventStatus.OBSERVED:
            first_contact_group = _group(contact_event.target_identity or "unknown")
            evidence_ids.update(contact_event.evidence_frame_ids)

        cushion_events = observe_cushion_contacts(
            _DeadlineFrames(
                frames,
                monotonic_ns=monotonic_ns,
                deadline_ns=deadline_ns,
                stage="cushion_contacts",
            ),
            rail_distance_mm=round(config.cushion_distance_mm),
        )
        _deadline(monotonic_ns, deadline_ns)
        cushion_observed = _coverage_observed(
            shot, "cushion_contacts", reasons, record_reason=False
        )
        observed_cushions = tuple(
            event for event in cushion_events if event.status is EventStatus.OBSERVED
        )

        pot_events = observe_pots(
            _DeadlineFrames(
                frames,
                monotonic_ns=monotonic_ns,
                deadline_ns=deadline_ns,
                stage="pots",
            ),
            persistence_frames=config.pot_missing_frames,
            pocket_radius_mm=config.pocket_distance_mm,
            table_length_mm=config.table_length_mm,
            table_width_mm=config.table_width_mm,
        )
        _deadline(monotonic_ns, deadline_ns)
        pots_observed = _coverage_observed(shot, "pots", reasons, record_reason=False)
        candidate_pots = (
            ()
            if not pots_observed
            else tuple(
                pot
                for event in pot_events
                if (
                    pot := _pot_fact(
                        event,
                        frames=frames,
                        frame_index=frame_index,
                        before_by_track=before_by_track,
                        after_by_track=after_by_track,
                        uncertain_by_track=uncertain_by_track,
                        pocket_radius_mm=config.pocket_distance_mm,
                        trusted_after_frame_id=trusted_after_frame_id,
                    )
                )
                is not None
            )
        )
        pots = candidate_pots
        for pot in pots:
            evidence_ids.update(pot.evidence_frame_ids)

        potted_tracks = {pot.track_id for pot in pots}
        controlled_lost = {
            (frame_id, pot.track_id)
            for pot in pots
            for frame_id, state in uncertain_by_track.get(pot.track_id, ())
            if state == "lost" and frame_id >= pot.first_missing_frame_id
        }
        conserved_tracks = {
            track_id
            for track_id, ball in before_by_track.items()
            if track_id in after_by_track
            and _group(ball.identity) is _group(after_by_track[track_id].identity)
        }
        inventory_complete = (
            set(before_by_track) == conserved_tracks | potted_tracks
            and set(after_by_track) == conserved_tracks
            and all(
                track_id in potted_tracks or track_id not in uncertain_by_track
                for track_id in before_by_track
            )
        )
        for frame_id, track_id, _state in coverage.uncertain_track_states:
            if (frame_id, track_id) not in controlled_lost and track_id not in conserved_tracks:
                reasons.add(f"tracker_uncertain:{track_id}")
        if pots:
            explicit_no_pot = FactTruth.FALSE
        elif pots_observed and inventory_complete and not coverage.uncertain_track_states:
            explicit_no_pot = FactTruth.TRUE

        cue_pots = tuple(pot for pot in pots if pot.group is BallGroup.CUE)
        before_cues = tuple(ball for ball in before.balls if _group(ball.identity) is BallGroup.CUE)
        if len(cue_pots) == 1:
            cue_ball_scratch = FactTruth.TRUE
        elif (
            not cue_pots
            and len(before_cues) == 1
            and before_cues[0].track_id in conserved_tracks
            and before_cues[0].track_id not in uncertain_by_track
        ):
            cue_ball_scratch = FactTruth.FALSE

        object_tracks = {
            track_id
            for track_id, ball in before_by_track.items()
            if _group(ball.identity) is not BallGroup.CUE
        }
        if all(
            track_id in conserved_tracks | potted_tracks
            and (track_id in potted_tracks or track_id not in uncertain_by_track)
            for track_id in object_tracks
        ) and not any(
            track_id not in before_by_track and _group(ball.identity) is not BallGroup.CUE
            for track_id, ball in after_by_track.items()
        ):
            object_ball_off_table = FactTruth.FALSE

        if first_contact_group is not None:
            contact_time = contact_event.timestamp_ms
            later_cushions = tuple(
                event
                for event in observed_cushions
                if contact_time is not None
                and event.timestamp_ms is not None
                and event.timestamp_ms > contact_time
            )
            later_pots = tuple(
                pot
                for pot in pots
                if frame_index[pot.first_missing_frame_id].timestamp_ms > (contact_time or -1)
            )
            same_time = any(
                event.timestamp_ms == contact_time for event in observed_cushions
            ) or any(
                frame_index[pot.first_missing_frame_id].timestamp_ms == contact_time for pot in pots
            )
            observed_cushion_keys = {
                (event.actor_identity, event.timestamp_ms) for event in later_cushions
            }

            def near_rail(ball: BallState) -> bool:
                return (
                    min(ball.x_mm, config.table_length_mm - ball.x_mm) <= config.cushion_distance_mm
                    or min(ball.y_mm, config.table_width_mm - ball.y_mm)
                    <= config.cushion_distance_mm
                )

            unexplained_rail_entries: set[str] = set()
            for previous, current in zip(frames, frames[1:], strict=False):
                if current.timestamp_ms <= (contact_time or -1):
                    continue
                previous_balls = previous.by_identity()
                for identity, current_ball in current.by_identity().items():
                    previous_ball = previous_balls.get(identity)
                    if (
                        previous_ball is not None
                        and not near_rail(previous_ball)
                        and near_rail(current_ball)
                        and (identity, current.timestamp_ms) not in observed_cushion_keys
                    ):
                        unexplained_rail_entries.add(identity)
            if later_cushions or later_pots:
                post_contact_rail_or_pot = FactTruth.TRUE
            elif (
                not same_time
                and cushion_observed
                and pots_observed
                and explicit_no_pot is FactTruth.TRUE
                and not unexplained_rail_entries
            ):
                post_contact_rail_or_pot = FactTruth.FALSE
            if not same_time and cushion_observed:
                object_rail_tracks = {
                    event.actor_identity
                    for event in later_cushions
                    if event.actor_identity not in {None, "cue"}
                }
                if not any(identity != "cue" for identity in unexplained_rail_entries):
                    distinct_break_object_rail_count = len(object_rail_tracks)
            for event in later_cushions:
                evidence_ids.update(event.evidence_frame_ids)

    if before is not None:
        evidence_ids.add(before.frame_id)
    if after is not None:
        evidence_ids.add(after.frame_id)
    abstention_reasons = tuple(sorted(reasons))
    normalized = {
        "session_id": session_id,
        "shot_id": shot.shot_id,
        "source_epoch_id": shot.source_epoch_id,
        "match_revision_at_start": match_revision,
        "actor_role": actor_role.value,
        "profile_id": profile_id,
        "required_cue_ball_placement": required_cue_ball_placement.value,
        "cue_ball_placement_observed": cue_ball_placement_observed.value,
        "start_frame_id": shot.start_frame_id,
        "settled_frame_id": shot.settled_frame_id,
        "settled_evidence_digest": settled_evidence_digest,
        "first_contact_group": None if first_contact_group is None else first_contact_group.value,
        "pots": [{**asdict(pot), "group": pot.group.value} for pot in pots],
        "explicit_no_pot": explicit_no_pot.value,
        "cue_ball_scratch": cue_ball_scratch.value,
        "post_contact_rail_or_pot": post_contact_rail_or_pot.value,
        "distinct_break_object_rail_count": distinct_break_object_rail_count,
        "before_counts": {group.value: count for group, count in before_counts.items()},
        "after_counts": {group.value: count for group, count in after_counts.items()},
        "object_ball_off_table": object_ball_off_table.value,
        "evidence_frame_ids": sorted(evidence_ids),
        "abstention_reasons": list(abstention_reasons),
        "event_config_digest": shot.event_config_digest,
    }
    normalization_preimage_json = _canonical_preimage_json(normalized)
    facts = RealtimeRuleFacts(
        session_id=session_id,
        shot_id=shot.shot_id,
        source_epoch_id=shot.source_epoch_id,
        match_revision_at_start=match_revision,
        actor_role=actor_role,
        start_frame_id=shot.start_frame_id,
        settled_frame_id=shot.settled_frame_id,
        settled_evidence_digest=settled_evidence_digest,
        first_contact_group=first_contact_group,
        pots=pots,
        explicit_no_pot=explicit_no_pot,
        cue_ball_scratch=cue_ball_scratch,
        post_contact_rail_or_pot=post_contact_rail_or_pot,
        distinct_break_object_rail_count=distinct_break_object_rail_count,
        before_counts=before_counts,
        after_counts=after_counts,
        object_ball_off_table=object_ball_off_table,
        evidence_frame_ids=tuple(sorted(evidence_ids)),
        abstention_reasons=abstention_reasons,
        digest=hashlib.sha256(normalization_preimage_json.encode()).hexdigest(),
        profile_id=profile_id,
        cue_ball_placement_observed=cue_ball_placement_observed,
    )
    return RealtimeRuleFactsBuild(facts, normalization_preimage_json)


def build_realtime_rule_facts(
    shot: _SettledShot,
    *,
    session_id: str,
    actor_role: WristbandRole,
    match_revision: int,
    profile_id: str,
    required_cue_ball_placement: CueBallPlacement = CueBallPlacement.NONE,
    deadline_ns: int,
    monotonic_ns: Callable[[], int],
) -> RealtimeRuleFacts:
    """Retain the original v1 facts-only API and byte representation."""
    return build_realtime_rule_facts_with_preimage(
        shot,
        session_id=session_id,
        actor_role=actor_role,
        match_revision=match_revision,
        profile_id=profile_id,
        required_cue_ball_placement=required_cue_ball_placement,
        deadline_ns=deadline_ns,
        monotonic_ns=monotonic_ns,
    ).facts


def build_abstained_realtime_rule_facts(
    shot: _SettledShot,
    *,
    session_id: str,
    actor_role: WristbandRole,
    match_revision: int,
    profile_id: str,
    reason: str,
    required_cue_ball_placement: CueBallPlacement = CueBallPlacement.NONE,
) -> RealtimeRuleFacts:
    """Retain the original v1 abstention API without fabricating observations."""
    return build_abstained_realtime_rule_facts_with_preimage(
        shot,
        session_id=session_id,
        actor_role=actor_role,
        match_revision=match_revision,
        profile_id=profile_id,
        reason=reason,
        required_cue_ball_placement=required_cue_ball_placement,
    ).facts
