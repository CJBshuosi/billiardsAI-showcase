"""Observer software controls only: constructed records are not perception evidence."""

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace as NS

import pytest

from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.rule_facts import FactTruth, RealtimeRuleFacts
from billiards_ai.lab.shot_local.models import BallGroup
from scripts.algorithm.offline_fullgame_observer import (
    CONTRACT,
    ORACLE_CONTRACT,
    FullgameFailure,
    FullgameObserver,
    InputOracle,
    fact_summary,
    load_oracle,
)

SID = "managed-11111111-2222-4333-8444-555555555555"
EPOCH = "a" * 32


def unit_json(value, *, ensure_ascii):
    """Independent UNIT serialization, never the observer's digest helper."""
    return json.dumps(
        value, ensure_ascii=ensure_ascii, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def unit_digest(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class Owner:
    session_id: str
    shot_id: str
    sequence: int
    source_epoch_id: str
    source_kind: str
    evidence_status: str
    actor_role: str
    actor_user_id: str
    match_revision: int


def oracle():
    return InputOracle(
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        1,
        "red",
        (("red", "continue_turn", "legal_break_object_pot"), ("red", "end", "legal_eight")),
    )


def shot_bundle(sequence, *, applied=True, durable=True):
    shot_id = f"shot-{sequence:04d}"
    expected = sequence + 2
    job_id = "job-" + str(sequence) * 64
    shot = NS(
        session_id=SID,
        shot_id=shot_id,
        sequence=sequence,
        settled_ms=sequence * 200,
        started_ms=sequence * 200 - 100,
        live_job_id=job_id,
        live_evidence_digest="e" * 64,
        media=NS(source_kind="usb_live", source_epoch_id=EPOCH),
    )
    owner = Owner(
        SID, shot_id, sequence, EPOCH, "usb_live", "bound_at_start", "red", "offline-red", expected
    )
    fact = NS(
        session_id=SID,
        shot_id=shot_id,
        sequence=sequence,
        actor_role="red",
        actor_user_id="offline-red",
        rule_revision=expected,
        start_frame_id=sequence * 100,
        settle_frame_id=sequence * 100 + 60,
    )
    action = "continue_turn" if sequence == 1 else "end"
    reason = "legal_break_object_pot" if sequence == 1 else "legal_eight"
    # Complete actual dataclass projection; values and matching normalization
    # basis are explicit UNIT constructions, not perception or legal-game proof.
    record = RealtimeRuleFacts(
        session_id=SID,
        shot_id=shot_id,
        source_epoch_id=EPOCH,
        match_revision_at_start=expected,
        actor_role=WristbandRole.RED,
        start_frame_id=fact.start_frame_id,
        settled_frame_id=fact.settle_frame_id,
        settled_evidence_digest="e" * 64,
        first_contact_group=BallGroup.SOLID if sequence == 1 else BallGroup.EIGHT,
        pots=(),
        explicit_no_pot=FactTruth.TRUE,
        cue_ball_scratch=FactTruth.FALSE,
        post_contact_rail_or_pot=FactTruth.TRUE,
        distinct_break_object_rail_count=None,
        before_counts={
            BallGroup.CUE: 1,
            BallGroup.EIGHT: 1,
            BallGroup.SOLID: 7,
            BallGroup.STRIPE: 7,
        },
        after_counts={
            BallGroup.CUE: 1,
            BallGroup.EIGHT: 1,
            BallGroup.SOLID: 7,
            BallGroup.STRIPE: 7,
        },
        object_ball_off_table=FactTruth.FALSE,
        evidence_frame_ids=(fact.start_frame_id, fact.settle_frame_id),
        abstention_reasons=(),
        digest="",
        profile_id="cbsa-chinese-pool-2017-observable-v1",
        cue_ball_placement_observed=FactTruth.TRUE,
    )
    facts = json.loads(json.dumps(asdict(record)))
    basis = {key: value for key, value in facts.items() if key != "digest"}
    basis.update(required_cue_ball_placement="none", event_config_digest="c" * 64)
    preimage = unit_json(basis, ensure_ascii=True)
    facts["digest"] = unit_digest(preimage)
    decision = dict(
        session_id=SID,
        shot_id=shot_id,
        source_epoch_id=EPOCH,
        expected_revision=expected,
        actor_role="red",
        fact_digest=facts["digest"],
        profile_id="cbsa-chinese-pool-2017-observable-v1",
        settled_evidence_digest="e" * 64,
        winner_role="red" if sequence == 2 else None,
        kind="apply" if applied else "abstain",
        action=action if applied else None,
        reason_code=reason if applied else "first_contact_unknown",
    )
    decision["decision_digest"] = unit_digest(unit_json(decision, ensure_ascii=False))
    rule = NS(
        session_id=SID,
        shot_id=shot_id,
        source_epoch_id=EPOCH,
        profile_id="cbsa-chinese-pool-2017-observable-v1",
        start_frame_id=fact.start_frame_id,
        settled_frame_id=fact.settle_frame_id,
        settled_evidence_digest="e" * 64,
        fact_digest=facts["digest"],
        decision_digest=decision["decision_digest"],
        facts_json=json.dumps(facts),
        decision_json=json.dumps(decision),
        status="applied" if applied else "abstained",
        applied_command_id=f"rule-{sequence}",
    )
    entry = NS(
        entry_id=rule.applied_command_id,
        session_id=SID,
        origin="rules_engine",
        principal_id=rule.profile_id,
        command_type="auto_" + action,
        payload_digest=rule.decision_digest,
        outcome="applied",
        result_code=reason,
        resulting_revision=expected + 1,
        ledger_order=sequence + 3,
        decision_ref_json=json.dumps(
            dict(
                session_id=SID,
                shot_id=shot_id,
                fact_digest=rule.fact_digest,
                decision_digest=rule.decision_digest,
            )
        ),
    )
    media = NS(
        session_id=SID,
        shot_id=shot_id,
        source_epoch_id=EPOCH,
        match_revision=expected,
        job_id=job_id,
        media_sha256="f" * 64,
    )
    return NS(
        shot=shot,
        owner=owner,
        fact=fact,
        rule=rule,
        entry=entry,
        media=media if durable else None,
        preimage=preimage,
    )


class Store:
    def __init__(self, bundles):
        self.bundles = bundles
        self.ended = False
        self.phones = [
            NS(
                entry_id=f"phone-{i}",
                command_type=kind,
                principal_id=actor,
                origin="phone",
                outcome="applied",
                result_code="APPLIED",
                ledger_order=i + 1,
                origin_sequence=i + 1,
                resulting_revision=i + 1,
                session_id=SID,
            )
            for i, (kind, actor) in enumerate(
                [
                    ("claim_role", "offline-red"),
                    ("claim_role", "offline-blue"),
                    ("set_breaker", "offline-red"),
                ]
            )
        ]

    def list_shots(self, session):
        assert session == SID
        return [b.shot for b in self.bundles]

    def list_command_ledger(self, _):
        return self.phones + [b.entry for b in self.bundles if b.rule.status == "applied"]

    def _get(self, shot, kind):
        return next(getattr(b, kind) for b in self.bundles if b.shot.shot_id == shot)

    def get_shot_ownership(self, _, shot):
        return self._get(shot, "owner")

    def get_live_shot_fact(self, _, shot):
        return self._get(shot, "fact")

    def get_rule_decision(self, _, shot):
        return self._get(shot, "rule")

    def get_rule_fact_preimage(self, session, shot):
        assert session == SID
        return self._get(shot, "preimage")

    def get_live_shot_media(self, _, shot):
        return self._get(shot, "media")

    def load_session(self, _):
        return NS(
            status="ended" if self.ended else "open",
            match=NS(
                rack_phase="ended" if self.ended else "assigned",
                winner_role="red" if self.ended else None,
                terminal_reason="legal_eight" if self.ended else None,
                revision=5,
            ),
        )

    def get_postgame_manifest(self, _):
        return (
            NS(expected_jobs=tuple(b.shot.live_job_id for b in self.bundles))
            if self.ended
            else None
        )


class Actor:
    def __init__(self, bundles):
        self.store = Store(bundles)
        self.session, self.epoch = SID, EPOCH
        self.report = dict(capture={"source_sha256": "2" * 64}, postgame={})
        self.remote = NS(
            snapshot=lambda: dict(birth_count=1, issued_count=3, completed_prefix=3, failure=None)
        )
        self.steps = []
        self.review_calls = 0
        self.cleanup = "VERIFIED"
        self.health_failure = None

    def fullgame_job(self, shot, media):
        return dict(
            job_id=media.job_id,
            status="pending",
            payload_sha256="9" * 64,
            session_id=SID,
            shot_id=shot.shot_id,
            source_epoch_id=EPOCH,
            media_sha256=media.media_sha256,
        )

    def health(self, *, ending):
        if self.health_failure:
            raise FullgameFailure(self.health_failure)

    def wait(self, probe, deadline, code):
        for step in self.steps or [lambda: None]:
            step()
            if probe():
                return True
        raise FullgameFailure(code)

    def review(self):
        self.review_calls += 1
        self.report["postgame"]["selection"] = dict(
            save="VERIFIED",
            idempotency="VERIFIED",
            completion="VERIFIED",
            cleanup=self.cleanup,
            unsaved_count=1 if self.cleanup == "VERIFIED" else 0,
        )


def test_natural_terminal_waits_for_all_shot_media_and_preserves_snapshots():
    first, final = shot_bundle(1), shot_bundle(2, durable=False)
    actor = Actor([first])
    observer = FullgameObserver(actor, oracle(), match_seconds=10)
    assert not observer.poll()
    earlier = copy.deepcopy(observer.report["snapshots"])
    actor.store.bundles.append(final)
    assert not observer.poll()
    final.media = shot_bundle(2).media
    actor.store.ended = True
    assert observer.poll()
    assert not observer.report["failures"]
    assert observer.report["snapshots"][: len(earlier)] == earlier
    assert observer.report["terminal"]["job_id"] == final.shot.live_job_id


def test_abstention_is_sticky_and_later_diagnostics_are_collected():
    actor = Actor([shot_bundle(1, applied=False)])
    observer = FullgameObserver(actor, oracle(), match_seconds=10)
    assert not observer.poll()
    assert "FULLGAME_RULE_NOT_APPLIED" in observer.report["failures"]
    actor.store.bundles.append(shot_bundle(2))
    assert not observer.poll()
    assert any(r["sequence"] == 2 for r in observer.report["snapshots"])
    assert any(
        r.get("rule", {}).get("reason_code") == "first_contact_unknown"
        for r in observer.report["snapshots"]
    )


@pytest.mark.parametrize("fault", ["owner", "epoch", "ledger", "decision", "oracle"])
def test_faults_cannot_pass_but_do_not_suppress_later_receipts(fault):
    bundles = [shot_bundle(1), shot_bundle(2)]
    actor = Actor(bundles)
    if fault == "owner":
        bundles[0].owner.actor_user_id = "offline-blue"
    elif fault == "epoch":
        bundles[0].media.source_epoch_id = "b" * 32
    elif fault == "ledger":
        bundles[0].entry.resulting_revision = 99
    elif fault == "decision":
        bundles[0].rule.decision_digest = "0" * 64
    else:
        observer_oracle = replace(
            oracle(), shots=(("blue", "continue_turn", "other"), oracle().shots[1])
        )
    observer = FullgameObserver(
        actor, observer_oracle if fault == "oracle" else oracle(), match_seconds=10
    )
    actor.store.ended = True
    assert observer.poll()
    assert observer.report["failures"]
    assert any(r["sequence"] == 2 for r in observer.report["snapshots"])


def test_eof_preserves_collected_rows_and_never_enters_postgame():
    actor = Actor([shot_bundle(1, applied=False)])
    actor.health_failure = "CAPTURE_EOF"
    observer = FullgameObserver(actor, oracle(), match_seconds=10)
    with pytest.raises(FullgameFailure, match="CAPTURE_EOF"):
        observer.run()
    assert observer.report["status"] == "FAILED"
    assert observer.report["snapshots"]
    assert "CAPTURE_EOF" in observer.report["failures"]
    assert actor.review_calls == 0


def test_eof_at_completed_terminal_poll_still_refuses_postgame():
    """Durable natural end does not waive the observer's final capture check."""
    from billiards_ai.edge.route_assistance.records import FieldSessionStatus
    from scripts.algorithm.run_offline_managed_blackbox import _Actor, _EpisodeFailure

    actor = Actor([shot_bundle(1), shot_bundle(2)])
    actor.store.ended = True
    load_session = actor.store.load_session
    actor.store.load_session = lambda sid: NS(
        **{**vars(load_session(sid)), "status": FieldSessionStatus.ENDED}
    )
    source = NS(
        subsystem_health=lambda: {
            "video_health": {"status": "degraded", "reason": "capture source ended unexpectedly"},
            "analysis_health": {"realtime_diagnostics": None},
        },
        failure_reason=None,
        restart_required=False,
    )
    actor.source = source
    actor.state = NS(session_id=SID, projection_source=source)
    actor.health = lambda *, ending: _Actor.health(actor, ending=ending)
    observer = FullgameObserver(actor, oracle(), match_seconds=10)

    with pytest.raises(_EpisodeFailure, match="CAPTURE_EOF"):
        observer.run()
    assert observer.report["status"] == "FAILED"
    assert observer.report["terminal"]["job_id"] == actor.store.bundles[-1].media.job_id
    assert len(observer.durable) == 2 and observer.report["snapshots"]
    assert actor.review_calls == 0


def test_eof_after_review_entry_does_not_add_a_new_observer_health_gate():
    """Controller boundary only; the integration test covers actual HTTP/media."""
    from billiards_ai.edge.route_assistance.records import FieldSessionStatus
    from scripts.algorithm.run_offline_managed_blackbox import _Actor, _EpisodeFailure

    actor = Actor([shot_bundle(1), shot_bundle(2)])
    actor.store.ended = True
    load_session = actor.store.load_session
    actor.store.load_session = lambda sid: NS(
        **{**vars(load_session(sid)), "status": FieldSessionStatus.ENDED}
    )
    video = {"status": "ready", "reason": None}
    source = NS(
        subsystem_health=lambda: {
            "video_health": video,
            "analysis_health": {"realtime_diagnostics": None},
        },
        failure_reason=None,
        restart_required=False,
    )
    actor.source = source
    actor.state = NS(session_id=SID, projection_source=source)
    health_calls = []

    def health(*, ending):
        health_calls.append(ending)
        _Actor.health(actor, ending=ending)

    actor.health = health
    complete_review = actor.review

    def review_after_eof():
        video.update(status="degraded", reason="capture source ended unexpectedly")
        complete_review()

    actor.review = review_after_eof
    observer = FullgameObserver(actor, oracle(), match_seconds=10)
    observer.run()
    assert health_calls == [True]
    assert actor.review_calls == 1 and observer.report["status"] == "OBSERVED"
    assert observer.report["perception_fact_quality"] == "UNASSESSED"
    # It is absence of a later health check, not a changed EOF permission.
    with pytest.raises(_EpisodeFailure, match="CAPTURE_EOF"):
        _Actor.health(actor, ending=True)


def test_timeout_keeps_abstention_and_never_forces_progression():
    actor = Actor([shot_bundle(1, applied=False)])
    observer = FullgameObserver(actor, oracle(), match_seconds=10)
    with pytest.raises(FullgameFailure, match="MATCH_TIMEOUT"):
        observer.run()
    assert observer.report["status"] == "FAILED"
    assert "FULLGAME_RULE_NOT_APPLIED" in observer.report["failures"]
    assert "FULLGAME_MATCH_TIMEOUT" in observer.report["failures"]
    assert actor.review_calls == 0


def test_transient_store_error_is_inspectable_and_later_records_survive():
    actor = Actor([shot_bundle(1)])
    observer = FullgameObserver(actor, oracle(), match_seconds=10)
    read = actor.store.list_shots
    actor.store.list_shots = lambda _: (_ for _ in ()).throw(ValueError("private data"))
    assert not observer.poll()
    actor.store.list_shots = read
    assert not observer.poll()
    assert observer.report["snapshots"]
    assert observer.report["failures"] == ["FULLGAME_STORE_UNAVAILABLE"]
    assert "private data" not in json.dumps(observer.report)


def test_natural_terminal_delegates_postgame_and_requires_real_sibling_gate():
    actor = Actor([shot_bundle(1), shot_bundle(2)])
    actor.store.ended = True
    observer = FullgameObserver(actor, oracle(), match_seconds=10)
    observer.run()
    assert actor.review_calls == 1
    assert observer.report["status"] == "OBSERVED"
    assert observer.report["perception_fact_quality"] == "UNASSESSED"
    assert actor.report["boundary"]["shot_id"] == "shot-0001"
    actor.cleanup = "NOT_EXERCISED"
    observer = FullgameObserver(actor, oracle(), match_seconds=10)
    with pytest.raises(FullgameFailure, match="UNSAVED_CLEANUP"):
        observer.run()
    assert observer.report["status"] == "FAILED"


def test_actual_fact_summary_is_bounded_and_omits_track_ids_and_private_suffixes():
    facts = json.loads(shot_bundle(1).rule.facts_json)
    facts["pots"] = [dict(group="solid", pocket_id="top-left", track_id="private-track")]
    facts["abstention_reasons"] = ["tracker_uncertain:private-track"]
    summary = fact_summary(facts)
    assert summary["pots"] == [dict(group="solid", pocket_id="top-left")]
    assert summary["abstention_reasons"] == ["tracker_uncertain"]
    assert "private" not in json.dumps(summary)
    facts["pots"] *= 17
    summary = fact_summary(facts)
    assert not summary["shape_valid"] and summary["pot_count"] == 16


def test_invalid_observed_facts_prevent_success_without_suppressing_later_shots():
    actor = Actor([shot_bundle(1), shot_bundle(2)])
    facts = json.loads(actor.store.bundles[0].rule.facts_json)
    facts["before_counts"]["solid"] = True
    actor.store.bundles[0].rule.facts_json = json.dumps(facts)
    observer = FullgameObserver(actor, oracle(), match_seconds=10)
    assert not observer.poll()
    assert "FULLGAME_FACT_SUMMARY_INVALID" in observer.report["failures"]
    assert any(r["sequence"] == 2 for r in observer.report["snapshots"])


def bind_unit_preimage(bundle, raw, *, facts=None):
    """Rebind every digest reference independently; no observer hashing helper."""
    bundle.preimage = raw
    digest = unit_digest(raw)
    facts = json.loads(bundle.rule.facts_json) if facts is None else facts
    facts["digest"] = digest
    decision = json.loads(bundle.rule.decision_json)
    decision["fact_digest"] = digest
    decision.pop("decision_digest")
    decision_digest = unit_digest(unit_json(decision, ensure_ascii=False))
    decision["decision_digest"] = decision_digest
    bundle.rule.fact_digest = digest
    bundle.rule.decision_digest = decision_digest
    bundle.rule.facts_json = json.dumps(facts)
    bundle.rule.decision_json = json.dumps(decision)
    bundle.entry.payload_digest = decision_digest
    bundle.entry.decision_ref_json = json.dumps(
        dict(
            session_id=SID,
            shot_id=bundle.shot.shot_id,
            fact_digest=digest,
            decision_digest=decision_digest,
        )
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "getter_error",
        "tamper",
        "rehash_conflict",
        "noncanonical",
        "duplicate",
        "nonfinite",
        "depth",
        "extra",
        "missing_evidence",
        "evidence_bool",
        "placement",
        "config_digest",
        "pot_confidence_bool",
        "pot_extra",
    ],
)
def test_rule_fact_preimage_rejects_missing_invalid_and_rehashed_conflicts(fault):
    first, later = shot_bundle(1), shot_bundle(2)
    facts = json.loads(first.rule.facts_json)
    basis = json.loads(first.preimage)
    if fault == "missing":
        first.preimage = None
    elif fault == "getter_error":
        pass
    elif fault == "tamper":
        basis["first_contact_group"] = "eight"
        first.preimage = unit_json(basis, ensure_ascii=True)
    elif fault == "rehash_conflict":
        basis["first_contact_group"] = "eight"
        bind_unit_preimage(first, unit_json(basis, ensure_ascii=True))
    elif fault == "noncanonical":
        # Matching raw SHA and matching parsed values still do not make these
        # bytes the producer's exact canonical normalized representation.
        bind_unit_preimage(first, "\n" + first.preimage)
    elif fault == "duplicate":
        raw = first.preimage.replace('"actor_role":', '"actor_role":"blue","actor_role":', 1)
        assert raw != first.preimage and raw.count('"actor_role":') == 2
        bind_unit_preimage(first, raw)
    elif fault == "nonfinite":
        raw = first.preimage.replace(
            '"event_config_digest":"' + "c" * 64 + '"', '"event_config_digest":NaN'
        )
        assert raw != first.preimage
        bind_unit_preimage(first, raw)
    elif fault == "depth":
        bind_unit_preimage(first, "[" * 1100 + "0" + "]" * 1100)
    else:
        if fault == "extra":
            basis["unexpected"] = True
        elif fault == "missing_evidence":
            del basis["evidence_frame_ids"]
            del facts["evidence_frame_ids"]
        elif fault == "evidence_bool":
            facts["evidence_frame_ids"] = basis["evidence_frame_ids"] = [True]
        elif fault == "placement":
            basis["required_cue_ball_placement"] = "unobserved"
        elif fault == "config_digest":
            basis["event_config_digest"] = 42
        else:
            pot = dict(
                track_id="UNIT-track",
                group="solid",
                pocket_id="top-left",
                last_seen_frame_id=150,
                first_missing_frame_id=151,
                confirmed_through_frame_id=160,
                evidence_frame_ids=[150, 151, 160],
                confidence=True if fault == "pot_confidence_bool" else 0.99,
            )
            if fault == "pot_extra":
                pot["unexpected"] = "UNIT-only"
            facts["pots"] = basis["pots"] = [pot]
        bind_unit_preimage(first, unit_json(basis, ensure_ascii=True), facts=facts)
    actor = Actor([first, later])
    if fault == "getter_error":
        read = actor.store.get_rule_fact_preimage

        def damaged(session, shot):
            if shot == first.shot.shot_id:
                raise ValueError("UNIT damaged stored basis")
            return read(session, shot)

        actor.store.get_rule_fact_preimage = damaged
    actor.store.ended = True
    observer = FullgameObserver(actor, oracle(), match_seconds=10)
    assert observer.poll()
    expected = (
        "FULLGAME_FACT_PREIMAGE_MISSING" if fault == "missing" else "FULLGAME_FACT_PREIMAGE_INVALID"
    )
    assert expected in observer.report["failures"]
    assert expected in observer.report["snapshots"][0]["errors"]
    assert any(row["sequence"] == 2 for row in observer.report["snapshots"])
    assert 1 not in observer.durable
    assert observer.report["perception_fact_quality"] == "UNASSESSED"
    assert "damaged stored basis" not in json.dumps(observer.report)


def test_rule_fact_preimage_matching_unicode_basis_is_integrity_only():
    first = shot_bundle(1)
    facts = json.loads(first.rule.facts_json)
    facts["pots"] = [
        dict(
            track_id="UNIT-台球",
            group="solid",
            pocket_id="top-left",
            last_seen_frame_id=150,
            first_missing_frame_id=151,
            confirmed_through_frame_id=160,
            evidence_frame_ids=[150, 151, 160],
            confidence=0.99,
        )
    ]
    basis = {key: value for key, value in facts.items() if key != "digest"}
    basis.update(required_cue_ball_placement="none", event_config_digest="c" * 64)
    raw = unit_json(basis, ensure_ascii=True)
    assert "\\u53f0" in raw and raw != unit_json(basis, ensure_ascii=False)
    bind_unit_preimage(first, raw, facts=facts)
    observer = FullgameObserver(Actor([first]), oracle(), match_seconds=10)
    assert not observer.poll()
    assert observer.report["failures"] == []
    assert 1 in observer.durable
    assert observer.report["perception_fact_quality"] == "UNASSESSED"
    assert "台球" not in json.dumps(observer.report, ensure_ascii=False)


def test_manual_end_or_incomplete_game_never_counts_as_natural_terminal():
    actor = Actor([shot_bundle(1)])
    actor.store.ended = True
    observer = FullgameObserver(actor, oracle(), match_seconds=10)
    with pytest.raises(FullgameFailure, match="OBSERVATION_FAILED"):
        observer.run()
    assert "FULLGAME_NATURAL_TERMINAL_MISMATCH" in observer.report["failures"]
    assert actor.review_calls == 0


def oracle_file(tmp_path):
    video = tmp_path / "pixels.avi"
    video.write_bytes(b"not decoded by oracle control")
    data = dict(
        contract=ORACLE_CONTRACT,
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        physics_manifest_sha256="3" * 64,
        independent_rule_review_sha256="4" * 64,
        selected_sequence=1,
        winner="red",
        shots=[
            dict(actor_role=actor, action=action, reason_code=reason)
            for actor, action, reason in oracle().shots
        ],
    )
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(data))
    return path, video, data


def test_frozen_oracle_requires_exact_hash_video_and_strict_contract(tmp_path):
    path, video, data = oracle_file(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_oracle(path, digest, video)
    assert loaded.projection()["binding_only"] is True
    assert loaded.shots == oracle().shots
    with pytest.raises(ValueError):
        load_oracle(path, "0" * 64, video)
    video.write_bytes(b"different input")
    with pytest.raises(ValueError):
        load_oracle(path, digest, video)
    assert CONTRACT != ORACLE_CONTRACT


@pytest.mark.parametrize("change", ["extra", "early_end", "winner", "one_shot", "duplicate"])
def test_invalid_oracle_never_becomes_observer_truth(tmp_path, change):
    path, video, data = oracle_file(tmp_path)
    if change == "extra":
        data["inject_facts"] = True
    elif change == "early_end":
        data["shots"][0]["action"] = "end"
    elif change == "winner":
        data["winner"] = "blue"
    elif change == "one_shot":
        data["shots"] = data["shots"][-1:]
    raw = json.dumps(data)
    if change == "duplicate":
        raw = raw.replace('{"contract":', '{"contract":"ignored","contract":', 1)
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_oracle(path, hashlib.sha256(path.read_bytes()).hexdigest(), video)
