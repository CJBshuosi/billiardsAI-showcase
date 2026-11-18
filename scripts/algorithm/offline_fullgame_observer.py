"""Versioned assertions over actual product state; never supplies perception facts.

An oracle hash binds an independently reviewed input, but cannot establish its
legality. Formal acceptance fields remain outside this software observer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import stat
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

CONTRACT = "offline_fullgame_observer_v1"
ORACLE_CONTRACT = "offline_fullgame_input_oracle_v1"
PROFILE = "cbsa-chinese-pool-2017-observable-v1"
MAX_SHOTS = 128
MAX_SNAPSHOTS = 4096
MAX_MATCH_SECONDS = 600
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")


class FullgameFailure(Exception):
    """Only fixed public diagnostic codes leave this observer."""


def _require(condition, code):
    if not condition:
        raise FullgameFailure(code)


def _sha(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _integer(value, lo=0, hi=2**53 - 1):
    return type(value) is int and lo <= value <= hi


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def fact_summary(facts):
    """Project actual facts for later quality review; omit track IDs and raw payloads."""
    groups = {"cue", "eight", "solid", "stripe", "unknown"}
    pockets = {
        "top-left",
        "top-middle",
        "top-right",
        "bottom-left",
        "bottom-middle",
        "bottom-right",
    }
    valid = True
    result = {}
    contact = facts.get("first_contact_group")
    result["first_contact_group"] = (
        contact
        if contact is None or (isinstance(contact, str) and contact in groups)
        else "INVALID"
    )
    valid &= result["first_contact_group"] != "INVALID"
    for key in (
        "explicit_no_pot",
        "cue_ball_scratch",
        "post_contact_rail_or_pot",
        "object_ball_off_table",
        "cue_ball_placement_observed",
    ):
        value = facts.get(key)
        result[key] = (
            value if isinstance(value, str) and value in {"true", "false", "unknown"} else "INVALID"
        )
        valid &= result[key] != "INVALID"
    rails = facts.get("distinct_break_object_rail_count")
    result["distinct_break_object_rail_count"] = (
        rails if rails is None or _integer(rails, 0, 15) else None
    )
    valid &= rails is None or _integer(rails, 0, 15)
    for key in ("before_counts", "after_counts"):
        counts = facts.get(key)
        ok = (
            type(counts) is dict
            and set(counts) <= groups
            and all(_integer(n, 0, 16) for n in counts.values())
            and sum(counts.values()) <= 16
        )
        result[key] = dict(counts) if ok else {}
        valid &= ok
    pots = facts.get("pots")
    valid &= type(pots) is list and len(pots) <= 16
    result["pots"] = []
    for pot in pots[:16] if type(pots) is list else []:
        ok = (
            type(pot) is dict
            and isinstance(pot.get("group"), str)
            and pot["group"] in groups
            and isinstance(pot.get("pocket_id"), str)
            and pot["pocket_id"] in pockets
        )
        result["pots"].append(
            dict(group=pot["group"], pocket_id=pot["pocket_id"])
            if ok
            else dict(group="INVALID", pocket_id="INVALID")
        )
        valid &= ok
    result["pot_count"] = len(result["pots"])
    reasons = facts.get("abstention_reasons")
    valid &= type(reasons) is list and len(reasons) <= 64
    result["abstention_reasons"] = []
    for reason in reasons[:64] if type(reasons) is list else []:
        prefix = reason.partition(":")[0] if isinstance(reason, str) else ""
        result["abstention_reasons"].append(
            prefix if _CODE.fullmatch(prefix) else "UNEXPORTED_REASON"
        )
    result["shape_valid"] = bool(valid)
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate oracle field")
        result[key] = value
    return result


def _constant(_):
    raise ValueError("nonfinite JSON")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite JSON")
    return result


def _json(raw):
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="strict")
    if len(raw) > 1024**2:
        raise ValueError("JSON size limit")
    return json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=_object,
        parse_constant=_constant,
        parse_float=_float,
    )


def _fact_preimage_valid(raw, facts, digest):
    """Check the separate v1 normalization basis without a producer validator.

    This verifies representation, shared fact fields and digest binding only.
    It cannot establish whether perception observed the physical event correctly.
    """
    fields = {
        "session_id",
        "shot_id",
        "source_epoch_id",
        "match_revision_at_start",
        "actor_role",
        "profile_id",
        "required_cue_ball_placement",
        "cue_ball_placement_observed",
        "start_frame_id",
        "settled_frame_id",
        "settled_evidence_digest",
        "first_contact_group",
        "pots",
        "explicit_no_pot",
        "cue_ball_scratch",
        "post_contact_rail_or_pot",
        "distinct_break_object_rail_count",
        "before_counts",
        "after_counts",
        "object_ball_off_table",
        "evidence_frame_ids",
        "abstention_reasons",
        "event_config_digest",
    }
    shared = fields - {"required_cue_ball_placement", "event_config_digest"}

    def text(value):
        if type(value) is not str or not value or "\0" in value:
            return False
        value.encode("utf-8", errors="strict")
        return True

    def frame_ids(value):
        return type(value) is list and all(_integer(frame) for frame in value)

    try:
        if type(raw) is not str or type(facts) is not dict or set(facts) != shared | {"digest"}:
            return False
        basis = _json(raw)
        if type(basis) is not dict or set(basis) != fields:
            return False
        canonical = json.dumps(
            basis, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if raw != canonical or not _sha(digest) or facts["digest"] != digest:
            return False
        if hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
            return False
        if any(_canonical(basis[key]) != _canonical(facts[key]) for key in shared):
            return False
        if not fact_summary(basis)["shape_valid"]:
            return False
        if not all(text(basis[key]) for key in ("session_id", "shot_id", "source_epoch_id")):
            return False
        if basis["actor_role"] not in ("red", "blue") or basis["profile_id"] != PROFILE:
            return False
        if basis["required_cue_ball_placement"] not in ("none", "anywhere", "behind_line"):
            return False
        if basis["event_config_digest"] is not None and not _sha(basis["event_config_digest"]):
            return False
        if not _sha(basis["settled_evidence_digest"]) or not all(
            _integer(basis[key])
            for key in ("match_revision_at_start", "start_frame_id", "settled_frame_id")
        ):
            return False
        if not basis["start_frame_id"] < basis["settled_frame_id"]:
            return False
        if not frame_ids(basis["evidence_frame_ids"]) or not all(
            text(reason) for reason in basis["abstention_reasons"]
        ):
            return False
        pot_fields = {
            "track_id",
            "group",
            "pocket_id",
            "last_seen_frame_id",
            "first_missing_frame_id",
            "confirmed_through_frame_id",
            "evidence_frame_ids",
            "confidence",
        }
        for pot in basis["pots"]:
            if set(pot) != pot_fields or not text(pot["track_id"]):
                return False
            if not all(
                _integer(pot[key])
                for key in (
                    "last_seen_frame_id",
                    "first_missing_frame_id",
                    "confirmed_through_frame_id",
                )
            ) or not frame_ids(pot["evidence_frame_ids"]):
                return False
            confidence = pot["confidence"]
            if (
                type(confidence) not in (int, float)
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                return False
        return True
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return False


def _plain(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.casefold() == ".project-archive" for p in path.parts):
        raise ValueError("unsafe input")
    if any(p.is_symlink() for p in (*path.parents, path)):
        raise ValueError("unsafe input")
    return path


@dataclass(frozen=True)
class InputOracle:
    sha256: str
    video_sha256: str
    physics_manifest_sha256: str
    independent_rule_review_sha256: str
    selected_sequence: int
    winner: str
    shots: tuple[tuple[str, str, str], ...]

    def projection(self):
        return dict(
            sha256=self.sha256,
            video_sha256=self.video_sha256,
            physics_manifest_sha256=self.physics_manifest_sha256,
            independent_rule_review_sha256=self.independent_rule_review_sha256,
            expected_shot_count=len(self.shots),
            selected_sequence=self.selected_sequence,
            binding_only=True,
        )


def load_oracle(path, expected_sha256, video):
    """Bound three reviewed identities before allocating a run or starting CLI."""
    try:
        if not _sha(expected_sha256):
            raise ValueError("oracle identity required")
        path = _plain(path)
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1024**2:
            raise ValueError("oracle size")
        with path.open("rb") as stream:
            raw = stream.read(1024**2 + 1)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("oracle changed")
        if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("oracle identity")
        data = _json(raw)
        keys = {
            "contract",
            "video_sha256",
            "physics_manifest_sha256",
            "independent_rule_review_sha256",
            "selected_sequence",
            "winner",
            "shots",
        }
        if type(data) is not dict or set(data) != keys or data["contract"] != ORACLE_CONTRACT:
            raise ValueError("oracle contract")
        if not all(_sha(data[k]) for k in keys if k.endswith("sha256")):
            raise ValueError("oracle digest")
        shots = data["shots"]
        if type(shots) is not list or not 2 <= len(shots) <= MAX_SHOTS:
            raise ValueError("oracle count")
        expected = []
        for row in shots:
            if (
                type(row) is not dict
                or set(row) != {"actor_role", "action", "reason_code"}
                or row["actor_role"] not in {"red", "blue"}
                or row["action"] not in {"continue_turn", "switch_turn", "end"}
                or not isinstance(row["reason_code"], str)
                or _CODE.fullmatch(row["reason_code"]) is None
            ):
                raise ValueError("oracle shot")
            expected.append((row["actor_role"], row["action"], row["reason_code"]))
        if (
            any(s[1] == "end" for s in expected[:-1])
            or expected[-1][1:] != ("end", "legal_eight")
            or data["winner"] not in {"red", "blue"}
            or data["winner"] != expected[-1][0]
            or not _integer(data["selected_sequence"], 1, len(shots))
        ):
            raise ValueError("oracle terminal")
        video = _plain(video)
        if not stat.S_ISREG(video.stat().st_mode):
            raise ValueError("video missing")
        video_before = video.stat()
        if video_before.st_size > 2 * 1024**3:
            raise ValueError("video size limit")
        digest = hashlib.sha256()
        read = 0
        with video.open("rb") as stream:
            while chunk := stream.read(1024**2):
                read += len(chunk)
                if read > video_before.st_size:
                    raise ValueError("video changed")
                digest.update(chunk)
        video_after = video.stat()
        if (read, video_before.st_mtime_ns) != (
            video_after.st_size,
            video_after.st_mtime_ns,
        ) or read != video_before.st_size:
            raise ValueError("video changed")
        observed = digest.hexdigest()
        if observed != data["video_sha256"]:
            raise ValueError("video identity")
        return InputOracle(
            expected_sha256,
            observed,
            data["physics_manifest_sha256"],
            data["independent_rule_review_sha256"],
            data["selected_sequence"],
            data["winner"],
            tuple(expected),
        )
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        raise ValueError("invalid frozen fullgame input oracle") from None


class FullgameObserver:
    """Append state-change receipts while preserving sticky diagnostic failures."""

    def __init__(self, actor, oracle, *, match_seconds, clock=time.monotonic):
        if (
            type(match_seconds) not in (int, float)
            or not math.isfinite(match_seconds)
            or not 0 < match_seconds <= MAX_MATCH_SECONDS
        ):
            raise ValueError("invalid fullgame match budget")
        self.actor, self.oracle, self.clock = actor, oracle, clock
        self.match_seconds = match_seconds
        self.last = {}
        self.durable = {}
        self.report = actor.report["fullgame"] = dict(
            contract=CONTRACT,
            oracle=oracle.projection(),
            status="OBSERVING",
            failures=[],
            snapshots=[],
            terminal={},
            observed_shot_count=0,
            perception_fact_quality="UNASSESSED",
        )

    def fail(self, code):
        if code not in self.report["failures"]:
            self.report["failures"].append(code)

    def _append(self, receipt):
        sequence = receipt["sequence"]
        signature = _canonical(receipt)
        if self.last.get(sequence) == signature:
            return
        _require(len(self.report["snapshots"]) < MAX_SNAPSHOTS, "FULLGAME_SNAPSHOT_LIMIT")
        # New dictionaries on every scan: later states never overwrite earlier evidence.
        self.report["snapshots"].append(json.loads(signature))
        self.last[sequence] = signature

    def _shot(self, shot, ledger):
        a = self.actor
        r = dict(
            sequence=shot.sequence,
            shot_id=shot.shot_id,
            settled=shot.settled_ms is not None,
            state="STARTED",
            errors=[],
        )

        def require(ok, code="FULLGAME_SHOT_LINEAGE"):
            if not ok:
                if code not in r["errors"]:
                    r["errors"].append(code)
                self.fail(code)

        require(shot.session_id == a.session and _integer(shot.sequence, 1, MAX_SHOTS))
        require(shot.media.source_kind == "usb_live" and shot.media.source_epoch_id == a.epoch)
        require(
            _integer(shot.started_ms)
            and (
                shot.settled_ms is None
                or (_integer(shot.settled_ms) and shot.settled_ms > shot.started_ms)
            )
        )
        r["started_ms"], r["settled_ms"] = shot.started_ms, shot.settled_ms
        owner = a.store.get_shot_ownership(a.session, shot.shot_id)
        if owner is None:
            require(False, "FULLGAME_OWNERSHIP_MISSING")
            return r
        role = str(owner.actor_role)
        require(
            owner.session_id == a.session
            and owner.shot_id == shot.shot_id
            and owner.sequence == shot.sequence
            and owner.source_epoch_id == a.epoch
            and owner.source_kind == "usb_live"
            and owner.evidence_status == "bound_at_start"
            and role in {"red", "blue"}
            and owner.actor_user_id == "offline-" + role
            and _integer(owner.match_revision)
            and _integer(owner.sequence, 1, MAX_SHOTS)
        )
        r["ownership"] = dict(
            actor_role=role if role in {"red", "blue"} else "UNKNOWN",
            owner=owner.actor_user_id
            if role in {"red", "blue"} and owner.actor_user_id == "offline-" + role
            else "UNKNOWN",
            match_revision=owner.match_revision,
            source_epoch_id=a.epoch,
            receipt_sha256=_digest(asdict(owner)),
        )
        if 1 <= shot.sequence <= len(self.oracle.shots):
            require(role == self.oracle.shots[shot.sequence - 1][0], "FULLGAME_ORACLE_MISMATCH")
        else:
            require(False, "FULLGAME_ORACLE_MISMATCH")
        if shot.settled_ms is None:
            return r
        r["state"] = "SETTLED"
        fact = a.store.get_live_shot_fact(a.session, shot.shot_id)
        rule = a.store.get_rule_decision(a.session, shot.shot_id)
        if fact is None or rule is None:
            # Product settlement writes these together; preserve partial receipt until next poll.
            return r
        facts, decision = _json(rule.facts_json), _json(rule.decision_json)
        r["facts"] = fact_summary(facts)
        require(r["facts"]["shape_valid"], "FULLGAME_FACT_SUMMARY_INVALID")
        getter = getattr(a.store, "get_rule_fact_preimage", None)
        try:
            preimage = getter(a.session, shot.shot_id) if callable(getter) else None
        except ValueError:
            require(False, "FULLGAME_FACT_PREIMAGE_INVALID")
        else:
            require(preimage is not None, "FULLGAME_FACT_PREIMAGE_MISSING")
            if preimage is not None:
                require(
                    _fact_preimage_valid(preimage, facts, rule.fact_digest),
                    "FULLGAME_FACT_PREIMAGE_INVALID",
                )
        require(
            fact.session_id == a.session
            and fact.shot_id == shot.shot_id
            and fact.sequence == shot.sequence
            and str(fact.actor_role) == role
            and fact.actor_user_id == owner.actor_user_id
            and fact.rule_revision == owner.match_revision
        )
        require(
            rule.session_id == a.session
            and rule.shot_id == shot.shot_id
            and rule.source_epoch_id == a.epoch
            and rule.profile_id == PROFILE
            and _integer(rule.start_frame_id)
            and _integer(rule.settled_frame_id)
            and rule.start_frame_id < rule.settled_frame_id
            and (rule.start_frame_id, rule.settled_frame_id)
            == (fact.start_frame_id, fact.settle_frame_id)
            and shot.live_evidence_digest == rule.settled_evidence_digest
            and all(
                _sha(x)
                for x in (rule.fact_digest, rule.decision_digest, rule.settled_evidence_digest)
            )
        )
        require(
            all(
                x.get("session_id") == a.session
                and x.get("shot_id") == shot.shot_id
                and x.get("source_epoch_id") == a.epoch
                for x in (facts, decision)
            )
        )
        require(
            facts.get("digest") == decision.get("fact_digest") == rule.fact_digest
            and decision.get("decision_digest") == rule.decision_digest
            and _digest({k: v for k, v in decision.items() if k != "decision_digest"})
            == rule.decision_digest
            and decision.get("expected_revision") == owner.match_revision
            and decision.get("actor_role") == role
            and decision.get("profile_id") == PROFILE
            and facts.get("profile_id") == PROFILE
            and facts.get("match_revision_at_start") == owner.match_revision
            and facts.get("actor_role") == role
            and (facts.get("start_frame_id"), facts.get("settled_frame_id"))
            == (rule.start_frame_id, rule.settled_frame_id)
            and facts.get("settled_evidence_digest")
            == decision.get("settled_evidence_digest")
            == rule.settled_evidence_digest
            and all(
                _integer(facts.get(k))
                for k in ("match_revision_at_start", "start_frame_id", "settled_frame_id")
            )
            and _integer(decision.get("expected_revision"))
        )
        action = decision.get("action")
        reason = decision.get("reason_code")
        r["rule"] = dict(
            status=rule.status
            if rule.status in {"applied", "manual", "stale", "abstained"}
            else "UNKNOWN",
            action=action if action in {"end", "continue_turn", "switch_turn"} else "UNKNOWN",
            reason_code=reason
            if isinstance(reason, str) and _CODE.fullmatch(reason)
            else "UNKNOWN",
            fact_digest=rule.fact_digest,
            decision_digest=rule.decision_digest,
            settled_evidence_digest=rule.settled_evidence_digest,
            start_frame_id=rule.start_frame_id,
            settle_frame_id=rule.settled_frame_id,
            placement_observed=facts.get("cue_ball_placement_observed")
            if facts.get("cue_ball_placement_observed") in {"true", "false", "unknown"}
            else "UNKNOWN",
            resulting_revision=None,
        )
        r["rule"]["winner"] = (
            decision.get("winner_role") if decision.get("winner_role") in {"red", "blue"} else None
        )
        reasons = facts.get("abstention_reasons", [])
        r["rule"]["abstention_reasons"] = [
            reason if isinstance(reason, str) and _CODE.fullmatch(reason) else "UNEXPORTED_REASON"
            for reason in reasons[:64]
        ]
        require(
            rule.status == "applied" and decision.get("kind") == "apply",
            "FULLGAME_RULE_NOT_APPLIED",
        )
        if 1 <= shot.sequence <= len(self.oracle.shots):
            require(
                (role, action, reason) == self.oracle.shots[shot.sequence - 1],
                "FULLGAME_ORACLE_MISMATCH",
            )
        if rule.status == "applied":
            entries = [entry for entry in ledger if entry.entry_id == rule.applied_command_id]
            require(len(entries) == 1, "FULLGAME_RULE_LEDGER")
            if len(entries) == 1:
                entry = entries[0]
                reference = _json(entry.decision_ref_json)
                require(
                    entry.session_id == a.session
                    and entry.origin == "rules_engine"
                    and entry.principal_id == PROFILE
                    and entry.command_type == "auto_" + action
                    and entry.payload_digest == rule.decision_digest
                    and entry.outcome == "applied"
                    and entry.result_code == reason
                    and entry.resulting_revision == owner.match_revision + 1
                    and reference
                    == dict(
                        session_id=a.session,
                        shot_id=shot.shot_id,
                        fact_digest=rule.fact_digest,
                        decision_digest=rule.decision_digest,
                    ),
                    "FULLGAME_RULE_LEDGER",
                )
                r["rule"]["resulting_revision"] = entry.resulting_revision
                r["rule"]["ledger_order"] = entry.ledger_order
        r["state"] = "RULE_RECORDED"
        media = a.store.get_live_shot_media(a.session, shot.shot_id)
        if media is None:
            return r
        require(
            media.session_id == a.session
            and media.shot_id == shot.shot_id
            and media.source_epoch_id == a.epoch
            and media.match_revision == owner.match_revision
            and media.job_id == shot.live_job_id
            and _sha(media.media_sha256)
        )
        job = a.fullgame_job(shot, media)
        require(
            job.get("job_id") == media.job_id
            and job.get("session_id") == a.session
            and job.get("shot_id") == shot.shot_id
            and job.get("source_epoch_id") == a.epoch
            and job.get("media_sha256") == media.media_sha256
            and _sha(job.get("payload_sha256")),
            "FULLGAME_JOB_LINEAGE",
        )
        r["media"] = dict(
            job_id=media.job_id,
            media_sha256=media.media_sha256,
            source_epoch_id=media.source_epoch_id,
            match_revision=media.match_revision,
        )
        r["job"] = job
        r["state"] = "MEDIA_DURABLE"
        if not r["errors"]:
            self.durable[shot.sequence] = json.loads(_canonical(r))
        return r

    def poll(self):
        # Share the product's authority lock when observing several independently
        # opened read transactions. Never combine half of two live transitions.
        lock = getattr(getattr(self.actor, "state", None), "match_lock", nullcontext())
        with lock:
            try:
                return self._poll()
            except (ValueError, TypeError, KeyError, AttributeError, OSError, sqlite3.Error):
                self.fail("FULLGAME_STORE_UNAVAILABLE")
                return False

    def _poll(self):
        a = self.actor
        shots = a.store.list_shots(a.session)
        _require(len(shots) <= MAX_SHOTS, "FULLGAME_SHOT_LIMIT")
        self.report["observed_shot_count"] = len(shots)
        ledger = a.store.list_command_ledger(a.session)
        _require(len(ledger) <= MAX_SHOTS + 3, "FULLGAME_LEDGER_LIMIT")
        orders = [x.ledger_order for x in ledger]
        if orders != list(range(1, len(orders) + 1)):
            self.fail("FULLGAME_LEDGER_ORDER")
        phone = [x for x in ledger if x.origin == "phone"]
        if (
            [x.command_type for x in phone] != ["claim_role", "claim_role", "set_breaker"]
            or [x.principal_id for x in phone] != ["offline-red", "offline-blue", "offline-red"]
            or any(x.origin not in {"phone", "rules_engine"} for x in ledger)
        ):
            self.fail("FULLGAME_SETUP_LEDGER")
        if [x.sequence for x in shots] != list(range(1, len(shots) + 1)):
            self.fail("FULLGAME_SHOT_ORDER")
        for shot in shots:
            try:
                receipt = self._shot(shot, ledger)
            except (ValueError, TypeError, KeyError, AttributeError, OSError, sqlite3.Error):
                self.fail("FULLGAME_RECEIPT_INVALID")
                receipt = dict(
                    sequence=shot.sequence,
                    shot_id=shot.shot_id,
                    state="INVALID",
                    errors=["FULLGAME_RECEIPT_INVALID"],
                )
            self._append(receipt)
        for sequence in sorted(self.durable):
            previous = self.durable.get(sequence - 1)
            if (
                previous is not None
                and self.durable[sequence]["ownership"]["match_revision"]
                != previous["rule"]["resulting_revision"]
            ):
                self.fail("FULLGAME_REVISION_CHAIN")
        session = a.store.load_session(a.session)
        match = session.match
        if str(session.status) != "ended":
            return False
        terminal = dict(
            session_status="ended",
            rack_phase=str(match.rack_phase),
            winner=str(match.winner_role),
            terminal_reason=match.terminal_reason,
            match_revision=match.revision,
        )
        # Only enums/code shapes are retained; arbitrary DB strings are not diagnostics.
        terminal["rack_phase"] = "ended" if terminal["rack_phase"] == "ended" else "UNKNOWN"
        terminal["winner"] = (
            terminal["winner"] if terminal["winner"] in {"red", "blue"} else "UNKNOWN"
        )
        if not isinstance(terminal["terminal_reason"], str) or not _CODE.fullmatch(
            terminal["terminal_reason"]
        ):
            terminal["terminal_reason"] = "UNKNOWN"
        self.report["terminal"] = terminal
        if (
            len(shots) != len(self.oracle.shots)
            or set(self.durable) != set(range(1, len(shots) + 1))
            or not shots
            or terminal["rack_phase"] != "ended"
            or terminal["winner"] != self.oracle.winner
            or terminal["terminal_reason"] != "legal_eight"
        ):
            self.fail("FULLGAME_NATURAL_TERMINAL_MISMATCH")
        else:
            last = self.durable[len(shots)]
            if (
                last["rule"]["action"],
                last["rule"]["reason_code"],
                last["rule"]["winner"],
                last["rule"]["resulting_revision"],
            ) != ("end", "legal_eight", self.oracle.winner, match.revision):
                self.fail("FULLGAME_NATURAL_TERMINAL_MISMATCH")
            terminal["shot_id"] = last["shot_id"]
            terminal["job_id"] = last["media"]["job_id"]
        phone = [x for x in ledger if x.origin == "phone"]
        if (
            [x.command_type for x in phone] != ["claim_role", "claim_role", "set_breaker"]
            or [x.principal_id for x in phone] != ["offline-red", "offline-blue", "offline-red"]
            or any(x.outcome != "applied" or x.result_code != "APPLIED" for x in phone)
        ):
            self.fail("FULLGAME_SETUP_LEDGER")
        if (
            len(phone) != 3
            or [x.origin_sequence for x in phone] != [1, 2, 3]
            or [x.ledger_order for x in phone] != [1, 2, 3]
            or any(x.session_id != a.session for x in phone)
            or (
                self.durable.get(1) is not None
                and phone[-1].resulting_revision != self.durable[1]["ownership"]["match_revision"]
            )
        ):
            self.fail("FULLGAME_SETUP_LEDGER")
        applied = [x for x in ledger if x.origin == "rules_engine"]
        if (
            len(applied) != len(shots)
            or len(ledger) != len(shots) + 3
            or any(x.origin not in {"phone", "rules_engine"} for x in ledger)
        ):
            self.fail("FULLGAME_RULE_LEDGER")
        manifest = a.store.get_postgame_manifest(a.session)
        jobs = {r["media"]["job_id"] for r in self.durable.values()}
        if (
            manifest is None
            or len(jobs) != len(shots)
            or tuple(manifest.expected_jobs)
            != tuple(self.durable[i]["media"]["job_id"] for i in sorted(self.durable))
        ):
            self.fail("FULLGAME_MANIFEST_MISMATCH")
        return True

    def run(self):
        a = self.actor
        try:
            _require(
                a.report["capture"].get("source_sha256") == self.oracle.video_sha256,
                "FULLGAME_CAPTURE_IDENTITY",
            )

            def probe():
                done = self.poll()  # Retain current rows even when health then reports EOF.
                a.health(ending=done)
                return done

            a.wait(probe, self.clock() + self.match_seconds, "FULLGAME_MATCH_TIMEOUT")
            remote = a.remote.snapshot()
            if (
                remote["birth_count"],
                remote["issued_count"],
                remote["completed_prefix"],
                remote["failure"],
            ) != (1, 3, 3, None):
                self.fail("FULLGAME_REMOTE_PREFIX")
            _require(not self.report["failures"], "FULLGAME_OBSERVATION_FAILED")
            chosen = self.durable[self.oracle.selected_sequence]
            a.report["boundary"] = dict(
                chosen["rule"], shot_id=chosen["shot_id"], owner=chosen["ownership"]["owner"]
            )
            a.report["media"] = dict(chosen["media"])
            a.report["end"] = dict(self.report["terminal"], origin="rules_engine")
            a.review()  # Existing real large-screen selection/save/idempotency/complete/disposal.
            selection = a.report["postgame"].get("selection", {})
            _require(
                all(
                    selection.get(k) == "VERIFIED"
                    for k in ("save", "idempotency", "completion", "cleanup")
                )
                and _integer(selection.get("unsaved_count"), 1, MAX_SHOTS - 1),
                "FULLGAME_UNSAVED_CLEANUP_UNVERIFIED",
            )
            # Immutable lineage receipts were checked before review. Do not
            # reopen unsaved source intervals after the real disposal step.
            self.report["status"] = "OBSERVED"
        except BaseException as exc:
            self.report["status"] = "FAILED"
            if isinstance(exc, FullgameFailure):
                self.fail(exc.args[0])
            raise
