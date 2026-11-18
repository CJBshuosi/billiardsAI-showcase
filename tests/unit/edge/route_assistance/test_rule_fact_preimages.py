"""Normalization evidence controls; coordinates/facts here are explicitly UNIT data."""

import hashlib
import json
import sqlite3
from dataclasses import asdict, replace

import pytest

from billiards_ai.edge.route_assistance.control import CueBallPlacement
from billiards_ai.edge.route_assistance.eight_ball_rules import decide_eight_ball
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from tests.unit.edge.route_assistance.test_rule_persistence import (
    _facts,
    _match,
    _prepare_settled_shot,
    _store,
)


def _canonical(value, *, ensure_ascii=True):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ensure_ascii)


def _unit_preimage(facts=None, *, placement="none", payload_changes=None):
    facts = facts or _facts()
    normalized = asdict(facts)
    normalized.pop("digest")
    normalized.update(required_cue_ball_placement=placement, event_config_digest="c" * 64)
    normalized.update(payload_changes or {})
    raw = _canonical(normalized)
    return replace(facts, digest=hashlib.sha256(raw.encode()).hexdigest()), raw


def _apply(store, facts, preimage):
    return store.apply_automatic_rule_decision(
        facts,
        decide_eight_ball(_match(), facts),
        elapsed_ms=1,
        fact_preimage_json=preimage,
    )


def test_fresh_preimage_is_atomic_exact_and_idempotent_without_changing_v1_json(tmp_path):
    store = _store(tmp_path)
    _prepare_settled_shot(store)
    facts, raw = _unit_preimage()

    receipt = _apply(store, facts, raw)

    assert receipt.facts_json == _canonical(asdict(facts), ensure_ascii=False)
    assert receipt.fact_digest == hashlib.sha256(raw.encode()).hexdigest()
    assert store.get_rule_fact_preimage(facts.session_id, facts.shot_id) == raw
    assert _apply(store, facts, raw) == receipt
    assert len(store.list_command_ledger(facts.session_id)) == 1
    assert store.load_session(facts.session_id).match.revision == 2
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM field_rule_fact_preimages").fetchone()[0] == 1
    # Losing a previously required sidecar is a conflict, never a silent rewrite.
    with pytest.raises(ValueError, match="preimage"):
        _apply(store, facts, None)


@pytest.mark.parametrize("mutation", ["digest", "projection", "placement", "config_type"])
def test_invalid_fresh_preimage_never_mutates_match_ledger_or_rule_receipt(tmp_path, mutation):
    store = _store(tmp_path)
    _prepare_settled_shot(store)
    facts, raw = _unit_preimage()
    if mutation == "digest":
        raw = raw.replace('"actor_role":"red"', '"actor_role":"blue"')
    elif mutation == "projection":
        facts, raw = _unit_preimage(payload_changes={"actor_role": "blue"})
    elif mutation == "placement":
        facts, raw = _unit_preimage(placement=CueBallPlacement.ANYWHERE.value)
    else:
        facts, raw = _unit_preimage(payload_changes={"event_config_digest": True})

    with pytest.raises(ValueError, match="preimage|placement"):
        _apply(store, facts, raw)

    assert store.load_session(facts.session_id).match == _match()
    assert store.list_command_ledger(facts.session_id) == ()
    assert store.get_rule_decision(facts.session_id, facts.shot_id) is None
    assert store.get_rule_fact_preimage(facts.session_id, facts.shot_id) is None


def test_sidecar_insert_failure_rolls_back_settlement_rule_and_ledger(tmp_path, monkeypatch):
    store = _store(tmp_path)
    captured = []
    # Keep a real persisted start; obtain the fixture's settled record without
    # persisting it so the tested boundary must perform the actual settlement.
    with monkeypatch.context() as patch:
        patch.setattr(store, "record_shot_settle", lambda shot, **kw: captured.append(shot))
        _prepare_settled_shot(store)
    facts, raw = _unit_preimage()
    with sqlite3.connect(store.path) as db:
        db.execute("""CREATE TRIGGER unit_deny_preimage BEFORE INSERT ON field_rule_fact_preimages
                      BEGIN SELECT RAISE(ABORT, 'unit_reject'); END""")

    with pytest.raises(ValueError):
        store.settle_and_apply_automatic_rule_decision(
            captured[0],
            facts,
            decide_eight_ball(_match(), facts),
            source_epoch_id=facts.source_epoch_id,
            elapsed_ms=1,
            fact_preimage_json=raw,
        )

    assert store.list_shots(facts.session_id)[0].settled_ms is None
    assert store.load_session(facts.session_id).match == _match()
    assert store.list_command_ledger(facts.session_id) == ()
    assert store.get_rule_decision(facts.session_id, facts.shot_id) is None
    assert store.get_rule_fact_preimage(facts.session_id, facts.shot_id) is None


def test_schema25_migration_and_legacy_replay_never_backfill_a_preimage(tmp_path):
    store = _store(tmp_path)
    _prepare_settled_shot(store)
    facts, raw = _unit_preimage()
    legacy = store.apply_automatic_rule_decision(
        facts,
        decide_eight_ball(_match(), facts),
        elapsed_ms=1,
    )
    with sqlite3.connect(store.path) as db:
        old_rules = db.execute("SELECT * FROM field_rule_decisions").fetchall()
        old_ledger = db.execute("SELECT * FROM field_command_ledger").fetchall()
        old_match = db.execute("SELECT * FROM field_match_state").fetchall()
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=25")

    migrated = SQLiteFieldRecordStore(store.path)

    assert migrated.get_rule_decision(facts.session_id, facts.shot_id) == legacy
    assert migrated.get_rule_fact_preimage(facts.session_id, facts.shot_id) is None
    assert _apply(migrated, facts, raw) == legacy
    assert migrated.get_rule_fact_preimage(facts.session_id, facts.shot_id) is None
    with sqlite3.connect(store.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30
        assert db.execute("SELECT * FROM field_rule_decisions").fetchall() == old_rules
        assert db.execute("SELECT * FROM field_command_ledger").fetchall() == old_ledger
        assert db.execute("SELECT * FROM field_match_state").fetchall() == old_match
        assert db.execute("SELECT COUNT(*) FROM field_rule_fact_preimages").fetchone()[0] == 0


def test_corrupt_persisted_preimage_is_rejected_without_overwriting_original_receipt(tmp_path):
    store = _store(tmp_path)
    _prepare_settled_shot(store)
    facts, raw = _unit_preimage()
    receipt = _apply(store, facts, raw)
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE field_rule_fact_preimages SET normalization_preimage_json='{}'")

    with pytest.raises(ValueError, match="preimage"):
        store.get_rule_fact_preimage(facts.session_id, facts.shot_id)
    with pytest.raises(ValueError, match="preimage"):
        _apply(store, facts, raw)
    assert store.get_rule_decision(facts.session_id, facts.shot_id) == receipt
    assert len(store.list_command_ledger(facts.session_id)) == 1


def test_preimage_read_is_bound_to_the_authoritative_receipt_epoch(tmp_path):
    store = _store(tmp_path)
    _prepare_settled_shot(store)
    facts, raw = _unit_preimage()
    _apply(store, facts, raw)
    with sqlite3.connect(store.path) as db:
        # The preimage hash and facts_json still match each other; changing only
        # the authoritative receipt epoch must nevertheless invalidate the read.
        db.execute("UPDATE field_rule_decisions SET source_epoch_id='other-epoch'")
    with pytest.raises(ValueError, match="preimage"):
        store.get_rule_fact_preimage(facts.session_id, facts.shot_id)
