"""Fail-closed SQLite persistence for local field-assistance facts."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
import time
import uuid
from collections.abc import Callable
from contextlib import closing, contextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

from billiards_ai.edge.route_assistance import rule_facts as rule_fact_evidence
from billiards_ai.edge.route_assistance.commands import (
    FieldCommandApplication,
    FieldCommandEnvelope,
    FieldCommandReduction,
    reduce_field_command,
)
from billiards_ai.edge.route_assistance.control import (
    BallGroupAssignment,
    CueBallPlacement,
    MatchControlSnapshot,
    MatchFormat,
    MatchMode,
    RackPhase,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.eight_ball_rules import (
    PROFILE_ID,
    RuleDecision,
    RuleOutcomeKind,
    apply_rule_decision,
    decide_eight_ball,
)
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedOpenRequest,
    ManagedSessionBinding,
    ManagedSessionCloseState,
    ManagedSessionScope,
    validate_managed_artifact_id,
    validate_managed_rule_profile_id,
    validate_managed_scope_id,
    validate_managed_utc_timestamp,
    validate_managed_uuid,
)
from billiards_ai.edge.route_assistance.owner_media_deletion import (
    OwnerDeleteDeviceTask,
    OwnerDeleteLocalState,
)
from billiards_ai.edge.route_assistance.postgame import (
    MAX_SAVED_CLIP_BYTES,
    ClipSaveReceipt,
    ClipSaveStatus,
    LocalDiscardClaim,
    PostgameClip,
    PostgameSessionManifest,
    SavedMediaOutboxItem,
    clip_save_receipt_from_json,
    session_scoped_clip_id,
    validate_save_request_id,
)
from billiards_ai.edge.route_assistance.recording_capture import (
    RECORDING_CAPTURE_SCHEMA30,
    recording_capture_from_payload,
)
from billiards_ai.edge.route_assistance.records import (
    CommandLedgerEntry,
    FieldSessionRecord,
    FieldSessionStatus,
    FieldShotRecord,
    LiveShotFact,
    LiveShotFactOutboxItem,
    LiveShotMediaReference,
    RecordedMediaV1,
    RecordingBoundaryV1,
    RuleDecisionReceipt,
    SessionLimitCode,
    ShotMediaRange,
    ShotOwnershipReceipt,
    ownership_receipt_from_payload,
    ownership_receipt_sha256,
    recording_end_from_payload,
    recording_evidence_from_payload,
    require_session_policy,
    validate_shot_ownership,
)
from billiards_ai.edge.route_assistance.rule_facts import RealtimeRuleFacts
from billiards_ai.edge.route_assistance.runtime_resources import (
    MAX_RESOURCE_METADATA_BYTES,
    ManagedCaptureEpochRecord,
    ManagedResourceSpecRecord,
    ManagedRuntimeResourceSpec,
)
from billiards_ai.edge.route_assistance.shot_diagnostics import is_capture_interruption
from billiards_ai.edge.route_assistance.sync import SyncMediaItem, SyncOutboxJob
from billiards_ai.edge.route_assistance.turn_confirmation import (
    TurnConfirmationReceipt,
    TurnConfirmationRef,
)

_SCHEMA_VERSION = 30
_RECORDING_CLEANUP_SCHEMA29 = """
CREATE TABLE IF NOT EXISTS field_recording_cleanup (
  session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
  manifest_sha256 TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','leased','completed','blocked')),
  attempts INTEGER NOT NULL DEFAULT 0,
  claim_token TEXT,
  lease_until_utc TEXT,
  next_attempt_at_utc TEXT NOT NULL,
  inventory_json TEXT,
  inventory_sha256 TEXT,
  progress_json TEXT NOT NULL DEFAULT '{}',
  last_error_code TEXT
);
CREATE INDEX IF NOT EXISTS due_recording_cleanup
ON field_recording_cleanup(status,next_attempt_at_utc,session_id);
CREATE TABLE IF NOT EXISTS field_original_clip_sources (
  session_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  media_job_json TEXT NOT NULL,
  media_job_sha256 TEXT NOT NULL,
  artifact_manifest_sha256 TEXT NOT NULL,
  PRIMARY KEY(session_id,shot_id),
  FOREIGN KEY(session_id,shot_id)
    REFERENCES field_postgame_clip_availability(session_id,shot_id)
);
"""
_RECORDING_END_SCHEMA28 = """
CREATE TABLE field_recording_ends (
  session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
  phase TEXT NOT NULL CHECK(phase IN ('prepared','frozen')),
  payload_json TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL
);
"""
_RECORDING_SCHEMA27 = """
CREATE TABLE field_session_recording_policy (
  session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
  policy TEXT NOT NULL CHECK(policy IN ('legacy','recording-v1'))
);
CREATE TABLE field_recording_evidence (
  session_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  PRIMARY KEY(session_id,shot_id),
  FOREIGN KEY(session_id,shot_id) REFERENCES field_shots(session_id,shot_id)
);
"""
_PERSISTED_SAVED_MEDIA_INVALID_ERROR = "SYNC_PERSISTED_DESCRIPTOR_INVALID"

_OWNER_COMPLETION_SCHEMA25 = """
CREATE TABLE field_owner_completion_outbox (
  deletion_job_id TEXT PRIMARY KEY,
  clip_id TEXT NOT NULL UNIQUE,
  environment_id TEXT NOT NULL,
  device_id TEXT NOT NULL,
  table_id TEXT NOT NULL,
  task_sha256 TEXT NOT NULL,
  cloud_erase_receipt_sha256 TEXT NOT NULL,
  local_receipt_json TEXT NOT NULL,
  local_receipt_sha256 TEXT NOT NULL,
  completion_json TEXT NOT NULL,
  completion_sha256 TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','leased','acknowledged','blocked')),
  attempts INTEGER NOT NULL CHECK(attempts >= 0),
  claim_token TEXT,
  lease_until_utc TEXT,
  next_attempt_at_utc TEXT NOT NULL,
  last_error_code TEXT,
  ack_json TEXT
);
CREATE INDEX due_owner_completion_outbox
  ON field_owner_completion_outbox(environment_id,device_id,table_id,status,next_attempt_at_utc);
CREATE TABLE field_local_media_schedule_state (
  state_key TEXT PRIMARY KEY,
  cursor TEXT NOT NULL DEFAULT '',
  upper_key TEXT,
  turn TEXT NOT NULL
);
"""

_UPLOAD_RECEIPT_SCHEMA24 = """
CREATE TABLE field_upload_receipt_outbox (
  request_id TEXT PRIMARY KEY REFERENCES field_saved_upload_requests(request_id),
  environment_id TEXT NOT NULL,
  device_id TEXT NOT NULL,
  table_id TEXT NOT NULL,
  clip_id TEXT NOT NULL,
  cloud_job_id TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  receipt_sha256 TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','leased','acknowledged','blocked')),
  attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
  claim_token TEXT,
  lease_until_utc TEXT,
  next_attempt_at_utc TEXT,
  last_error_code TEXT
);
CREATE INDEX due_upload_receipts ON field_upload_receipt_outbox
  (environment_id,device_id,table_id,status,next_attempt_at_utc,request_id);
"""

_SAVED_UPLOAD_REQUEST_SCHEMA23 = """
CREATE TABLE field_saved_upload_requests (
  request_id TEXT PRIMARY KEY,
  clip_id TEXT NOT NULL REFERENCES field_postgame_clip_availability(clip_id),
  cloud_job_id TEXT NOT NULL,
  binding_json TEXT NOT NULL,
  binding_sha256 TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN
    ('requested','received','started','succeeded','unknown','retired')),
  upload_attempt_json TEXT,
  created_at_utc TEXT NOT NULL,
  started_at_utc TEXT,
  terminal_json TEXT,
  terminal_sha256 TEXT
);
CREATE UNIQUE INDEX one_active_saved_upload_request
  ON field_saved_upload_requests(clip_id) WHERE state != 'retired';
"""

_OWNER_DELETE_MARKER = "OWNER_CLIP_DELETE_REQUESTED"
_OWNER_DELETE_SCHEMA22 = """
CREATE TABLE field_owner_media_deletions (
  clip_id TEXT PRIMARY KEY REFERENCES field_postgame_clip_availability(clip_id),
  deletion_job_id TEXT NOT NULL UNIQUE,
  session_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  task_json TEXT NOT NULL,
  task_sha256 TEXT NOT NULL,
  local_path TEXT NOT NULL,
  selection_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','leased','retry','blocked','completed')),
  attempts INTEGER NOT NULL CHECK(attempts >= 0),
  claim_token TEXT,
  lease_until_utc TEXT,
  next_attempt_at_utc TEXT NOT NULL,
  progress_json TEXT NOT NULL DEFAULT '{}',
  receipt_json TEXT,
  receipt_sha256 TEXT,
  last_error_code TEXT,
  created_at_utc TEXT NOT NULL,
  UNIQUE(session_id,shot_id),
  FOREIGN KEY(session_id,shot_id)
    REFERENCES field_postgame_clip_availability(session_id,shot_id)
);
CREATE INDEX due_owner_media_deletions
  ON field_owner_media_deletions(status,next_attempt_at_utc);
"""

_MANAGED_OPEN_SCHEMA = """
CREATE TABLE field_managed_open_requests (
  environment_id TEXT NOT NULL,
  device_id TEXT NOT NULL,
  table_id TEXT NOT NULL,
  artifact_id TEXT NOT NULL UNIQUE,
  rule_profile_id TEXT,
  open_request_id TEXT NOT NULL,
  phase TEXT NOT NULL CHECK(phase IN ('pending','bound','materialized')),
  session_id TEXT,
  birth_id TEXT,
  birth_contract TEXT,
  remote_status TEXT,
  created_at_utc TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL,
  PRIMARY KEY(environment_id,device_id,open_request_id),
  UNIQUE(environment_id,device_id,table_id),
  UNIQUE(environment_id,session_id),
  UNIQUE(environment_id,birth_id),
  CHECK(
    (phase='pending' AND session_id IS NULL AND birth_id IS NULL
      AND birth_contract IS NULL AND remote_status IS NULL)
    OR
    (phase IN ('bound','materialized') AND session_id IS NOT NULL AND birth_id IS NOT NULL
      AND birth_contract IS NOT NULL AND remote_status IS NOT NULL
      AND remote_status IN ('open','ended'))
  )
);
"""

# Keep the v19 definition above for the historical 18 -> 19 migration.
_MANAGED_OPEN_SCHEMA20 = """
CREATE TABLE field_managed_open_requests (
  environment_id TEXT NOT NULL,
  device_id TEXT NOT NULL,
  table_id TEXT NOT NULL,
  artifact_id TEXT NOT NULL,
  rule_profile_id TEXT,
  open_request_id TEXT NOT NULL,
  phase TEXT NOT NULL CHECK(phase IN ('pending','bound','materialized')),
  session_id TEXT,
  birth_id TEXT,
  birth_contract TEXT,
  remote_status TEXT,
  created_at_utc TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL,
  lifecycle TEXT NOT NULL DEFAULT 'active' CHECK(lifecycle IN ('active','retired')),
  retired_at_utc TEXT,
  sealed_close_json TEXT,
  reservation_order INTEGER NOT NULL CHECK(
    typeof(reservation_order)='integer' AND reservation_order > 0),
  PRIMARY KEY(environment_id,device_id,open_request_id),
  UNIQUE(environment_id,device_id,table_id,reservation_order),
  UNIQUE(environment_id,session_id),
  UNIQUE(environment_id,birth_id),
  CHECK((lifecycle='active' AND retired_at_utc IS NULL AND sealed_close_json IS NULL)
    OR (lifecycle='retired' AND phase='materialized' AND remote_status='ended'
      AND retired_at_utc IS NOT NULL AND sealed_close_json IS NOT NULL)),
  CHECK(
    (phase='pending' AND session_id IS NULL AND birth_id IS NULL
      AND birth_contract IS NULL AND remote_status IS NULL)
    OR
    (phase IN ('bound','materialized') AND session_id IS NOT NULL AND birth_id IS NOT NULL
      AND birth_contract IS NOT NULL AND remote_status IS NOT NULL
      AND remote_status IN ('open','ended'))
  )
);
"""
_MANAGED_OPEN_ACTIVE_INDEXES = """
CREATE UNIQUE INDEX field_managed_active_scope
  ON field_managed_open_requests(environment_id,device_id,table_id) WHERE lifecycle='active';
CREATE UNIQUE INDEX field_managed_active_artifact
  ON field_managed_open_requests(artifact_id) WHERE lifecycle='active';
"""

# Metadata only; permanent requests and epochs are never backfilled or reclaimed.
_MANAGED_RESOURCE_SCHEMA21 = """
CREATE TABLE field_managed_resource_specs (
  environment_id TEXT NOT NULL,
  device_id TEXT NOT NULL,
  table_id TEXT NOT NULL,
  open_request_id TEXT NOT NULL,
  spec_json TEXT NOT NULL,
  spec_sha256 TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY(environment_id,device_id,open_request_id),
  UNIQUE(environment_id,device_id,open_request_id,spec_sha256),
  FOREIGN KEY(environment_id,device_id,open_request_id)
    REFERENCES field_managed_open_requests(environment_id,device_id,open_request_id)
);
CREATE TABLE field_managed_capture_epochs (
  environment_id TEXT NOT NULL,
  device_id TEXT NOT NULL,
  table_id TEXT NOT NULL,
  open_request_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  birth_id TEXT NOT NULL,
  source_epoch_id TEXT NOT NULL,
  spec_sha256 TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY(environment_id,device_id,source_epoch_id),
  FOREIGN KEY(environment_id,device_id,open_request_id,spec_sha256)
    REFERENCES field_managed_resource_specs(
      environment_id,device_id,open_request_id,spec_sha256)
);
CREATE INDEX field_managed_capture_epochs_session
  ON field_managed_capture_epochs(environment_id,device_id,table_id,session_id);
"""

_LOCAL_DISCARD_SCHEMA = """
CREATE TABLE field_local_media_discards (
  session_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  postgame_manifest_sha256 TEXT NOT NULL,
  job_id TEXT NOT NULL,
  artifact_id TEXT NOT NULL,
  clip_id TEXT NOT NULL UNIQUE,
  local_path TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  byte_size INTEGER NOT NULL CHECK(byte_size > 0),
  content_type TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','leased','completed','blocked')),
  attempts INTEGER NOT NULL CHECK(attempts >= 0),
  claim_token TEXT,
  lease_until_utc TEXT,
  next_attempt_at_utc TEXT NOT NULL,
  last_error_code TEXT,
  progress_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(session_id,shot_id),
  FOREIGN KEY(session_id,shot_id)
    REFERENCES field_postgame_clip_availability(session_id,shot_id)
);
CREATE INDEX due_local_media_discards
  ON field_local_media_discards(status,next_attempt_at_utc);
"""

_POSTGAME_REVIEW_SCHEMA = """
CREATE TABLE field_postgame_reviews (
  queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL UNIQUE REFERENCES field_postgame_manifests(session_id),
  artifact_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','active','completed'))
);
CREATE UNIQUE INDEX field_postgame_one_active_review
  ON field_postgame_reviews(artifact_id) WHERE status='active';
"""

_SAVED_CLIP_SCHEMA = """
CREATE TABLE field_postgame_clip_availability (
  session_id TEXT NOT NULL REFERENCES field_postgame_manifests(session_id),
  shot_id TEXT NOT NULL,
  postgame_manifest_sha256 TEXT NOT NULL,
  job_id TEXT NOT NULL,
  artifact_id TEXT NOT NULL,
  clip_id TEXT NOT NULL UNIQUE,
  local_path TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  byte_size INTEGER NOT NULL CHECK(byte_size > 0),
  content_type TEXT NOT NULL,
  PRIMARY KEY(session_id,shot_id),
  FOREIGN KEY(session_id,shot_id) REFERENCES field_shots(session_id,shot_id)
);
CREATE TABLE field_clip_save_requests (
  request_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES field_postgame_manifests(session_id),
  shot_id TEXT NOT NULL,
  postgame_manifest_sha256 TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  FOREIGN KEY(session_id,shot_id) REFERENCES field_shots(session_id,shot_id)
);
CREATE INDEX saved_clip_selection ON field_clip_save_requests(session_id,shot_id);
CREATE TABLE field_saved_media_outbox (
  clip_id TEXT PRIMARY KEY REFERENCES field_postgame_clip_availability(clip_id),
  save_request_id TEXT NOT NULL REFERENCES field_clip_save_requests(request_id),
  status TEXT NOT NULL CHECK(status IN ('pending','leased','acknowledged','blocked')),
  attempts INTEGER NOT NULL CHECK(attempts >= 0),
  claim_token TEXT,
  lease_until_utc TEXT,
  next_attempt_at_utc TEXT NOT NULL,
  last_error_code TEXT
);
CREATE INDEX due_saved_media ON field_saved_media_outbox(status,next_attempt_at_utc);
"""

_SAVED_MEDIA_BLOCKED_MIGRATION = """
ALTER TABLE field_saved_media_outbox RENAME TO field_saved_media_outbox_v15;
CREATE TABLE field_saved_media_outbox (
  clip_id TEXT PRIMARY KEY REFERENCES field_postgame_clip_availability(clip_id),
  status TEXT NOT NULL CHECK(status IN ('pending','leased','acknowledged','blocked')),
  attempts INTEGER NOT NULL CHECK(attempts >= 0),
  claim_token TEXT,
  lease_until_utc TEXT,
  next_attempt_at_utc TEXT NOT NULL,
  last_error_code TEXT
);
INSERT INTO field_saved_media_outbox(
  clip_id,status,attempts,claim_token,lease_until_utc,next_attempt_at_utc,last_error_code
)
SELECT clip_id,status,attempts,claim_token,lease_until_utc,next_attempt_at_utc,last_error_code
FROM field_saved_media_outbox_v15;
DROP TABLE field_saved_media_outbox_v15;
CREATE INDEX due_saved_media ON field_saved_media_outbox(status,next_attempt_at_utc);
"""

_SAVED_MEDIA_SELECTION_MIGRATION = """
ALTER TABLE field_saved_media_outbox RENAME TO field_saved_media_outbox_v16;
CREATE TABLE field_saved_media_outbox (
  clip_id TEXT PRIMARY KEY REFERENCES field_postgame_clip_availability(clip_id),
  save_request_id TEXT NOT NULL REFERENCES field_clip_save_requests(request_id),
  status TEXT NOT NULL CHECK(status IN ('pending','leased','acknowledged','blocked')),
  attempts INTEGER NOT NULL CHECK(attempts >= 0),
  claim_token TEXT,
  lease_until_utc TEXT,
  next_attempt_at_utc TEXT NOT NULL,
  last_error_code TEXT
);
INSERT INTO field_saved_media_outbox(
  clip_id,save_request_id,status,attempts,claim_token,lease_until_utc,
  next_attempt_at_utc,last_error_code
)
SELECT o.clip_id,
  (SELECT r.request_id FROM field_clip_save_requests r
   JOIN field_postgame_clip_availability a
     ON a.session_id=r.session_id AND a.shot_id=r.shot_id
   WHERE a.clip_id=o.clip_id
     AND r.postgame_manifest_sha256=a.postgame_manifest_sha256
   ORDER BY r.rowid LIMIT 1),
  o.status,o.attempts,o.claim_token,o.lease_until_utc,o.next_attempt_at_utc,o.last_error_code
FROM field_saved_media_outbox_v16 o;
DROP TABLE field_saved_media_outbox_v16;
CREATE INDEX due_saved_media ON field_saved_media_outbox(status,next_attempt_at_utc);
"""

_LIVE_SHOT_FACT_SCHEMA = """
CREATE TABLE field_live_shot_facts (
  idempotency_key TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  UNIQUE(session_id, shot_id),
  FOREIGN KEY(session_id, shot_id) REFERENCES field_shots(session_id, shot_id)
);
CREATE TABLE field_live_shot_fact_outbox (
  idempotency_key TEXT PRIMARY KEY REFERENCES field_live_shot_facts(idempotency_key),
  status TEXT NOT NULL CHECK(status IN ('pending','leased','acknowledged')),
  attempts INTEGER NOT NULL CHECK(attempts >= 0),
  lease_until_utc TEXT,
  updated_at_utc TEXT NOT NULL
);
CREATE INDEX due_field_live_shot_facts
  ON field_live_shot_fact_outbox(status, lease_until_utc);
"""

_UNIFIED_COMMAND_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS field_command_ledger (
  entry_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES field_sessions(session_id),
  ledger_order INTEGER NOT NULL CHECK(ledger_order > 0),
  origin TEXT NOT NULL CHECK(origin IN ('phone','rules_engine')),
  principal_id TEXT NOT NULL,
  origin_sequence INTEGER,
  command_type TEXT NOT NULL,
  payload_digest TEXT NOT NULL,
  outcome TEXT NOT NULL CHECK(outcome IN ('applied','rejected')),
  result_code TEXT NOT NULL,
  resulting_revision INTEGER NOT NULL CHECK(resulting_revision >= 0),
  decision_ref_json TEXT,
  completed_at_utc TEXT NOT NULL,
  UNIQUE(session_id, ledger_order),
  CHECK(
    (origin='phone' AND origin_sequence IS NOT NULL AND decision_ref_json IS NULL)
    OR
    (origin='rules_engine' AND origin_sequence IS NULL AND decision_ref_json IS NOT NULL)
  )
);
"""

_RULE_DECISION_SCHEMA = """
CREATE TABLE IF NOT EXISTS field_rule_decisions (
  session_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  source_epoch_id TEXT NOT NULL,
  start_frame_id INTEGER NOT NULL,
  settled_frame_id INTEGER NOT NULL,
  settled_evidence_digest TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  facts_json TEXT NOT NULL,
  fact_digest TEXT NOT NULL,
  decision_json TEXT NOT NULL,
  decision_digest TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('applied','abstained','manual','stale')),
  applied_command_id TEXT,
  elapsed_ms REAL NOT NULL CHECK(elapsed_ms >= 0),
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY(session_id, shot_id),
  FOREIGN KEY(session_id, shot_id) REFERENCES field_shots(session_id, shot_id),
  FOREIGN KEY(applied_command_id) REFERENCES field_command_ledger(entry_id),
  CHECK((status='applied') = (applied_command_id IS NOT NULL))
);
"""

_RULE_FACT_PREIMAGE_SCHEMA26 = """
CREATE TABLE field_rule_fact_preimages (
  session_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  fact_digest TEXT NOT NULL,
  normalization_preimage_json TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY(session_id, shot_id),
  FOREIGN KEY(session_id, shot_id) REFERENCES field_rule_decisions(session_id, shot_id)
);
"""

_LIVE_MEDIA_SCHEMA = """
CREATE TABLE field_live_shot_media (
  session_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY(session_id, shot_id),
  FOREIGN KEY(session_id, shot_id) REFERENCES field_shots(session_id, shot_id)
);
"""

_POSTGAME_MANIFEST_SCHEMA = """
CREATE TABLE field_postgame_manifests (
  session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
  ended_at_utc TEXT NOT NULL,
  deadline_at_utc TEXT NOT NULL,
  expected_jobs_json TEXT NOT NULL
);
"""
_POSTGAME_FINALIZATION_SCHEMA = """
CREATE TABLE field_postgame_finalizations (
  session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
  manifest_sha256 TEXT NOT NULL,
  manifest_json TEXT NOT NULL
);
"""
_LOCAL_POSTGAME_OUTBOX_SCHEMA = """
CREATE TABLE local_postgame_outbox (
  session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
  manifest_sha256 TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','complete'))
);
"""

_RECOVERY_SCHEMA = """
CREATE TABLE field_session_artifacts (
  session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
  artifact_path TEXT NOT NULL
);
CREATE TABLE field_cloud_close_intents (
  session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
  status TEXT NOT NULL CHECK(status IN ('open','pending','complete')),
  updated_at_utc TEXT NOT NULL
);
"""

_TURN_CONFIRMATION_SCHEMA = """
CREATE TABLE field_turn_confirmations (
  session_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  evidence_digest TEXT NOT NULL,
  match_revision INTEGER NOT NULL,
  source_mainline1_artifact_id TEXT,
  status TEXT NOT NULL CHECK(status IN ('pending','stale','resolved')),
  stale_reason TEXT,
  resolved_command_id TEXT REFERENCES field_applied_commands(command_id),
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY(session_id,shot_id),
  FOREIGN KEY(session_id,shot_id) REFERENCES field_shots(session_id,shot_id),
  CHECK((status='resolved') = (resolved_command_id IS NOT NULL)),
  CHECK((status='stale') = (stale_reason IS NOT NULL))
);
"""

_OWNERSHIP_SCHEMA = """
CREATE TABLE field_shot_ownership (
  session_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY(session_id, shot_id),
  FOREIGN KEY(session_id, shot_id) REFERENCES field_shots(session_id, shot_id)
);
"""

_COMMAND_LEDGER_SCHEMA = """
CREATE TABLE field_applied_commands (
  command_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES field_sessions(session_id),
  server_sequence INTEGER NOT NULL CHECK(server_sequence > 0),
  actor_user_id TEXT NOT NULL,
  command_type TEXT NOT NULL,
  payload_digest TEXT NOT NULL,
  outcome TEXT NOT NULL CHECK(outcome IN ('applied','rejected')),
  result_code TEXT NOT NULL,
  resulting_revision INTEGER NOT NULL CHECK(resulting_revision >= 0),
  completed_at_utc TEXT NOT NULL,
  UNIQUE(session_id, server_sequence)
);
"""

_ADMIN_AUDIT_SCHEMA = """
CREATE TABLE field_admin_audit (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES field_sessions(session_id),
  command_type TEXT NOT NULL,
  reason TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  resulting_revision INTEGER NOT NULL CHECK(resulting_revision >= 0),
  created_at_utc TEXT NOT NULL
);
"""


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(_json_value(key)): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("persistence clock must return an aware datetime")
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _match_payload(match: MatchControlSnapshot) -> dict[str, object]:
    return asdict(match)


def _match_from_payload(raw: str) -> MatchControlSnapshot:
    try:
        payload = json.loads(raw)
        return MatchControlSnapshot(
            red_user_id=payload["red_user_id"],
            blue_user_id=payload["blue_user_id"],
            breaker=(None if payload["breaker"] is None else WristbandRole(payload["breaker"])),
            active_role=(
                None if payload["active_role"] is None else WristbandRole(payload["active_role"])
            ),
            red_group=(
                None if payload["red_group"] is None else BallGroupAssignment(payload["red_group"])
            ),
            blue_group=(
                None
                if payload["blue_group"] is None
                else BallGroupAssignment(payload["blue_group"])
            ),
            revision=int(payload["revision"]),
            rule_profile_id=payload.get("rule_profile_id"),
            rack_phase=RackPhase(payload.get("rack_phase", RackPhase.BREAK.value)),
            cue_ball_placement=CueBallPlacement(
                payload.get("cue_ball_placement", CueBallPlacement.NONE.value)
            ),
            winner_role=(
                None
                if payload.get("winner_role") is None
                else WristbandRole(payload["winner_role"])
            ),
            terminal_reason=payload.get("terminal_reason"),
            mode=(None if payload.get("mode") is None else MatchMode(payload["mode"])),
            match_format=(
                None
                if payload.get("match_format") is None
                else MatchFormat(payload["match_format"])
            ),
            race_target=payload.get("race_target"),
            ranking_intent=bool(payload.get("ranking_intent", False)),
            settings_revision=int(payload.get("settings_revision", 0)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("persisted match state is invalid") from exc


def _shot_payload(shot: FieldShotRecord) -> dict[str, object]:
    media = asdict(shot.media)
    if shot.media.source_epoch_id is None:
        del media["source_epoch_id"]
    payload = {
        "session_id": shot.session_id,
        "shot_id": shot.shot_id,
        "sequence": shot.sequence,
        "actor_role": None if shot.actor_role is None else shot.actor_role.value,
        "started_ms": shot.started_ms,
        "settled_ms": shot.settled_ms,
        "recommendation_id": shot.recommendation_id,
        "power_band": shot.power_band,
        "power_percent": shot.power_percent,
        "media": media,
        "limitations": [item.value for item in shot.limitations],
    }
    if shot.live_evidence_digest is not None:
        payload["live_evidence_digest"] = shot.live_evidence_digest
        payload["live_job_id"] = shot.live_job_id
    return payload


def _shot_from_payload(raw: str) -> FieldShotRecord:
    try:
        payload: dict[str, Any] = json.loads(raw)
        media = payload["media"]
        source_kind = media["source_kind"]
        if source_kind == "development_presentation":
            source_kind = "development_replay"
        return FieldShotRecord(
            session_id=str(payload["session_id"]),
            shot_id=str(payload["shot_id"]),
            sequence=int(payload["sequence"]),
            actor_role=(
                None if payload["actor_role"] is None else WristbandRole(payload["actor_role"])
            ),
            started_ms=int(payload["started_ms"]),
            settled_ms=(None if payload["settled_ms"] is None else int(payload["settled_ms"])),
            recommendation_id=payload["recommendation_id"],
            power_band=payload["power_band"],
            power_percent=(
                None if payload["power_percent"] is None else int(payload["power_percent"])
            ),
            media=ShotMediaRange(
                artifact_id=str(media["artifact_id"]),
                source_locator=str(media["source_locator"]),
                source_kind=source_kind,
                start_ms=int(media["start_ms"]),
                end_ms=int(media["end_ms"]),
                source_epoch_id=media.get("source_epoch_id"),
            ),
            limitations=tuple(SessionLimitCode(item) for item in payload["limitations"]),
            live_evidence_digest=payload.get("live_evidence_digest"),
            live_job_id=payload.get("live_job_id"),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("persisted shot state is invalid") from exc


def _live_shot_fact_from_payload(raw: str) -> LiveShotFact:
    try:
        payload = json.loads(raw)
        payload["actor_role"] = WristbandRole(payload["actor_role"])
        return LiveShotFact(**payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("persisted live shot fact is invalid") from exc


class SQLiteFieldRecordStore:
    """Persist local match and shot facts with optimistic revisions."""

    def __init__(
        self,
        path: Path,
        *,
        now_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
        session_id_factory: Callable[[], str] = lambda: f"session-{uuid.uuid4().hex}",
        sync_job_id_factory: Callable[[], str] = lambda: f"sync-{uuid.uuid4().hex}",
        managed_open_request_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self.path = path.resolve()
        self._recording_end_deadlines = {}
        self._now_utc = now_utc
        self._session_id_factory = session_id_factory
        self._sync_job_id_factory = sync_job_id_factory
        self._managed_open_request_id_factory = managed_open_request_id_factory
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._initialize()
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def _connect(self, *, deadline=None) -> sqlite3.Connection:
        timeout = 5.0 if deadline is None else min(5.0, deadline - time.monotonic())
        if timeout <= 0:
            raise ValueError("RECORDING_DRAIN_TIMEOUT")
        connection = sqlite3.connect(self.path, timeout=timeout)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {*range(_SCHEMA_VERSION + 1)}:
                raise ValueError(f"field record schema version {version} is unsupported")
            if version == _SCHEMA_VERSION:
                return
            if version == 1:
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE field_sync_jobs (
                      sync_job_id TEXT PRIMARY KEY,
                      session_id TEXT NOT NULL UNIQUE REFERENCES field_sessions(session_id),
                      manifest_hash TEXT NOT NULL,
                      manifest_json TEXT NOT NULL,
                      status TEXT NOT NULL CHECK(status IN
                        ('pending','claimed','partial','complete','blocked')),
                      retry_count INTEGER NOT NULL CHECK(retry_count >= 0),
                      next_attempt_at_utc TEXT NOT NULL,
                      last_error_code TEXT,
                      created_at_utc TEXT NOT NULL,
                      updated_at_utc TEXT NOT NULL
                    );
                    CREATE INDEX due_field_sync_jobs
                      ON field_sync_jobs(status, next_attempt_at_utc);
                    CREATE TABLE field_sync_media (
                      sync_job_id TEXT NOT NULL REFERENCES field_sync_jobs(sync_job_id),
                      clip_id TEXT NOT NULL,
                      shot_id TEXT NOT NULL,
                      local_path TEXT NOT NULL,
                      content_type TEXT NOT NULL,
                      content_sha256 TEXT NOT NULL,
                      byte_size INTEGER NOT NULL CHECK(byte_size > 0),
                      status TEXT NOT NULL CHECK(status IN ('pending','uploaded','complete')),
                      PRIMARY KEY(sync_job_id, clip_id)
                    );
                    PRAGMA user_version=2;
                    COMMIT;
                    """
                )
                version = 2
            if version == 2:
                connection.executescript(
                    f"""
                    BEGIN IMMEDIATE;
                    {_COMMAND_LEDGER_SCHEMA}
                    PRAGMA user_version=3;
                    COMMIT;
                    """
                )
                version = 3
            if version == 3:
                connection.executescript(
                    f"""
                    BEGIN IMMEDIATE;
                    {_ADMIN_AUDIT_SCHEMA}
                    PRAGMA user_version=4;
                    COMMIT;
                    """
                )
                version = 4
            if version == 4:
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_OWNERSHIP_SCHEMA} PRAGMA user_version=5; COMMIT;"
                )
                version = 5
            if version == 5:
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_TURN_CONFIRMATION_SCHEMA} PRAGMA user_version=6; COMMIT;"
                )
                version = 6
            if version == 6:
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_RECOVERY_SCHEMA} PRAGMA user_version=7; COMMIT;"
                )
                version = 7
            if version == 7:
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_LIVE_MEDIA_SCHEMA} PRAGMA user_version=8; COMMIT;"
                )
                version = 8
            if version == 8:
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_POSTGAME_MANIFEST_SCHEMA} PRAGMA user_version=9; COMMIT;"
                )
                version = 9
            if version == 9:
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_POSTGAME_FINALIZATION_SCHEMA} "
                    "PRAGMA user_version=10; COMMIT;"
                )
                version = 10
            if version == 10:
                connection.executescript(
                    # Module-defined schema only; no caller input enters migration SQL.
                    f"BEGIN IMMEDIATE; {_LOCAL_POSTGAME_OUTBOX_SCHEMA} "  # nosec B608
                    "INSERT INTO local_postgame_outbox("
                    "session_id,manifest_sha256,manifest_json,status) "
                    "SELECT session_id,manifest_sha256,manifest_json,'pending' "
                    "FROM field_postgame_finalizations; "
                    "PRAGMA user_version=11; COMMIT;"
                )
                version = 11
            if version == 11:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(_UNIFIED_COMMAND_LEDGER_SCHEMA)
                connection.execute(_RULE_DECISION_SCHEMA)
                legacy_commands = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='field_applied_commands'"
                ).fetchone()
                if legacy_commands is not None:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO field_command_ledger(
                          entry_id,session_id,ledger_order,origin,principal_id,
                          origin_sequence,command_type,payload_digest,outcome,result_code,
                          resulting_revision,decision_ref_json,completed_at_utc
                        )
                        SELECT command_id,session_id,server_sequence,'phone',actor_user_id,
                          server_sequence,command_type,payload_digest,outcome,result_code,
                          resulting_revision,NULL,completed_at_utc
                        FROM field_applied_commands
                        ORDER BY session_id,server_sequence
                        """
                    )
                connection.execute("PRAGMA user_version=12")
                connection.commit()
                version = 12
            if version == 12:
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_LIVE_SHOT_FACT_SCHEMA} PRAGMA user_version=13; COMMIT;"
                )
                version = 13
            if version == 13:
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_SAVED_CLIP_SCHEMA} PRAGMA user_version=14; COMMIT;"
                )
                version = 14
            if version == 14:
                connection.executescript(
                    # Module-defined schema only; no caller input enters migration SQL.
                    f"BEGIN IMMEDIATE; {_POSTGAME_REVIEW_SCHEMA} "  # nosec B608
                    "INSERT INTO field_postgame_reviews(session_id,artifact_id,status) "
                    "SELECT s.session_id,s.artifact_id,'pending' FROM field_sessions s "
                    "JOIN field_postgame_manifests m USING(session_id) WHERE s.status='ended' "
                    "ORDER BY m.ended_at_utc,s.session_id; "
                    "UPDATE field_postgame_reviews SET status='active' WHERE queue_id IN "
                    "(SELECT MIN(queue_id) FROM field_postgame_reviews GROUP BY artifact_id); "
                    "PRAGMA user_version=15; COMMIT;"
                )
                version = 15
            if version == 15:
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_SAVED_MEDIA_BLOCKED_MIGRATION} "
                    "PRAGMA user_version=16; COMMIT;"
                )
                version = 16
            if version == 16:
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_SAVED_MEDIA_SELECTION_MIGRATION} "
                    "PRAGMA user_version=17; COMMIT;"
                )
                version = 17
            if version == 17:
                # Deliberately no backfill from historical completed reviews.
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_LOCAL_DISCARD_SCHEMA} PRAGMA user_version=18; COMMIT;"
                )
                version = 18
            if version == 18:
                # Metadata only: no adoption or backfill of historical sessions.
                connection.executescript(
                    f"BEGIN IMMEDIATE; {_MANAGED_OPEN_SCHEMA} PRAGMA user_version=19; COMMIT;"
                )
                version = 19
            if version == 19:
                self._migrate_managed_requests20(connection)
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 20:
                self._migrate_managed_resources21(connection)
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 21:
                self._migrate_owner_deletions22(connection)
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 22:
                self._migrate_saved_upload_requests23(connection)
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 23:
                self._migrate_upload_receipts24(connection)
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 24:
                self._migrate_owner_completion25(connection)
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 25:
                self._migrate_rule_fact_preimages26(connection)
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 26:
                self._migrate_recording27(connection)
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 27:
                self._migrate_recording_end28(connection)
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 28:
                self._migrate_recording_cleanup29(connection)
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 29:
                self._migrate_recording_capture30(connection)
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == _SCHEMA_VERSION:
                return
            existing = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if existing:
                raise ValueError("field record schema version is missing")
            connection.executescript(
                f"""
                CREATE TABLE field_sessions (
                  session_id TEXT PRIMARY KEY,
                  artifact_id TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('open','ended')),
                  created_at_utc TEXT NOT NULL,
                  ended_at_utc TEXT
                );
                CREATE UNIQUE INDEX one_open_session_per_artifact
                  ON field_sessions(artifact_id) WHERE status='open';
                CREATE TABLE field_match_state (
                  session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
                  payload_json TEXT NOT NULL,
                  revision INTEGER NOT NULL CHECK(revision >= 0)
                );
                CREATE TABLE field_shots (
                  session_id TEXT NOT NULL REFERENCES field_sessions(session_id),
                  shot_id TEXT NOT NULL,
                  sequence INTEGER NOT NULL CHECK(sequence > 0),
                  payload_json TEXT NOT NULL,
                  PRIMARY KEY(session_id, shot_id),
                  UNIQUE(session_id, sequence)
                );
                CREATE TABLE field_sync_jobs (
                  sync_job_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL UNIQUE REFERENCES field_sessions(session_id),
                  manifest_hash TEXT NOT NULL,
                  manifest_json TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN
                    ('pending','claimed','partial','complete','blocked')),
                  retry_count INTEGER NOT NULL CHECK(retry_count >= 0),
                  next_attempt_at_utc TEXT NOT NULL,
                  last_error_code TEXT,
                  created_at_utc TEXT NOT NULL,
                  updated_at_utc TEXT NOT NULL
                );
                CREATE INDEX due_field_sync_jobs
                  ON field_sync_jobs(status, next_attempt_at_utc);
                CREATE TABLE field_sync_media (
                  sync_job_id TEXT NOT NULL REFERENCES field_sync_jobs(sync_job_id),
                  clip_id TEXT NOT NULL,
                  shot_id TEXT NOT NULL,
                  local_path TEXT NOT NULL,
                  content_type TEXT NOT NULL,
                  content_sha256 TEXT NOT NULL,
                  byte_size INTEGER NOT NULL CHECK(byte_size > 0),
                  status TEXT NOT NULL CHECK(status IN ('pending','uploaded','complete')),
                  PRIMARY KEY(sync_job_id, clip_id)
                );
                {_COMMAND_LEDGER_SCHEMA}
                {_ADMIN_AUDIT_SCHEMA}
                {_OWNERSHIP_SCHEMA}
                {_TURN_CONFIRMATION_SCHEMA}
                {_RECOVERY_SCHEMA}
                {_LIVE_MEDIA_SCHEMA}
                {_POSTGAME_MANIFEST_SCHEMA}
                {_POSTGAME_FINALIZATION_SCHEMA}
                {_LOCAL_POSTGAME_OUTBOX_SCHEMA}
                {_UNIFIED_COMMAND_LEDGER_SCHEMA}
                {_RULE_DECISION_SCHEMA}
                {_RULE_FACT_PREIMAGE_SCHEMA26}
                {_RECORDING_SCHEMA27}
                {_RECORDING_END_SCHEMA28}
                {_LIVE_SHOT_FACT_SCHEMA}
                {_SAVED_CLIP_SCHEMA}
                {_LOCAL_DISCARD_SCHEMA}
                {_POSTGAME_REVIEW_SCHEMA}
                {_RECORDING_CLEANUP_SCHEMA29}
                {RECORDING_CAPTURE_SCHEMA30}
                {_MANAGED_OPEN_SCHEMA20}
                {_MANAGED_OPEN_ACTIVE_INDEXES}
                {_MANAGED_RESOURCE_SCHEMA21}
                {_OWNER_DELETE_SCHEMA22}
                {_SAVED_UPLOAD_REQUEST_SCHEMA23}
                {_UPLOAD_RECEIPT_SCHEMA24}
                {_OWNER_COMPLETION_SCHEMA25}
                PRAGMA user_version=30;
                """
            )

    def _migrate_managed_requests20(self, connection: sqlite3.Connection) -> None:
        # No executescript here: it implicitly commits an existing transaction.
        # Validation, exact copying, replacement, indexes and version share one commit.
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if 20 <= version <= _SCHEMA_VERSION:
            # Another initializer may have finished both migrations while we waited.
            connection.commit()
            return
        if version != 19:
            raise ValueError("managed migration requires schema19")
        rows = connection.execute("SELECT * FROM field_managed_open_requests").fetchall()
        for row in rows:
            self._managed_request_from_row(row, legacy19=True)
        connection.execute(
            _MANAGED_OPEN_SCHEMA20.replace(
                "CREATE TABLE field_managed_open_requests (",
                "CREATE TABLE field_managed_open_requests20 (",
            )
        )
        connection.executemany(
            "INSERT INTO field_managed_open_requests20 VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,'active',NULL,NULL,1)",
            (tuple(row) for row in rows),
        )
        connection.execute("DROP TABLE field_managed_open_requests")
        connection.execute(
            "ALTER TABLE field_managed_open_requests20 RENAME TO field_managed_open_requests"
        )
        for statement in _MANAGED_OPEN_ACTIVE_INDEXES.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("PRAGMA user_version=20")
        connection.commit()

    def _migrate_managed_resources21(self, connection: sqlite3.Connection) -> None:
        # execute (not executescript) preserves atomic DDL and rollback on any fault.
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if 21 <= version <= _SCHEMA_VERSION:
            connection.commit()
            return
        if version != 20:
            raise ValueError("managed resource migration requires schema20")
        for statement in _MANAGED_RESOURCE_SCHEMA21.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("PRAGMA user_version=21")
        connection.commit()

    @staticmethod
    def _migrate_owner_deletions22(connection: sqlite3.Connection) -> None:
        # Metadata only: retain original save/fact/ownership bytes and adopt no rows.
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if 22 <= version <= _SCHEMA_VERSION:
            connection.commit()
            return
        if version != 21:
            raise ValueError("owner deletion migration requires schema21")
        for statement in _OWNER_DELETE_SCHEMA22.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("PRAGMA user_version=22")
        connection.commit()

    @staticmethod
    def _migrate_saved_upload_requests23(connection: sqlite3.Connection) -> None:
        # Permanent metadata only: no inferred historical authorization or transport.
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if 23 <= version <= _SCHEMA_VERSION:
            connection.commit()
            return
        if version != 22:
            raise ValueError("saved upload request migration requires schema22")
        for statement in _SAVED_UPLOAD_REQUEST_SCHEMA23.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("PRAGMA user_version=23")
        connection.commit()

    @staticmethod
    def _migrate_upload_receipts24(connection: sqlite3.Connection) -> None:
        # DDL only: transport facts require retained original metadata validation.
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if 24 <= version <= _SCHEMA_VERSION:
            connection.commit()
            return
        if version != 23:
            raise ValueError("upload receipt migration requires schema23")
        for statement in _UPLOAD_RECEIPT_SCHEMA24.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("PRAGMA user_version=24")
        connection.commit()

    @staticmethod
    def _migrate_owner_completion25(connection: sqlite3.Connection) -> None:
        # DDL only. Legacy completions require separate bounded reconciliation.
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if 25 <= version <= _SCHEMA_VERSION:
            connection.commit()
            return
        if version != 24:
            raise ValueError("owner completion migration requires schema24")
        for statement in _OWNER_COMPLETION_SCHEMA25.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("PRAGMA user_version=25")
        connection.commit()

    @staticmethod
    def _migrate_rule_fact_preimages26(connection: sqlite3.Connection) -> None:
        # Retain no reconstructed historical evidence. DDL and version share
        # one transaction, including a version recheck for concurrent startup.
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if 26 <= version <= _SCHEMA_VERSION:
            connection.commit()
            return
        if version != 25:
            raise ValueError("rule fact preimage migration requires schema25")
        connection.execute(_RULE_FACT_PREIMAGE_SCHEMA26)
        connection.execute("PRAGMA user_version=26")
        connection.commit()

    @staticmethod
    def _migrate_recording27(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version in {27, 28, 29, 30}:
            connection.commit()
            return
        if version != 26:
            raise ValueError("recording migration requires schema26")
        for statement in _RECORDING_SCHEMA27.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("PRAGMA user_version=27")
        connection.commit()

    @staticmethod
    def _migrate_recording_end28(connection):
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in {28, 29, 30}:
            if version != 27:
                raise ValueError("recording end migration requires schema27")
            connection.execute(_RECORDING_END_SCHEMA28)
            connection.execute("PRAGMA user_version=28")
        connection.commit()

    @staticmethod
    def _migrate_recording_cleanup29(connection):
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in {29, 30}:
            if version != 28:
                raise ValueError("recording cleanup migration requires schema28")
            for statement in _RECORDING_CLEANUP_SCHEMA29.split(";"):
                if statement.strip():
                    connection.execute(statement)
            connection.execute("PRAGMA user_version=29")
        connection.commit()

    @staticmethod
    def _migrate_recording_capture30(connection):
        connection.execute("BEGIN IMMEDIATE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != 30:
            if version != 29:
                raise ValueError("recording capture migration requires schema29")
            for statement in RECORDING_CAPTURE_SCHEMA30.split(";"):
                if statement.strip():
                    connection.execute(statement)
            connection.execute("PRAGMA user_version=30")
        connection.commit()

    @staticmethod
    def _session_policy_in(connection, session_id):
        row = connection.execute(
            "SELECT policy FROM field_session_recording_policy WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return "legacy" if row is None else require_session_policy(row[0])

    def get_session_policy(self, session_id: str) -> str:
        with self._connect() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM field_sessions WHERE session_id=?",
                    (session_id,),
                ).fetchone()
                is None
            ):
                raise ValueError("field session does not exist")
            return self._session_policy_in(connection, session_id)

    @staticmethod
    def _managed_scope_values(scope: ManagedSessionScope) -> tuple[str, str, str]:
        if not isinstance(scope, ManagedSessionScope):
            raise ValueError("invalid managed session scope")
        # Revalidate at the persistence boundary without normalizing identities.
        ManagedSessionScope(scope.environment_id, scope.device_id, scope.table_id)
        return scope.environment_id, scope.device_id, scope.table_id

    @staticmethod
    def _managed_request_from_row(
        row: sqlite3.Row, *, legacy19: bool = False
    ) -> ManagedOpenRequest:
        scope = ManagedSessionScope(row["environment_id"], row["device_id"], row["table_id"])
        binding_values = tuple(
            row[key] for key in ("session_id", "birth_id", "birth_contract", "remote_status")
        )
        binding = None
        if any(value is not None for value in binding_values):
            binding = ManagedSessionBinding(
                "1.0", *binding_values[:3], scope.table_id, scope.device_id, binding_values[3]
            )
        retirement = {}
        if not legacy19:
            retirement_columns = {
                "lifecycle",
                "retired_at_utc",
                "sealed_close_json",
                "reservation_order",
            }
            if not retirement_columns.issubset(row.keys()):
                raise ValueError("managed retirement metadata is missing")
            order = row["reservation_order"]
            if type(order) is not int or not 1 <= order <= 9223372036854775807:
                raise ValueError("invalid managed reservation order")
            close = None
            if row["sealed_close_json"] is not None:
                payload = json.loads(row["sealed_close_json"])
                close = ManagedSessionCloseState.from_payload(payload, binding=binding)
                if _canonical(asdict(close)) != row["sealed_close_json"]:
                    raise ValueError("managed sealed close is not canonical")
            retirement = dict(
                lifecycle=row["lifecycle"], retired_at_utc=row["retired_at_utc"], sealed_close=close
            )
        request = ManagedOpenRequest(
            scope,
            row["artifact_id"],
            row["rule_profile_id"],
            row["open_request_id"],
            row["phase"],
            binding,
            row["created_at_utc"],
            row["updated_at_utc"],
            **retirement,
        )
        return request

    def _managed_request_in(
        self, connection: sqlite3.Connection, scope: ManagedSessionScope, open_request_id: str
    ) -> ManagedOpenRequest:
        values = self._managed_scope_values(scope)
        validate_managed_uuid(open_request_id, name="open request ID")
        row = connection.execute(
            "SELECT * FROM field_managed_open_requests "
            "WHERE environment_id=? AND device_id=? AND table_id=? AND open_request_id=?",
            (*values, open_request_id),
        ).fetchone()
        if row is None:
            raise ValueError("managed open request does not exist")
        return self._managed_request_from_row(row)

    def get_managed_open_request(
        self, scope: ManagedSessionScope, open_request_id: str
    ) -> ManagedOpenRequest:
        try:
            with self._connect() as connection:
                return self._managed_request_in(connection, scope, open_request_id)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def get_materialized_managed_open_request(
        self, scope: ManagedSessionScope, session_id: str
    ) -> ManagedOpenRequest:
        request = self.get_managed_open_request_for_session(scope, session_id)
        if request.phase != "materialized":
            raise ValueError("managed request is not materialized")
        return request

    def get_active_managed_open_request(
        self, scope: ManagedSessionScope
    ) -> ManagedOpenRequest | None:
        return self._get_managed_lifecycle_request(scope, "active")

    def get_latest_retired_managed_open_request(
        self, scope: ManagedSessionScope
    ) -> ManagedOpenRequest | None:
        return self._get_managed_lifecycle_request(scope, "retired")

    def _get_managed_lifecycle_request(
        self, scope: ManagedSessionScope, lifecycle: str
    ) -> ManagedOpenRequest | None:
        values = self._managed_scope_values(scope)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM field_managed_open_requests WHERE "
                    "environment_id=? AND device_id=? AND table_id=? "
                    "ORDER BY reservation_order DESC",
                    values,
                ).fetchall()
                requests = [self._managed_request_from_row(row) for row in rows]
                matches = [request for request in requests if request.lifecycle == lifecycle]
                if lifecycle == "active" and len(matches) > 1:
                    raise ValueError("ambiguous active managed request")
                return matches[0] if matches else None
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def get_managed_open_request_for_session(
        self, scope: ManagedSessionScope, session_id: str
    ) -> ManagedOpenRequest:
        values = self._managed_scope_values(scope)
        if not isinstance(session_id, str) or not session_id.startswith("managed-"):
            raise ValueError("invalid managed session ID")
        validate_managed_uuid(session_id.removeprefix("managed-"), name="birth ID")
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM field_managed_open_requests WHERE "
                    "environment_id=? AND device_id=? AND table_id=? AND session_id=?",
                    (*values, session_id),
                ).fetchall()
                if len(rows) != 1:
                    raise ValueError("managed request is missing or ambiguous")
                return self._managed_request_from_row(rows[0])
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def find_open_session(self, artifact_id: str) -> FieldSessionRecord | None:
        validate_managed_artifact_id(artifact_id)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM field_sessions WHERE artifact_id=? AND status='open'",
                    (artifact_id,),
                ).fetchall()
                if len(rows) > 1:
                    raise ValueError("ambiguous open field session")
                if not rows:
                    return None
                return self._session_with_checked_revision(connection, rows[0])
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def _session_with_checked_revision(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> FieldSessionRecord:
        state = connection.execute(
            "SELECT payload_json,revision FROM field_match_state WHERE session_id=?",
            (row["session_id"],),
        ).fetchone()
        if state is None:
            raise ValueError("persisted session match state is missing")
        payload = json.loads(state["payload_json"])
        if not isinstance(payload, dict) or type(payload.get("revision")) is not int:
            raise ValueError("persisted match revision is invalid")
        record = self._session_from_row(connection, row)
        if type(state["revision"]) is not int or state["revision"] != record.match.revision:
            raise ValueError("persisted match revision is invalid")
        return record

    def _refuse_managed_predecessor_in(
        self,
        connection: sqlite3.Connection,
        artifact_id: str,
        scope: ManagedSessionScope,
        *,
        current_request_id: str | None = None,
    ) -> None:
        managed_rows = connection.execute(
            "SELECT * FROM field_managed_open_requests WHERE "
            "(environment_id=? AND device_id=? AND table_id=?) OR artifact_id=?",
            (*self._managed_scope_values(scope), artifact_id),
        ).fetchall()
        retired_sessions = set()
        for row in managed_rows:
            request = self._managed_request_from_row(row)
            if request.lifecycle == "retired":
                self._retired_local_session_in(connection, request)
                retired_sessions.add(request.binding.session_id)
            elif not (request.scope == scope and request.open_request_id == current_request_id):
                raise ValueError("unretired managed predecessor blocks reservation")
        rows = connection.execute(
            "SELECT s.*,c.status AS close_status,c.updated_at_utc AS close_updated "
            "FROM field_sessions s LEFT JOIN field_cloud_close_intents c USING(session_id) "
            "WHERE s.artifact_id=?",
            (artifact_id,),
        ).fetchall()
        for row in rows:
            if row["session_id"] in retired_sessions:
                continue
            self._session_with_checked_revision(connection, row)
            validate_managed_utc_timestamp(row["created_at_utc"], name="session timestamp")
            if row["status"] not in ("open", "ended"):
                raise ValueError("invalid persisted predecessor status")
            if (row["status"] == "open") != (row["ended_at_utc"] is None):
                raise ValueError("invalid persisted predecessor end state")
            if row["ended_at_utc"] is not None:
                validate_managed_utc_timestamp(row["ended_at_utc"], name="ended timestamp")
            if row["close_status"] is not None:
                if row["close_status"] not in ("open", "pending", "complete"):
                    raise ValueError("invalid predecessor close marker")
                validate_managed_utc_timestamp(row["close_updated"], name="close timestamp")
            if row["status"] == "open" or row["close_status"] in ("open", "pending"):
                raise ValueError("existing session or pending predecessor close blocks reservation")

    def reserve_managed_open(
        self,
        scope: ManagedSessionScope,
        *,
        artifact_id: str,
        rule_profile_id: str | None = None,
        resource_spec: ManagedRuntimeResourceSpec | None = None,
    ) -> ManagedOpenRequest:
        values = self._managed_scope_values(scope)
        validate_managed_artifact_id(artifact_id)
        validate_managed_rule_profile_id(rule_profile_id)
        if resource_spec is not None:
            if not isinstance(resource_spec, ManagedRuntimeResourceSpec):
                raise ValueError("managed resource specification must be typed")
            resource_spec = ManagedRuntimeResourceSpec.from_payload(resource_spec.as_payload())
            if (
                resource_spec.artifact_id != artifact_id
                or resource_spec.rule_profile_id != rule_profile_id
            ):
                raise ValueError("managed resource specification conflicts with request")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    "SELECT * FROM field_managed_open_requests WHERE "
                    "(environment_id=? AND device_id=? AND table_id=?) OR artifact_id=?",
                    (*values, artifact_id),
                ).fetchall()
                # Validate history too: damaged retirement metadata cannot create authority.
                requests = tuple(
                    request
                    for row in rows
                    if (request := self._managed_request_from_row(row)).lifecycle == "active"
                )
                if requests:
                    if len(requests) != 1 or (
                        requests[0].scope != scope
                        or requests[0].artifact_id != artifact_id
                        or requests[0].rule_profile_id != rule_profile_id
                    ):
                        raise ValueError("managed reservation conflicts with persisted state")
                    if resource_spec is not None:
                        original = self._managed_resource_spec_in(connection, requests[0])
                        if original.spec != resource_spec:
                            raise ValueError("managed resource specification conflicts with intent")
                    return requests[0]
                self._refuse_managed_predecessor_in(connection, artifact_id, scope)
                maximum = connection.execute(
                    "SELECT MAX(reservation_order) FROM field_managed_open_requests WHERE "
                    "environment_id=? AND device_id=? AND table_id=?",
                    values,
                ).fetchone()[0]
                if maximum == 9223372036854775807:
                    raise ValueError("managed reservation order exhausted")
                order = 1 if maximum is None else maximum + 1
                request_id = validate_managed_uuid(
                    self._managed_open_request_id_factory(), name="open request ID"
                )
                created = _timestamp(self._now_utc())
                connection.execute(
                    "INSERT INTO field_managed_open_requests VALUES"
                    "(?,?,?,?,?,?,'pending',NULL,NULL,NULL,NULL,?,?,'active',NULL,NULL,?)",
                    (*values, artifact_id, rule_profile_id, request_id, created, created, order),
                )
                if resource_spec is not None:
                    connection.execute(
                        "INSERT INTO field_managed_resource_specs VALUES(?,?,?,?,?,?,?)",
                        (
                            *values,
                            request_id,
                            resource_spec.canonical_json,
                            resource_spec.spec_sha256,
                            created,
                        ),
                    )
                return self._managed_request_in(connection, scope, request_id)
        except sqlite3.DatabaseError as exc:
            raise ValueError("managed reservation conflicts with persisted state") from exc

    def _managed_resource_spec_in(
        self, connection: sqlite3.Connection, request: ManagedOpenRequest
    ) -> ManagedResourceSpecRecord:
        row = connection.execute(
            "SELECT * FROM field_managed_resource_specs WHERE "
            "environment_id=? AND device_id=? AND open_request_id=?",
            (request.scope.environment_id, request.scope.device_id, request.open_request_id),
        ).fetchone()
        if row is None:
            raise ValueError("managed resource specification is missing")
        raw = row["spec_json"]
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_RESOURCE_METADATA_BYTES:
            raise ValueError("invalid managed resource specification metadata")
        spec = ManagedRuntimeResourceSpec.from_payload(json.loads(raw))
        record = ManagedResourceSpecRecord(
            ManagedSessionScope(row["environment_id"], row["device_id"], row["table_id"]),
            row["open_request_id"],
            spec,
            row["spec_sha256"],
            row["created_at_utc"],
        )
        if (
            record.scope != request.scope
            or spec.canonical_json != raw
            or spec.artifact_id != request.artifact_id
            or spec.rule_profile_id != request.rule_profile_id
            or record.created_at_utc != request.created_at_utc
        ):
            raise ValueError("managed resource specification conflicts with retained request")
        return record

    def get_managed_resource_spec(
        self, scope: ManagedSessionScope, open_request_id: str
    ) -> ManagedResourceSpecRecord:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN")
                request = self._managed_request_in(connection, scope, open_request_id)
                return self._managed_resource_spec_in(connection, request)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def _managed_resource_session_in(
        self, connection: sqlite3.Connection, scope: ManagedSessionScope, session_id: str
    ) -> ManagedOpenRequest:
        values = self._managed_scope_values(scope)
        if not isinstance(session_id, str) or not session_id.startswith("managed-"):
            raise ValueError("invalid managed session ID")
        validate_managed_uuid(session_id.removeprefix("managed-"), name="birth ID")
        rows = connection.execute(
            "SELECT * FROM field_managed_open_requests WHERE environment_id=? AND device_id=? "
            "AND table_id=? AND session_id=?",
            (*values, session_id),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("managed request is missing or ambiguous")
        request = self._managed_request_from_row(rows[0])
        if request.phase != "materialized" or request.binding is None:
            raise ValueError("managed resource lookup requires materialized binding")
        local = connection.execute(
            "SELECT status FROM field_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if local is not None and local["status"] == "open" and request.lifecycle == "active":
            self._materialized_session_in(connection, request)
        else:
            self._retired_local_session_in(connection, request)
        return request

    def get_managed_resource_spec_for_session(
        self, scope: ManagedSessionScope, session_id: str
    ) -> ManagedResourceSpecRecord:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN")
                request = self._managed_resource_session_in(connection, scope, session_id)
                return self._managed_resource_spec_in(connection, request)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def register_managed_capture_epoch(
        self, scope: ManagedSessionScope, open_request_id: str, source_epoch_id: str
    ) -> ManagedCaptureEpochRecord:
        values = self._managed_scope_values(scope)
        validate_managed_scope_id(source_epoch_id, name="source epoch ID")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                request = self._managed_request_in(connection, scope, open_request_id)
                if (
                    request.lifecycle != "active"
                    or request.phase != "materialized"
                    or request.binding is None
                    or request.binding.status != "open"
                ):
                    raise ValueError("capture epoch requires active materialized open request")
                self._materialized_session_in(connection, request)
                spec = self._managed_resource_spec_in(connection, request)
                if (
                    connection.execute(
                        "SELECT 1 FROM field_managed_capture_epochs WHERE "
                        "environment_id=? AND device_id=? AND source_epoch_id=?",
                        (*values[:2], source_epoch_id),
                    ).fetchone()
                    is not None
                ):
                    raise ValueError("managed capture epoch is permanently consumed")
                record = ManagedCaptureEpochRecord(
                    scope,
                    open_request_id,
                    request.binding.session_id,
                    request.binding.birth_id,
                    source_epoch_id,
                    spec.spec_sha256,
                    _timestamp(self._now_utc()),
                )
                connection.execute(
                    "INSERT INTO field_managed_capture_epochs VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        *values,
                        open_request_id,
                        record.session_id,
                        record.birth_id,
                        source_epoch_id,
                        record.spec_sha256,
                        record.created_at_utc,
                    ),
                )
                return record
        except sqlite3.DatabaseError as exc:
            raise ValueError("managed capture epoch conflicts with retained state") from exc

    def get_managed_capture_epoch(
        self, scope: ManagedSessionScope, session_id: str, source_epoch_id: str
    ) -> ManagedCaptureEpochRecord:
        values = self._managed_scope_values(scope)
        validate_managed_scope_id(source_epoch_id, name="source epoch ID")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN")
                request = self._managed_resource_session_in(connection, scope, session_id)
                spec = self._managed_resource_spec_in(connection, request)
                row = connection.execute(
                    "SELECT * FROM field_managed_capture_epochs WHERE environment_id=? "
                    "AND device_id=? AND source_epoch_id=?",
                    (*values[:2], source_epoch_id),
                ).fetchone()
                if row is None:
                    raise ValueError("managed capture epoch is missing")
                record = ManagedCaptureEpochRecord(
                    ManagedSessionScope(row["environment_id"], row["device_id"], row["table_id"]),
                    row["open_request_id"],
                    row["session_id"],
                    row["birth_id"],
                    row["source_epoch_id"],
                    row["spec_sha256"],
                    row["created_at_utc"],
                )
                if (
                    record.scope != scope
                    or record.session_id != session_id
                    or record.open_request_id != request.open_request_id
                    or record.birth_id != request.binding.birth_id
                    or record.spec_sha256 != spec.spec_sha256
                ):
                    raise ValueError("managed capture epoch conflicts with retained binding")
                return record
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def record_managed_binding(
        self, scope: ManagedSessionScope, open_request_id: str, binding: ManagedSessionBinding
    ) -> ManagedOpenRequest:
        self._managed_scope_values(scope)
        if not isinstance(binding, ManagedSessionBinding):
            raise ValueError("invalid managed session binding")
        binding = ManagedSessionBinding.from_payload(
            asdict(binding), expected_table_id=scope.table_id, expected_device_id=scope.device_id
        )
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                request = self._managed_request_in(connection, scope, open_request_id)
                if request.binding is not None:
                    if request.binding == binding:
                        return request
                    if request.lifecycle == "retired":
                        raise ValueError("retired managed binding is immutable")
                    if (
                        request.binding.status != "open"
                        or binding.status != "ended"
                        or asdict(request.binding) | {"status": "ended"} != asdict(binding)
                    ):
                        raise ValueError("managed binding identity or status conflicts")
                updated = _timestamp(self._now_utc())
                connection.execute(
                    "UPDATE field_managed_open_requests SET phase=?,session_id=?,birth_id=?,"
                    "birth_contract=?,remote_status=?,updated_at_utc=? WHERE "
                    "environment_id=? AND device_id=? AND table_id=? AND open_request_id=?",
                    (
                        "bound" if request.phase == "pending" else request.phase,
                        binding.session_id,
                        binding.birth_id,
                        binding.birth_contract,
                        binding.status,
                        updated,
                        *self._managed_scope_values(scope),
                        open_request_id,
                    ),
                )
                return self._managed_request_in(connection, scope, open_request_id)
        except sqlite3.DatabaseError as exc:
            raise ValueError("managed binding conflicts with persisted state") from exc

    def _materialized_session_in(
        self, connection: sqlite3.Connection, request: ManagedOpenRequest
    ) -> FieldSessionRecord:
        assert request.binding is not None
        session_id = request.binding.session_id
        row = connection.execute(
            "SELECT * FROM field_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None or (
            row["artifact_id"] != request.artifact_id
            or row["status"] != "open"
            or row["ended_at_utc"] is not None
        ):
            raise ValueError("managed local session is missing, ended or mismatched")
        validate_managed_utc_timestamp(row["created_at_utc"], name="session timestamp")
        state = connection.execute(
            "SELECT * FROM field_match_state WHERE session_id=?", (session_id,)
        ).fetchone()
        marker = connection.execute(
            "SELECT * FROM field_cloud_close_intents WHERE session_id=?", (session_id,)
        ).fetchone()
        if state is None or marker is None or marker["status"] != "open":
            raise ValueError("managed match or open cloud marker is missing or invalid")
        validate_managed_utc_timestamp(marker["updated_at_utc"], name="close timestamp")
        record = self._session_from_row(connection, row)
        if (
            type(state["revision"]) is not int
            or state["revision"] < 0
            or record.match.revision != state["revision"]
            or record.match.rule_profile_id != request.rule_profile_id
            or _canonical(json.loads(state["payload_json"]))
            != _canonical(_match_payload(record.match))
        ):
            raise ValueError("managed match state is invalid or mismatched")
        return record

    def _retired_local_session_in(
        self, connection: sqlite3.Connection, request: ManagedOpenRequest
    ) -> FieldSessionRecord:
        if request.phase != "materialized" or request.binding is None:
            raise ValueError("managed retirement requires materialized binding")
        row = connection.execute(
            "SELECT * FROM field_sessions WHERE session_id=?", (request.binding.session_id,)
        ).fetchone()
        if row is None or row["status"] != "ended" or row["artifact_id"] != request.artifact_id:
            raise ValueError("managed retirement requires matching local ended session")
        validate_managed_utc_timestamp(row["created_at_utc"], name="session timestamp")
        validate_managed_utc_timestamp(row["ended_at_utc"], name="ended timestamp")
        if datetime.fromisoformat(row["ended_at_utc"]) < datetime.fromisoformat(
            row["created_at_utc"]
        ):
            raise ValueError("invalid managed local end timestamp order")
        marker = connection.execute(
            "SELECT * FROM field_cloud_close_intents WHERE session_id=?", (row["session_id"],)
        ).fetchone()
        if marker is None or marker["status"] not in ("pending", "complete"):
            raise ValueError("managed retirement requires pending or complete close marker")
        validate_managed_utc_timestamp(marker["updated_at_utc"], name="close timestamp")
        record = self._session_with_checked_revision(connection, row)
        state = connection.execute(
            "SELECT payload_json FROM field_match_state WHERE session_id=?", (row["session_id"],)
        ).fetchone()
        if record.match.rule_profile_id != request.rule_profile_id or _canonical(
            json.loads(state["payload_json"])
        ) != _canonical(_match_payload(record.match)):
            raise ValueError("managed retirement local match is invalid or mismatched")
        return record

    def validate_managed_local_session(
        self, scope: ManagedSessionScope, open_request_id: str
    ) -> FieldSessionRecord:
        """Read retained authority strictly, without changing lifecycle or history."""
        self._managed_scope_values(scope)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN")
                request = self._managed_request_in(connection, scope, open_request_id)
                if request.phase != "materialized" or request.binding is None:
                    raise ValueError("managed local validation requires materialized binding")
                row = connection.execute(
                    "SELECT status FROM field_sessions WHERE session_id=?",
                    (request.binding.session_id,),
                ).fetchone()
                if row is not None and row["status"] == "open":
                    if request.lifecycle == "retired":
                        raise ValueError("retired managed local session cannot be open")
                    return self._materialized_session_in(connection, request)
                return self._retired_local_session_in(connection, request)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def retire_managed_open(
        self,
        scope: ManagedSessionScope,
        open_request_id: str,
        *,
        sealed_close: ManagedSessionCloseState,
    ) -> ManagedOpenRequest:
        self._managed_scope_values(scope)
        if not isinstance(sealed_close, ManagedSessionCloseState):
            raise ValueError("managed retirement requires typed sealed close")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                request = self._managed_request_in(connection, scope, open_request_id)
                close = ManagedSessionCloseState.from_payload(
                    asdict(sealed_close),
                    binding=request.binding,
                )
                if close.coverage != "sealed":
                    raise ValueError("managed retirement requires sealed close")
                self._retired_local_session_in(connection, request)
                if request.lifecycle == "retired":
                    if request.sealed_close != close:
                        raise ValueError("managed retirement seal conflicts")
                    return request
                retired = _timestamp(self._now_utc())
                if datetime.fromisoformat(retired) < datetime.fromisoformat(request.updated_at_utc):
                    raise ValueError("managed retirement precedes last update")
                connection.execute(
                    "UPDATE field_managed_open_requests SET remote_status='ended',"
                    "lifecycle='retired',retired_at_utc=?,sealed_close_json=?,updated_at_utc=? "
                    "WHERE environment_id=? AND device_id=? AND table_id=? AND open_request_id=?",
                    (
                        retired,
                        _canonical(asdict(close)),
                        retired,
                        *self._managed_scope_values(scope),
                        open_request_id,
                    ),
                )
                return self._managed_request_in(connection, scope, open_request_id)
        except sqlite3.DatabaseError as exc:
            raise ValueError("managed retirement conflicts with persisted state") from exc

    def materialize_managed_session(
        self, scope: ManagedSessionScope, open_request_id: str, *, session_policy: str = "legacy"
    ) -> FieldSessionRecord:
        require_session_policy(session_policy)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                request = self._managed_request_in(connection, scope, open_request_id)
                if (
                    request.lifecycle != "active"
                    or request.binding is None
                    or request.binding.status != "open"
                ):
                    raise ValueError("managed materialization requires bound open status")
                if session_policy == "recording-v1" and request.rule_profile_id is not None:
                    raise ValueError("recording policy does not request a rule profile")
                if request.phase == "materialized":
                    if (
                        self._session_policy_in(connection, request.binding.session_id)
                        != session_policy
                    ):
                        raise ValueError("session recording policy changed on resume")
                    return self._materialized_session_in(connection, request)
                # A preexisting scene must never be adopted, even with the issued name.
                if (
                    connection.execute(
                        "SELECT 1 FROM field_sessions WHERE session_id=?",
                        (request.binding.session_id,),
                    ).fetchone()
                    is not None
                ):
                    raise ValueError("managed local session already exists without materialization")
                self._refuse_managed_predecessor_in(
                    connection,
                    request.artifact_id,
                    scope,
                    current_request_id=open_request_id,
                )
                created = _timestamp(self._now_utc())
                session_id = request.binding.session_id
                connection.execute(
                    "INSERT INTO field_sessions VALUES(?,?,'open',?,NULL)",
                    (session_id, request.artifact_id, created),
                )
                connection.execute(
                    "INSERT INTO field_session_recording_policy VALUES(?,?)",
                    (session_id, session_policy),
                )
                match = MatchControlSnapshot(rule_profile_id=request.rule_profile_id)
                connection.execute(
                    "INSERT INTO field_match_state VALUES(?,?,?)",
                    (session_id, _canonical(_match_payload(match)), match.revision),
                )
                connection.execute(
                    "INSERT INTO field_cloud_close_intents VALUES(?,'open',?)",
                    (session_id, created),
                )
                connection.execute(
                    "UPDATE field_managed_open_requests SET phase='materialized',updated_at_utc=? "
                    "WHERE environment_id=? AND device_id=? AND table_id=? AND open_request_id=?",
                    (created, *self._managed_scope_values(scope), open_request_id),
                )
                request = self._managed_request_in(connection, scope, open_request_id)
                return self._materialized_session_in(connection, request)
        except sqlite3.DatabaseError as exc:
            raise ValueError("managed materialization conflicts with persisted state") from exc

    def _session_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> FieldSessionRecord:
        match_row = connection.execute(
            "SELECT payload_json FROM field_match_state WHERE session_id=?",
            (row["session_id"],),
        ).fetchone()
        if match_row is None:
            raise ValueError("persisted session match state is missing")
        return FieldSessionRecord(
            session_id=row["session_id"],
            artifact_id=row["artifact_id"],
            status=FieldSessionStatus(row["status"]),
            created_at_utc=row["created_at_utc"],
            ended_at_utc=row["ended_at_utc"],
            match=_match_from_payload(match_row["payload_json"]),
        )

    def open_or_resume_session(
        self,
        *,
        artifact_id: str,
        rule_profile_id: str | None = None,
        fresh_session_on_start: bool = False,
        session_policy: str = "legacy",
    ) -> FieldSessionRecord:
        """Open a session, optionally ending the previous one in the same transaction.

        Fresh startup is for sources that restart at frame zero. Callers must own
        their listener before choosing it; managed sessions require verified close.
        """
        require_session_policy(session_policy)
        if session_policy == "recording-v1" and rule_profile_id is not None:
            raise ValueError("recording policy does not request a rule profile")
        if not artifact_id.strip():
            raise ValueError("artifact ID must not be empty")
        if rule_profile_id not in {None, PROFILE_ID}:
            raise ValueError("unsupported rule profile")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                managed = connection.execute(
                    "SELECT 1 FROM field_managed_open_requests "
                    "WHERE artifact_id=? AND lifecycle='active' LIMIT 1",
                    (artifact_id,),
                ).fetchone()
                if managed is not None:
                    raise ValueError("managed artifact requires explicit managed startup")
                row = connection.execute(
                    "SELECT * FROM field_sessions WHERE artifact_id=? AND status='open'",
                    (artifact_id,),
                ).fetchone()
                if row is not None:
                    if row["session_id"].startswith("managed-"):
                        raise ValueError("managed session requires explicit managed startup")
                    self._require_recording_recovery_complete_in(connection, row["session_id"])
                    if not fresh_session_on_start:
                        if self._session_policy_in(connection, row["session_id"]) != session_policy:
                            raise ValueError("session recording policy changed on resume")
                        return self._session_from_row(connection, row)
                    shots = tuple(
                        _shot_from_payload(item["payload_json"])
                        for item in connection.execute(
                            "SELECT payload_json FROM field_shots "
                            "WHERE session_id=? ORDER BY sequence",
                            (row["session_id"],),
                        ).fetchall()
                    )
                    unsettled = tuple(shot for shot in shots if shot.settled_ms is None)
                    if unsettled and (len(unsettled) != 1 or unsettled[0] != shots[-1]):
                        raise ValueError("persisted shot recovery is ambiguous")
                    self._end_session_in(connection, row)
                session_id = self._session_id_factory().strip()
                if not session_id:
                    raise ValueError("session ID factory returned an empty value")
                created = _timestamp(self._now_utc())
                connection.execute(
                    "INSERT INTO field_sessions VALUES(?,?,?,?,NULL)",
                    (session_id, artifact_id, FieldSessionStatus.OPEN.value, created),
                )
                connection.execute(
                    "INSERT INTO field_session_recording_policy VALUES(?,?)",
                    (session_id, session_policy),
                )
                match = MatchControlSnapshot(rule_profile_id=rule_profile_id)
                connection.execute(
                    "INSERT INTO field_match_state VALUES(?,?,?)",
                    (session_id, _canonical(_match_payload(match)), match.revision),
                )
                row = connection.execute(
                    "SELECT * FROM field_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                assert row is not None
                return self._session_from_row(connection, row)
        except sqlite3.IntegrityError as exc:
            raise ValueError("field session identity conflicts with persisted state") from exc
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def load_session(self, session_id: str) -> FieldSessionRecord:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM field_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                if row is None:
                    raise ValueError("field session does not exist")
                return self._session_from_row(connection, row)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def media_encoding_state(self, session_id: str, shot_id: str | None = None) -> dict:
        """One fresh short snapshot per check; no connection or read snapshot escapes."""
        with closing(self._connect()) as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT status FROM field_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise ValueError("field session does not exist")
            policy = self._session_policy_in(connection, session_id)
            banned = shot_id is not None and (
                connection.execute(
                    "SELECT 1 FROM field_owner_media_deletions WHERE session_id=? AND shot_id=?",
                    (session_id, shot_id),
                ).fetchone()
                is not None
                or self._is_postgame_discarded_in(connection, session_id, shot_id)
            )
            return {"status": row[0], "policy": policy, "banned": bool(banned)}

    @contextmanager
    def media_publication_guard(self, job):
        """Serialize final task completion with field-store revocation commits.

        Lock order is field store then jobs store, as in deletion admission. No
        match lock, media I/O, model work or long-lived snapshot belongs here.
        """
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM field_sessions WHERE session_id=?", (job.session_id,)
            ).fetchone()
            policy = self._session_policy_in(connection, job.session_id)
            kind = job.source.kind
            allowed = row is not None and (
                row[0] == "ended"
                or (row[0] == "open" and policy == "recording-v1" and kind == "recording_interval")
            )
            allowed = allowed and not self._is_postgame_discarded_in(
                connection, job.session_id, job.shot_id
            )
            allowed = (
                allowed
                and connection.execute(
                    "SELECT 1 FROM field_owner_media_deletions WHERE session_id=? AND shot_id=?",
                    (job.session_id, job.shot_id),
                ).fetchone()
                is None
            )
            if allowed and kind in {"recording_interval", "recording_capture", "recording_end"}:
                evidence = self._recording_evidence_in(connection, job.session_id, job.shot_id)
                allowed = (
                    policy == "recording-v1"
                    and sha256(_canonical(evidence).encode()).hexdigest()
                    == job.source.shot_evidence_sha256
                )
                capture = self._recording_capture_in(connection, job.session_id)
                frozen = self._recording_end_in(connection, job.session_id)
                if capture is not None:
                    cutoff = capture["cutoff"]
                    allowed = (
                        allowed
                        and cutoff is not None
                        and capture["source_epoch_id"] == job.source_epoch_id
                        and job.end_frame_id <= cutoff["frame_id"]
                    )
                if kind == "recording_capture":
                    allowed = (
                        allowed
                        and capture is not None
                        and capture["capture_end_sha256"] == job.source.capture_end_sha256
                    )
                elif kind == "recording_end":
                    allowed = (
                        allowed
                        and frozen is not None
                        and sha256(_canonical(frozen).encode()).hexdigest()
                        == job.source.recording_end_sha256
                    )
                elif row[0] == "ended" and capture is None:
                    allowed = (
                        allowed
                        and frozen is not None
                        and any(
                            item["shot_id"] == job.shot_id
                            and item["payload_sha256"] == job.source.shot_evidence_sha256
                            for item in frozen["shots"]
                        )
                    )
            elif allowed:
                manifest = self._postgame_manifest_in(connection, job.session_id)
                allowed = (
                    policy == "legacy"
                    and kind == "legacy_frozen_job"
                    and manifest.manifest_sha256 == job.source.postgame_manifest_sha256
                    and job.source.source_job_id in manifest.expected_jobs
                )
            yield bool(allowed)

    def ended_session_allows_media_encoding(self, session_id: str) -> bool:
        """Fresh ended-session/live-priority snapshot, with one explicitly closed reader."""
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN")
                row = connection.execute(
                    "SELECT * FROM field_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                if row is None:
                    raise ValueError("field session does not exist")
                session = self._session_from_row(connection, row)
                return (
                    session.status is FieldSessionStatus.ENDED
                    and connection.execute(
                        "SELECT 1 FROM field_sessions WHERE status='open' LIMIT 1"
                    ).fetchone()
                    is None
                )
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def has_open_session(self) -> bool:
        """Non-blocking live-priority probe for independent postgame workers."""
        with self._connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM field_sessions WHERE status='open' LIMIT 1"
                ).fetchone()
                is not None
            )

    def has_managed_session_identity(self, session_id: str) -> bool:
        """An exclusion probe, never authority to adopt an unbound session."""
        if session_id.startswith("managed-"):
            return True
        with self._connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM field_managed_open_requests WHERE session_id=? LIMIT 1",
                    (session_id,),
                ).fetchone()
                is not None
            )

    def list_ended_sessions_without_sync(
        self, artifact_id: str | None = None
    ) -> tuple[FieldSessionRecord, ...]:
        """Return durable ended sessions whose outbox creation still needs retrying."""
        try:
            with self._connect() as connection:
                query = (
                    "SELECT s.* FROM field_sessions AS s "
                    "LEFT JOIN field_sync_jobs AS j ON j.session_id=s.session_id "
                    "WHERE s.status='ended' AND j.session_id IS NULL "
                )
                parameters: tuple[str, ...] = ()
                if artifact_id is not None:
                    query += "AND s.artifact_id=? "
                    parameters = (artifact_id,)
                rows = connection.execute(
                    query + "ORDER BY s.created_at_utc, s.session_id", parameters
                ).fetchall()
                return tuple(self._session_from_row(connection, row) for row in rows)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def list_ended_session_ids_requiring_postgame(
        self, *, managed_scope: ManagedSessionScope
    ) -> tuple[str, ...]:
        """Scoped candidates only; callers must resolve each retained context strictly.

        Filter before decoding scene/match metadata so foreign malformed rows cannot
        starve eligible work. This grants no launch, claim, or cleanup authority.
        """
        values = self._managed_scope_values(managed_scope)
        with self._connect() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT s.session_id FROM field_sessions s "
                    "JOIN field_managed_open_requests m ON m.session_id=s.session_id "
                    "LEFT JOIN field_postgame_finalizations f ON f.session_id=s.session_id "
                    "LEFT JOIN field_sync_jobs j ON j.session_id=s.session_id "
                    "WHERE m.environment_id=? AND m.device_id=? AND m.table_id=? "
                    "AND s.status='ended' AND (f.session_id IS NULL "
                    "OR j.status IN ('pending','partial','claimed')) "
                    "ORDER BY s.created_at_utc,s.session_id",
                    values,
                )
            )

    def list_ended_sessions_requiring_postgame(self) -> tuple[FieldSessionRecord, ...]:
        """Return ended sessions whose analysis or retryable publication is unfinished."""

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT s.* FROM field_sessions AS s "
                    "LEFT JOIN field_postgame_finalizations AS f "
                    "ON f.session_id=s.session_id "
                    "LEFT JOIN field_sync_jobs AS j ON j.session_id=s.session_id "
                    "WHERE s.status='ended' AND (f.session_id IS NULL "
                    "OR j.status IN ('pending','partial','claimed')) "
                    "ORDER BY s.created_at_utc, s.session_id"
                ).fetchall()
                return tuple(self._session_from_row(connection, row) for row in rows)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def bind_artifact_path(self, artifact_id: str, artifact_path: Path) -> None:
        resolved = artifact_path.resolve()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                conflicts = connection.execute(
                    "SELECT a.artifact_path FROM field_session_artifacts AS a "
                    "JOIN field_sessions AS s ON s.session_id=a.session_id "
                    "WHERE s.artifact_id=? AND a.artifact_path<>? LIMIT 1",
                    (artifact_id, str(resolved)),
                ).fetchone()
                if conflicts is not None:
                    raise ValueError("artifact ID is already bound to a different path")
                connection.execute(
                    "INSERT OR IGNORE INTO field_session_artifacts(session_id, artifact_path) "
                    "SELECT session_id, ? FROM field_sessions WHERE artifact_id=?",
                    (str(resolved), artifact_id),
                )
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def artifact_path_for_session(self, session_id: str) -> Path:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT artifact_path FROM field_session_artifacts WHERE session_id=?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    raise ValueError("field session artifact path is unavailable")
                return Path(row["artifact_path"])
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def has_sync_job(self, session_id: str) -> bool:
        try:
            with self._connect() as connection:
                return (
                    connection.execute(
                        "SELECT 1 FROM field_sync_jobs WHERE session_id=?", (session_id,)
                    ).fetchone()
                    is not None
                )
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def is_session_sync_complete(self, session_id: str) -> bool:
        """Return true only after the immutable session and media reached CloudBase."""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT status FROM field_sync_jobs WHERE session_id=?",
                    (session_id,),
                ).fetchone()
                return row is not None and row["status"] == "complete"
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def enqueue_cloud_close_intent(self, session_id: str) -> None:
        now = _timestamp(self._now_utc())
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO field_cloud_close_intents VALUES(?, 'pending', ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET status='pending', updated_at_utc=? "
                    "WHERE field_cloud_close_intents.status<>'complete'",
                    (session_id, now, now),
                )
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def record_cloud_session_open(self, session_id: str) -> None:
        now = _timestamp(self._now_utc())
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO field_cloud_close_intents VALUES(?, 'open', ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET status='open', updated_at_utc=?",
                    (session_id, now, now),
                )
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def complete_cloud_close_intent(self, session_id: str) -> None:
        now = _timestamp(self._now_utc())
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE field_cloud_close_intents SET status='complete', updated_at_utc=? "
                    "WHERE session_id=? AND status='pending'",
                    (now, session_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("pending cloud close intent does not exist")
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def list_pending_cloud_close_session_ids(self) -> tuple[str, ...]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT session_id FROM field_cloud_close_intents "
                    "WHERE status='pending' ORDER BY updated_at_utc, session_id"
                ).fetchall()
                return tuple(row["session_id"] for row in rows)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def save_match(
        self,
        session_id: str,
        match: MatchControlSnapshot,
        *,
        expected_revision: int,
    ) -> FieldSessionRecord:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                session = connection.execute(
                    "SELECT status FROM field_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                state = connection.execute(
                    "SELECT revision FROM field_match_state WHERE session_id=?", (session_id,)
                ).fetchone()
                if session is None or state is None:
                    raise ValueError("field session does not exist")
                if session["status"] != FieldSessionStatus.OPEN.value:
                    raise ValueError("ended field session cannot change match state")
                if state["revision"] != expected_revision:
                    raise ValueError("stale match revision cannot overwrite persisted state")
                if match.revision <= expected_revision:
                    raise ValueError("new match revision must advance persisted state")
                connection.execute(
                    "UPDATE field_match_state SET payload_json=?, revision=? WHERE session_id=?",
                    (_canonical(_match_payload(match)), match.revision, session_id),
                )
                self._stale_confirmations(connection, session_id, "MATCH_REVISION_CHANGED")
            return self.load_session(session_id)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    @staticmethod
    def _command_application(row: sqlite3.Row) -> FieldCommandApplication:
        return FieldCommandApplication(
            command_id=row["command_id"],
            outcome=row["outcome"],
            result_code=row["result_code"],
            resulting_revision=int(row["resulting_revision"]),
        )

    @staticmethod
    def _stale_confirmations(connection: sqlite3.Connection, session_id: str, reason: str) -> None:
        connection.execute(
            "UPDATE field_turn_confirmations SET status='stale', stale_reason=? "
            "WHERE session_id=? AND status='pending'",
            (reason, session_id),
        )

    @staticmethod
    def _valid_confirmation(
        connection: sqlite3.Connection,
        envelope: FieldCommandEnvelope,
        reference: TurnConfirmationRef,
    ) -> bool:
        if envelope.command_type not in {"continue_turn", "switch_turn"}:
            return False
        if (
            reference.session_id != envelope.session_id
            or reference.match_revision != envelope.expected_revision
        ):
            return False
        row = connection.execute(
            "SELECT c.*, s.sequence FROM field_turn_confirmations c "
            "JOIN field_shots s USING(session_id,shot_id) "
            "WHERE c.session_id=? AND c.shot_id=?",
            (reference.session_id, reference.shot_id),
        ).fetchone()
        latest = connection.execute(
            "SELECT MAX(sequence) FROM field_shots WHERE session_id=?", (envelope.session_id,)
        ).fetchone()[0]
        return bool(
            row is not None
            and row["status"] == "pending"
            and row["evidence_digest"] == reference.evidence_digest
            and row["match_revision"] == reference.match_revision
            and row["sequence"] == latest
        )

    def get_turn_confirmation(
        self, session_id: str, shot_id: str
    ) -> TurnConfirmationReceipt | None:
        """Read the local abstention; resolved does not mean reviewed algorithm facts."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM field_turn_confirmations WHERE session_id=? AND shot_id=?",
                    (session_id, shot_id),
                ).fetchone()
            if row is None:
                return None
            return TurnConfirmationReceipt(
                reference=TurnConfirmationRef(
                    session_id, shot_id, row["evidence_digest"], row["match_revision"]
                ),
                source_mainline1_artifact_id=row["source_mainline1_artifact_id"],
                status=row["status"],
                stale_reason=row["stale_reason"],
                resolved_command_id=row["resolved_command_id"],
                created_at_utc=row["created_at_utc"],
            )
        except sqlite3.DatabaseError as exc:
            raise ValueError("invalid turn confirmation receipt") from exc

    @staticmethod
    def _has_unsettled_shot(connection: sqlite3.Connection, session_id: str) -> bool:
        rows = connection.execute(
            "SELECT payload_json FROM field_shots WHERE session_id=?", (session_id,)
        ).fetchall()
        return any(_shot_from_payload(row["payload_json"]).settled_ms is None for row in rows)

    def is_new_ordered_field_command(self, envelope: FieldCommandEnvelope) -> bool:
        """Read-only gate before a live service freezes physical capture.

        The single service owner must hold its command/match lock across this
        check, preparation and apply_field_command. The latter still validates
        and persists the authoritative result. Any existing ID (including an
        identical retry) has no new physical side effect to perform.
        """
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM field_applied_commands WHERE command_id=?",
                (envelope.command_id,),
            ).fetchone()
            next_sequence = connection.execute(
                "SELECT COALESCE(MAX(server_sequence), 0) + 1 "
                "FROM field_applied_commands WHERE session_id=?",
                (envelope.session_id,),
            ).fetchone()[0]
            return existing is None and envelope.server_sequence == next_sequence

    @staticmethod
    def _command_digest(envelope, confirmation_ref):
        # Preserve legacy hashes exactly when no internal reference was supplied.
        digest = envelope.payload_digest
        if confirmation_ref is not None:
            digest = sha256(
                _canonical({"command": digest, "confirmation": asdict(confirmation_ref)}).encode()
            ).hexdigest()
        return digest

    def get_field_command_application(
        self,
        envelope: FieldCommandEnvelope,
        *,
        confirmation_ref: TurnConfirmationRef | None = None,
    ) -> FieldCommandApplication | None:
        """Read only an exact immutable receipt; never admit a new command.

        Open the existing database read-only rather than using the writer setup
        path. Normal SQLite WAL visibility is required for committed receipts.
        """
        digest = self._command_digest(envelope, confirmation_ref)
        try:
            with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                existing = connection.execute(
                    "SELECT * FROM field_applied_commands WHERE command_id=?",
                    (envelope.command_id,),
                ).fetchone()
                if existing is None or existing["payload_digest"] != digest:
                    return None
                return self._command_application(existing)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def apply_field_command(
        self,
        envelope: FieldCommandEnvelope,
        reducer: Callable[..., FieldCommandReduction],
        *,
        confirmation_ref: TurnConfirmationRef | None = None,
    ) -> FieldCommandApplication:
        """Persist one ordered command outcome and its fact mutation atomically."""

        digest = self._command_digest(envelope, confirmation_ref)
        try:
            with self._connect(
                deadline=self._recording_end_deadlines.get(envelope.session_id)
            ) as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT * FROM field_applied_commands WHERE command_id=?",
                    (envelope.command_id,),
                ).fetchone()
                if existing is not None:
                    if existing["payload_digest"] == digest:
                        return self._command_application(existing)
                    current = connection.execute(
                        "SELECT revision FROM field_match_state WHERE session_id=?",
                        (envelope.session_id,),
                    ).fetchone()
                    return FieldCommandApplication(
                        envelope.command_id,
                        "rejected",
                        "COMMAND_IDEMPOTENCY_CONFLICT",
                        0 if current is None else int(current["revision"]),
                    )

                state_row = connection.execute(
                    """
                    SELECT s.status, m.payload_json, m.revision
                    FROM field_sessions AS s
                    JOIN field_match_state AS m ON m.session_id=s.session_id
                    WHERE s.session_id=?
                    """,
                    (envelope.session_id,),
                ).fetchone()
                if state_row is None:
                    raise ValueError("field session does not exist")
                current_revision = int(state_row["revision"])
                next_sequence = int(
                    connection.execute(
                        """
                        SELECT COALESCE(MAX(server_sequence), 0) + 1
                        FROM field_applied_commands WHERE session_id=?
                        """,
                        (envelope.session_id,),
                    ).fetchone()[0]
                )
                if envelope.server_sequence != next_sequence:
                    return FieldCommandApplication(
                        envelope.command_id,
                        "rejected",
                        "COMMAND_OUT_OF_ORDER",
                        current_revision,
                    )

                outcome = "applied"
                result_code = "APPLIED"
                resulting_revision = current_revision
                reduction: FieldCommandReduction | None = None
                if state_row["status"] != FieldSessionStatus.OPEN.value:
                    outcome, result_code = "rejected", "SESSION_ENDED"
                elif current_revision != envelope.expected_revision:
                    outcome, result_code = "rejected", "STALE_REVISION"
                elif confirmation_ref is not None and not self._valid_confirmation(
                    connection, envelope, confirmation_ref
                ):
                    outcome, result_code = "rejected", "TURN_CONFIRMATION_MISMATCH"
                else:
                    match = _match_from_payload(state_row["payload_json"])
                    shot_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM field_shots WHERE session_id=?",
                            (envelope.session_id,),
                        ).fetchone()[0]
                    )
                    try:
                        reduction = reducer(
                            match,
                            envelope,
                            has_unsettled_shot=self._has_unsettled_shot(
                                connection, envelope.session_id
                            ),
                            has_started=shot_count > 0,
                        )
                    except (TypeError, ValueError) as exc:
                        outcome = "rejected"
                        message = str(exc).strip()
                        result_code = (
                            message
                            if message.isascii()
                            and message.replace("_", "").isalnum()
                            and message.upper() == message
                            else "COMMAND_INVALID"
                        )
                    else:
                        resulting_revision = reduction.match.revision
                        result_code = reduction.result_code
                        if reduction.match != match:
                            if reduction.match.revision <= current_revision:
                                raise ValueError("field command reduction must advance revision")
                            connection.execute(
                                """
                                UPDATE field_match_state
                                SET payload_json=?, revision=? WHERE session_id=?
                                """,
                                (
                                    _canonical(_match_payload(reduction.match)),
                                    reduction.match.revision,
                                    envelope.session_id,
                                ),
                            )
                        if reduction.end_session:
                            ended_at = _timestamp(self._now_utc())
                            connection.execute(
                                """
                                UPDATE field_sessions SET status='ended', ended_at_utc=?
                                WHERE session_id=?
                                """,
                                (ended_at, envelope.session_id),
                            )
                            self._freeze_postgame_manifest(
                                connection, envelope.session_id, ended_at
                            )
                            connection.execute(
                                "UPDATE field_cloud_close_intents "
                                "SET status='pending', updated_at_utc=? "
                                "WHERE session_id=? AND status='open'",
                                (_timestamp(self._now_utc()), envelope.session_id),
                            )

                completed = _timestamp(self._now_utc())
                connection.execute(
                    """
                    INSERT INTO field_applied_commands VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        envelope.command_id,
                        envelope.session_id,
                        envelope.server_sequence,
                        envelope.actor_user_id,
                        envelope.command_type,
                        digest,
                        outcome,
                        result_code,
                        resulting_revision,
                        completed,
                    ),
                )
                ledger_order = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(ledger_order),0)+1 "
                        "FROM field_command_ledger WHERE session_id=?",
                        (envelope.session_id,),
                    ).fetchone()[0]
                )
                connection.execute(
                    """
                    INSERT INTO field_command_ledger(
                      entry_id,session_id,ledger_order,origin,principal_id,
                      origin_sequence,command_type,payload_digest,outcome,result_code,
                      resulting_revision,decision_ref_json,completed_at_utc
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        envelope.command_id,
                        envelope.session_id,
                        ledger_order,
                        "phone",
                        envelope.actor_user_id,
                        envelope.server_sequence,
                        envelope.command_type,
                        digest,
                        outcome,
                        result_code,
                        resulting_revision,
                        None,
                        completed,
                    ),
                )
                if outcome == "applied" and reduction is not None:
                    if confirmation_ref is not None:
                        connection.execute(
                            "UPDATE field_turn_confirmations SET status='resolved', "
                            "resolved_command_id=? "
                            "WHERE session_id=? AND shot_id=? AND status='pending'",
                            (
                                envelope.command_id,
                                confirmation_ref.session_id,
                                confirmation_ref.shot_id,
                            ),
                        )
                    if reduction.end_session or resulting_revision != current_revision:
                        self._stale_confirmations(
                            connection,
                            envelope.session_id,
                            "SESSION_ENDED" if reduction.end_session else "MATCH_REVISION_CHANGED",
                        )
                return FieldCommandApplication(
                    envelope.command_id,
                    outcome,
                    result_code,
                    resulting_revision,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("field command conflicts with persisted state") from exc
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def apply_admin_override(
        self,
        session_id: str,
        match: MatchControlSnapshot,
        *,
        expected_revision: int,
        command_type: str,
        reason: str,
        payload: dict[str, object],
    ) -> FieldSessionRecord:
        """Apply a pre-start administrator correction with durable audit evidence."""

        normalized_reason = reason.strip()
        if not 8 <= len(normalized_reason) <= 300:
            raise ValueError("administrator emergency reason is required")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                state = connection.execute(
                    """
                    SELECT s.status, m.revision FROM field_sessions AS s
                    JOIN field_match_state AS m ON m.session_id=s.session_id
                    WHERE s.session_id=?
                    """,
                    (session_id,),
                ).fetchone()
                if state is None:
                    raise ValueError("field session does not exist")
                if state["status"] != FieldSessionStatus.OPEN.value:
                    raise ValueError("ended field session cannot change match state")
                if int(state["revision"]) != expected_revision:
                    raise ValueError("stale match revision cannot overwrite persisted state")
                if connection.execute(
                    "SELECT 1 FROM field_shots WHERE session_id=? LIMIT 1", (session_id,)
                ).fetchone():
                    raise ValueError("administrator emergency correction is frozen after start")
                if match.revision <= expected_revision:
                    raise ValueError("new match revision must advance persisted state")
                connection.execute(
                    "UPDATE field_match_state SET payload_json=?, revision=? WHERE session_id=?",
                    (_canonical(_match_payload(match)), match.revision, session_id),
                )
                connection.execute(
                    """
                    INSERT INTO field_admin_audit(
                      session_id,command_type,reason,payload_json,resulting_revision,created_at_utc
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        session_id,
                        command_type,
                        normalized_reason,
                        _canonical(payload),
                        match.revision,
                        _timestamp(self._now_utc()),
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM field_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                assert row is not None
                return self._session_from_row(connection, row)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def list_admin_audit(self, session_id: str) -> tuple[dict[str, object], ...]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT command_type,reason,payload_json,resulting_revision,created_at_utc
                    FROM field_admin_audit WHERE session_id=? ORDER BY event_id
                    """,
                    (session_id,),
                ).fetchall()
                return tuple(
                    {
                        "command_type": row["command_type"],
                        "reason": row["reason"],
                        "payload": json.loads(row["payload_json"]),
                        "resulting_revision": int(row["resulting_revision"]),
                        "created_at_utc": row["created_at_utc"],
                    }
                    for row in rows
                )
        except (json.JSONDecodeError, sqlite3.DatabaseError) as exc:
            raise ValueError("field record database is invalid") from exc

    def record_shot_start(
        self,
        shot: FieldShotRecord,
        *,
        source_mainline1_artifact_id: str | None = None,
        ownership_ambiguous: bool = False,
        source_start_event_id: str | None = None,
        ownership_reason: str = "SOURCE_START_UNVERIFIED",
    ) -> None:
        if shot.settled_ms is not None:
            raise ValueError("shot start evidence must remain unsettled")
        payload = _canonical(_shot_payload(shot))
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM field_recording_ends WHERE session_id=?", (shot.session_id,)
                ).fetchone():
                    raise ValueError("recording evidence is sealed")
                session = connection.execute(
                    "SELECT status, artifact_id FROM field_sessions WHERE session_id=?",
                    (shot.session_id,),
                ).fetchone()
                if session is None or session["status"] != FieldSessionStatus.OPEN.value:
                    raise ValueError("shot start requires an open field session")
                if session["artifact_id"] != shot.media.artifact_id:
                    raise ValueError("shot artifact does not match session")
                existing = connection.execute(
                    "SELECT payload_json FROM field_shots WHERE session_id=? AND shot_id=?",
                    (shot.session_id, shot.shot_id),
                ).fetchone()
                if existing is not None:
                    original_start = json.loads(existing["payload_json"])
                    original_start["settled_ms"] = None
                    original_start.pop("live_evidence_digest", None)
                    original_start.pop("live_job_id", None)
                    if _canonical(original_start) == payload:
                        ownership = connection.execute(
                            "SELECT payload_json FROM field_shot_ownership "
                            "WHERE session_id=? AND shot_id=?",
                            (shot.session_id, shot.shot_id),
                        ).fetchone()
                        if (
                            ownership is not None
                            and json.loads(ownership["payload_json"])[
                                "source_mainline1_artifact_id"
                            ]
                            != source_mainline1_artifact_id
                        ):
                            raise ValueError("conflicting ownership source")
                        if (
                            ownership is not None
                            and source_start_event_id is not None
                            and json.loads(ownership[0]).get("source_start_event_id")
                            != source_start_event_id
                        ):
                            raise ValueError("conflicting ownership start event")
                        return
                    raise ValueError("conflicting shot evidence cannot overwrite persisted state")
                connection.execute(
                    "INSERT INTO field_shots VALUES(?,?,?,?)",
                    (shot.session_id, shot.shot_id, shot.sequence, payload),
                )
                match_row = connection.execute(
                    "SELECT payload_json FROM field_match_state WHERE session_id=?",
                    (shot.session_id,),
                ).fetchone()
                match = _match_from_payload(match_row["payload_json"])
                if match.rule_profile_id is not None:
                    if match.rule_profile_id != PROFILE_ID:
                        raise ValueError("unsupported persisted rule profile")
                    if (
                        match.red_user_id is None
                        or match.blue_user_id is None
                        or match.breaker is None
                        or match.active_role is None
                    ):
                        raise ValueError("profiled match requires participants and breaker")
                    if (
                        match.cue_ball_placement is not CueBallPlacement.NONE
                        and shot.media.source_kind != "usb_live"
                    ):
                        raise ValueError("cue ball placement requires live placement evidence")
                    previous = connection.execute(
                        "SELECT shot_id,payload_json FROM field_shots "
                        "WHERE session_id=? AND shot_id<>? ORDER BY sequence DESC LIMIT 1",
                        (shot.session_id, shot.shot_id),
                    ).fetchone()
                    if previous is not None:
                        previous_shot = _shot_from_payload(previous["payload_json"])
                        previous_rule = connection.execute(
                            "SELECT status FROM field_rule_decisions "
                            "WHERE session_id=? AND shot_id=?",
                            (shot.session_id, previous["shot_id"]),
                        ).fetchone()
                        if (
                            previous_shot.settled_ms is None
                            or previous_rule is None
                            or previous_rule["status"] not in {"applied", "abstained"}
                        ):
                            raise ValueError(
                                "previous profiled shot is not authoritatively resolved"
                            )
                usb_live = shot.media.source_kind == "usb_live"
                if usb_live and shot.actor_role is not match.active_role:
                    raise ValueError("stale live shot actor")
                user = (
                    match.red_user_id
                    if shot.actor_role is WristbandRole.RED
                    else match.blue_user_id
                    if shot.actor_role is WristbandRole.BLUE
                    else None
                )
                known = (
                    not ownership_ambiguous
                    and (source_mainline1_artifact_id is not None or usb_live)
                    and source_start_event_id is not None
                    and shot.actor_role is match.active_role
                    and user is not None
                )
                receipt = ShotOwnershipReceipt(
                    session_id=shot.session_id,
                    shot_id=shot.shot_id,
                    sequence=shot.sequence,
                    actor_user_id=user if known else None,
                    actor_role=shot.actor_role if known else None,
                    match_revision=match.revision,
                    source_artifact_id=shot.media.artifact_id,
                    source_mainline1_artifact_id=source_mainline1_artifact_id,
                    source_started_ms=shot.started_ms,
                    recorded_at_utc=_timestamp(self._now_utc()),
                    evidence_status="bound_at_start" if known else "unknown",
                    source_start_event_id=source_start_event_id,
                    evidence_reason="SOURCE_START_VERIFIED" if known else ownership_reason,
                    source_kind="usb_live" if usb_live else "development_replay",
                    source_epoch_id=shot.media.source_epoch_id,
                )
                connection.execute(
                    "INSERT INTO field_shot_ownership VALUES(?,?,?)",
                    (shot.session_id, shot.shot_id, _canonical(asdict(receipt))),
                )
                self._stale_confirmations(connection, shot.session_id, "NEW_SHOT_STARTED")
        except sqlite3.IntegrityError as exc:
            raise ValueError("conflicting shot evidence cannot overwrite persisted state") from exc
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    @staticmethod
    def _recording_evidence_in(connection, session_id, shot_id):
        if SQLiteFieldRecordStore._session_policy_in(connection, session_id) != "recording-v1":
            raise ValueError("recording evidence requires explicit recording policy")
        row = connection.execute(
            "SELECT payload_json,payload_sha256 FROM field_recording_evidence "
            "WHERE session_id=? AND shot_id=?",
            (session_id, shot_id),
        ).fetchone()
        if row is None:
            raise ValueError("recording evidence has no persisted start")
        if sha256(row[0].encode()).hexdigest() != row[1]:
            raise ValueError("recording metadata hash changed")
        result = recording_evidence_from_payload(json.loads(row[0]))
        if _canonical(result) != row[0]:
            raise ValueError("recording metadata is not canonical")
        if result["session_id"] != session_id or result["shot_id"] != shot_id:
            raise ValueError("recording evidence lineage mismatch")
        shot_row = connection.execute(
            "SELECT payload_json FROM field_shots WHERE session_id=? AND shot_id=?",
            (session_id, shot_id),
        ).fetchone()
        if shot_row is None:
            raise ValueError("recording evidence has no authoritative shot")
        shot = _shot_from_payload(shot_row[0])
        start = result["start"]
        boundary = result["boundary"]
        expected_settled = (
            boundary["ended_ms"]
            if boundary is not None
            and boundary["ended_ms"] > start["started_ms"]
            and boundary["reason"] not in {"USER_ENDED_RECORDING", "PROCESS_INTERRUPTED"}
            else None
        )
        if (
            (
                shot.session_id,
                shot.shot_id,
                shot.started_ms,
                shot.media.source_epoch_id,
                shot.media.source_kind,
                shot.settled_ms,
            )
            != (
                session_id,
                shot_id,
                start["started_ms"],
                start["source_epoch_id"],
                "usb_live",
                expected_settled,
            )
            or shot.actor_role is not None
            or shot.live_job_id is not None
        ):
            raise ValueError("recording metadata conflicts with authoritative shot")
        return result

    @staticmethod
    def _write_recording_evidence_in(connection, value):
        if connection.execute(
            "SELECT 1 FROM field_recording_ends WHERE session_id=?", (value["session_id"],)
        ).fetchone():
            raise ValueError("recording evidence is sealed")
        raw = _canonical(recording_evidence_from_payload(value))
        connection.execute(
            "UPDATE field_recording_evidence SET payload_json=?,payload_sha256=? "
            "WHERE session_id=? AND shot_id=?",
            (raw, sha256(raw.encode()).hexdigest(), value["session_id"], value["shot_id"]),
        )

    @staticmethod
    def _recording_end_in(connection, session_id):
        row = connection.execute(
            "SELECT * FROM field_recording_ends WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        if sha256(row["payload_json"].encode()).hexdigest() != row["payload_sha256"]:
            raise ValueError("recording end metadata hash changed")
        payload = recording_end_from_payload(json.loads(row["payload_json"]))
        if payload["session_id"] != session_id or _canonical(payload) != row["payload_json"]:
            raise ValueError("recording end metadata conflicts")
        if (row["phase"] == "frozen") != (payload["ended_at_utc"] is not None):
            raise ValueError("recording end phase conflicts")
        return payload

    @staticmethod
    def _recording_shot_refs_in(
        connection,
        session_id,
        epoch,
        cutoff,
        *,
        deadline,
        raw_manifest_path,
        raw_manifest,
        end_reason="USER_ENDED_RECORDING",
    ):
        refs = []
        incomplete = False
        rows = connection.execute(
            "SELECT shot_id FROM field_shots WHERE session_id=? ORDER BY sequence", (session_id,)
        ).fetchall()
        for row in rows:
            if time.monotonic() >= deadline:
                raise ValueError("RECORDING_DRAIN_TIMEOUT")
            evidence = SQLiteFieldRecordStore._recording_evidence_in(connection, session_id, row[0])
            boundary = evidence["boundary"]
            if (
                boundary is None
                or evidence["media_status"] not in {"RECORDED", "FAILED"}
                or boundary["source_epoch_id"] != epoch
                or cutoff is None
                or boundary["end_frame_id"] > cutoff["frame_id"]
            ):
                raise ValueError("recording shot evidence is not terminal at cutoff")
            # Media I/O/continuity failures are not successful user ends. Only
            # a real truncated window or the active user-ended tail is expected.
            if (
                evidence["media_status"] == "FAILED"
                and evidence["media"]["failure_code"] != "INCOMPLETE_RECORDING_WINDOW"
            ):
                raise ValueError("recording media failed before end")
            if not boundary["complete"] and boundary["reason"] != end_reason:
                raise ValueError("recording boundary failed before user end")
            SQLiteFieldRecordStore._validate_recording_end_reference(
                evidence, raw_manifest_path, raw_manifest, deadline=deadline
            )
            incomplete |= evidence["media_status"] == "FAILED"
            refs.append(
                dict(
                    shot_id=row[0], payload_sha256=sha256(_canonical(evidence).encode()).hexdigest()
                )
            )
        return refs, incomplete

    @staticmethod
    def _validate_recording_end_reference(evidence, raw_manifest_path, manifest, *, deadline):
        """Recheck original interval metadata without re-reading raw frame bytes."""
        from billiards_ai.lab.mainline1_realtime_ring import _same_file_identity
        from billiards_ai.lab.mainline1_session_recording import RecordedShotInterval

        media = evidence["media"]
        if media["reference_path"] is None:
            return
        if time.monotonic() >= deadline:
            raise ValueError("RECORDING_DRAIN_TIMEOUT")
        path = Path(media["reference_path"])
        try:
            if not path.is_absolute() or path.resolve(strict=True) != path or path.is_symlink():
                raise ValueError("recording reference is missing or unsafe")
            # Do not follow a swapped final symlink or block on a swapped FIFO.
            flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            with os.fdopen(os.open(path, flags), "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 * 1024:
                    raise ValueError("recording reference is unsafe or exceeds limit")
                raw = stream.read(1024 * 1024 + 1)
                after = os.fstat(stream.fileno())
            current = path.stat(follow_symlinks=False)

            def identity(value):
                return (
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                )

            if (
                not stat.S_ISREG(current.st_mode)
                or path.resolve(strict=True) != path
                or identity(before) != identity(after)
                # Keep strict dev/inode equality even if an inode is zero;
                # only the path/fd clock comparison differs on Windows.
                or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
                or not _same_file_identity(after, current, path_and_handle=True)
            ):
                raise ValueError("recording reference changed while reading")
        except OSError as exc:
            raise ValueError("recording reference is missing or unsafe") from exc
        if time.monotonic() >= deadline:
            raise ValueError("RECORDING_DRAIN_TIMEOUT")
        if len(raw) > 1024 * 1024 or sha256(raw).hexdigest() != media["reference_sha256"]:
            raise ValueError("recording reference hash mismatch")
        try:
            reference = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("recording reference is invalid") from exc
        fields = {
            "schema_version",
            "session_id",
            "shot_id",
            "source_epoch_id",
            "width",
            "height",
            "start_frame_id",
            "end_frame_id",
            "frame_count",
            "media_sha256",
            "segment_sha256",
            "storage_kind",
            "segment_root",
        }
        boundary = evidence["boundary"]
        if (
            not isinstance(reference, dict)
            or set(reference) != fields
            or reference["schema_version"] != "recording-reference/v1"
            or reference["storage_kind"] != "session_segment_reference"
            or reference["session_id"] != evidence["session_id"]
            or reference["shot_id"] != evidence["shot_id"]
            or reference["source_epoch_id"] != boundary["source_epoch_id"]
            or reference["segment_root"] != str(Path(raw_manifest_path).parent)
            or reference["width"] != manifest["width"]
            or reference["height"] != manifest["height"]
            or reference["start_frame_id"] != media["start_frame_id"]
            or reference["end_frame_id"] != media["end_frame_id"]
            or reference["frame_count"] != media["end_frame_id"] - media["start_frame_id"] + 1
            or reference["media_sha256"] != media["media_sha256"]
            or not isinstance(reference["segment_sha256"], dict)
        ):
            raise ValueError("recording reference lineage mismatch")
        # Compare the exact overlapping original segments, not just an arbitrary
        # subset of hashes. Raw verification separately validates their receipts.
        selected = {}
        for segment in manifest["segments"]:
            if time.monotonic() >= deadline:
                raise ValueError("RECORDING_DRAIN_TIMEOUT")
            if (
                segment["last_frame_id"] >= media["start_frame_id"]
                and segment["first_frame_id"] <= media["end_frame_id"]
            ):
                selected[segment["name"]] = segment["sha256"]
        if (
            not selected
            or reference["segment_sha256"] != selected
            or media["start_frame_id"] < manifest["segments"][0]["first_frame_id"]
            or media["end_frame_id"] > manifest["segments"][-1]["last_frame_id"]
        ):
            raise ValueError("recording reference segment lineage mismatch")
        interval = RecordedShotInterval(
            session_id=evidence["session_id"],
            shot_id=evidence["shot_id"],
            root=Path(reference["segment_root"]),
            source_epoch_id=boundary["source_epoch_id"],
            width=reference["width"],
            height=reference["height"],
            start_frame_id=media["start_frame_id"],
            end_frame_id=media["end_frame_id"],
            segment_sha256=reference["segment_sha256"],
        )
        if interval.media_sha256 != media["media_sha256"]:
            raise ValueError("recording reference media digest mismatch")

    @staticmethod
    def _validate_recording_raw(payload, *, deadline):
        from billiards_ai.lab.mainline1_session_recording import (
            ContinuousSessionRecorder,
            SessionSegment,
        )

        def budget():
            if time.monotonic() >= deadline:
                raise ValueError("RECORDING_DRAIN_TIMEOUT")

        budget()

        path = Path(payload["raw_manifest_path"])
        if (
            not path.is_absolute()
            or path.is_symlink()
            or path.resolve() != path
            or path.name != "session-recording.json"
            or not path.is_file()
            or path.stat().st_size > 8 * 1024 * 1024
        ):
            raise ValueError("recording end raw manifest is unsafe")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != payload["raw_manifest_sha256"]:
            raise ValueError("recording end raw manifest hash changed")
        manifest = json.loads(raw)
        if (manifest.get("schema_version"), manifest.get("kind"), manifest.get("session_id")) != (
            "1.0",
            "mainline1_session_recording",
            payload["session_id"],
        ):
            raise ValueError("recording end raw lineage mismatch")
        segments = tuple(SessionSegment(**item) for item in manifest["segments"])
        if not segments:
            if (
                payload["cutoff"] is not None
                or payload["raw_segments"]
                or manifest["source_epoch_id"]
                or manifest["width"]
                or manifest["height"]
            ):
                raise ValueError("recording empty raw manifest conflicts")
            return manifest
        if (
            manifest["source_epoch_id"],
            manifest["width"],
            manifest["height"],
            segments[0].first_frame_id,
            segments[-1].last_frame_id,
            segments[-1].last_captured_at_ns,
        ) != (
            payload["source_epoch_id"],
            payload["width"],
            payload["height"],
            payload["first_frame_id"],
            payload["last_frame_id"],
            payload["cutoff"]["captured_at_ns"],
        ):
            raise ValueError("recording end raw cutoff conflicts")
        for previous, current in zip(segments, segments[1:], strict=False):
            budget()
            if current.first_frame_id != previous.last_frame_id + 1:
                raise ValueError("recording end raw continuity failed")
        if len(payload["raw_segments"]) != len(segments):
            raise ValueError("recording segment receipt count conflicts")
        for segment, receipt in zip(segments, payload["raw_segments"], strict=True):
            budget()
            if (receipt["name"], receipt["sha256"]) != (segment.name, segment.sha256):
                raise ValueError("recording segment receipt lineage conflicts")
            original = {
                key: value for key, value in receipt.items() if key not in {"name", "sha256"}
            }
            if (
                ContinuousSessionRecorder.segment_file_identity(path.parent / segment.name)
                != original
            ):
                raise ValueError("recording segment changed since durable write")
        budget()
        return manifest

    @staticmethod
    def _recording_capture_in(connection, session_id):
        row = connection.execute(
            "SELECT * FROM field_recording_captures WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        raw = row["payload_json"]
        if sha256(raw.encode()).hexdigest() != row["payload_sha256"]:
            raise ValueError("recording capture receipt hash changed")
        payload = recording_capture_from_payload(json.loads(raw))
        if payload["session_id"] != session_id:
            raise ValueError("recording capture session conflicts")
        return payload | dict(
            tail_status=row["tail_status"],
            processing_status=row["processing_status"],
            failure_code=row["failure_code"],
            capture_end_sha256=row["payload_sha256"],
        )

    def get_recording_capture_end(self, session_id):
        with self._connect() as connection:
            return self._recording_capture_in(connection, session_id)

    def record_recording_capture_end(
        self,
        session_id,
        source_epoch_id,
        cutoff,
        raw_manifest_path,
        *,
        first_frame_id,
        width,
        height,
        last_frame_sha256,
        deadline=None,
        interruption_manifest_sha256=None,
    ):
        interruption = interruption_manifest_sha256 is not None
        payload = recording_capture_from_payload(
            dict(
                contract="recording-capture-end/v2" if interruption else "recording-capture-end/v1",
                session_id=session_id,
                source_epoch_id=source_epoch_id,
                end_reason="PROCESS_INTERRUPTED" if interruption else "USER_ENDED_RECORDING",
                cutoff=cutoff,
                raw_manifest_path=str(raw_manifest_path),
                first_frame_id=first_frame_id,
                width=width,
                height=height,
                last_frame_sha256=last_frame_sha256,
                accepted_at_utc=_timestamp(self._now_utc()),
                **(
                    {
                        "tail_completeness": "UNKNOWN",
                        "raw_manifest_sha256": interruption_manifest_sha256,
                    }
                    if interruption
                    else {}
                ),
            )
        )
        with self._connect(deadline=deadline) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._session_policy_in(connection, session_id) != "recording-v1":
                raise ValueError("capture end requires explicit recording policy")
            existing = self._recording_capture_in(connection, session_id)
            if existing is not None:
                original = {key: existing[key] for key in payload if key != "accepted_at_utc"}
                candidate = {
                    key: value for key, value in payload.items() if key != "accepted_at_utc"
                }
                if original != candidate:
                    raise ValueError("recording capture cutoff conflicts with original receipt")
                return existing
            row = connection.execute(
                "SELECT * FROM field_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row is None or row["status"] != "open":
                raise ValueError("capture end requires original open session")
            path = Path(raw_manifest_path)
            if path.resolve() != path or not path.parent.is_dir():
                raise ValueError("recording capture root is missing or unsafe")
            if (
                interruption
                and sha256(path.read_bytes()).hexdigest() != interruption_manifest_sha256
            ):
                raise ValueError("committed interruption manifest changed")
            raw = _canonical(payload)
            connection.execute(
                "INSERT INTO field_recording_captures VALUES(?,?,?,'PENDING','PROCESSING',NULL)",
                (session_id, raw, sha256(raw.encode()).hexdigest()),
            )
            if interruption:
                self._end_session_in(connection, row, ended_at=payload["accepted_at_utc"])
            return payload | dict(
                tail_status="PENDING",
                processing_status="PROCESSING",
                failure_code=None,
                capture_end_sha256=sha256(raw.encode()).hexdigest(),
            )

    def recording_interruption_candidates(self, artifact_id):
        """Only original open ordinary sessions lacking a durable stop receipt."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT s.session_id FROM field_sessions s WHERE s.artifact_id=? "
                "AND s.status='open' AND NOT EXISTS (SELECT 1 FROM field_recording_captures c "
                "WHERE c.session_id=s.session_id)",
                (artifact_id,),
            ).fetchall()
            result = []
            for row in rows:
                sid = row[0]
                if self._session_policy_in(connection, sid) != "recording-v1":
                    continue
                evidence = [
                    self._recording_evidence_in(connection, sid, x[0])
                    for x in connection.execute(
                        "SELECT shot_id FROM field_recording_evidence WHERE session_id=?",
                        (sid,),
                    ).fetchall()
                ]
                if not any(x["media_status"] == "PENDING" for x in evidence):
                    continue
                epochs = {x["start"]["source_epoch_id"] for x in evidence}
                if len(epochs) != 1:
                    raise ValueError("interrupted recording epoch is ambiguous")
                result.append((sid, epochs.pop()))
            return tuple(result)

    def resume_recording_capture_intents(self, artifact_id):
        """Finish only original durable stop intents; never resume their capture."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT s.* FROM field_sessions s JOIN field_recording_captures c "
                "ON c.session_id=s.session_id WHERE s.artifact_id=? AND s.status='open'",
                (artifact_id,),
            ).fetchall()
            for row in rows:
                receipt = self._recording_capture_in(connection, row["session_id"])
                if self._session_policy_in(connection, row["session_id"]) != "recording-v1":
                    raise ValueError("capture recovery requires original ordinary session")
                self._end_session_in(connection, row, ended_at=receipt["accepted_at_utc"])

    def pending_recording_captures(self, artifact_id):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT c.session_id FROM field_recording_captures c "
                "JOIN field_sessions s ON s.session_id=c.session_id "
                "WHERE s.artifact_id=? AND s.status='ended' "
                "AND c.processing_status!='COMPLETE' ORDER BY s.created_at_utc,c.session_id",
                (artifact_id,),
            ).fetchall()
            return tuple(self._recording_capture_in(connection, row[0]) for row in rows)

    def set_recording_capture_progress(
        self,
        session_id,
        *,
        tail_status=None,
        processing_status=None,
        failure_code=None,
    ):
        if tail_status not in {None, "PENDING", "SAVED", "FAILED"} or processing_status not in {
            None,
            "PROCESSING",
            "FAILED",
        }:
            raise ValueError("recording capture progress is invalid")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            receipt = self._recording_capture_in(connection, session_id)
            if receipt is None:
                raise ValueError("recording capture receipt is missing")
            if receipt["processing_status"] == "COMPLETE":
                raise ValueError("recording processing is already complete")
            connection.execute(
                "UPDATE field_recording_captures "
                "SET tail_status=?,processing_status=?,failure_code=? "
                "WHERE session_id=?",
                (
                    tail_status or receipt["tail_status"],
                    processing_status or receipt["processing_status"],
                    failure_code,
                    session_id,
                ),
            )

    def prepare_recording_end(
        self,
        session_id,
        source_epoch_id,
        cutoff,
        raw_manifest_path,
        raw_segments,
        *,
        deadline=None,
        background=False,
    ):
        deadline = time.monotonic() + 6 if deadline is None else deadline
        if not background:
            self._recording_end_deadlines[session_id] = deadline
        path = Path(raw_manifest_path)
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("recording end raw manifest exceeds limit")
        raw = path.read_bytes()
        manifest = json.loads(raw)
        segments = manifest["segments"]
        with self._connect(deadline=deadline) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._session_policy_in(connection, session_id) != "recording-v1":
                raise ValueError("recording end requires recording policy")
            capture = self._recording_capture_in(connection, session_id)
            interrupted = capture is not None and capture["contract"] == "recording-capture-end/v2"
            if interrupted and sha256(raw).hexdigest() != capture["raw_manifest_sha256"]:
                raise ValueError("committed interruption manifest changed")
            end_reason = "PROCESS_INTERRUPTED" if interrupted else "USER_ENDED_RECORDING"
            refs, incomplete = self._recording_shot_refs_in(
                connection,
                session_id,
                source_epoch_id,
                cutoff,
                deadline=deadline,
                raw_manifest_path=path,
                raw_manifest=manifest,
                end_reason=end_reason,
            )
            payload = recording_end_from_payload(
                dict(
                    contract="recording-end/v2" if interrupted else "recording-end/v1",
                    session_id=session_id,
                    source_epoch_id=source_epoch_id,
                    policy="recording-v1",
                    end_reason=end_reason,
                    cutoff=cutoff,
                    raw_manifest_path=str(path),
                    raw_manifest_sha256=sha256(raw).hexdigest(),
                    raw_segments=raw_segments,
                    first_frame_id=segments[0]["first_frame_id"] if segments else None,
                    last_frame_id=segments[-1]["last_frame_id"] if segments else None,
                    width=manifest["width"],
                    height=manifest["height"],
                    shots=refs,
                    media_status="EMPTY"
                    if not segments
                    else "RETAINED_INCOMPLETE"
                    if incomplete
                    else "RECORDED",
                    analysis_status="NOT_REQUESTED",
                    encoding_status="NOT_SCHEDULED",
                    ended_at_utc=None,
                    **({"tail_completeness": "UNKNOWN"} if interrupted else {}),
                )
            )
            self._validate_recording_raw(payload, deadline=deadline)
            existing = self._recording_end_in(connection, session_id)
            if existing is not None:
                if {**existing, "ended_at_utc": None} != payload:
                    raise ValueError("recording end preparation conflicts")
                return
            session = connection.execute(
                "SELECT status,ended_at_utc FROM field_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            capture = self._recording_capture_in(connection, session_id)
            if background:
                if (
                    session is None
                    or session[0] != "ended"
                    or capture is None
                    or capture["source_epoch_id"] != source_epoch_id
                    or capture["cutoff"] != cutoff
                    or capture["raw_manifest_path"] != str(path)
                ):
                    raise ValueError("background recording end lacks original capture cutoff")
                payload["ended_at_utc"] = session["ended_at_utc"]
            elif session is None or session[0] != "open":
                raise ValueError("recording end requires open session")
            raw_payload = _canonical(payload)
            connection.execute(
                "INSERT INTO field_recording_ends VALUES(?,?,?,?)",
                (
                    session_id,
                    "frozen" if background else "prepared",
                    raw_payload,
                    sha256(raw_payload.encode()).hexdigest(),
                ),
            )
            if background:
                self._freeze_postgame_manifest(connection, session_id, session["ended_at_utc"])
                connection.execute(
                    "UPDATE field_recording_captures SET tail_status='SAVED',"
                    "processing_status='COMPLETE',failure_code=NULL WHERE session_id=?",
                    (session_id,),
                )

    def _freeze_recording_end_in(self, connection, session_id, ended_at):
        if self._session_policy_in(connection, session_id) != "recording-v1":
            return
        deadline = self._recording_end_deadlines.get(session_id, time.monotonic() + 6)
        if time.monotonic() >= deadline:
            raise ValueError("RECORDING_DRAIN_TIMEOUT")
        payload = self._recording_end_in(connection, session_id)
        if payload is None:
            raise ValueError("RECORDING_DRAIN_REQUIRED")
        if payload["ended_at_utc"] is not None:
            if payload["ended_at_utc"] != ended_at:
                raise ValueError("recording end timestamp conflicts")
            return
        manifest = self._validate_recording_raw(payload, deadline=deadline)
        refs, _ = self._recording_shot_refs_in(
            connection,
            session_id,
            payload["source_epoch_id"],
            payload["cutoff"],
            deadline=deadline,
            raw_manifest_path=payload["raw_manifest_path"],
            raw_manifest=manifest,
        )
        if refs != payload["shots"]:
            raise ValueError("recording end shot identities changed")
        payload["ended_at_utc"] = ended_at
        raw = _canonical(recording_end_from_payload(payload))
        connection.execute(
            "UPDATE field_recording_ends SET phase='frozen',payload_json=?,payload_sha256=? "
            "WHERE session_id=?",
            (raw, sha256(raw.encode()).hexdigest(), session_id),
        )

    def get_recording_end_manifest(self, session_id):
        with self._connect() as connection:
            payload = self._recording_end_in(connection, session_id)
            if payload is None or payload["ended_at_utc"] is None:
                return None
            row = connection.execute(
                "SELECT status,ended_at_utc FROM field_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None or tuple(row) != ("ended", payload["ended_at_utc"]):
                raise ValueError("recording end session conflicts")
            self._recording_end_deadlines.pop(session_id, None)
            return {**payload, "manifest_sha256": sha256(_canonical(payload).encode()).hexdigest()}

    def get_recording_evidence(self, session_id, shot_id):
        with self._connect() as connection:
            return self._recording_evidence_in(connection, session_id, shot_id)

    def record_recording_start(self, session_id, shot_id, source_epoch_id, frame_id, started_ms):
        if (
            type(shot_id) is not str
            or not shot_id
            or type(source_epoch_id) is not str
            or not source_epoch_id
            or type(frame_id) is not int
            or not 0 <= frame_id <= 9_007_199_254_740_991
            or type(started_ms) is not int
            or not 0 <= started_ms < 9_007_199_254_740_991
        ):
            raise ValueError("invalid recording start")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._session_policy_in(connection, session_id) != "recording-v1":
                raise ValueError("recording start requires explicit recording policy")
            state = connection.execute(
                "SELECT * FROM field_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            start = dict(source_epoch_id=source_epoch_id, frame_id=frame_id, started_ms=started_ms)
            existing = connection.execute(
                "SELECT 1 FROM field_recording_evidence WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchone()
            if existing is not None:
                if self._recording_evidence_in(connection, session_id, shot_id)["start"] != start:
                    raise ValueError("conflicting recording start")
                return
            if connection.execute(
                "SELECT 1 FROM field_recording_ends WHERE session_id=?", (session_id,)
            ).fetchone():
                raise ValueError("recording evidence is sealed")
            capture = self._recording_capture_in(connection, session_id)
            if capture is not None:
                cutoff = capture["cutoff"]
                if (
                    source_epoch_id != capture["source_epoch_id"]
                    or cutoff is None
                    or frame_id < capture["first_frame_id"]
                    or frame_id > cutoff["frame_id"]
                    or started_ms * 1_000_000 > cutoff["captured_at_ns"] + 500_000
                ):
                    raise ValueError("recording start is outside original capture cutoff/epoch")
            if state is None or (state["status"] != "open" and capture is None):
                raise ValueError(
                    "recording start requires open session or stopped original capture"
                )
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 FROM field_shots WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
            shot = FieldShotRecord(
                session_id,
                shot_id,
                sequence,
                None,
                started_ms,
                None,
                None,
                None,
                None,
                ShotMediaRange(
                    state["artifact_id"],
                    f"usb:{source_epoch_id}",
                    "usb_live",
                    started_ms,
                    started_ms + 1,
                    source_epoch_id,
                ),
                (SessionLimitCode.NO_AUTOMATIC_FOUL_VERDICT,),
            )
            connection.execute(
                "INSERT INTO field_shots VALUES(?,?,?,?)",
                (session_id, shot_id, sequence, _canonical(_shot_payload(shot))),
            )
            match = self._session_from_row(connection, state).match
            receipt = ShotOwnershipReceipt(
                session_id=session_id,
                shot_id=shot_id,
                sequence=sequence,
                actor_role=None,
                actor_user_id=None,
                match_revision=match.revision,
                source_artifact_id=state["artifact_id"],
                source_mainline1_artifact_id=None,
                source_started_ms=started_ms,
                recorded_at_utc=_timestamp(self._now_utc()),
                evidence_status="unknown",
                evidence_reason="RECORDING_ACTOR_UNPROVEN",
                source_start_event_id=sha256(
                    _canonical(
                        start
                        | dict(
                            session_id=session_id,
                            shot_id=shot_id,
                        )
                    ).encode()
                ).hexdigest(),
                source_kind="usb_live",
                source_epoch_id=source_epoch_id,
            )
            connection.execute(
                "INSERT INTO field_shot_ownership VALUES(?,?,?)",
                (session_id, shot_id, _canonical(asdict(receipt))),
            )
            evidence = dict(
                contract="recording-evidence/v1",
                session_id=session_id,
                shot_id=shot_id,
                start=start,
                boundary=None,
                media=None,
                media_status="PENDING",
                analysis_status="NOT_REQUESTED",
                encoding_status="NOT_SCHEDULED",
            )
            raw = _canonical(evidence)
            connection.execute(
                "INSERT INTO field_recording_evidence VALUES(?,?,?,?)",
                (session_id, shot_id, raw, sha256(raw.encode()).hexdigest()),
            )

    def record_recording_boundary(self, session_id, boundary: RecordingBoundaryV1):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            data = self._recording_evidence_in(connection, session_id, boundary.shot_id)
            if data["start"] != dict(
                source_epoch_id=boundary.source_epoch_id,
                frame_id=boundary.start_frame_id,
                started_ms=boundary.started_ms,
            ):
                raise ValueError("recording boundary lineage mismatch")
            payload = asdict(boundary)
            if data["boundary"] is not None:
                if data["boundary"] != payload:
                    raise ValueError("conflicting recording boundary")
                return
            row = connection.execute(
                "SELECT payload_json FROM field_shots WHERE session_id=? AND shot_id=?",
                (session_id, boundary.shot_id),
            ).fetchone()
            original = _shot_from_payload(row[0])
            settled = (
                replace(
                    original,
                    settled_ms=boundary.ended_ms,
                    media=replace(original.media, end_ms=boundary.ended_ms),
                )
                if (
                    boundary.ended_ms > boundary.started_ms
                    and boundary.reason not in {"USER_ENDED_RECORDING", "PROCESS_INTERRUPTED"}
                )
                else original
            )
            connection.execute(
                "UPDATE field_shots SET payload_json=? WHERE session_id=? AND shot_id=?",
                (_canonical(_shot_payload(settled)), session_id, boundary.shot_id),
            )
            data["boundary"] = payload
            self._write_recording_evidence_in(connection, data)

    def record_recording_media(self, session_id, media: RecordedMediaV1):
        if not isinstance(media, RecordedMediaV1):
            raise ValueError("recording media must be typed")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            original = self._recording_evidence_in(connection, session_id, media.boundary.shot_id)
            if original["boundary"] != asdict(media.boundary):
                raise ValueError("recording media boundary mismatch")
            if original["media"] is not None:
                if original["media"] != asdict(media):
                    raise ValueError("conflicting immutable recording media")
                return
        if media.reference_path is not None:
            # Reuse the recorded interval's original segment/hash/contiguity verifier.
            from billiards_ai.lab.mainline1_session_recording import RecordedShotInterval

            path = Path(media.reference_path)
            if path.is_symlink() or path.resolve() != path or not path.is_file():
                raise ValueError("recording reference is missing or unsafe")
            with path.open("rb") as stream:
                raw = stream.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024 or sha256(raw).hexdigest() != media.reference_sha256:
                raise ValueError("recording reference hash mismatch")
            reference = json.loads(raw)
            expected_keys = {
                "schema_version",
                "session_id",
                "shot_id",
                "source_epoch_id",
                "width",
                "height",
                "start_frame_id",
                "end_frame_id",
                "frame_count",
                "media_sha256",
                "segment_sha256",
                "storage_kind",
                "segment_root",
            }
            if (
                type(reference) is not dict
                or set(reference) != expected_keys
                or (
                    reference["schema_version"] != "recording-reference/v1"
                    or reference["session_id"] != session_id
                    or reference["shot_id"] != media.boundary.shot_id
                    or reference["source_epoch_id"] != media.boundary.source_epoch_id
                    or reference["start_frame_id"] != media.start_frame_id
                    or reference["end_frame_id"] != media.end_frame_id
                    or reference["frame_count"] != media.end_frame_id - media.start_frame_id + 1
                    or reference["storage_kind"] != "session_segment_reference"
                    or reference["media_sha256"] != media.media_sha256
                    or any(
                        type(reference[k]) is not int or reference[k] <= 0
                        for k in ("width", "height", "frame_count")
                    )
                    or type(reference["segment_root"]) is not str
                    or type(reference["segment_sha256"]) is not dict
                    or not 1 <= len(reference["segment_sha256"]) <= 4096
                )
            ):
                raise ValueError("recording reference lineage mismatch")
            root = Path(reference["segment_root"])
            if not root.is_absolute() or root.resolve() != root or not root.is_dir():
                raise ValueError("recording segment root is unsafe")
            manifest_path = root / "session-recording.json"
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError("recording source manifest is missing or unsafe")
            with manifest_path.open("rb") as stream:
                manifest_raw = stream.read(16 * 1024 * 1024 + 1)
            if len(manifest_raw) > 16 * 1024 * 1024:
                raise ValueError("recording source manifest exceeds limit")
            manifest = json.loads(manifest_raw)
            if type(manifest) is not dict or (
                manifest.get("schema_version") != "1.0"
                or manifest.get("kind") != "mainline1_session_recording"
                or manifest.get("session_id") != session_id
                or manifest.get("source_epoch_id") != media.boundary.source_epoch_id
                or manifest.get("width") != reference["width"]
                or manifest.get("height") != reference["height"]
                or type(manifest.get("segments")) is not list
            ):
                raise ValueError("recording source manifest lineage mismatch")
            source_pins = {row["name"]: row["sha256"] for row in manifest["segments"]}
            if any(
                source_pins.get(name) != digest
                for name, digest in reference["segment_sha256"].items()
            ):
                raise ValueError("recording segment was not committed by source manifest")
            for name, digest in reference["segment_sha256"].items():
                if (
                    type(name) is not str
                    or not name
                    or Path(name).name != name
                    or name in {".", ".."}
                    or "\\" in name
                    or type(digest) is not str
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or (root / name).is_symlink()
                ):
                    raise ValueError("recording segment identity is unsafe")
            interval = RecordedShotInterval(
                session_id=session_id,
                shot_id=media.boundary.shot_id,
                root=root,
                source_epoch_id=media.boundary.source_epoch_id,
                width=reference["width"],
                height=reference["height"],
                start_frame_id=media.start_frame_id,
                end_frame_id=media.end_frame_id,
                segment_sha256=reference["segment_sha256"],
            )
            if interval.media_sha256 != media.media_sha256:
                raise ValueError("recording segment digest mismatch")
            try:
                for _ in interval.frames():
                    pass
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError("recording segment verification failed") from exc
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            data = self._recording_evidence_in(connection, session_id, media.boundary.shot_id)
            if data["boundary"] != asdict(media.boundary):
                raise ValueError("recording media boundary mismatch")
            payload = asdict(media)
            if data["media"] is not None:
                if data["media"] != payload:
                    raise ValueError("conflicting immutable recording media")
                return
            data["media"] = payload
            data["media_status"] = media.status
            self._write_recording_evidence_in(connection, data)

    @staticmethod
    def _require_recording_recovery_complete_in(connection, session_id):
        if SQLiteFieldRecordStore._session_policy_in(connection, session_id) == "recording-v1":
            if connection.execute(
                "SELECT 1 FROM field_recording_ends WHERE session_id=? AND phase='prepared'",
                (session_id,),
            ).fetchone():
                raise ValueError("prepared recording end requires explicit recovery before restart")
            rows = connection.execute(
                "SELECT shot_id FROM field_recording_evidence WHERE session_id=?",
                (session_id,),
            ).fetchall()
            for row in rows:
                evidence = SQLiteFieldRecordStore._recording_evidence_in(
                    connection, session_id, row[0]
                )
                if evidence["media_status"] == "PENDING":
                    raise ValueError("recording evidence requires explicit recovery before restart")

    def require_recording_recovery_complete(self, session_id):
        with self._connect() as connection:
            self._require_recording_recovery_complete_in(connection, session_id)

    def discard_latest_unsettled_shot(
        self,
        session_id: str,
        shot_id: str,
        *,
        reason: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        """Remove only the latest incomplete boundary and retain an audit tombstone."""

        normalized_reason = reason.strip()
        if not 8 <= len(normalized_reason) <= 300:
            raise ValueError("incomplete shot discard reason is required")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM field_recording_ends WHERE session_id=?", (session_id,)
                ).fetchone():
                    raise ValueError("recording evidence is sealed")
                state = connection.execute(
                    "SELECT s.status,m.revision FROM field_sessions AS s "
                    "JOIN field_match_state AS m USING(session_id) WHERE s.session_id=?",
                    (session_id,),
                ).fetchone()
                latest = connection.execute(
                    "SELECT shot_id,payload_json FROM field_shots WHERE session_id=? "
                    "ORDER BY sequence DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                if state is None or state["status"] != FieldSessionStatus.OPEN.value:
                    raise ValueError("incomplete shot discard requires an open session")
                if latest is None or latest["shot_id"] != shot_id:
                    raise ValueError("only the latest shot can be discarded")
                if _shot_from_payload(latest["payload_json"]).settled_ms is not None:
                    raise ValueError("a settled shot cannot be discarded")
                connection.execute(
                    "DELETE FROM field_live_shot_fact_outbox WHERE idempotency_key IN "
                    "(SELECT idempotency_key FROM field_live_shot_facts "
                    "WHERE session_id=? AND shot_id=?)",
                    (session_id, shot_id),
                )
                for table in (
                    "field_live_shot_facts",
                    "field_rule_decisions",
                    "field_turn_confirmations",
                    "field_live_shot_media",
                    "field_shot_ownership",
                ):
                    connection.execute(
                        # Table comes from the fixed tuple above; both IDs are bound parameters.
                        f"DELETE FROM {table} WHERE session_id=? AND shot_id=?",  # nosec B608
                        (session_id, shot_id),
                    )
                connection.execute(
                    "DELETE FROM field_shots WHERE session_id=? AND shot_id=?",
                    (session_id, shot_id),
                )
                connection.execute(
                    "INSERT INTO field_admin_audit("
                    "session_id,command_type,reason,payload_json,resulting_revision,created_at_utc"
                    ") VALUES(?,?,?,?,?,?)",
                    (
                        session_id,
                        "discard_incomplete_shot",
                        normalized_reason,
                        _canonical({"shot_id": shot_id, **(payload or {})}),
                        int(state["revision"]),
                        _timestamp(self._now_utc()),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("incomplete shot has durable dependent evidence") from exc
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    @staticmethod
    def _settle_core(shot: FieldShotRecord) -> tuple[object, ...]:
        return (
            shot.session_id,
            shot.shot_id,
            shot.sequence,
            shot.actor_role,
            shot.started_ms,
            shot.recommendation_id,
            shot.power_band,
            shot.power_percent,
            shot.media.artifact_id,
            shot.media.source_locator,
            shot.media.source_kind,
            shot.media.start_ms,
            shot.media.source_epoch_id,
            shot.limitations,
        )

    def get_shot_ownership(self, session_id: str, shot_id: str) -> ShotOwnershipReceipt | None:
        """No legacy backfill or current-roster inference on read."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT payload_json FROM field_shot_ownership "
                    "WHERE session_id=? AND shot_id=?",
                    (session_id, shot_id),
                ).fetchone()
            if row is None:
                return None
            receipt = ownership_receipt_from_payload(
                json.loads(row["payload_json"]), allow_legacy=True
            )
            if (receipt.session_id, receipt.shot_id) != (session_id, shot_id):
                raise ValueError("ownership identity mismatch")
            return receipt
        except (sqlite3.DatabaseError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("invalid ownership receipt") from exc

    def record_shot_settle(
        self,
        shot: FieldShotRecord,
        *,
        source_epoch_id: str | None = None,
        live_fact: LiveShotFact | None = None,
        _connection: sqlite3.Connection | None = None,
    ) -> None:
        if shot.settled_ms is None:
            raise ValueError("shot settle evidence requires a settled timestamp")
        payload = _canonical(_shot_payload(shot))
        try:
            with self._connect() if _connection is None else nullcontext(_connection) as connection:
                if _connection is None:
                    connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM field_recording_ends WHERE session_id=?", (shot.session_id,)
                ).fetchone():
                    raise ValueError("recording evidence is sealed")
                row = connection.execute(
                    "SELECT payload_json FROM field_shots WHERE session_id=? AND shot_id=?",
                    (shot.session_id, shot.shot_id),
                ).fetchone()
                if row is None:
                    raise ValueError("shot settle evidence has no persisted start")
                existing = _shot_from_payload(row["payload_json"])
                ownership_row = None
                if shot.media.source_kind == "usb_live":
                    ownership_row = connection.execute(
                        "SELECT payload_json FROM field_shot_ownership "
                        "WHERE session_id=? AND shot_id=?",
                        (shot.session_id, shot.shot_id),
                    ).fetchone()
                    ownership = None if ownership_row is None else json.loads(ownership_row[0])
                    if (
                        source_epoch_id is None
                        or source_epoch_id != existing.media.source_epoch_id
                        or source_epoch_id != shot.media.source_epoch_id
                        or ownership is None
                        or ownership.get("source_kind") != "usb_live"
                        or ownership.get("source_epoch_id") != source_epoch_id
                        or ownership.get("session_id") != shot.session_id
                        or ownership.get("shot_id") != shot.shot_id
                    ):
                        raise ValueError("USB settlement source epoch lineage mismatch")
                if existing.settled_ms is not None:
                    if row["payload_json"] == payload:
                        if live_fact is not None:
                            self._insert_live_shot_fact(connection, shot, live_fact)
                        return
                    raise ValueError("conflicting shot evidence cannot overwrite persisted state")
                if self._settle_core(existing) != self._settle_core(shot):
                    raise ValueError("conflicting shot evidence cannot overwrite persisted state")
                if shot.media.source_kind == "usb_live":
                    attachment = connection.execute(
                        "SELECT payload_json FROM field_live_shot_media "
                        "WHERE session_id=? AND shot_id=?",
                        (shot.session_id, shot.shot_id),
                    ).fetchone()
                    if (
                        attachment is not None
                        and json.loads(attachment[0])["job_id"] != shot.live_job_id
                    ):
                        raise ValueError("USB settlement job lineage mismatch")
                    # Settlement records physical evidence against the frozen start.
                    # Rule application validates its own expected match revision.
                connection.execute(
                    "UPDATE field_shots SET payload_json=? WHERE session_id=? AND shot_id=?",
                    (payload, shot.session_id, shot.shot_id),
                )
                state = connection.execute(
                    "SELECT m.revision, s.status FROM field_match_state m "
                    "JOIN field_sessions s USING(session_id) WHERE session_id=?",
                    (shot.session_id,),
                ).fetchone()
                ownership = connection.execute(
                    "SELECT payload_json FROM field_shot_ownership "
                    "WHERE session_id=? AND shot_id=?",
                    (shot.session_id, shot.shot_id),
                ).fetchone()
                source = (
                    None
                    if ownership is None
                    else json.loads(ownership[0])["source_mainline1_artifact_id"]
                )
                digest = sha256(
                    _canonical(
                        {"shot": json.loads(payload), "source_mainline1_artifact_id": source}
                    ).encode()
                ).hexdigest()
                latest = connection.execute(
                    "SELECT MAX(sequence) FROM field_shots WHERE session_id=?", (shot.session_id,)
                ).fetchone()[0]
                stale_reason = (
                    "SESSION_ENDED"
                    if state["status"] != "open"
                    else "NEW_SHOT_STARTED"
                    if latest != shot.sequence
                    else None
                )
                connection.execute(
                    "INSERT INTO field_turn_confirmations VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        shot.session_id,
                        shot.shot_id,
                        digest,
                        state["revision"],
                        source,
                        "pending" if stale_reason is None else "stale",
                        stale_reason,
                        None,
                        _timestamp(self._now_utc()),
                    ),
                )
                if live_fact is not None:
                    self._insert_live_shot_fact(connection, shot, live_fact)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def _insert_live_shot_fact(
        self,
        connection: sqlite3.Connection,
        shot: FieldShotRecord,
        fact: LiveShotFact,
    ) -> None:
        if (
            shot.settled_ms is None
            or fact.session_id != shot.session_id
            or fact.shot_id != shot.shot_id
            or fact.sequence != shot.sequence
            or fact.actor_role is not shot.actor_role
        ):
            raise ValueError("live shot fact does not match settled shot")
        if shot.media.source_kind == "usb_live":
            row = connection.execute(
                "SELECT payload_json FROM field_shot_ownership WHERE session_id=? AND shot_id=?",
                (shot.session_id, shot.shot_id),
            ).fetchone()
            if row is None:
                raise ValueError("live shot fact ownership is unavailable")
            ownership = validate_shot_ownership(
                ownership_receipt_from_payload(json.loads(row[0])), shot
            )
            if fact.actor_user_id != ownership.actor_user_id:
                raise ValueError("live shot fact ownership does not match start")
        payload = _canonical(_json_value(asdict(fact)))
        existing = connection.execute(
            "SELECT payload_json FROM field_live_shot_facts WHERE session_id=? AND shot_id=?",
            (fact.session_id, fact.shot_id),
        ).fetchone()
        if existing is not None:
            if existing["payload_json"] != payload:
                raise ValueError("conflicting live shot fact cannot replace persisted state")
            return
        connection.execute(
            "INSERT INTO field_live_shot_facts VALUES(?,?,?,?)",
            (fact.idempotency_key, fact.session_id, fact.shot_id, payload),
        )
        now = _timestamp(self._now_utc())
        connection.execute(
            "INSERT INTO field_live_shot_fact_outbox VALUES(?,?,?,?,?)",
            (fact.idempotency_key, "pending", 0, None, now),
        )

    @staticmethod
    def _live_shot_outbox_from_row(row: sqlite3.Row) -> LiveShotFactOutboxItem:
        return LiveShotFactOutboxItem(
            fact=_live_shot_fact_from_payload(row["payload_json"]),
            payload_json=row["payload_json"],
            status=row["status"],
            attempts=int(row["attempts"]),
            lease_until_utc=row["lease_until_utc"],
        )

    def get_live_shot_fact(self, session_id: str, shot_id: str) -> LiveShotFact | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT payload_json FROM field_live_shot_facts "
                    "WHERE session_id=? AND shot_id=?",
                    (session_id, shot_id),
                ).fetchone()
            return None if row is None else _live_shot_fact_from_payload(row["payload_json"])
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def claim_due_live_shot_fact(
        self,
        now: datetime,
        *,
        lease_seconds: int = 60,
    ) -> LiveShotFactOutboxItem | None:
        if lease_seconds <= 0:
            raise ValueError("live shot fact lease must be positive")
        now_text = _timestamp(now)
        lease_until = _timestamp(now + timedelta(seconds=lease_seconds))
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT f.payload_json,o.* FROM field_live_shot_fact_outbox o "
                    "JOIN field_live_shot_facts f USING(idempotency_key) "
                    "WHERE o.status='pending' OR "
                    "(o.status='leased' AND o.lease_until_utc<=?) "
                    "ORDER BY f.session_id,f.shot_id LIMIT 1",
                    (now_text,),
                ).fetchone()
                if row is None:
                    return None
                connection.execute(
                    "UPDATE field_live_shot_fact_outbox SET status='leased', "
                    "attempts=attempts+1,lease_until_utc=?,updated_at_utc=? "
                    "WHERE idempotency_key=?",
                    (lease_until, now_text, row["idempotency_key"]),
                )
                claimed = connection.execute(
                    "SELECT f.payload_json,o.* FROM field_live_shot_fact_outbox o "
                    "JOIN field_live_shot_facts f USING(idempotency_key) "
                    "WHERE idempotency_key=?",
                    (row["idempotency_key"],),
                ).fetchone()
                assert claimed is not None
                return self._live_shot_outbox_from_row(claimed)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def retry_live_shot_fact(self, idempotency_key: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE field_live_shot_fact_outbox SET status='pending',"
                "lease_until_utc=NULL,updated_at_utc=? "
                "WHERE idempotency_key=? AND status='leased'",
                (_timestamp(self._now_utc()), idempotency_key),
            )
            if cursor.rowcount != 1:
                raise ValueError("only a leased live shot fact can be retried")

    def acknowledge_live_shot_fact(self, idempotency_key: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM field_live_shot_fact_outbox WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if row is None:
                raise ValueError("live shot fact does not exist")
            if row["status"] == "acknowledged":
                return
            if row["status"] != "leased":
                raise ValueError("only a leased live shot fact can be acknowledged")
            connection.execute(
                "UPDATE field_live_shot_fact_outbox SET status='acknowledged',"
                "lease_until_utc=NULL,updated_at_utc=? WHERE idempotency_key=?",
                (_timestamp(self._now_utc()), idempotency_key),
            )

    def attach_live_shot_media(self, media: LiveShotMediaReference) -> None:
        """Attach once to the original actor snapshot, never rebind from today's roster."""
        payload = _canonical(asdict(media))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            ownership = connection.execute(
                "SELECT payload_json FROM field_shot_ownership WHERE session_id=? AND shot_id=?",
                (media.session_id, media.shot_id),
            ).fetchone()
            receipt = None if ownership is None else json.loads(ownership[0])
            if receipt is None or (
                receipt["source_kind"] != "usb_live"
                or receipt.get("source_epoch_id") != media.source_epoch_id
                or receipt["match_revision"] != media.match_revision
            ):
                raise ValueError("USB media lineage does not match shot start")
            shot = _shot_from_payload(
                connection.execute(
                    "SELECT payload_json FROM field_shots WHERE session_id=? AND shot_id=?",
                    (media.session_id, media.shot_id),
                ).fetchone()[0]
            )
            if shot.live_job_id is not None and shot.live_job_id != media.job_id:
                raise ValueError("USB media job lineage does not match settlement")
            existing = connection.execute(
                "SELECT payload_json FROM field_live_shot_media WHERE session_id=? AND shot_id=?",
                (media.session_id, media.shot_id),
            ).fetchone()
            if existing is not None:
                if existing[0] != payload:
                    raise ValueError("conflicting immutable USB media")
                return
            connection.execute(
                "INSERT INTO field_live_shot_media VALUES(?,?,?)",
                (media.session_id, media.shot_id, payload),
            )

    def get_live_shot_media(self, session_id: str, shot_id: str) -> LiveShotMediaReference | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM field_live_shot_media WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchone()
        return None if row is None else LiveShotMediaReference(**json.loads(row[0]))

    def apply_confirmed_turn(
        self,
        command_id: str,
        action: str,
        reference: TurnConfirmationRef,
    ) -> MatchControlSnapshot:
        """Local confirmation adapter over the same ordered, idempotent command ledger."""
        if action not in {"continue_turn", "switch_turn"}:
            raise ValueError("unsupported turn action")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT actor_user_id, server_sequence FROM field_applied_commands "
                "WHERE command_id=?",
                (command_id,),
            ).fetchone()
            match = self.load_session(reference.session_id).match
            actor = (
                match.red_user_id if match.active_role is WristbandRole.RED else match.blue_user_id
            )
            sequence = connection.execute(
                "SELECT COALESCE(MAX(server_sequence),0)+1 FROM field_applied_commands "
                "WHERE session_id=?",
                (reference.session_id,),
            ).fetchone()[0]
        if existing is not None:
            actor, sequence = existing["actor_user_id"], existing["server_sequence"]
        if actor is None:
            raise ValueError("confirmation requires a known active participant")
        result = self.apply_field_command(
            FieldCommandEnvelope(
                command_id,
                reference.session_id,
                actor,
                action,
                {},
                reference.match_revision,
                sequence,
            ),
            reduce_field_command,
            confirmation_ref=reference,
        )
        if result.outcome != "applied":
            label = "conflict" if result.result_code == "COMMAND_IDEMPOTENCY_CONFLICT" else "stale"
            raise ValueError(f"{label} turn confirmation: {result.result_code}")
        return self.load_session(reference.session_id).match

    @staticmethod
    def _rule_receipt(row: sqlite3.Row) -> RuleDecisionReceipt:
        return RuleDecisionReceipt(
            session_id=row["session_id"],
            shot_id=row["shot_id"],
            source_epoch_id=row["source_epoch_id"],
            start_frame_id=int(row["start_frame_id"]),
            settled_frame_id=int(row["settled_frame_id"]),
            settled_evidence_digest=row["settled_evidence_digest"],
            profile_id=row["profile_id"],
            facts_json=row["facts_json"],
            fact_digest=row["fact_digest"],
            decision_json=row["decision_json"],
            decision_digest=row["decision_digest"],
            status=row["status"],
            applied_command_id=row["applied_command_id"],
            elapsed_ms=float(row["elapsed_ms"]),
            created_at_utc=row["created_at_utc"],
        )

    def get_rule_decision(self, session_id: str, shot_id: str) -> RuleDecisionReceipt | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM field_rule_decisions WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchone()
        return None if row is None else self._rule_receipt(row)

    @staticmethod
    def _validate_rule_fact_preimage(
        facts: RealtimeRuleFacts, fact_preimage_json: str | None
    ) -> dict[str, object] | None:
        if fact_preimage_json is None:
            return None
        return rule_fact_evidence.validate_realtime_rule_fact_preimage(
            _json_value(asdict(facts)), fact_preimage_json, expected_digest=facts.digest
        )

    @staticmethod
    def _read_rule_fact_preimage(
        connection: sqlite3.Connection, session_id: str, shot_id: str
    ) -> str | None:
        row = connection.execute(
            "SELECT p.fact_digest AS preimage_digest,p.normalization_preimage_json,"
            "r.* FROM field_rule_fact_preimages p "
            "LEFT JOIN field_rule_decisions r ON r.session_id=p.session_id AND r.shot_id=p.shot_id "
            "WHERE p.session_id=? AND p.shot_id=?",
            (session_id, shot_id),
        ).fetchone()
        if row is None:
            return None
        try:
            if row["preimage_digest"] != row["fact_digest"]:
                raise ValueError("digest conflict")
            normalized = rule_fact_evidence.validate_realtime_rule_fact_preimage(
                json.loads(row["facts_json"]),
                row["normalization_preimage_json"],
                expected_digest=row["fact_digest"],
            )
            for name in (
                "session_id",
                "shot_id",
                "source_epoch_id",
                "start_frame_id",
                "settled_frame_id",
                "settled_evidence_digest",
                "profile_id",
            ):
                if _canonical(normalized[name]) != _canonical(row[name]):
                    raise ValueError("receipt lineage conflict")
        except (TypeError, ValueError) as exc:
            raise ValueError("persisted rule fact normalization preimage is invalid") from exc
        return row["normalization_preimage_json"]

    def get_rule_fact_preimage(self, session_id: str, shot_id: str) -> str | None:
        """Read the retained normalization bytes; legacy absence stays unavailable."""
        try:
            with self._connect() as connection:
                return self._read_rule_fact_preimage(connection, session_id, shot_id)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def _check_existing_rule_fact_preimage(
        self,
        connection: sqlite3.Connection,
        facts: RealtimeRuleFacts,
        fact_preimage_json: str | None,
    ) -> None:
        original = self._read_rule_fact_preimage(connection, facts.session_id, facts.shot_id)
        if original is not None and original != fact_preimage_json:
            raise ValueError("rule fact normalization preimage conflict")
        # Never manufacture a historical sidecar through a replay, even if the
        # newly supplied bytes happen to reproduce the legacy digest.

    def settle_and_apply_automatic_rule_decision(
        self,
        shot: FieldShotRecord,
        facts: RealtimeRuleFacts,
        decision: RuleDecision | None,
        *,
        source_epoch_id: str,
        elapsed_ms: float,
        live_fact: LiveShotFact | None = None,
        fact_preimage_json: str | None = None,
    ) -> RuleDecisionReceipt:
        """Commit USB settlement and its rule receipt as one idempotent boundary."""

        facts_json = _canonical(_json_value(asdict(facts)))
        self._validate_rule_fact_preimage(facts, fact_preimage_json)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self.record_shot_settle(
                    shot,
                    source_epoch_id=source_epoch_id,
                    live_fact=live_fact,
                    _connection=connection,
                )
                existing = connection.execute(
                    "SELECT * FROM field_rule_decisions WHERE session_id=? AND shot_id=?",
                    (facts.session_id, facts.shot_id),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["facts_json"] == facts_json
                        and existing["fact_digest"] == facts.digest
                        and existing["source_epoch_id"] == facts.source_epoch_id
                        and int(existing["start_frame_id"]) == facts.start_frame_id
                        and int(existing["settled_frame_id"]) == facts.settled_frame_id
                        and existing["settled_evidence_digest"] == facts.settled_evidence_digest
                        and existing["profile_id"] == facts.profile_id
                    ):
                        self._check_existing_rule_fact_preimage(
                            connection, facts, fact_preimage_json
                        )
                        return self._rule_receipt(existing)
                    raise ValueError("rule decision conflict for settled shot")
                if decision is None:
                    raise ValueError("new rule settlement requires a deterministic decision")
                return self.apply_automatic_rule_decision(
                    facts,
                    decision,
                    elapsed_ms=elapsed_ms,
                    fact_preimage_json=fact_preimage_json,
                    _connection=connection,
                )
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def list_command_ledger(self, session_id: str) -> tuple[CommandLedgerEntry, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM field_command_ledger WHERE session_id=? ORDER BY ledger_order",
                (session_id,),
            ).fetchall()
        return tuple(
            CommandLedgerEntry(
                entry_id=row["entry_id"],
                session_id=row["session_id"],
                ledger_order=int(row["ledger_order"]),
                origin=row["origin"],
                principal_id=row["principal_id"],
                origin_sequence=(
                    None if row["origin_sequence"] is None else int(row["origin_sequence"])
                ),
                command_type=row["command_type"],
                payload_digest=row["payload_digest"],
                outcome=row["outcome"],
                result_code=row["result_code"],
                resulting_revision=int(row["resulting_revision"]),
                decision_ref_json=row["decision_ref_json"],
                completed_at_utc=row["completed_at_utc"],
            )
            for row in rows
        )

    def apply_automatic_rule_decision(
        self,
        facts: RealtimeRuleFacts,
        decision: RuleDecision,
        *,
        elapsed_ms: float,
        fact_preimage_json: str | None = None,
        _connection: sqlite3.Connection | None = None,
    ) -> RuleDecisionReceipt:
        """Persist one deterministic rules result and its match mutation atomically."""

        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("rule evaluation elapsed time is invalid")
        facts_json = _canonical(_json_value(asdict(facts)))
        normalized = self._validate_rule_fact_preimage(facts, fact_preimage_json)
        decision_json = _canonical(_json_value(asdict(decision)))
        if (
            facts.session_id != decision.session_id
            or facts.shot_id != decision.shot_id
            or facts.source_epoch_id != decision.source_epoch_id
            or facts.profile_id != decision.profile_id
            or facts.digest != decision.fact_digest
            or facts.settled_evidence_digest != decision.settled_evidence_digest
        ):
            raise ValueError("rule decision lineage conflict")
        try:
            with self._connect() if _connection is None else nullcontext(_connection) as connection:
                if _connection is None:
                    connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT * FROM field_rule_decisions WHERE session_id=? AND shot_id=?",
                    (facts.session_id, facts.shot_id),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["facts_json"] == facts_json
                        and existing["fact_digest"] == facts.digest
                        and existing["decision_json"] == decision_json
                        and existing["decision_digest"] == decision.decision_digest
                    ):
                        self._check_existing_rule_fact_preimage(
                            connection, facts, fact_preimage_json
                        )
                        return self._rule_receipt(existing)
                    raise ValueError("rule decision conflict for settled shot")

                row = connection.execute(
                    """
                    SELECT s.status,m.payload_json,m.revision,sh.sequence,
                           sh.payload_json AS shot_json,
                           o.payload_json AS ownership_json
                    FROM field_sessions AS s
                    JOIN field_match_state AS m USING(session_id)
                    JOIN field_shots AS sh USING(session_id)
                    JOIN field_shot_ownership AS o
                      ON o.session_id=sh.session_id AND o.shot_id=sh.shot_id
                    WHERE s.session_id=? AND sh.shot_id=?
                    """,
                    (facts.session_id, facts.shot_id),
                ).fetchone()
                if row is None:
                    raise ValueError("rule decision has no authoritative shot")
                latest = connection.execute(
                    "SELECT MAX(sequence) FROM field_shots WHERE session_id=?",
                    (facts.session_id,),
                ).fetchone()[0]
                match = _match_from_payload(row["payload_json"])
                shot = _shot_from_payload(row["shot_json"])
                ownership = json.loads(row["ownership_json"])
                if row["status"] != FieldSessionStatus.OPEN.value:
                    raise ValueError("stale rule decision for ended session")
                if int(row["sequence"]) != int(latest):
                    raise ValueError("stale rule decision for non-latest shot")
                if shot.settled_ms is None or shot.live_evidence_digest is None:
                    raise ValueError("rule decision requires settled USB evidence")
                if (
                    shot.media.source_kind != "usb_live"
                    or shot.media.source_epoch_id != facts.source_epoch_id
                    or shot.live_evidence_digest != facts.settled_evidence_digest
                    or ownership.get("source_epoch_id") != facts.source_epoch_id
                    or ownership.get("match_revision") != facts.match_revision_at_start
                    or ownership.get("actor_role") != facts.actor_role.value
                    or match.revision != facts.match_revision_at_start
                    or decision.expected_revision != match.revision
                    or match.rule_profile_id != facts.profile_id
                    or facts.profile_id != PROFILE_ID
                ):
                    raise ValueError("stale or conflicting rule decision lineage")
                if (
                    normalized is not None
                    and normalized["required_cue_ball_placement"] != match.cue_ball_placement.value
                ):
                    raise ValueError("rule fact preimage placement context mismatch")
                expected = decide_eight_ball(match, facts)
                if expected != decision:
                    raise ValueError("rule decision does not match deterministic reducer")

                created = _timestamp(self._now_utc())
                applied_command_id: str | None = None
                status = {
                    RuleOutcomeKind.APPLY: "applied",
                    RuleOutcomeKind.ABSTAIN: "abstained",
                    RuleOutcomeKind.MANUAL_CHOICE: "manual",
                }[decision.kind]
                if decision.kind is RuleOutcomeKind.APPLY:
                    updated = apply_rule_decision(match, decision)
                    connection.execute(
                        "UPDATE field_match_state SET payload_json=?,revision=? WHERE session_id=?",
                        (_canonical(_match_payload(updated)), updated.revision, facts.session_id),
                    )
                    applied_command_id = f"rule-{decision.decision_digest}"
                    ledger_order = int(
                        connection.execute(
                            "SELECT COALESCE(MAX(ledger_order),0)+1 "
                            "FROM field_command_ledger WHERE session_id=?",
                            (facts.session_id,),
                        ).fetchone()[0]
                    )
                    decision_ref = _canonical(
                        {
                            "session_id": facts.session_id,
                            "shot_id": facts.shot_id,
                            "fact_digest": facts.digest,
                            "decision_digest": decision.decision_digest,
                        }
                    )
                    connection.execute(
                        """
                        INSERT INTO field_command_ledger(
                          entry_id,session_id,ledger_order,origin,principal_id,
                          origin_sequence,command_type,payload_digest,outcome,result_code,
                          resulting_revision,decision_ref_json,completed_at_utc
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            applied_command_id,
                            facts.session_id,
                            ledger_order,
                            "rules_engine",
                            PROFILE_ID,
                            None,
                            f"auto_{decision.action}",
                            decision.decision_digest,
                            "applied",
                            decision.reason_code,
                            updated.revision,
                            decision_ref,
                            created,
                        ),
                    )
                    self._stale_confirmations(
                        connection,
                        facts.session_id,
                        "MATCH_REVISION_CHANGED",
                    )

                connection.execute(
                    """
                    INSERT INTO field_rule_decisions(
                      session_id,shot_id,source_epoch_id,start_frame_id,settled_frame_id,
                      settled_evidence_digest,profile_id,facts_json,fact_digest,
                      decision_json,decision_digest,status,applied_command_id,elapsed_ms,
                      created_at_utc
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        facts.session_id,
                        facts.shot_id,
                        facts.source_epoch_id,
                        facts.start_frame_id,
                        facts.settled_frame_id,
                        facts.settled_evidence_digest,
                        facts.profile_id,
                        facts_json,
                        facts.digest,
                        decision_json,
                        decision.decision_digest,
                        status,
                        applied_command_id,
                        elapsed_ms,
                        created,
                    ),
                )
                if fact_preimage_json is not None:
                    connection.execute(
                        "INSERT INTO field_rule_fact_preimages("
                        "session_id,shot_id,fact_digest,"
                        "normalization_preimage_json,created_at_utc) "
                        "VALUES(?,?,?,?,?)",
                        (
                            facts.session_id,
                            facts.shot_id,
                            facts.digest,
                            fact_preimage_json,
                            created,
                        ),
                    )
                result = connection.execute(
                    "SELECT * FROM field_rule_decisions WHERE session_id=? AND shot_id=?",
                    (facts.session_id, facts.shot_id),
                ).fetchone()
                assert result is not None
                return self._rule_receipt(result)
        except sqlite3.IntegrityError as exc:
            raise ValueError("rule decision conflicts with persisted state") from exc
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def finalize_automatic_terminal_decision(
        self,
        session_id: str,
        shot_id: str,
        job_id: str,
    ) -> None:
        """End only after the terminal shot's immutable media job is attached."""

        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT s.status,s.ended_at_utc,m.payload_json,
                           sh.sequence,sh.payload_json AS shot_json,
                           r.status AS rule_status,r.decision_json,
                           lm.payload_json AS media_json
                    FROM field_sessions AS s
                    JOIN field_match_state AS m USING(session_id)
                    JOIN field_shots AS sh USING(session_id)
                    JOIN field_rule_decisions AS r
                      ON r.session_id=sh.session_id AND r.shot_id=sh.shot_id
                    JOIN field_live_shot_media AS lm
                      ON lm.session_id=sh.session_id AND lm.shot_id=sh.shot_id
                    WHERE s.session_id=? AND sh.shot_id=?
                    """,
                    (session_id, shot_id),
                ).fetchone()
                if row is None:
                    raise ValueError("terminal rule decision lacks durable USB media")
                shot = _shot_from_payload(row["shot_json"])
                match = _match_from_payload(row["payload_json"])
                decision_payload = json.loads(row["decision_json"])
                media_payload = json.loads(row["media_json"])
                latest = connection.execute(
                    "SELECT MAX(sequence) FROM field_shots WHERE session_id=?",
                    (session_id,),
                ).fetchone()[0]
                if (
                    int(row["sequence"]) != int(latest)
                    or row["rule_status"] != "applied"
                    or decision_payload.get("action") != "end"
                    or match.rack_phase is not RackPhase.ENDED
                    or shot.media.source_kind != "usb_live"
                    or shot.live_job_id != job_id
                    or media_payload.get("job_id") != job_id
                ):
                    raise ValueError("terminal rule decision durable lineage mismatch")
                if row["status"] == FieldSessionStatus.ENDED.value:
                    self._freeze_postgame_manifest(connection, session_id, row["ended_at_utc"])
                    return
                if row["status"] != FieldSessionStatus.OPEN.value:
                    raise ValueError("terminal rule decision session status is invalid")
                ended_at = _timestamp(self._now_utc())
                connection.execute(
                    "UPDATE field_sessions SET status='ended',ended_at_utc=? WHERE session_id=?",
                    (ended_at, session_id),
                )
                self._freeze_postgame_manifest(connection, session_id, ended_at)
                self._stale_confirmations(connection, session_id, "SESSION_ENDED")
                connection.execute(
                    "UPDATE field_cloud_close_intents "
                    "SET status='pending', updated_at_utc=? "
                    "WHERE session_id=? AND status='open'",
                    (_timestamp(self._now_utc()), session_id),
                )
        except (json.JSONDecodeError, sqlite3.DatabaseError) as exc:
            raise ValueError("field record database is invalid") from exc

    def list_shots(self, session_id: str) -> tuple[FieldShotRecord, ...]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT payload_json FROM field_shots WHERE session_id=? ORDER BY sequence",
                    (session_id,),
                ).fetchall()
                return tuple(_shot_from_payload(row["payload_json"]) for row in rows)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def end_session(self, session_id: str) -> FieldSessionRecord:
        try:
            with self._connect(
                deadline=self._recording_end_deadlines.get(session_id)
            ) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM field_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                if row is None:
                    raise ValueError("field session does not exist")
                self._end_session_in(connection, row)
            self._recording_end_deadlines.pop(session_id, None)
            return self.load_session(session_id)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def _end_session_in(
        self, connection: sqlite3.Connection, row: sqlite3.Row, *, ended_at=None
    ) -> None:
        session_id = row["session_id"]
        if row["status"] == FieldSessionStatus.OPEN.value:
            ended_at = ended_at or _timestamp(self._now_utc())
            connection.execute(
                "UPDATE field_sessions SET status='ended', ended_at_utc=? WHERE session_id=?",
                (ended_at, session_id),
            )
            self._freeze_postgame_manifest(connection, session_id, ended_at)
            self._stale_confirmations(connection, session_id, "SESSION_ENDED")
            connection.execute(
                "UPDATE field_cloud_close_intents "
                "SET status='pending', updated_at_utc=? "
                "WHERE session_id=? AND status='open'",
                (_timestamp(self._now_utc()), session_id),
            )
        else:
            self._freeze_postgame_manifest(connection, session_id, row["ended_at_utc"])

    def _freeze_postgame_manifest(
        self, connection: sqlite3.Connection, session_id: str, ended_at: str
    ) -> None:
        """Create the one immutable job head in the same end transaction."""
        capture = self._recording_capture_in(connection, session_id)
        # This immutable head identifies the review, not its growing shot count.
        # Ordinary capture has no analysis jobs; only recording-end/v1 freezes
        # the final detected inventory and permits completion.
        if capture is None or self._recording_end_in(connection, session_id) is not None:
            self._freeze_recording_end_in(connection, session_id, ended_at)
        expected_jobs = tuple(
            shot.live_job_id
            for shot_row in connection.execute(
                "SELECT payload_json FROM field_shots WHERE session_id=? ORDER BY sequence",
                (session_id,),
            ).fetchall()
            if (shot := _shot_from_payload(shot_row[0])).settled_ms is not None
            and shot.media.source_kind == "usb_live"
            and shot.live_job_id
            and not is_capture_interruption(shot)
        )
        if len(expected_jobs) != len(set(expected_jobs)):
            raise ValueError("ended session has duplicate USB job identities")
        existing = connection.execute(
            "SELECT ended_at_utc, expected_jobs_json FROM field_postgame_manifests "
            "WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if existing is not None:
            if existing["ended_at_utc"] != ended_at or existing["expected_jobs_json"] != _canonical(
                expected_jobs
            ):
                raise ValueError("postgame manifest conflicts with ended session")
            return
        deadline = _timestamp(
            datetime.fromisoformat(ended_at.replace("Z", "+00:00")) + timedelta(seconds=120)
        )
        connection.execute(
            "INSERT INTO field_postgame_manifests "
            "(session_id, ended_at_utc, deadline_at_utc, expected_jobs_json) VALUES(?,?,?,?)",
            (session_id, ended_at, deadline, _canonical(expected_jobs)),
        )
        connection.execute(
            "INSERT INTO field_postgame_reviews(session_id,artifact_id,status) "
            "SELECT session_id,artifact_id,CASE WHEN EXISTS "
            "(SELECT 1 FROM field_postgame_reviews r WHERE r.artifact_id=s.artifact_id "
            "AND r.status='active') THEN 'pending' ELSE 'active' END "
            "FROM field_sessions s WHERE session_id=?",
            (session_id,),
        )

    def active_postgame_review(self, artifact_id: str) -> PostgameSessionManifest | None:
        """Recover review authority independently of live sessions and sync state."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT session_id FROM field_postgame_reviews "
                "WHERE artifact_id=? AND status='active'",
                (artifact_id,),
            ).fetchone()
            if row is None:
                return None
            manifest = self._postgame_manifest_in(connection, row["session_id"])
            session = connection.execute(
                "SELECT artifact_id FROM field_sessions WHERE session_id=?",
                (manifest.session_id,),
            ).fetchone()
            if session["artifact_id"] != artifact_id:
                raise ValueError("review artifact identity conflicts")
            return manifest

    def active_postgame_review_for_scope(
        self, scope: ManagedSessionScope
    ) -> PostgameSessionManifest | None:
        """Select retained same-scope review authority across original artifacts."""
        values = self._managed_scope_values(scope)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT m.* FROM field_managed_open_requests m "
                "JOIN field_postgame_reviews r ON r.session_id=m.session_id "
                "WHERE m.environment_id=? AND m.device_id=? AND m.table_id=? "
                "AND r.status='active' ORDER BY m.reservation_order",
                values,
            ).fetchall()
            for row in rows:
                request = self._managed_request_from_row(row)
                if request.phase != "materialized":
                    raise ValueError("managed review requires retained materialized binding")
                session = self.validate_managed_local_session(scope, request.open_request_id)
                manifest = self._postgame_manifest_in(connection, session.session_id)
                review = connection.execute(
                    "SELECT artifact_id FROM field_postgame_reviews WHERE session_id=?",
                    (session.session_id,),
                ).fetchone()
                if review["artifact_id"] != session.artifact_id:
                    raise ValueError("managed review artifact identity conflicts")
                return manifest
            return None

    def pending_postgame_review_session_ids_for_scope(
        self, scope: ManagedSessionScope
    ) -> tuple[str, ...]:
        """Read blocked scoped queues without promoting or adopting another scope."""
        values = self._managed_scope_values(scope)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT m.* FROM field_managed_open_requests m "
                "JOIN field_postgame_reviews r ON r.session_id=m.session_id "
                "WHERE m.environment_id=? AND m.device_id=? AND m.table_id=? "
                "AND r.status='pending' ORDER BY m.reservation_order",
                values,
            ).fetchall()
            result = []
            for row in rows:
                request = self._managed_request_from_row(row)
                session = self.validate_managed_local_session(scope, request.open_request_id)
                self._postgame_manifest_in(connection, session.session_id)
                result.append(session.session_id)
            return tuple(result)

    def complete_postgame_review(
        self, artifact_id: str, session_id: str, manifest_sha256: str, *, media_jobs=None
    ) -> None:
        """Finish the active review and promote the next queued end atomically."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            manifest = self._postgame_manifest_in(connection, session_id)
            review = connection.execute(
                "SELECT artifact_id,status FROM field_postgame_reviews WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if (
                review is None
                or review["artifact_id"] != artifact_id
                or review["status"] not in {"active", "completed"}
            ):
                raise ValueError("session is not the active review")
            if manifest.manifest_sha256 != manifest_sha256:
                raise ValueError("stale postgame manifest digest")
            if review["status"] == "completed":
                return
            ordinary = self._session_policy_in(connection, session_id) == "recording-v1"
            if ordinary:
                recording = self._recording_end_in(connection, session_id)
                if recording is None or recording["ended_at_utc"] is None:
                    raise ValueError("recording processing is not complete")
                capture = self._recording_capture_in(connection, session_id)
                if capture is not None:
                    if capture["contract"] == "recording-capture-end/v2":
                        raise ValueError(
                            "interrupted recording tail is unknown; original media retained"
                        )
                    if capture["processing_status"] != "COMPLETE":
                        raise ValueError("recording processing is not complete")
                    recorded = any(
                        self._recording_evidence_in(connection, session_id, ref["shot_id"])[
                            "media_status"
                        ]
                        == "RECORDED"
                        for ref in recording["shots"]
                    )
                    if recorded and media_jobs is None:
                        raise ValueError("recording encoding is not ready")
                    if media_jobs is not None:
                        from billiards_ai.lab.mainline1_recording_media import (
                            project_recording_playback,
                        )

                        playback = project_recording_playback(self, media_jobs, manifest)
                        if any(
                            row.playback_clip is None and not row.terminal_failure
                            for row in playback
                        ):
                            raise ValueError("recording encoding is not ready")
            available = connection.execute(
                "SELECT * FROM field_postgame_clip_availability WHERE session_id=?",
                (session_id,),
            ).fetchall()
            finalized = connection.execute(
                "SELECT 1 FROM field_postgame_finalizations WHERE session_id=?", (session_id,)
            ).fetchone()
            ready = not manifest.expected_jobs or (
                finalized is not None
                and len(available) == len(manifest.expected_jobs)
                and {r["job_id"] for r in available} == set(manifest.expected_jobs)
            )
            for item in available:
                expected_job = self._validate_postgame_clip_identity(
                    connection, session_id, item["shot_id"], manifest_sha256
                )
                if expected_job != item["job_id"]:
                    raise ValueError("postgame review is not ready")
            if ordinary or not ready:
                connection.execute(
                    "INSERT INTO field_recording_cleanup "
                    "(session_id,manifest_sha256,status,next_attempt_at_utc) "
                    "VALUES(?,?,'pending',?)",
                    (session_id, manifest_sha256, _timestamp(self._now_utc())),
                )
            else:
                connection.execute(
                    "INSERT INTO field_local_media_discards "
                    "(session_id,shot_id,postgame_manifest_sha256,job_id,artifact_id,clip_id,"
                    "local_path,content_sha256,byte_size,content_type,status,attempts,"
                    "next_attempt_at_utc) "
                    "SELECT a.*, 'pending',0,? FROM field_postgame_clip_availability a "
                    "WHERE a.session_id=? AND NOT EXISTS "
                    "(SELECT 1 FROM field_clip_save_requests r "
                    "WHERE r.session_id=a.session_id AND r.shot_id=a.shot_id)",
                    (_timestamp(self._now_utc()), session_id),
                )
            connection.execute(
                "UPDATE field_postgame_reviews SET status='completed' WHERE session_id=?",
                (session_id,),
            )
            connection.execute(
                "UPDATE field_postgame_reviews SET status='active' WHERE queue_id="
                "(SELECT MIN(queue_id) FROM field_postgame_reviews "
                "WHERE artifact_id=? AND status='pending')",
                (artifact_id,),
            )

    def get_postgame_manifest(self, session_id: str) -> PostgameSessionManifest | None:
        """Read the end-transaction snapshot; open sessions deliberately have none."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT ended_at_utc, deadline_at_utc, expected_jobs_json "
                "FROM field_postgame_manifests WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return PostgameSessionManifest(
            session_id=session_id,
            ended_at_utc=datetime.fromisoformat(row["ended_at_utc"].replace("Z", "+00:00")),
            deadline_at_utc=datetime.fromisoformat(row["deadline_at_utc"].replace("Z", "+00:00")),
            expected_jobs=tuple(json.loads(row["expected_jobs_json"])),
        )

    @staticmethod
    def _postgame_manifest_in(
        connection: sqlite3.Connection, session_id: str
    ) -> PostgameSessionManifest:
        row = connection.execute(
            "SELECT m.* FROM field_postgame_manifests m JOIN field_sessions s "
            "USING(session_id) WHERE session_id=? AND s.status='ended'",
            (session_id,),
        ).fetchone()
        if row is None:
            raise ValueError("postgame clips require an ended session manifest")
        return PostgameSessionManifest(
            session_id,
            datetime.fromisoformat(row["ended_at_utc"].replace("Z", "+00:00")),
            datetime.fromisoformat(row["deadline_at_utc"].replace("Z", "+00:00")),
            tuple(json.loads(row["expected_jobs_json"])),
        )

    @classmethod
    def _validate_postgame_clip_identity(
        cls, connection: sqlite3.Connection, session_id: str, shot_id: str, digest: str
    ) -> str:
        manifest = cls._postgame_manifest_in(connection, session_id)
        if digest != manifest.manifest_sha256:
            raise ValueError("stale postgame manifest digest")
        shot = connection.execute(
            "SELECT payload_json FROM field_shots WHERE session_id=? AND shot_id=?",
            (session_id, shot_id),
        ).fetchone()
        if shot is None:
            raise ValueError("postgame shot does not exist")
        payload = json.loads(shot["payload_json"])
        job_id = payload.get("live_job_id")
        if (
            job_id not in manifest.expected_jobs
            or payload.get("settled_ms") is None
            or payload["media"]["source_kind"] != "usb_live"
        ):
            raise ValueError("postgame shot is outside the frozen job manifest")
        return job_id

    @staticmethod
    def _private_clip_ownership_in(
        connection: sqlite3.Connection, session_id: str, shot_id: str
    ) -> ShotOwnershipReceipt:
        try:
            shot_row = connection.execute(
                "SELECT payload_json FROM field_shots WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchone()
            ownership_row = connection.execute(
                "SELECT payload_json FROM field_shot_ownership WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchone()
            media_row = connection.execute(
                "SELECT payload_json FROM field_live_shot_media WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchone()
            if shot_row is None or ownership_row is None or media_row is None:
                raise ValueError("shot ownership is unavailable")
            return validate_shot_ownership(
                ownership_receipt_from_payload(json.loads(ownership_row[0])),
                _shot_from_payload(shot_row[0]),
                LiveShotMediaReference(**json.loads(media_row[0])),
            )
        except (TypeError, KeyError, AttributeError, ValueError) as exc:
            raise ValueError("shot ownership is unavailable") from exc

    @staticmethod
    def _validate_save_ownership(receipt: ClipSaveReceipt, ownership: ShotOwnershipReceipt) -> None:
        if (
            receipt.schema_version != "2.0"
            or receipt.owner_user_id != ownership.actor_user_id
            or receipt.actor_role != ownership.actor_role
            or receipt.ownership_receipt_sha256 != ownership_receipt_sha256(ownership)
        ):
            raise ValueError("clip save ownership conflicts with original request")

    @staticmethod
    def _enqueue_selected_clip(
        connection: sqlite3.Connection, session_id: str, shot_id: str, now: str
    ) -> None:
        # The two independent durable facts meet here, under the caller's write lock.
        SQLiteFieldRecordStore._reject_owner_deleted_in(connection, session_id, shot_id)
        connection.execute(
            "INSERT OR IGNORE INTO field_saved_media_outbox "
            "(clip_id,save_request_id,status,attempts,next_attempt_at_utc) "
            "SELECT a.clip_id,r.request_id,'pending',0,? "
            "FROM field_postgame_clip_availability a "
            "JOIN field_clip_save_requests r ON r.session_id=a.session_id "
            "AND r.shot_id=a.shot_id "
            "AND r.postgame_manifest_sha256=a.postgame_manifest_sha256 "
            "WHERE a.session_id=? AND a.shot_id=? AND r.rowid=("
            "SELECT MIN(original.rowid) FROM field_clip_save_requests original "
            "WHERE original.session_id=a.session_id AND original.shot_id=a.shot_id "
            "AND original.postgame_manifest_sha256=a.postgame_manifest_sha256)",
            (now, session_id, shot_id),
        )

    def save_postgame_clip(
        self, *, request_id: str, session_id: str, shot_id: str, postgame_manifest_sha256: str
    ) -> ClipSaveReceipt:
        """Persist screen intent; exact request replay returns the original receipt."""
        validate_save_request_id(request_id)
        if self.get_session_policy(session_id) == "recording-v1":
            raise ValueError("shot ownership is unavailable")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM field_clip_save_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if existing is not None and (
                existing["session_id"],
                existing["shot_id"],
                existing["postgame_manifest_sha256"],
            ) != (session_id, shot_id, postgame_manifest_sha256):
                raise ValueError("clip save request payload conflicts with original request")
            self._reject_owner_deleted_in(connection, session_id, shot_id)
            self._validate_postgame_clip_identity(
                connection, session_id, shot_id, postgame_manifest_sha256
            )
            if existing is not None:
                receipt = clip_save_receipt_from_json(existing["receipt_json"])
                if receipt.schema_version == "2.0":
                    ownership = self._private_clip_ownership_in(connection, session_id, shot_id)
                    self._validate_save_ownership(receipt, ownership)
                return receipt
            ownership = self._private_clip_ownership_in(connection, session_id, shot_id)
            for previous in connection.execute(
                "SELECT receipt_json FROM field_clip_save_requests "
                "WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchall():
                self._validate_save_ownership(clip_save_receipt_from_json(previous[0]), ownership)
            self._reject_discarded_in(connection, session_id, shot_id)
            blocked = connection.execute(
                "SELECT 1 FROM field_saved_media_outbox o "
                "JOIN field_postgame_clip_availability a USING(clip_id) "
                "WHERE a.session_id=? AND a.shot_id=? AND o.status='blocked'",
                (session_id, shot_id),
            ).fetchone()
            if blocked is not None:
                raise ValueError("saved clip is blocked without a recovery contract")
            available = connection.execute(
                "SELECT clip_id FROM field_postgame_clip_availability "
                "WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchone()
            receipt = ClipSaveReceipt(
                request_id,
                session_id,
                shot_id,
                postgame_manifest_sha256,
                ClipSaveStatus.SAVED if available is not None else ClipSaveStatus.SAVE_PENDING,
                available["clip_id"] if available is not None else None,
                schema_version="2.0",
                owner_user_id=ownership.actor_user_id,
                actor_role=ownership.actor_role,
                ownership_receipt_sha256=ownership_receipt_sha256(ownership),
            )
            connection.execute(
                "INSERT INTO field_clip_save_requests VALUES(?,?,?,?,?)",
                (
                    request_id,
                    session_id,
                    shot_id,
                    postgame_manifest_sha256,
                    _canonical(asdict(receipt)),
                ),
            )
            self._enqueue_selected_clip(
                connection, session_id, shot_id, _timestamp(self._now_utc())
            )
            return receipt

    def record_postgame_clip_available(
        self,
        *,
        session_id: str,
        shot_id: str,
        postgame_manifest_sha256: str,
        job_id: str,
        artifact_id: str,
        local_path: Path,
        content_sha256: str,
        byte_size: int,
        content_type: str = "video/mp4",
    ) -> None:
        """Attach immutable local output independently of a user's save selection."""
        if (
            not isinstance(artifact_id, str)
            or len(artifact_id) != 64
            or any(c not in "0123456789abcdef" for c in artifact_id)
            or not isinstance(local_path, Path)
            or not local_path.is_absolute()
            or not isinstance(content_sha256, str)
            or len(content_sha256) != 64
            or any(c not in "0123456789abcdef" for c in content_sha256)
            or type(byte_size) is not int
            or byte_size <= 0
            or byte_size > MAX_SAVED_CLIP_BYTES
            or content_type != "video/mp4"
        ):
            raise ValueError("postgame clip media metadata is invalid")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._reject_discarded_in(connection, session_id, shot_id)
            self._reject_owner_deleted_in(connection, session_id, shot_id)
            expected_job = self._validate_postgame_clip_identity(
                connection, session_id, shot_id, postgame_manifest_sha256
            )
            if expected_job != job_id:
                raise ValueError("postgame clip job conflicts with frozen shot lineage")
            for selected in connection.execute(
                "SELECT receipt_json FROM field_clip_save_requests "
                "WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchall():
                receipt = clip_save_receipt_from_json(selected[0])
                if receipt.schema_version == "2.0":
                    ownership = self._private_clip_ownership_in(connection, session_id, shot_id)
                    self._validate_save_ownership(receipt, ownership)
            values = (
                session_id,
                shot_id,
                postgame_manifest_sha256,
                job_id,
                artifact_id,
                session_scoped_clip_id(session_id, shot_id, job_id, artifact_id),
                str(local_path),
                content_sha256,
                byte_size,
                content_type,
            )
            existing = connection.execute(
                "SELECT * FROM field_postgame_clip_availability WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchone()
            if existing is not None and tuple(existing) != values:
                original = connection.execute(
                    "SELECT 1 FROM field_original_clip_sources WHERE session_id=? AND shot_id=?",
                    (session_id, shot_id),
                ).fetchone()
                if original is not None:
                    return
                raise ValueError("postgame clip availability conflicts with original output")
            if existing is None:
                connection.execute(
                    "INSERT INTO field_postgame_clip_availability VALUES(?,?,?,?,?,?,?,?,?,?)",
                    values,
                )
            self._enqueue_selected_clip(
                connection, session_id, shot_id, _timestamp(self._now_utc())
            )

    def record_selected_original_clip_available(self, media_job, source_job, artifact):
        """Publish verified original bytes only for an existing trusted selection.

        V2 saved descriptors bind the original frozen shot/job and authenticated
        kiosk media assertion; they do not assert successful semantic analysis.
        The typed source retained here lets local erasure verify that distinction.
        """
        from billiards_ai.lab.mainline1_recording_media import verify_recording_playback_artifact
        from billiards_ai.lab.mainline1_shot_jobs import LegacyFrozenJobMediaSourceV1

        source = media_job.source
        if not isinstance(source, LegacyFrozenJobMediaSourceV1):
            raise ValueError("original private media requires original trusted shot source")
        if (
            source_job.job_id != source.source_job_id
            or source_job.session_id != media_job.session_id
            or source_job.shot_id != media_job.shot_id
            or source_job.source_epoch_id != media_job.source_epoch_id
            or source_job.start_frame_id != media_job.start_frame_id
            or source_job.end_frame_id != media_job.end_frame_id
            or source_job.media_root != media_job.media_root
            or source_job.media_sha256 != media_job.media_sha256
            or sha256(_canonical(source_job.identity()).encode()).hexdigest()
            != source.source_job_payload_sha256
        ):
            raise ValueError("original selected media source conflicts")
        root = self.artifact_path_for_session(media_job.session_id)
        if artifact.path != root / "postgame" / media_job.session_id / artifact.artifact_id:
            raise ValueError("original selected media root conflicts")
        verified = verify_recording_playback_artifact(
            artifact.path, media_job, validate_video=False
        )
        if verified != artifact:
            raise ValueError("original selected media changed")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            expected = self._validate_postgame_clip_identity(
                connection,
                media_job.session_id,
                media_job.shot_id,
                source.postgame_manifest_sha256,
            )
            if expected != source_job.job_id:
                raise ValueError("original selected job conflicts with frozen shot")
            self._reject_discarded_in(connection, media_job.session_id, media_job.shot_id)
            self._reject_owner_deleted_in(connection, media_job.session_id, media_job.shot_id)
            selections = connection.execute(
                "SELECT receipt_json FROM field_clip_save_requests "
                "WHERE session_id=? AND shot_id=?",
                (media_job.session_id, media_job.shot_id),
            ).fetchall()
            if not selections:
                return
            ownership = self._private_clip_ownership_in(
                connection,
                media_job.session_id,
                media_job.shot_id,
            )
            for row in selections:
                self._validate_save_ownership(clip_save_receipt_from_json(row[0]), ownership)
            if connection.execute(
                "SELECT 1 FROM field_postgame_clip_availability WHERE session_id=? AND shot_id=?",
                (media_job.session_id, media_job.shot_id),
            ).fetchone():
                return
            artifact_id = artifact.artifact_id.removeprefix("recording-playback-")
            if len(artifact_id) != 64 or any(c not in "0123456789abcdef" for c in artifact_id):
                raise ValueError("original selected artifact identity is invalid")
            clip_id = session_scoped_clip_id(
                media_job.session_id,
                media_job.shot_id,
                source_job.job_id,
                artifact_id,
            )
            connection.execute(
                "INSERT INTO field_postgame_clip_availability VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    media_job.session_id,
                    media_job.shot_id,
                    source.postgame_manifest_sha256,
                    source_job.job_id,
                    artifact_id,
                    clip_id,
                    str(artifact.media_path),
                    artifact.content_sha256,
                    artifact.byte_size,
                    "video/mp4",
                ),
            )
            raw_job = _canonical(media_job.identity())
            connection.execute(
                "INSERT INTO field_original_clip_sources VALUES(?,?,?,?,?)",
                (
                    media_job.session_id,
                    media_job.shot_id,
                    raw_job,
                    sha256(raw_job.encode()).hexdigest(),
                    artifact.manifest_sha256,
                ),
            )
            self._enqueue_selected_clip(
                connection,
                media_job.session_id,
                media_job.shot_id,
                _timestamp(self._now_utc()),
            )

    @staticmethod
    def _reject_owner_deleted_in(connection, session_id: str, shot_id: str) -> None:
        if (
            connection.execute(
                "SELECT 1 FROM field_owner_media_deletions WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchone()
            is not None
        ):
            raise ValueError(_OWNER_DELETE_MARKER)

    @staticmethod
    def _reject_owner_clip_in(connection, clip_id: str) -> None:
        if (
            connection.execute(
                "SELECT 1 FROM field_owner_media_deletions WHERE clip_id=?", (clip_id,)
            ).fetchone()
            is not None
        ):
            raise ValueError(_OWNER_DELETE_MARKER)

    @staticmethod
    def _owner_json(value: object) -> str:
        return json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @staticmethod
    def _owner_delete_from_row(row: sqlite3.Row) -> OwnerDeleteLocalState:
        task = OwnerDeleteDeviceTask.from_payload(json.loads(row["task_json"]))
        if (
            task.canonical_json() != row["task_json"]
            or task.task_sha256 != row["task_sha256"]
            or task.clip_id != row["clip_id"]
            or task.deletion_job_id != row["deletion_job_id"]
            or task.session_id != row["session_id"]
            or task.shot_id != row["shot_id"]
            or not Path(row["local_path"]).is_absolute()
        ):
            raise ValueError("persisted owner deletion identity conflicts")
        return OwnerDeleteLocalState(
            task,
            row["task_sha256"],
            Path(row["local_path"]),
            row["status"],
            row["attempts"],
            row["claim_token"],
            row["lease_until_utc"],
            row["progress_json"],
            row["receipt_json"],
            row["receipt_sha256"],
        )

    def _validate_owner_delete_authority_in(self, connection, task, values):
        """Revalidate retained accepted authority without opening another connection."""
        binding = connection.execute(
            "SELECT * FROM field_managed_open_requests WHERE session_id=? "
            "AND environment_id=? AND device_id=? AND table_id=?",
            (task.session_id, *values),
        ).fetchone()
        if binding is None:
            raise ValueError("owner deletion requires retained managed binding")
        request = self._managed_request_from_row(binding)
        self._retired_local_session_in(connection, request)
        expected_job = self._validate_postgame_clip_identity(
            connection, task.session_id, task.shot_id, task.postgame_manifest_sha256
        )
        self._reject_discarded_in(connection, task.session_id, task.shot_id)
        selected = connection.execute(
            "SELECT a.*,o.*,s.payload_json AS shot_payload_json,"
            "r.receipt_json AS save_receipt_json,"
            "w.payload_json AS ownership_json,m.payload_json AS live_media_json "
            "FROM field_saved_media_outbox o "
            "JOIN field_postgame_clip_availability a USING(clip_id) "
            "JOIN field_clip_save_requests r ON r.request_id=o.save_request_id "
            "JOIN field_shots s ON s.session_id=a.session_id AND s.shot_id=a.shot_id "
            "LEFT JOIN field_shot_ownership w "
            "ON w.session_id=a.session_id AND w.shot_id=a.shot_id "
            "LEFT JOIN field_live_shot_media m "
            "ON m.session_id=a.session_id AND m.shot_id=a.shot_id "
            "WHERE o.clip_id=?",
            (task.clip_id,),
        ).fetchone()
        if selected is None:
            raise ValueError("owner deletion requires original saved selection")
        for name in (
            "shot_payload_json",
            "save_receipt_json",
            "ownership_json",
            "live_media_json",
        ):
            raw = selected[name]
            if not isinstance(raw, str) or len(raw.encode("utf-8")) > 1024 * 1024:
                raise ValueError("owner deletion metadata is unavailable or oversized")
        item = self._saved_media_from_row(selected, connection=connection)
        descriptor = json.loads(item.descriptor_json)
        first = connection.execute(
            "SELECT request_id FROM field_clip_save_requests WHERE session_id=? "
            "AND shot_id=? AND postgame_manifest_sha256=? ORDER BY rowid LIMIT 1",
            (task.session_id, task.shot_id, task.postgame_manifest_sha256),
        ).fetchone()
        if first[0] != task.save_request_id or expected_job != item.job_id:
            raise ValueError("owner deletion selection conflicts")
        for name in (
            "clip_id",
            "session_id",
            "shot_id",
            "artifact_id",
            "owner_user_id",
            "actor_role",
            "content_sha256",
            "byte_size",
            "ownership_receipt_sha256",
            "save_request_id",
            "save_receipt_sha256",
            "postgame_manifest_sha256",
        ):
            if getattr(task, name) != descriptor[name]:
                raise ValueError("owner deletion immutable descriptor conflicts")
        if task.source_job_id != item.job_id:
            raise ValueError("owner deletion job conflicts")
        selection = self._owner_json(
            {
                "descriptor": descriptor,
                "authority_sha256": {
                    name: sha256(selected[name].encode("utf-8")).hexdigest()
                    for name in (
                        "shot_payload_json",
                        "save_receipt_json",
                        "ownership_json",
                        "live_media_json",
                    )
                },
            }
        )
        if len(selection.encode("utf-8")) > 1024 * 1024:
            raise ValueError("owner deletion selection metadata limit")
        return selection, item

    def accept_owner_delete_task(
        self, task: OwnerDeleteDeviceTask, *, managed_scope: ManagedSessionScope
    ) -> OwnerDeleteLocalState:
        """Trusted internal receiver, after authenticated upstream cloud erasure.

        This local admission validates identity, not cloud erasure proof. There is
        deliberately no public receiver or task issuer in this implementation.
        """
        if not isinstance(task, OwnerDeleteDeviceTask):
            raise ValueError("invalid owner delete task")
        # Revalidate direct dataclass construction as well as parsed deliveries.
        task = OwnerDeleteDeviceTask.from_payload(asdict(task))
        values = self._managed_scope_values(managed_scope)
        if values != (task.environment_id, task.device_id, task.table_id):
            raise ValueError("owner deletion scope conflicts")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                selection, item = self._validate_owner_delete_authority_in(connection, task, values)
                existing = connection.execute(
                    "SELECT * FROM field_owner_media_deletions WHERE clip_id=?", (task.clip_id,)
                ).fetchone()
                if existing is not None:
                    state = self._owner_delete_from_row(existing)
                    if (
                        state.task_sha256 != task.task_sha256
                        or state.task != task
                        or existing["selection_json"] != selection
                        or state.local_path != item.local_path
                        or item.status != "blocked"
                        or item.last_error_code != _OWNER_DELETE_MARKER
                        or item.claim_token is not None
                        or item.lease_until_utc is not None
                    ):
                        raise ValueError("owner deletion replay conflicts with original authority")
                    return state
                if item.status not in {"pending", "leased", "acknowledged"}:
                    raise ValueError("owner deletion saved selection is blocked")
                now = _timestamp(self._now_utc())
                connection.execute(
                    "INSERT INTO field_owner_media_deletions "
                    "(clip_id,deletion_job_id,session_id,shot_id,task_json,task_sha256,local_path,"
                    "selection_json,status,attempts,next_attempt_at_utc,created_at_utc) "
                    "VALUES(?,?,?,?,?,?,?,?,'pending',0,?,?)",
                    (
                        task.clip_id,
                        task.deletion_job_id,
                        task.session_id,
                        task.shot_id,
                        task.canonical_json(),
                        task.task_sha256,
                        str(item.local_path),
                        selection,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE field_saved_media_outbox SET status='blocked',claim_token=NULL,"
                    "lease_until_utc=NULL,last_error_code=? WHERE clip_id=?",
                    (_OWNER_DELETE_MARKER, task.clip_id),
                )
                return self._owner_delete_from_row(
                    connection.execute(
                        "SELECT * FROM field_owner_media_deletions WHERE clip_id=?", (task.clip_id,)
                    ).fetchone()
                )
        except (sqlite3.DatabaseError, TypeError, KeyError, AttributeError, IndexError) as exc:
            raise ValueError("owner deletion authority is invalid") from exc

    def get_owner_delete(self, clip_id: str) -> OwnerDeleteLocalState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM field_owner_media_deletions WHERE clip_id=?", (clip_id,)
            ).fetchone()
            return None if row is None else self._owner_delete_from_row(row)

    def owner_delete_for_shot(self, session_id: str, shot_id: str) -> OwnerDeleteLocalState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM field_owner_media_deletions WHERE session_id=? AND shot_id=?",
                (session_id, shot_id),
            ).fetchone()
            return None if row is None else self._owner_delete_from_row(row)

    def list_pending_owner_delete_session_ids(
        self, *, managed_scope: ManagedSessionScope
    ) -> tuple[str, ...]:
        values = self._managed_scope_values(managed_scope)
        with self._connect() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT d.session_id FROM field_owner_media_deletions d "
                    "JOIN field_managed_open_requests m ON m.session_id=d.session_id "
                    "WHERE m.environment_id=? AND m.device_id=? AND m.table_id=? "
                    "AND m.phase='materialized' AND d.status IN ('pending','leased','retry') "
                    "ORDER BY d.session_id",
                    values,
                )
            )

    def claim_due_owner_delete(
        self,
        worker_id: str,
        now_utc: datetime,
        *,
        lease_seconds: int = 30,
        session_ids: tuple[str, ...] | None = None,
    ) -> OwnerDeleteLocalState | None:
        if (
            not isinstance(worker_id, str)
            or not worker_id.strip()
            or len(worker_id.encode("utf-8")) > 256
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in worker_id)
            or type(lease_seconds) is not int
            or lease_seconds <= 0
        ):
            raise ValueError("owner deletion lease is invalid")
        selection = ""
        selected = ()
        if session_ids is not None:
            if not isinstance(session_ids, tuple) or any(
                not isinstance(s, str) or not s for s in session_ids
            ):
                raise ValueError("owner deletion session selection is invalid")
            if not session_ids:
                return None
            selection = "AND session_id IN (" + ",".join("?" for _ in session_ids) + ") "
            selected = session_ids
        now = _timestamp(now_utc)
        until = _timestamp(now_utc + timedelta(seconds=lease_seconds))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                # Selection contains fixed SQL and placeholders; session IDs stay bound.
                "SELECT * FROM field_owner_media_deletions WHERE "  # nosec B608
                "((status IN ('pending','retry') AND next_attempt_at_utc<=?) "
                "OR (status='leased' AND lease_until_utc<=?)) "
                + selection
                + "ORDER BY next_attempt_at_utc,session_id,shot_id LIMIT 1",
                (now, now, *selected),
            ).fetchone()
            if row is None:
                return None
            self._owner_delete_from_row(row)
            connection.execute(
                "UPDATE field_owner_media_deletions SET status='leased',attempts=attempts+1,"
                "claim_token=?,lease_until_utc=?,last_error_code=NULL WHERE clip_id=?",
                (f"{worker_id}:{uuid.uuid4().hex}", until, row["clip_id"]),
            )
            return self._owner_delete_from_row(
                connection.execute(
                    "SELECT * FROM field_owner_media_deletions WHERE clip_id=?", (row["clip_id"],)
                ).fetchone()
            )

    @classmethod
    def _owner_progress(cls, claim: OwnerDeleteLocalState, raw: str, *, terminal=False) -> str:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 1024 * 1024:
            raise ValueError("owner deletion journal metadata limit")
        try:
            value = json.loads(raw)
            progress = cls._owner_json(value)
            if not isinstance(value, dict) or len(progress.encode("utf-8")) > 64 * 1024:
                raise ValueError("owner deletion journal limit")
            if value == {} and not terminal:
                return progress
            objects = value.get("objects")
            if (
                type(value.get("version")) is not int
                or value["version"] != 1
                or value.get("clip_id") != claim.clip_id
                or not isinstance(objects, list)
                or not 1 <= len(objects) <= 64
            ):
                raise ValueError("owner deletion journal identity or object limit")
            paths = set()
            for obj in objects:
                if not isinstance(obj, dict):
                    raise ValueError("owner deletion journal object")
                size = obj.get("size")
                identity = obj.get("identity")
                state = obj.get("state")
                if (
                    type(size) is not int
                    or not 0 <= size <= 8 * 1024**3
                    or type(obj.get("reclaimed_bytes")) is not int
                    or obj["reclaimed_bytes"] not in {0, size}
                    or state not in {"planned", "detached", "unlinking", "deleted", "absent"}
                    or state != "deleted"
                    and obj["reclaimed_bytes"] != 0
                    or terminal
                    and state not in {"deleted", "absent"}
                    or not isinstance(identity, list)
                    or len(identity) != 4
                    or any(type(v) is not int for v in identity)
                    or identity[2] != size
                    or not identity[1]
                    or obj.get("root") not in {"artifacts", "jobs"}
                ):
                    raise ValueError("owner deletion journal object state")
                for name in ("path", "staged_path"):
                    path = obj.get(name)
                    if not isinstance(path, str) or not Path(path).is_absolute() or path in paths:
                        raise ValueError("owner deletion journal object path")
                    paths.add(path)
                digest = obj.get("sha256")
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError("owner deletion journal digest")
            return progress
        except (TypeError, KeyError, RecursionError) as exc:
            raise ValueError("owner deletion journal is invalid") from exc

    @staticmethod
    def _owner_claim_identity(claim: OwnerDeleteLocalState) -> tuple[str, str, str, str] | None:
        if (
            not isinstance(claim, OwnerDeleteLocalState)
            or claim.status != "leased"
            or not claim.claim_token
            or claim.task.task_sha256 != claim.task_sha256
        ):
            return None
        return (claim.clip_id, claim.task_sha256, claim.task.canonical_json(), claim.claim_token)

    def update_owner_delete_progress(
        self, claim: OwnerDeleteLocalState, progress_json: str
    ) -> bool:
        identity = self._owner_claim_identity(claim)
        if identity is None:
            return False
        progress = self._owner_progress(claim, progress_json)
        with self._connect() as connection:
            return (
                connection.execute(
                    "UPDATE field_owner_media_deletions SET progress_json=? WHERE clip_id=? "
                    "AND task_sha256=? AND task_json=? AND claim_token=? AND status='leased'",
                    (progress, *identity),
                ).rowcount
                == 1
            )

    def _finish_owner_delete(
        self,
        claim: OwnerDeleteLocalState,
        status: str,
        error_code: str | None,
        next_attempt_at_utc: datetime | None = None,
        *,
        progress_json: str | None = None,
    ) -> bool:
        identity = self._owner_claim_identity(claim)
        if identity is None:
            return False
        if status != "completed" and (
            not isinstance(error_code, str)
            or not error_code.strip()
            or len(error_code.encode("utf-8")) > 256
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in error_code)
        ):
            raise ValueError("owner deletion error code is invalid")
        progress = (
            None
            if progress_json is None
            else self._owner_progress(claim, progress_json, terminal=status == "completed")
        )
        if status == "completed" and progress is None:
            raise ValueError("owner deletion completion requires terminal journal")
        receipt = None
        if status == "completed":
            receipt = self._owner_json(
                dict(
                    contract="local_owner_erase_receipt_v1",
                    deletion_job_id=claim.task.deletion_job_id,
                    clip_id=claim.clip_id,
                    task_sha256=claim.task_sha256,
                    cloud_erase_receipt_sha256=claim.task.cloud_erase_receipt_sha256,
                    journal_sha256=sha256(progress.encode("utf-8")).hexdigest(),
                    reclaimed_bytes=sum(
                        obj["reclaimed_bytes"] for obj in json.loads(progress)["objects"]
                    ),
                )
            )
        with self._connect() as connection:
            changed = (
                connection.execute(
                    "UPDATE field_owner_media_deletions SET status=?,claim_token=NULL,"
                    "lease_until_utc=NULL,last_error_code=?,next_attempt_at_utc=?,"
                    "progress_json=COALESCE(?,progress_json),receipt_json=?,receipt_sha256=? "
                    "WHERE clip_id=? AND task_sha256=? AND task_json=? AND claim_token=? "
                    "AND status='leased'",
                    (
                        status,
                        error_code,
                        _timestamp(next_attempt_at_utc or self._now_utc()),
                        progress,
                        receipt,
                        None if receipt is None else sha256(receipt.encode("utf-8")).hexdigest(),
                        *identity,
                    ),
                ).rowcount
                == 1
            )
            if changed and status == "completed":
                from billiards_ai.edge.route_assistance.owner_device_delivery import (
                    enqueue_owner_completion_in,
                )

                enqueue_owner_completion_in(self, connection, claim.clip_id)
            return changed

    def retry_owner_delete(
        self,
        claim: OwnerDeleteLocalState,
        error_code: str,
        next_attempt_at_utc: datetime,
        *,
        progress_json: str | None = None,
    ) -> bool:
        return self._finish_owner_delete(
            claim, "retry", error_code, next_attempt_at_utc, progress_json=progress_json
        )

    def block_owner_delete(
        self,
        claim: OwnerDeleteLocalState,
        error_code: str,
        *,
        progress_json: str | None = None,
    ) -> bool:
        return self._finish_owner_delete(claim, "blocked", error_code, progress_json=progress_json)

    def complete_owner_delete(self, claim: OwnerDeleteLocalState, *, progress_json: str) -> bool:
        """Internal pinned-executor port; journal terminal states are physical evidence."""
        return self._finish_owner_delete(claim, "completed", None, progress_json=progress_json)

    def is_saved_media_claim_current(self, clip_id: str, claim_token: str) -> bool:
        with self._connect() as connection:
            return (
                bool(claim_token)
                and connection.execute(
                    "SELECT 1 FROM field_saved_media_outbox o WHERE clip_id=? AND claim_token=? "
                    "AND status='leased' AND lease_until_utc>? AND NOT EXISTS "
                    "(SELECT 1 FROM field_owner_media_deletions d WHERE d.clip_id=o.clip_id)",
                    (clip_id, claim_token, _timestamp(self._now_utc())),
                ).fetchone()
                is not None
            )

    @staticmethod
    def _reject_discarded_in(connection, session_id: str, shot_id: str) -> None:
        if SQLiteFieldRecordStore._is_postgame_discarded_in(connection, session_id, shot_id):
            raise ValueError("postgame clip has been discarded")

    @staticmethod
    def _is_postgame_discarded_in(connection, session_id: str, shot_id: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM field_local_media_discards WHERE session_id=? AND shot_id=? "
                "UNION ALL SELECT 1 FROM field_postgame_reviews r "
                "JOIN field_recording_cleanup c USING(session_id) "
                "JOIN field_shots s USING(session_id) WHERE r.session_id=? AND s.shot_id=? "
                "AND r.status='completed' AND NOT EXISTS "
                "(SELECT 1 FROM field_clip_save_requests v WHERE v.session_id=r.session_id "
                "AND v.shot_id=s.shot_id) LIMIT 1",
                (session_id, shot_id, session_id, shot_id),
            ).fetchone()
            is not None
        )

    def is_postgame_discarded(self, session_id: str, shot_id: str) -> bool:
        with self._connect() as connection:
            return self._is_postgame_discarded_in(connection, session_id, shot_id)

    def is_postgame_work_fenced(self, session_id: str) -> bool:
        """Prove all frozen work revoked before a supervisor reads model resources."""
        with self._connect() as connection:
            session = connection.execute(
                "SELECT s.status,m.expected_jobs_json,r.status AS review_status "
                "FROM field_sessions s LEFT JOIN field_postgame_manifests m USING(session_id) "
                "LEFT JOIN field_postgame_reviews r USING(session_id) WHERE s.session_id=?",
                (session_id,),
            ).fetchone()
            if (
                session is None
                or session["status"] != "ended"
                or session["expected_jobs_json"] is None
            ):
                return False
            expected = set(json.loads(session["expected_jobs_json"]))
            if not expected:
                return (
                    session["review_status"] == "completed"
                    and connection.execute(
                        "SELECT 1 FROM field_recording_cleanup WHERE session_id=?",
                        (session_id,),
                    ).fetchone()
                    is not None
                )
            heads = {
                shot.live_job_id: shot.shot_id
                for row in connection.execute(
                    "SELECT payload_json FROM field_shots WHERE session_id=?",
                    (session_id,),
                ).fetchall()
                if (shot := _shot_from_payload(row[0])).live_job_id in expected
            }
            if set(heads) != expected:
                return False
            return all(
                self._is_postgame_discarded_in(connection, session_id, shot_id)
                or connection.execute(
                    "SELECT 1 FROM field_owner_media_deletions WHERE session_id=? AND shot_id=?",
                    (session_id, shot_id),
                ).fetchone()
                is not None
                for shot_id in heads.values()
            )

    def list_pending_local_discard_session_ids(self) -> tuple[str, ...]:
        """Inspect retained work for recovery diagnostics without claiming it."""
        with self._connect() as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT session_id FROM field_local_media_discards "
                    "WHERE status IN ('pending','leased') ORDER BY session_id"
                )
            )

    def claim_due_local_discard(
        self,
        worker_id: str,
        now_utc: datetime,
        *,
        lease_seconds: float = 30,
        session_ids: tuple[str, ...] | None = None,
        exclude_managed: bool = False,
        artifact_root: Path | None = None,
    ) -> LocalDiscardClaim | None:
        if not worker_id or not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("local discard lease is invalid")
        now = _timestamp(now_utc)
        until = _timestamp(now_utc + timedelta(seconds=lease_seconds))
        selection = ""
        selected = ()
        if session_ids is not None:
            if not isinstance(session_ids, tuple) or any(
                not isinstance(s, str) or not s for s in session_ids
            ):
                raise ValueError("local discard session selection is invalid")
            if not session_ids:
                return None
            selection = "AND session_id IN (" + ",".join("?" for _ in session_ids) + ") "
            selected = session_ids
        if exclude_managed:
            selection += (
                "AND session_id NOT LIKE 'managed-%' AND NOT EXISTS "
                "(SELECT 1 FROM field_managed_open_requests m "
                "WHERE m.session_id=field_local_media_discards.session_id) "
            )
        if artifact_root is not None:
            if not isinstance(artifact_root, Path):
                raise ValueError("local discard artifact root is invalid")
            resolved_root = artifact_root.resolve()
            if not resolved_root.is_dir():
                raise ValueError("local discard artifact root is invalid")
            selection += (
                "AND EXISTS (SELECT 1 FROM field_session_artifacts a "
                "WHERE a.session_id=field_local_media_discards.session_id "
                "AND a.artifact_path=?) "
            )
            selected = (*selected, str(resolved_root))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                # Only fixed predicates/placeholders are concatenated; values stay bound.
                "SELECT * FROM field_local_media_discards WHERE "  # nosec B608
                "((status='pending' AND next_attempt_at_utc<=?) OR "
                "(status='leased' AND lease_until_utc<=?)) "
                + selection
                + "ORDER BY next_attempt_at_utc,session_id,shot_id LIMIT 1",
                (now, now, *selected),
            ).fetchone()
            if row is None:
                return None
            token = f"{worker_id}:{uuid.uuid4().hex}"
            connection.execute(
                "UPDATE field_local_media_discards SET status='leased',claim_token=?,"
                "lease_until_utc=?,attempts=attempts+1 WHERE session_id=? AND shot_id=?",
                (token, until, row["session_id"], row["shot_id"]),
            )
            return LocalDiscardClaim(
                row["session_id"],
                row["shot_id"],
                row["postgame_manifest_sha256"],
                row["job_id"],
                row["artifact_id"],
                row["clip_id"],
                Path(row["local_path"]),
                row["content_sha256"],
                row["byte_size"],
                row["content_type"],
                row["attempts"] + 1,
                token,
                until,
                row["progress_json"],
            )

    def update_local_discard_progress(self, claim: LocalDiscardClaim, progress_json: str) -> bool:
        """Checkpoint object recovery under the caller's exclusive session pin."""
        progress = _canonical(json.loads(progress_json))
        if not isinstance(json.loads(progress), dict) or len(progress.encode("utf-8")) > 64 * 1024:
            raise ValueError("local discard progress is invalid")
        with self._connect() as connection:
            return (
                connection.execute(
                    "UPDATE field_local_media_discards SET progress_json=? "
                    "WHERE session_id=? AND shot_id=? AND status='leased' AND claim_token=?",
                    (progress, claim.session_id, claim.shot_id, claim.claim_token),
                ).rowcount
                == 1
            )

    def _finish_local_discard(
        self,
        claim: LocalDiscardClaim,
        status: str,
        error_code: str | None,
        next_attempt_at_utc: datetime | None = None,
        *,
        progress_json: str | None = None,
    ) -> bool:
        progress = None if progress_json is None else _canonical(json.loads(progress_json))
        if progress is not None and (
            not isinstance(json.loads(progress), dict) or len(progress.encode("utf-8")) > 64 * 1024
        ):
            raise ValueError("local discard progress is invalid")
        with self._connect() as connection:
            return (
                connection.execute(
                    "UPDATE field_local_media_discards SET status=?,claim_token=NULL,"
                    "lease_until_utc=NULL,next_attempt_at_utc=?,last_error_code=?,"
                    "progress_json=COALESCE(?,progress_json) WHERE session_id=? AND shot_id=? "
                    "AND status='leased' AND claim_token=?",
                    (
                        status,
                        _timestamp(next_attempt_at_utc or self._now_utc()),
                        error_code,
                        progress,
                        claim.session_id,
                        claim.shot_id,
                        claim.claim_token,
                    ),
                ).rowcount
                == 1
            )

    def complete_local_discard(
        self, claim: LocalDiscardClaim, *, progress_json: str | None = None
    ) -> bool:
        return self._finish_local_discard(claim, "completed", None, progress_json=progress_json)

    def retry_local_discard(
        self,
        claim: LocalDiscardClaim,
        error_code: str,
        next_attempt_at_utc: datetime,
        *,
        progress_json: str | None = None,
    ) -> bool:
        return self._finish_local_discard(
            claim, "pending", error_code, next_attempt_at_utc, progress_json=progress_json
        )

    def block_local_discard(
        self, claim: LocalDiscardClaim, error_code: str, *, progress_json: str | None = None
    ) -> bool:
        return self._finish_local_discard(claim, "blocked", error_code, progress_json=progress_json)

    def list_postgame_clips(self, session_id: str) -> tuple[PostgameClip, ...]:
        """Project review rows, including processing shots and unselected local outputs."""
        with self._connect() as connection:
            manifest = self._postgame_manifest_in(connection, session_id)
            result = []
            for row in connection.execute(
                "SELECT s.shot_id,s.payload_json,a.*,o.status AS outbox_status,o.last_error_code,"
                "d.status AS owner_status,d.receipt_json AS owner_receipt,"
                "d.receipt_sha256 AS owner_receipt_sha256,"
                "EXISTS(SELECT 1 FROM field_clip_save_requests r WHERE r.session_id=s.session_id "
                "AND r.shot_id=s.shot_id) AS selected "
                "FROM field_shots s LEFT JOIN field_postgame_clip_availability a "
                "ON a.session_id=s.session_id AND a.shot_id=s.shot_id "
                "LEFT JOIN field_saved_media_outbox o ON o.clip_id=a.clip_id "
                "LEFT JOIN field_owner_media_deletions d "
                "ON d.session_id=s.session_id AND d.shot_id=s.shot_id "
                "WHERE s.session_id=? ORDER BY s.sequence",
                (session_id,),
            ).fetchall():
                job_id = (
                    row["job_id"]
                    if row["outbox_status"] == "blocked"
                    else json.loads(row["payload_json"]).get("live_job_id")
                )
                if job_id not in manifest.expected_jobs:
                    continue
                status = ClipSaveStatus.NOT_SAVED
                if row["selected"]:
                    if row["outbox_status"] is None:
                        status = ClipSaveStatus.SAVE_PENDING
                    elif row["outbox_status"] == "blocked":
                        status = ClipSaveStatus.SAVE_BLOCKED
                    elif row["last_error_code"] is not None:
                        status = ClipSaveStatus.SAVE_FAILED_RETRYING
                    else:
                        status = ClipSaveStatus.SAVED
                try:
                    ownership = self._private_clip_ownership_in(
                        connection, session_id, row["shot_id"]
                    )
                except ValueError:
                    ownership = None
                result.append(
                    PostgameClip(
                        session_id,
                        row["shot_id"],
                        manifest.manifest_sha256,
                        job_id,
                        status,
                        row["clip_id"],
                        row["artifact_id"],
                        Path(row["local_path"])
                        if row["local_path"] is not None and row["owner_status"] is None
                        else None,
                        row["content_sha256"],
                        row["byte_size"],
                        row["content_type"],
                        sequence=json.loads(row["payload_json"])["sequence"],
                        actor_role=ownership.actor_role if ownership else None,
                        owner_user_id=ownership.actor_user_id if ownership else None,
                        can_save=ownership is not None and row["owner_status"] is None,
                        ownership_unavailable_reason=None
                        if ownership
                        else "SHOT_OWNERSHIP_UNAVAILABLE",
                        owner_delete_status=(
                            "completed"
                            if row["owner_status"] == "completed"
                            and row["owner_receipt"] is not None
                            and row["owner_receipt_sha256"] is not None
                            else "pending"
                        )
                        if row["owner_status"] is not None
                        else None,
                    )
                )
            return tuple(result)

    def _validate_original_clip_provenance_in(self, connection, row):
        """Bind selected original metadata to its retained typed output and source."""
        original = connection.execute(
            "SELECT * FROM field_original_clip_sources WHERE session_id=? AND shot_id=?",
            (row["session_id"], row["shot_id"]),
        ).fetchone()
        path = Path(row["local_path"])
        is_original = path.name == "original.mp4" or path.parent.name.startswith(
            "recording-playback-"
        )
        if original is None and not is_original:
            return
        try:
            from billiards_ai.lab.mainline1_recording_media import recording_playback_inventory
            from billiards_ai.lab.mainline1_shot_jobs import (
                LegacyFrozenJobMediaSourceV1,
                _media_job_from_payload,
            )

            if original is None or not is_original:
                raise ValueError("original provenance missing")
            raw = original["media_job_json"]
            if sha256(raw.encode()).hexdigest() != original["media_job_sha256"]:
                raise ValueError("original provenance changed")
            job = _media_job_from_payload(json.loads(raw))
            if not isinstance(job.source, LegacyFrozenJobMediaSourceV1):
                raise ValueError("original source kind changed")
            media = LiveShotMediaReference(**json.loads(row["live_media_json"]))
            if (
                job.session_id != row["session_id"]
                or job.shot_id != row["shot_id"]
                or job.source.source_job_id != row["job_id"]
                or job.source.postgame_manifest_sha256 != row["postgame_manifest_sha256"]
                or job.media_root != Path(media.media_path)
                or job.source_epoch_id != media.source_epoch_id
                or job.media_sha256 != media.media_sha256
                or path
                != self.artifact_path_for_session(job.session_id)
                / "postgame"
                / job.session_id
                / ("recording-playback-" + row["artifact_id"])
                / "original.mp4"
            ):
                raise ValueError("original source conflicts")
            artifact = recording_playback_inventory(path.parent, job)
            if (
                artifact.media_path != path
                or artifact.content_sha256 != row["content_sha256"]
                or artifact.byte_size != row["byte_size"]
                or artifact.manifest_sha256 != original["artifact_manifest_sha256"]
            ):
                raise ValueError("original output changed")
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            raise ValueError("persisted original clip provenance is invalid") from exc

    def _saved_media_from_row(self, row: sqlite3.Row, *, connection=None) -> SavedMediaOutboxItem:
        if connection is None:
            with self._connect() as original_connection:
                return self._saved_media_from_row(row, connection=original_connection)
        self._validate_original_clip_provenance_in(connection, row)
        try:
            shot = _shot_from_payload(row["shot_payload_json"])
            save_receipt = clip_save_receipt_from_json(row["save_receipt_json"])
            ownership = validate_shot_ownership(
                ownership_receipt_from_payload(json.loads(row["ownership_json"])),
                shot,
                LiveShotMediaReference(**json.loads(row["live_media_json"])),
            )
            SQLiteFieldRecordStore._validate_save_ownership(save_receipt, ownership)
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise ValueError("persisted saved media ownership is invalid") from exc
        if (
            shot.live_job_id != row["job_id"]
            or shot.media.source_kind != "usb_live"
            or shot.settled_ms is None
            or shot.session_id != row["session_id"]
            or shot.shot_id != row["shot_id"]
            or _canonical(asdict(save_receipt)) != row["save_receipt_json"]
            or save_receipt.request_id != row["save_request_id"]
            or save_receipt.session_id != row["session_id"]
            or save_receipt.shot_id != row["shot_id"]
            or save_receipt.postgame_manifest_sha256 != row["postgame_manifest_sha256"]
            or (
                save_receipt.status is ClipSaveStatus.SAVED
                and save_receipt.clip_id != row["clip_id"]
            )
        ):
            raise ValueError("persisted saved media lineage is invalid")
        return SavedMediaOutboxItem(
            clip_id=row["clip_id"],
            session_id=row["session_id"],
            shot_id=row["shot_id"],
            postgame_manifest_sha256=row["postgame_manifest_sha256"],
            job_id=row["job_id"],
            artifact_id=row["artifact_id"],
            local_path=Path(row["local_path"]),
            content_sha256=row["content_sha256"],
            byte_size=row["byte_size"],
            content_type=row["content_type"],
            source_kind=shot.media.source_kind,
            start_ms=shot.media.start_ms,
            end_ms=shot.media.end_ms,
            duration_ms=shot.media.duration_ms,
            save_request_id=row["save_request_id"],
            save_receipt_sha256=sha256(row["save_receipt_json"].encode("utf-8")).hexdigest(),
            status=row["status"],
            attempts=row["attempts"],
            claim_token=row["claim_token"],
            lease_until_utc=row["lease_until_utc"],
            next_attempt_at_utc=row["next_attempt_at_utc"],
            last_error_code=row["last_error_code"],
            owner_user_id=save_receipt.owner_user_id,
            actor_role=save_receipt.actor_role,
            ownership_receipt_sha256=save_receipt.ownership_receipt_sha256,
            ownership_receipt=ownership,
        )

    def get_saved_media_outbox(self, clip_id: str) -> SavedMediaOutboxItem | None:
        """Read retained selected metadata without requiring a live saved-media lease."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT a.*,o.*,s.payload_json AS shot_payload_json,"
                "r.receipt_json AS save_receipt_json,"
                "w.payload_json AS ownership_json,m.payload_json AS live_media_json "
                "FROM field_saved_media_outbox o "
                "JOIN field_postgame_clip_availability a USING(clip_id) "
                "JOIN field_clip_save_requests r ON r.request_id=o.save_request_id "
                "JOIN field_shots s ON s.session_id=a.session_id AND s.shot_id=a.shot_id "
                "LEFT JOIN field_shot_ownership w "
                "ON w.session_id=a.session_id AND w.shot_id=a.shot_id "
                "LEFT JOIN field_live_shot_media m "
                "ON m.session_id=a.session_id AND m.shot_id=a.shot_id "
                "WHERE o.clip_id=?",
                (clip_id,),
            ).fetchone()
            return None if row is None else self._saved_media_from_row(row, connection=connection)

    def list_saved_media_outbox(self) -> tuple[SavedMediaOutboxItem, ...]:
        with self._connect() as connection:
            return tuple(
                self._saved_media_from_row(row, connection=connection)
                for row in connection.execute(
                    "SELECT a.*,o.*,s.payload_json AS shot_payload_json,"
                    "r.receipt_json AS save_receipt_json,"
                    "w.payload_json AS ownership_json,m.payload_json AS live_media_json "
                    "FROM field_saved_media_outbox o "
                    "JOIN field_postgame_clip_availability a USING(clip_id) "
                    "JOIN field_clip_save_requests r ON r.request_id=o.save_request_id "
                    "JOIN field_shots s ON s.session_id=a.session_id AND s.shot_id=a.shot_id "
                    "LEFT JOIN field_shot_ownership w "
                    "ON w.session_id=a.session_id AND w.shot_id=a.shot_id "
                    "LEFT JOIN field_live_shot_media m "
                    "ON m.session_id=a.session_id AND m.shot_id=a.shot_id "
                    "ORDER BY clip_id"
                ).fetchall()
            )

    def claim_due_saved_media(
        self,
        now: datetime,
        *,
        lease_seconds: int = 60,
        managed_scope: ManagedSessionScope | None = None,
        exclude_managed: bool = False,
    ) -> SavedMediaOutboxItem | None:
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("saved media lease must be positive")
        now_text = _timestamp(now)
        selection = ""
        values = ()
        if managed_scope is not None:
            values = self._managed_scope_values(managed_scope)
            selection = (
                "AND EXISTS (SELECT 1 FROM field_managed_open_requests m "
                "JOIN field_postgame_clip_availability a ON a.session_id=m.session_id "
                "WHERE a.clip_id=o.clip_id AND m.environment_id=? AND m.device_id=? "
                "AND m.table_id=?) "
            )
        if exclude_managed:
            selection += (
                "AND EXISTS (SELECT 1 FROM field_postgame_clip_availability a "
                "WHERE a.clip_id=o.clip_id AND a.session_id NOT LIKE 'managed-%' "
                "AND NOT EXISTS (SELECT 1 FROM field_managed_open_requests m "
                "WHERE m.session_id=a.session_id)) "
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            while True:
                row = connection.execute(
                    # Selection uses fixed predicates/placeholders; scope values stay bound.
                    "SELECT clip_id FROM field_saved_media_outbox o "  # nosec B608
                    "WHERE ((status='pending' AND next_attempt_at_utc<=?) "
                    "OR (status='leased' AND lease_until_utc<=?)) "
                    "AND NOT EXISTS (SELECT 1 FROM field_owner_media_deletions d "
                    "WHERE d.clip_id=o.clip_id) " + selection + "ORDER BY clip_id LIMIT 1",
                    (now_text, now_text, *values),
                ).fetchone()
                if row is None:
                    return None
                if managed_scope is not None:
                    request_row = connection.execute(
                        "SELECT m.* FROM field_managed_open_requests m "
                        "JOIN field_postgame_clip_availability a ON a.session_id=m.session_id "
                        "WHERE a.clip_id=? AND m.environment_id=? AND m.device_id=? "
                        "AND m.table_id=?",
                        (row["clip_id"], *values),
                    ).fetchone()
                    request = self._managed_request_from_row(request_row)
                    if request.phase != "materialized":
                        raise ValueError("saved media requires retained managed binding")
                    self._retired_local_session_in(connection, request)
                claim_token = uuid.uuid4().hex
                connection.execute(
                    "UPDATE field_saved_media_outbox SET status='leased',attempts=attempts+1,"
                    "claim_token=?,lease_until_utc=?,last_error_code=NULL WHERE clip_id=?",
                    (
                        claim_token,
                        _timestamp(now + timedelta(seconds=lease_seconds)),
                        row["clip_id"],
                    ),
                )
                claimed = connection.execute(
                    "SELECT a.*,o.*,s.payload_json AS shot_payload_json,"
                    "r.receipt_json AS save_receipt_json,"
                    "w.payload_json AS ownership_json,m.payload_json AS live_media_json "
                    "FROM field_saved_media_outbox o "
                    "LEFT JOIN field_postgame_clip_availability a USING(clip_id) "
                    "LEFT JOIN field_clip_save_requests r ON r.request_id=o.save_request_id "
                    "LEFT JOIN field_shots s "
                    "ON s.session_id=a.session_id AND s.shot_id=a.shot_id "
                    "LEFT JOIN field_shot_ownership w "
                    "ON w.session_id=a.session_id AND w.shot_id=a.shot_id "
                    "LEFT JOIN field_live_shot_media m "
                    "ON m.session_id=a.session_id AND m.shot_id=a.shot_id "
                    "WHERE o.clip_id=?",
                    (row["clip_id"],),
                ).fetchone()
                try:
                    return self._saved_media_from_row(claimed, connection=connection)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError):
                    connection.execute(
                        "UPDATE field_saved_media_outbox SET status='blocked',"
                        "lease_until_utc=NULL,last_error_code=? "
                        "WHERE clip_id=? AND status='leased' AND claim_token=?",
                        (
                            _PERSISTED_SAVED_MEDIA_INVALID_ERROR,
                            row["clip_id"],
                            claim_token,
                        ),
                    )

    def acknowledge_saved_media(self, clip_id: str, claim_token: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._reject_owner_clip_in(connection, clip_id)
            row = connection.execute(
                "SELECT * FROM field_saved_media_outbox WHERE clip_id=?", (clip_id,)
            ).fetchone()
            if row is None or not claim_token or row["claim_token"] != claim_token:
                raise ValueError("saved media lease owner is stale or missing")
            if row["status"] == "acknowledged":
                return
            if row["status"] != "leased":
                raise ValueError("saved media lease is not active")
            if row["lease_until_utc"] <= _timestamp(self._now_utc()):
                raise ValueError("saved media lease has expired")
            connection.execute(
                "UPDATE field_saved_media_outbox SET status='acknowledged',lease_until_utc=NULL,"
                "last_error_code=NULL WHERE clip_id=?",
                (clip_id,),
            )

    def retry_saved_media(
        self,
        clip_id: str,
        claim_token: str,
        *,
        error_code: str,
        next_attempt_at_utc: datetime | None = None,
    ) -> None:
        if not isinstance(error_code, str) or not error_code.strip():
            raise ValueError("saved media retry error must be non-empty")
        retry_at = _timestamp(next_attempt_at_utc or self._now_utc())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._reject_owner_clip_in(connection, clip_id)
            row = connection.execute(
                "SELECT * FROM field_saved_media_outbox WHERE clip_id=?", (clip_id,)
            ).fetchone()
            if row is None or not claim_token or row["claim_token"] != claim_token:
                raise ValueError("saved media lease owner is stale or missing")
            if (
                row["status"] == "pending"
                and row["last_error_code"] == error_code
                and (next_attempt_at_utc is None or row["next_attempt_at_utc"] == retry_at)
            ):
                return
            if row["status"] != "leased":
                raise ValueError("saved media lease is not active or retry payload conflicts")
            if row["lease_until_utc"] <= _timestamp(self._now_utc()):
                raise ValueError("saved media lease has expired")
            connection.execute(
                "UPDATE field_saved_media_outbox SET status='pending',lease_until_utc=NULL,"
                "next_attempt_at_utc=?,last_error_code=? WHERE clip_id=?",
                (retry_at, error_code, clip_id),
            )

    def block_saved_media(self, clip_id: str, claim_token: str, *, error_code: str) -> None:
        """Persist one permanent attempt outcome under the active lease owner."""
        if not isinstance(error_code, str) or not error_code.strip():
            raise ValueError("saved media blocked error must be non-empty")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._reject_owner_clip_in(connection, clip_id)
            row = connection.execute(
                "SELECT * FROM field_saved_media_outbox WHERE clip_id=?", (clip_id,)
            ).fetchone()
            if row is None or not claim_token or row["claim_token"] != claim_token:
                raise ValueError("saved media lease owner is stale or missing")
            if row["status"] == "blocked":
                if row["last_error_code"] != error_code:
                    raise ValueError("saved media blocked replay conflicts with original error")
                return
            if row["status"] != "leased":
                raise ValueError("saved media lease is not active")
            if row["lease_until_utc"] <= _timestamp(self._now_utc()):
                raise ValueError("saved media lease has expired")
            connection.execute(
                "UPDATE field_saved_media_outbox SET status='blocked',lease_until_utc=NULL,"
                "last_error_code=? WHERE clip_id=?",
                (error_code, clip_id),
            )

    def record_postgame_finalization(self, session_id: str, manifest_json: str) -> None:
        digest = sha256(manifest_json.encode()).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT manifest_sha256, manifest_json FROM field_postgame_finalizations "
                "WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["manifest_sha256"] != digest
                    or existing["manifest_json"] != manifest_json
                ):
                    raise ValueError("postgame finalization cannot be replaced")
                return
            connection.execute(
                "INSERT INTO field_postgame_finalizations VALUES(?,?,?)",
                (session_id, digest, manifest_json),
            )
            connection.execute(
                "INSERT INTO local_postgame_outbox VALUES(?,?,?,?)",
                (session_id, digest, manifest_json, "pending"),
            )

    def is_postgame_finalized(self, session_id: str) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM field_postgame_finalizations WHERE session_id=?", (session_id,)
                ).fetchone()
                is not None
            )

    def _sync_job_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> SyncOutboxJob:
        media_rows = connection.execute(
            "SELECT * FROM field_sync_media WHERE sync_job_id=? ORDER BY clip_id",
            (row["sync_job_id"],),
        ).fetchall()
        return SyncOutboxJob(
            sync_job_id=row["sync_job_id"],
            session_id=row["session_id"],
            manifest_hash=row["manifest_hash"],
            manifest_json=row["manifest_json"],
            status=row["status"],
            retry_count=int(row["retry_count"]),
            next_attempt_at_utc=row["next_attempt_at_utc"],
            last_error_code=row["last_error_code"],
            media=tuple(
                SyncMediaItem(
                    clip_id=item["clip_id"],
                    shot_id=item["shot_id"],
                    local_path=Path(item["local_path"]),
                    content_type=item["content_type"],
                    content_sha256=item["content_sha256"],
                    byte_size=int(item["byte_size"]),
                    status=item["status"],
                )
                for item in media_rows
            ),
        )

    def enqueue_sync_job(
        self,
        *,
        session_id: str,
        manifest_hash: str,
        manifest_json: str,
        media: tuple[SyncMediaItem, ...] = (),
    ) -> SyncOutboxJob:
        if len(manifest_hash) != 64 or any(
            character not in "0123456789abcdef" for character in manifest_hash
        ):
            raise ValueError("manifest hash must be lowercase SHA-256")
        try:
            parsed = json.loads(manifest_json)
        except json.JSONDecodeError as exc:
            raise ValueError("sync manifest must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("sync manifest must be a JSON object")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                session = connection.execute(
                    "SELECT status FROM field_sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                if session is None or session["status"] != FieldSessionStatus.ENDED.value:
                    raise ValueError("sync outbox requires an ended field session")
                existing = connection.execute(
                    "SELECT * FROM field_sync_jobs WHERE session_id=?", (session_id,)
                ).fetchone()
                if existing is not None:
                    if (
                        existing["manifest_hash"] != manifest_hash
                        or existing["manifest_json"] != manifest_json
                    ):
                        raise ValueError("conflicting sync manifest cannot replace outbox state")
                    return self._sync_job_from_row(connection, existing)
                job_id = self._sync_job_id_factory().strip()
                if not job_id:
                    raise ValueError("sync job ID factory returned an empty value")
                now = _timestamp(self._now_utc())
                connection.execute(
                    "INSERT INTO field_sync_jobs VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        job_id,
                        session_id,
                        manifest_hash,
                        manifest_json,
                        "pending",
                        0,
                        now,
                        None,
                        now,
                        now,
                    ),
                )
                for item in media:
                    if not item.local_path.is_absolute():
                        raise ValueError("sync media path must be absolute")
                    connection.execute(
                        "INSERT INTO field_sync_media VALUES(?,?,?,?,?,?,?,?)",
                        (
                            job_id,
                            item.clip_id,
                            item.shot_id,
                            str(item.local_path),
                            item.content_type,
                            item.content_sha256,
                            item.byte_size,
                            item.status,
                        ),
                    )
                row = connection.execute(
                    "SELECT * FROM field_sync_jobs WHERE sync_job_id=?", (job_id,)
                ).fetchone()
                assert row is not None
                return self._sync_job_from_row(connection, row)
        except sqlite3.IntegrityError as exc:
            raise ValueError("sync outbox state conflicts with persisted facts") from exc
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def load_sync_job(self, sync_job_id: str) -> SyncOutboxJob:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM field_sync_jobs WHERE sync_job_id=?", (sync_job_id,)
                ).fetchone()
                if row is None:
                    raise ValueError("sync job does not exist")
                return self._sync_job_from_row(connection, row)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def claim_due_sync_job(self, now: datetime) -> SyncOutboxJob | None:
        return self._claim_due_sync_job(now, session_id=None)

    def claim_due_sync_job_for_session(
        self, session_id: str, now: datetime
    ) -> SyncOutboxJob | None:
        """Claim only this ended session, so one worker cannot drain another session."""
        return self._claim_due_sync_job(now, session_id=session_id)

    def _claim_due_sync_job(self, now: datetime, *, session_id: str | None) -> SyncOutboxJob | None:
        now_text = _timestamp(now)
        stale_claim = _timestamp(now - timedelta(minutes=5))
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                session_clause = "" if session_id is None else "session_id=? AND "
                parameters: tuple[str, ...] = (
                    (now_text, stale_claim)
                    if session_id is None
                    else (session_id, now_text, stale_claim)
                )
                row = connection.execute(
                    # Session clause is a fixed literal; session/time values stay bound.
                    "SELECT * FROM field_sync_jobs "  # nosec B608
                    f"WHERE {session_clause}"
                    "((status IN ('pending','partial') AND next_attempt_at_utc<=?) "
                    "OR (status='claimed' AND updated_at_utc<=?) "
                    ") ORDER BY next_attempt_at_utc, created_at_utc LIMIT 1",
                    parameters,
                ).fetchone()
                if row is None:
                    return None
                connection.execute(
                    "UPDATE field_sync_jobs SET status='claimed', updated_at_utc=? "
                    "WHERE sync_job_id=?",
                    (now_text, row["sync_job_id"]),
                )
                claimed = connection.execute(
                    "SELECT * FROM field_sync_jobs WHERE sync_job_id=?",
                    (row["sync_job_id"],),
                ).fetchone()
                assert claimed is not None
                return self._sync_job_from_row(connection, claimed)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def mark_sync_partial(
        self,
        sync_job_id: str,
        *,
        error_code: str,
        next_attempt_at: datetime,
    ) -> SyncOutboxJob:
        _require_error = error_code.strip()
        if not _require_error or any(ord(character) < 32 for character in error_code):
            raise ValueError("sync error code is invalid")
        now = _timestamp(self._now_utc())
        due = _timestamp(next_attempt_at)
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE field_sync_jobs SET status='partial', retry_count=retry_count+1, "
                    "next_attempt_at_utc=?, last_error_code=?, updated_at_utc=? "
                    "WHERE sync_job_id=? AND status='claimed'",
                    (due, error_code, now, sync_job_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("only a claimed sync job can become partial")
            return self.load_sync_job(sync_job_id)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def mark_sync_complete(self, sync_job_id: str) -> SyncOutboxJob:
        now = _timestamp(self._now_utc())
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE field_sync_jobs SET status='complete', last_error_code=NULL, "
                    "updated_at_utc=? WHERE sync_job_id=? AND status='claimed'",
                    (now, sync_job_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("only a claimed sync job can complete")
            return self.load_sync_job(sync_job_id)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc

    def mark_sync_blocked(self, sync_job_id: str, *, error_code: str) -> SyncOutboxJob:
        if not error_code.strip() or any(ord(character) < 32 for character in error_code):
            raise ValueError("sync error code is invalid")
        now = _timestamp(self._now_utc())
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE field_sync_jobs SET status='blocked', last_error_code=?, "
                    "updated_at_utc=? WHERE sync_job_id=? AND status='claimed'",
                    (error_code, now, sync_job_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("only a claimed sync job can become blocked")
            return self.load_sync_job(sync_job_id)
        except sqlite3.DatabaseError as exc:
            raise ValueError("field record database is invalid") from exc
