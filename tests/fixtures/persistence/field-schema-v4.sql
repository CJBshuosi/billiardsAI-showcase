-- Authentic fresh field schema v4; test-only, no runtime Git dependency.
-- Source commit: 1418812a69c6ee4976da3ab561b9a57fd9cc40b6
-- Source path: src/billiards_ai/edge/route_assistance/persistence.py
-- Source Git blob: 9434851b33e8f2ebe3598b5bc3ad96e81d4f3aca
-- Source SHA-256: 510a68cd1155fff60ce00285eb838da13e88056deaf34d5d2ea86cbcc331cc52
-- Origin: fresh initializer SQL plus its referenced literal schema constants.
-- Extracted with AST/literal inspection only; historical Python was never executed.
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


CREATE TABLE field_admin_audit (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES field_sessions(session_id),
  command_type TEXT NOT NULL,
  reason TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  resulting_revision INTEGER NOT NULL CHECK(resulting_revision >= 0),
  created_at_utc TEXT NOT NULL
);

PRAGMA user_version=4;
