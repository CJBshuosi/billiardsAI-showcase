-- Authentic fresh field schema v7; test-only, no runtime Git dependency.
-- Source commit: daa7bc1b9ae53d19290f31c21dda6ef610074bc1
-- Source path: src/billiards_ai/edge/route_assistance/persistence.py
-- Source Git blob: 35cf2f5f0a55a565c4d6a8a09fb28be702141896
-- Source SHA-256: 24b56ac1abe1073cce9868b7fd618f10b2ae233b0a47253e61933937ba059c80
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


CREATE TABLE field_shot_ownership (
  session_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY(session_id, shot_id),
  FOREIGN KEY(session_id, shot_id) REFERENCES field_shots(session_id, shot_id)
);


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


CREATE TABLE field_session_artifacts (
  session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
  artifact_path TEXT NOT NULL
);
CREATE TABLE field_cloud_close_intents (
  session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
  status TEXT NOT NULL CHECK(status IN ('open','pending','complete')),
  updated_at_utc TEXT NOT NULL
);

PRAGMA user_version=7;
