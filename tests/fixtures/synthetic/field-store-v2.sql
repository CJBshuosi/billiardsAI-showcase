-- Synthetic v2 database; DDL frozen from published commit c30e9c8.
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
                PRAGMA user_version=2;
INSERT INTO field_sessions VALUES('session-1','artifact-1','open','2026-08-26T00:00:00Z',NULL);
INSERT INTO field_match_state VALUES('session-1','{"red_user_id":null,"blue_user_id":null,"breaker":null,"active_role":null,"red_group":null,"blue_group":null,"revision":0}',0);
