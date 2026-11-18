// The original V1 ingest fact allowlists and canonical digest. Shared with the
// derived shot projector so publication and recovery cannot drift.
const crypto = require('crypto')
const { timestamp, ordinaryExpiry } = require('./retention-policy')
const HASH_PATTERN = /^[0-9a-f]{64}$/u
const ID_PATTERN = /^[A-Za-z0-9._-]{1,128}$/u
const SESSION_FIELDS = Object.freeze(['schema_version', 'session_id', 'artifact_id', 'status', 'created_at_utc',
  'ended_at_utc', 'postgame_manifest_sha256', 'evidence_source_kind', 'participants',
  'breaker', 'active_role_at_end', 'shots', 'limitations'])
const SHOT_FIELDS = Object.freeze(['schema_version', 'shot_id', 'sequence', 'actor_role', 'started_ms',
  'settled_ms', 'recommendation_id', 'live_job_id', 'power', 'limitations'])
function onlyKeys(value, keys) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || Object.keys(value).some(key => !keys.includes(key))) throw new Error('invalid session facts')
}

function contentDigest(value) {
  const canonical = item => Array.isArray(item) ? item.map(canonical)
    : item && typeof item === 'object'
      ? Object.fromEntries(Object.keys(item).sort().map(key => [key, canonical(item[key])]))
      : item
  return crypto.createHash('sha256').update(JSON.stringify(canonical(value))).digest('hex')
}

function validateSessionFacts(session, { factsOnlyUsb = false } = {}) {
  if (!ID_PATTERN.test(session?.session_id || '')) throw new Error('invalid session facts')
  onlyKeys(session, SESSION_FIELDS)
  if (!ID_PATTERN.test(session.artifact_id || '') || session.status !== 'ended'
      || (session.postgame_manifest_sha256 != null
        && !HASH_PATTERN.test(session.postgame_manifest_sha256))
      || (factsOnlyUsb && !HASH_PATTERN.test(session.postgame_manifest_sha256 || ''))
      || !['development_replay', 'usb_live'].includes(session.evidence_source_kind)
      || !Array.isArray(session.participants) || session.participants.length > 2
      || !Array.isArray(session.shots) || !Array.isArray(session.limitations)) {
    throw new Error('invalid session facts')
  }
  try {
    if (timestamp(session.ended_at_utc) < timestamp(session.created_at_utc)) {
      throw new Error('session ends before it starts')
    }
    ordinaryExpiry(session.ended_at_utc)
  } catch (_error) {
    throw new Error('invalid session facts')
  }
  const users = new Set()
  const roles = new Set()
  for (const p of session.participants) {
    onlyKeys(p, ['schema_version', 'user_id', 'role', 'group'])
    if (!p || !ID_PATTERN.test(p.user_id || '') || !['red', 'blue'].includes(p.role)
        || users.has(p.user_id) || roles.has(p.role)) throw new Error('invalid session facts')
    users.add(p.user_id)
    roles.add(p.role)
  }
  const shots = new Set()
  for (const [index, shot] of session.shots.entries()) {
    onlyKeys(shot, SHOT_FIELDS)
    if (shot.power != null) onlyKeys(shot.power, ['schema_version', 'band', 'approximate_percent', 'personalized'])
    if (!shot || !ID_PATTERN.test(shot.shot_id || '')
        || (shot.live_job_id != null && !ID_PATTERN.test(shot.live_job_id))
        || (factsOnlyUsb && !ID_PATTERN.test(shot.live_job_id || ''))
        || shots.has(shot.shot_id)
        || shot.sequence !== index + 1) throw new Error('invalid session facts')
    shots.add(shot.shot_id)
  }
  return shots
}

function pickFacts(value, fields) {
  return Object.fromEntries(fields.filter(key => Object.hasOwn(value, key)).map(key => [key, value[key]]))
}

module.exports = { SESSION_FIELDS, SHOT_FIELDS, contentDigest, validateSessionFacts, pickFacts }
