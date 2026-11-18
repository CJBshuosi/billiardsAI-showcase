const HOUR_MS = 60 * 60 * 1000
const DAY_MS = 24 * HOUR_MS

const ORDINARY_RETENTION_MS = 48 * HOUR_MS
const FAVORITE_RETENTION_MS = 30 * DAY_MS
const FAVORITE_RELEASE_GRACE_MS = 24 * HOUR_MS
const EVIDENCE_RETENTION_MS = 90 * DAY_MS
const FAVORITE_LIMIT = 20
const SAVED_UNTIL_DELETED_POLICY = 'saved_until_deleted_v1'


function isSavedUntilDeletedClip(clip) {
  return Boolean(clip && clip.retention_policy === SAVED_UNTIL_DELETED_POLICY
    && clip.publication_kind === 'saved_postgame_clip'
    && clip.retention_state === 'active'
    && clip.ordinary_expires_at === null && clip.retention_expires_at === null)
}


function timestamp(value) {
  if (typeof value !== 'string'
      || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/u.test(value)) {
    throw new Error('invalid retention timestamp')
  }
  const milliseconds = Date.parse(value)
  // SQLite facts retain canonical whole seconds. Compare an exact calendar
  // round trip without changing their signed source bytes or saved receipts.
  const roundTrip = value.length === 20 ? value.replace(/Z$/u, '.000Z') : value
  if (!Number.isFinite(milliseconds) || new Date(milliseconds).toISOString() !== roundTrip) {
    throw new Error('invalid retention timestamp')
  }
  return milliseconds
}

function expiry(value, durationMs) {
  return new Date(timestamp(value) + durationMs).toISOString()
}


function ordinaryExpiry(endedAt) {
  return expiry(endedAt, ORDINARY_RETENTION_MS)
}


function favoriteExpiry(now) {
  return expiry(now, FAVORITE_RETENTION_MS)
}


function releaseGraceExpiry(now) {
  return expiry(now, FAVORITE_RELEASE_GRACE_MS)
}


function evidenceExpiry(now) {
  return expiry(now, EVIDENCE_RETENTION_MS)
}


function retentionStatus({ availableUntil, now }) {
  const deadline = timestamp(availableUntil)
  const current = timestamp(now)
  if (deadline <= current) return { state: 'expired', remaining_seconds: 0 }
  return { state: 'active', remaining_seconds: Math.floor((deadline - current) / 1000) }
}


module.exports = {
  EVIDENCE_RETENTION_MS,
  FAVORITE_LIMIT,
  FAVORITE_RELEASE_GRACE_MS,
  FAVORITE_RETENTION_MS,
  ORDINARY_RETENTION_MS,
  SAVED_UNTIL_DELETED_POLICY,
  evidenceExpiry,
  favoriteExpiry,
  ordinaryExpiry,
  isSavedUntilDeletedClip,
  releaseGraceExpiry,
  retentionStatus,
  timestamp,
}
