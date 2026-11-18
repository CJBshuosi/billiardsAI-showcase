const crypto = require('crypto')
const { SESSION_FIELDS, SHOT_FIELDS, contentDigest, validateSessionFacts, pickFacts } = require('./session-facts')
const { timestamp } = require('./retention-policy')

const CONTRACT = 'shot_projection_v1'
const HASH = /^[0-9a-f]{64}$/u
const ID = /^[A-Za-z0-9._-]{1,128}$/u
const JOB_ID = /^shot-projection-[0-9a-f]{64}$/u
const CURSOR_FIELDS = ['contract', 'job_id', 'session_id', 'table_id', 'device_id', 'publication_sha256',
  'source_session_sha256', 'expected_shot_count', 'next_sequence', 'status', 'blocked_reason', 'created_at', 'updated_at']
const IMMUTABLE_FIELDS = ['contract', 'job_id', 'session_id', 'table_id', 'device_id', 'publication_sha256',
  'source_session_sha256', 'expected_shot_count']
const POINTER_FIELDS = ['contract', 'session_id', 'publication_sha256', 'source_sha256']
const digestId = value => crypto.createHash('sha256').update(value).digest('hex')
const jobIdentity = sessionId => `shot-projection-${digestId(sessionId)}`
const validString = (pattern, value) => typeof value === 'string' && pattern.test(value)

class ShotProjectionError extends Error {
  constructor(code) { super(code); this.name = 'ShotProjectionError'; this.code = code }
}
const invalid = () => { throw new ShotProjectionError('SHOT_PROJECTION_INVALID') }
function exactRecord(value, fields, allowId = false) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || (Object.hasOwn(value, '_id') && (!allowId || typeof value._id !== 'string'))
      || fields.some(key => !Object.hasOwn(value, key))
      || Object.keys(value).some(key => !fields.includes(key) && !(allowId && key === '_id'))) invalid()
}
function validTime(value) { try { timestamp(value); return true } catch (_) { return false } }
function validateCursor(cursor, requestedId) {
  exactRecord(cursor, CURSOR_FIELDS, true)
  if (cursor.contract !== CONTRACT || !validString(JOB_ID, cursor.job_id)
      || cursor.job_id !== requestedId || !validString(ID, cursor.session_id)
      || cursor.job_id !== jobIdentity(cursor.session_id)
      || !validString(ID, cursor.table_id) || !validString(ID, cursor.device_id)
      || !validString(HASH, cursor.publication_sha256) || !validString(HASH, cursor.source_session_sha256)
      || !Number.isSafeInteger(cursor.expected_shot_count) || cursor.expected_shot_count < 0
      || !Number.isSafeInteger(cursor.next_sequence) || cursor.next_sequence < 1
      || cursor.next_sequence > cursor.expected_shot_count + 1
      || !validTime(cursor.created_at) || !validTime(cursor.updated_at)
      || timestamp(cursor.updated_at) < timestamp(cursor.created_at)) invalid()
  const done = cursor.next_sequence === cursor.expected_shot_count + 1
  if (!['pending', 'completed', 'blocked'].includes(cursor.status)
      || (cursor.status === 'completed' && !done)
      || (cursor.status === 'pending' && done)
      || (cursor.status === 'blocked'
        ? !['source_unavailable', 'source_conflict', 'row_conflict'].includes(cursor.blocked_reason)
        : cursor.blocked_reason !== null)) invalid()
}
function sourceIdentity(session) {
  if (!session || typeof session !== 'object' || Array.isArray(session)
      || (Object.hasOwn(session, '_id') && typeof session._id !== 'string')) invalid()
  if (!validString(ID, session.session_id)) invalid()
  const facts = pickFacts(session, SESSION_FIELDS)
  try { validateSessionFacts(facts) } catch (_) { invalid() }
  exactRecord(session._sync_publication, ['table_id', 'device_id', 'content_sha256'])
  const fence = session._sync_publication
  if (session.sync_status !== 'published' || !validTime(session.published_at)
      || !validString(ID, fence.table_id) || !validString(ID, fence.device_id)
      || !validString(HASH, fence.content_sha256)) invalid()
  return { contract: CONTRACT, job_id: jobIdentity(facts.session_id), session_id: facts.session_id,
    table_id: fence.table_id, device_id: fence.device_id, publication_sha256: fence.content_sha256,
    source_session_sha256: contentDigest(facts), expected_shot_count: facts.shots.length }
}
async function optional(reference) {
  try { return (await reference.get()).data ?? null } catch (error) {
    if (/document with _id .+ does not exist/iu.test(error?.message || '')) return null
    throw error
  }
}
const equalFields = (a, b, fields) => fields.every(field => a[field] === b[field])

// Called only by the publisher inside its existing authoritative transaction.
async function enqueuePublishedShots(transaction, session) {
  const identity = sourceIdentity(session)
  const ref = transaction.collection('shot_projection_jobs').doc(identity.job_id)
  const existing = await optional(ref)
  if (existing !== null) {
    validateCursor(existing, identity.job_id)
    if (!equalFields(existing, identity, IMMUTABLE_FIELDS)) throw new ShotProjectionError('SHOT_PROJECTION_CONFLICT')
  } else {
    await ref.set({ data: { ...identity, next_sequence: 1,
      status: identity.expected_shot_count ? 'pending' : 'completed', blocked_reason: null,
      created_at: session.published_at, updated_at: session.published_at } })
  }
  return identity.job_id
}

function createCloudShotProjectionRepository(db, { nowIso = () => new Date().toISOString() } = {}) {
  return {
    async findPending() {
      const result = await db.collection('shot_projection_jobs').where({ status: 'pending' })
        .orderBy('updated_at', 'asc').orderBy('job_id', 'asc').limit(1).get()
      return result.data?.[0] || null
    },
    async projectBatch(input) {
      if (!input || typeof input !== 'object' || Array.isArray(input)) invalid()
      const { jobId } = input
      if (!validString(JOB_ID, jobId)) invalid()
      return db.runTransaction(async transaction => {
        const cursorRef = transaction.collection('shot_projection_jobs').doc(jobId)
        const cursor = await optional(cursorRef)
        if (cursor === null) throw new ShotProjectionError('SHOT_PROJECTION_INCOMPLETE')
        validateCursor(cursor, jobId)
        const result = (outcome, count = 0, next = cursor.next_sequence) => ({
          outcome, job_id: jobId, projected_count: count, next_sequence: next,
        })
        if (cursor.status === 'blocked') return result('blocked')
        const now = nowIso()
        if (!validTime(now) || timestamp(now) < timestamp(cursor.created_at)) invalid()
        const block = async reason => {
          await cursorRef.update({ data: { status: 'blocked', blocked_reason: reason, updated_at: now } })
          return result('blocked')
        }
        const source = await optional(transaction.collection('sessions').doc(cursor.session_id))
        if (source === null) return block('source_unavailable')
        let identity
        try { identity = sourceIdentity(source) } catch (error) {
          if (!(error instanceof ShotProjectionError)) throw error
          return block('source_conflict')
        }
        if (!equalFields(cursor, identity, IMMUTABLE_FIELDS) || source._shot_projection_job_id !== jobId) {
          return block('source_conflict')
        }
        if (cursor.status === 'completed') return result('completed')
        const batch = source.shots.slice(cursor.next_sequence - 1, cursor.next_sequence + 15)
        const writes = []
        // All reads and conflicts are resolved before any derived row writes.
        for (const shot of batch) {
          const row = { ...shot, session_id: cursor.session_id }
          const pointer = { contract: CONTRACT, session_id: cursor.session_id,
            publication_sha256: cursor.publication_sha256, source_sha256: contentDigest(row) }
          const ref = transaction.collection('shots').doc(digestId(`${cursor.session_id}:${shot.shot_id}`))
          const existing = await optional(ref)
          if (existing !== null) {
            if (typeof existing !== 'object' || Array.isArray(existing)
                || (Object.hasOwn(existing, '_id') && typeof existing._id !== 'string')
                || contentDigest(pickFacts(existing, [...SHOT_FIELDS, 'session_id'])) !== contentDigest(row)) {
              return block('row_conflict')
            }
            if (Object.hasOwn(existing, '_shot_projection')) {
              try { exactRecord(existing._shot_projection, POINTER_FIELDS) } catch (_) { return block('row_conflict') }
              if (contentDigest(existing._shot_projection) !== contentDigest(pointer)) return block('row_conflict')
            } else writes.push(() => ref.update({ data: { _shot_projection: pointer } }))
          } else writes.push(() => ref.set({ data: { ...row, _shot_projection: pointer } }))
        }
        for (const write of writes) await write()
        const next = cursor.next_sequence + batch.length
        const status = next === cursor.expected_shot_count + 1 ? 'completed' : 'pending'
        await cursorRef.update({ data: { next_sequence: next, status, updated_at: now } })
        return result(status === 'pending' ? 'progress' : 'completed', batch.length, next)
      })
    },
  }
}

module.exports = { enqueuePublishedShots, createCloudShotProjectionRepository, ShotProjectionError }
